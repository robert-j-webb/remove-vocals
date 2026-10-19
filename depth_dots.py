#!/usr/bin/env python3
"""Image or video -> depth map -> dot (stipple) art.

Closer areas get more dots, and density follows the depth gradient smoothly.
Videos are processed frame by frame with fresh random dots each frame (a
flickering, grainy look), or with --fixed-dots dots that stay put. The first
audio track is kept when ffmpeg is installed.

Everything after decoding runs on the GPU when there is one (CUDA > Apple MPS > CPU):
preprocessing, batched depth inference (fp16 where supported), normalization,
smoothing, detail/shade enhancement, dot sampling and a vectorized antialiased
dot rasterizer. Video frames are decoded and encoded on background threads, and
encoded by piping raw frames straight into ffmpeg.

Setup:  pip install torch transformers pillow numpy opencv-python   (+ ffmpeg for audio)
Usage:  python depth_dots.py photo.jpg  [options]
        python depth_dots.py clip.mp4   [options]  -> clip_dots_base_r518_n60000_g1.5_s1.2_x2.mp4
Without -o the output name lists the settings used; --save-depth side outputs get
<stem>_depth_<model>_r<res> and <stem>_density_<full tag>.
Options: -o OUT  --dots 60000  --gamma 1.5  --size 1.2  --scale 2
         --size-by-depth  --color  --dark  --save-depth  --smooth 0.6  --max-frames N
         --fixed-dots  --model large  --res 1036  --detail 1.5  --shade 0.3
Speed:   --batch 8  --fp32  --device auto|cuda|mps|cpu  --hw-encode  --preset medium
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops run on CPU instead of failing

import argparse, math, queue, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
from PIL import Image

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torch
import torch.nn.functional as F

_CPU_ONLY = set()  # ops that failed on the accelerator and now run on CPU


def pick_device(name=None):
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_oom(e):
    return isinstance(e, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def _safe(key, fn, *args):
    """Run fn on the tensors' device; if the op isn't supported there, redo it on CPU from then on."""
    dev = next((x.device for x in args if isinstance(x, torch.Tensor)), torch.device("cpu"))
    if dev.type != "cpu" and key not in _CPU_ONLY:
        try:
            return fn(*args)
        except (RuntimeError, NotImplementedError) as e:
            if _is_oom(e):
                raise
            _CPU_ONLY.add(key)
            print(f"\n[{key}] not supported on {dev.type} ({str(e).splitlines()[0][:120]}); using CPU for it.")
    out = fn(*[x.cpu() if isinstance(x, torch.Tensor) else x for x in args])
    return out.to(dev) if isinstance(out, torch.Tensor) else out


# ----------------------------------------------------------------------------- depth model
class DepthModel:
    """Depth Anything V2 run directly so we can pick the inference resolution.

    The default pipeline works at ~518 px, which blurs small things like faces.
    A higher --res (multiple of 14) gives much finer depth, at the cost of speed.
    Preprocessing (resize + ImageNet normalize) is done with torch on the device,
    and frames are processed in batches.
    """

    def __init__(self, model, res=518, device=None, fp32=False):
        from transformers import AutoModelForDepthEstimation
        self.torch = torch
        self.dev = pick_device(device)
        name = MODELS.get(model, model)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        try:  # only read the normalization constants; the processor itself isn't used per frame
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(name)
            mean = tuple(getattr(proc, "image_mean", None) or mean)
            std = tuple(getattr(proc, "image_std", None) or std)
        except Exception:
            pass
        self.mean = torch.tensor(mean, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.dev).view(1, 3, 1, 1)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.dev).eval()
        self.res = int(res) // 14 * 14
        self.dtype = torch.float32
        if self.dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if not fp32 and self.dev.type in ("cuda", "mps"):
            self._try_half()
        print(f"Depth model: {name} on {self.dev.type} ({str(self.dtype).replace('torch.', '')})")

    def _try_half(self):
        try:
            self.model.half()
            with torch.inference_mode():
                x = torch.randn(1, 3, 14 * 12, 14 * 16, device=self.dev, dtype=torch.float16)
                d = self.model(pixel_values=x).predicted_depth
                ok = bool(torch.isfinite(d).all())
            if ok:
                self.dtype = torch.float16
                return
            print("fp16 produced NaNs on this device, using fp32.")
        except Exception as e:
            print(f"fp16 not usable on {self.dev.type} ({str(e)[:100]}), using fp32.")
        self.model.float()

    def size_for(self, w, h):
        s = self.res / min(w, h)  # shortest side -> res, keep aspect ratio
        return max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)

    def preprocess(self, frames):
        """frames: (B, h, w, 3) uint8 tensor on device -> normalized (B, 3, H14, W14)."""
        _, h, w, _ = frames.shape
        x = frames.permute(0, 3, 1, 2).float().div_(255)
        size = self.size_for(w, h)
        if size != (h, w):
            try:
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
            except (RuntimeError, NotImplementedError):
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
            x = x.clamp_(0, 1)  # like resizing 8-bit images
        return ((x - self.mean) / self.std).to(self.dtype).contiguous()

    def batch(self, frames):
        """frames: (B, h, w, 3) uint8 tensor -> (B, h, w) float32 depth on device (bigger = closer)."""
        frames = frames.to(self.dev, non_blocking=True)
        _, h, w, _ = frames.shape
        with torch.inference_mode():
            d = self.model(pixel_values=self.preprocess(frames)).predicted_depth
            if d.ndim == 3:
                d = d[:, None]
            return F.interpolate(d.float(), size=(h, w), mode="bicubic", align_corners=False)[:, 0]

    def __call__(self, img):
        """Back-compat: PIL image -> numpy depth."""
        t = torch.from_numpy(np.array(img.convert("RGB")))[None]
        return self.batch(t)[0].cpu().numpy()


# ----------------------------------------------------------------------------- depth processing
def percentile(x, qs):
    """np.percentile (linear) of a tensor, returned as a tensor on x.device (no host sync on GPU)."""
    flat = x.reshape(-1).float()
    if flat.device.type == "cpu":
        return torch.from_numpy(np.percentile(flat.numpy(), qs).astype(np.float32))
    s = flat.sort().values
    n = s.numel()
    pos = torch.tensor(qs, dtype=torch.float32, device=s.device) / 100 * (n - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    f = pos - lo.float()
    return s[lo] * (1 - f) + s[hi] * f


def normalize(d, lo, hi):
    return ((d - lo) / (hi - lo + 1e-8)).clamp(0, 1)


def _blur(a, sh, sw):
    h, w = a.shape
    small = F.interpolate(a[None, None], size=(sh, sw), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0, 0]


def blur(a, radius):
    """Cheap large blur: shrink (box average) then grow (bilinear)."""
    h, w = a.shape
    f = max(1.0, radius)
    return _safe("blur", _blur, a, max(1, int(h / f)), max(1, int(w / f)))


def to_gray(rgb):
    """PIL 'L' conversion of a (h, w, 3) uint8 tensor, exactly."""
    c = rgb.to(torch.int32)
    return ((c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16).float()


def enhance(d01, detail, detail_radius, img=None, shade=0.0):
    """Boost local depth relief (noses, eye sockets, mouths) and optionally mix in image shading.

    img: (h, w, 3) uint8 tensor of the source frame (only needed for shade)."""
    out = d01
    if detail > 0:
        hp = d01 - blur(d01, detail_radius)            # local relief only
        hp = hp / (percentile(hp.abs(), [99])[0].to(hp.device) + 1e-6)
        out = out + detail * 0.25 * hp
    if shade > 0 and img is not None:
        dark = 1 - to_gray(img.to(d01.device)) / 255   # dark features (eyes, brows, lips) get more dots
        out = (1 - shade) * out + shade * dark * (d01 * 1.5).clamp(0, 1)
    return out.clamp(0, 1)


# ----------------------------------------------------------------------------- stipplers
def _sample(p, n):
    """n indices drawn with replacement, P(i) proportional to p[i]."""
    if p.numel() <= 2 ** 24 and "multinomial" not in _CPU_ONLY:
        try:
            return torch.multinomial(p, n, replacement=True)
        except (RuntimeError, NotImplementedError) as e:
            if _is_oom(e):
                raise
            _CPU_ONLY.add("multinomial")
    # inverse CDF (no category limit)
    cdf = p.double().cumsum(0) if p.device.type != "mps" else p.cumsum(0)
    u = torch.rand(n, device=p.device, dtype=cdf.dtype) * cdf[-1]
    return torch.searchsorted(cdf, u, right=True).clamp_(max=p.numel() - 1)


class RandomStippler:
    """Fresh random dots every call: each pixel's chance of a dot follows its closeness.

    On video this re-rolls the dots every frame, giving a flickering, grainy look.
    """

    def __init__(self, w, h, dots, gamma, floor, seed=0, device=None):
        torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators
        self.dots, self.gamma, self.floor = dots, gamma, floor

    def points(self, depth01):
        h, w = depth01.shape
        p = depth01.clamp(0, 1) ** self.gamma
        p = self.floor + (1 - self.floor) * p
        idx = _sample(p.reshape(-1), self.dots)
        iy, ix = idx // w, idx % w
        xs = ix.float() + torch.rand(self.dots, device=p.device)  # jitter inside the pixel
        ys = iy.float() + torch.rand(self.dots, device=p.device)
        return xs, ys, ix, iy


class FixedStippler:
    """Fixed random candidate points, each with a fixed random threshold.

    A point is drawn when its threshold is below the local closeness, so density
    tracks depth while the same dots stay put between frames (no flicker).
    """

    def __init__(self, w, h, dots, gamma, floor, seed=0, oversample=6, device=None):
        g = torch.Generator().manual_seed(seed)   # CPU generator: same dots on every device
        n = int(dots * oversample)
        dev = device or torch.device("cpu")
        self.x = (torch.rand(n, generator=g, dtype=torch.float64) * w).float().to(dev)
        self.y = (torch.rand(n, generator=g, dtype=torch.float64) * h).float().to(dev)
        self.u = torch.rand(n, generator=g).to(dev)
        self.ix = self.x.long().clamp_(max=w - 1)
        self.iy = self.y.long().clamp_(max=h - 1)
        self.dots, self.gamma, self.floor = dots, gamma, floor

    def points(self, depth01):
        w = depth01.clamp(0, 1) ** self.gamma
        w = self.floor + (1 - self.floor) * w
        wc = w[self.iy, self.ix]
        k = torch.minimum(self.dots / wc.sum(), 1.0 / wc.max())  # aim for ~dots total
        keep = self.u < wc * k
        return self.x[keep], self.y[keep], self.ix[keep], self.iy[keep]


def make_stippler(a, w, h, device=None):
    cls = FixedStippler if a.fixed_dots else RandomStippler
    return cls(w, h, a.dots, a.gamma, a.floor, a.seed, device=device)


# ----------------------------------------------------------------------------- rasterizer
def _footprint(cx, cy, r, ar, W, H, R):
    """Pixels around each dot: linear index (n,k,k) (H*W = discard slot) and disc coverage."""
    k = ar.numel()
    px = torch.ceil(cx - 0.5 - R).long()[:, None] + ar    # (n,k)
    py = torch.ceil(cy - 0.5 - R).long()[:, None] + ar
    dx = px.float() + 0.5 - cx[:, None]                    # pixel i spans [i, i+1)
    dy = py.float() + 0.5 - cy[:, None]
    dist = torch.sqrt(dy[:, :, None] ** 2 + dx[:, None, :] ** 2)
    rr = r[:, None, None] if isinstance(r, torch.Tensor) else r
    # PIL's ellipse(cx-r .. cx+r) fills a bit more than pi*r^2 pixels; +0.85 (fading over the
    # last pixel) matches its average ink/footprint while antialiasing the edge
    cov = (rr + 0.85 - dist).clamp_(0, 1)
    valid = ((py >= 0) & (py < H))[:, :, None] & ((px >= 0) & (px < W))[:, None, :]
    lin = py[:, :, None] * W + px[:, None, :]
    lin = torch.where(valid, lin, torch.full_like(lin, H * W))
    cov = torch.where(valid, cov, torch.zeros_like(cov))
    return lin.reshape(-1), cov.reshape(-1)


def _splat(cx, cy, col, r, rmax, W, H, bg, max_samples=1 << 22):
    """Rasterize all dots at once. Each pixel takes the color of the dot covering it most
    (like opaque drawing) blended with the background by that coverage."""
    dev = cx.device
    R = rmax + 1.0                       # footprint bound (coverage is 0 beyond r + 0.85)
    k = int(math.floor(2 * R)) + 1
    ar = torch.arange(k, device=dev)
    HW = H * W
    n = cx.numel()
    per = max(1, max_samples // (k * k))
    chunks = [slice(s, s + per) for s in range(0, n, per)]
    rs = (lambda sl: r[sl]) if isinstance(r, torch.Tensor) else (lambda sl: r)
    covmax = torch.zeros(HW + 1, device=dev)
    cached = []
    for sl in chunks:
        lin, cov = _footprint(cx[sl], cy[sl], rs(sl), ar, W, H, R)
        covmax.scatter_reduce_(0, lin, cov, reduce="amax", include_self=True)
        if len(chunks) <= 2:
            cached.append((lin, cov))
    # second pass: remember which dot wins each pixel (an index, cheaper than scattering colors)
    winner = torch.full((HW + 1,), n, dtype=torch.long, device=dev)   # n = background
    for j, sl in enumerate(chunks):
        lin, cov = cached[j] if cached else _footprint(cx[sl], cy[sl], rs(sl), ar, W, H, R)
        win = (cov > 0) & (cov >= covmax[lin])
        tgt = torch.where(win, lin, torch.full_like(lin, HW))
        ids = torch.arange(sl.start, min(sl.stop, n), device=dev)[:, None].expand(-1, k * k).reshape(-1)
        winner.index_put_((tgt,), ids)
    colx = torch.cat([col.float(), torch.full((1, 3), float(bg), device=dev)])
    out = bg + covmax[:HW, None] * (colx[winner[:HW]] - bg)
    return out.round_().clamp_(0, 255).to(torch.uint8).view(H, W, 3)


def render(stip, depth01, size, scale, radius, dark, rgb=None, size_by_depth=False, flat=False,
           canvas_size=None):
    """Returns the dot image as an (H, W, 3) uint8 tensor on depth01's device.

    rgb: (h, w, 3) uint8 tensor for --color."""
    w, h = size
    W, H = canvas_size or (int(w * scale), int(h * scale))
    bg = 0 if dark else 255
    xs, ys, ix, iy = stip.points(depth01)
    dv = depth01[iy, ix]
    if size_by_depth:
        r, rmax = radius * (0.5 + dv), radius * 1.5
    else:
        r, rmax = float(radius), float(radius)
    if rgb is not None:
        col = rgb.to(depth01.device)[iy, ix].float()
    elif flat:
        col = torch.full((xs.numel(), 3), 255.0 - bg, device=depth01.device)
    else:
        # dot tone = depth 0-255: 0 = farthest (fades into background), 255 = closest (full ink)
        v = torch.round(dv * 255)
        g = v if dark else 255 - v
        col = g[:, None].expand(-1, 3).contiguous()
    return _safe("render", _splat, xs * scale, ys * scale, col, r, rmax, W, H, bg)


def to_u8(d01):
    return Image.fromarray((d01 * 255).to(torch.uint8).cpu().numpy())


# ----------------------------------------------------------------------------- image
def run_image(a, src, out):
    t0 = time.time()
    if not hasattr(a, "tag"):
        a.tag, a.dtag = param_tag(a), depth_tag(a)
    img = Image.open(src).convert("RGB")
    dev = pick_device(a.device)
    rgb_np = np.array(img)
    if a.depth_in:
        d = torch.from_numpy(np.asarray(Image.open(a.depth_in).convert("L").resize(img.size),
                                        dtype=np.float32).copy()).to(dev)
    else:
        print("Estimating depth (first run downloads the model)...")
        model = DepthModel(a.model, a.res, a.device, a.fp32)
        dev = model.dev
        d = model.batch(torch.from_numpy(rgb_np)[None])[0]
    rgb = torch.from_numpy(rgb_np).to(dev)
    lo, hi = percentile(d, [1, 99]).to(dev)
    d01 = normalize(d, lo, hi)
    if a.save_depth:
        to_u8(d01).save(src.with_name(f"{src.stem}_depth_{a.dtag}.png"))
    d01 = enhance(d01, a.detail, a.detail_radius, rgb, a.shade)
    if a.save_depth and (a.detail > 0 or a.shade > 0):
        to_u8(d01).save(src.with_name(f"{src.stem}_density_{a.tag}.png"))
    stip = make_stippler(a, *img.size, device=dev)
    canvas = render(stip, d01, img.size, a.scale, a.size, a.dark, rgb if a.color else None,
                    a.size_by_depth, a.flat_dots)
    Image.fromarray(canvas.cpu().numpy()).save(out)
    print(f"Dots: {out}  ({time.time() - t0:.2f}s)")


# ----------------------------------------------------------------------------- video I/O
class FrameReader(threading.Thread):
    """Decodes frames (cv2) on a background thread into a bounded queue of RGB arrays."""

    def __init__(self, cap, max_frames=0, qsize=16):
        super().__init__(daemon=True)
        import cv2
        self.cv2, self.cap, self.max_frames = cv2, cap, max_frames
        self.q = queue.Queue(maxsize=qsize)
        self.stop_flag = False

    def run(self):
        n = 0
        try:
            while not self.stop_flag and not (self.max_frames and n >= self.max_frames):
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.q.put(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
                n += 1
        finally:
            self.q.put(None)

    def batch(self, n):
        out = []
        while len(out) < n:
            f = self.q.get()
            if f is None:
                self.q.put(None)  # keep the end marker for the next call
                break
            out.append(f)
        return out


class _ThreadedWriter:
    """Consumes frames from a bounded queue on a background thread."""

    def __init__(self, qsize=8):
        self.q = queue.Queue(maxsize=qsize)
        self.err = None
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            if self.err is None:
                try:
                    self._write(f)
                except Exception as e:  # keep draining so the producer never blocks
                    self.err = e

    def write(self, frame):
        if self.err is not None:
            raise RuntimeError(f"video writer failed: {self.err}")
        self.q.put(frame)

    def close(self):
        self.q.put(None)
        self.t.join()
        self._finish()
        if self.err is not None:
            raise RuntimeError(f"video writer failed: {self.err}")


class FFmpegWriter(_ThreadedWriter):
    """Pipes raw frames into an ffmpeg subprocess."""

    def __init__(self, path, w, h, fps, venc, gray=False, vf=None):
        self.log = tempfile.TemporaryFile()
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "gray" if gray else "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-an"] + (["-vf", vf] if vf else []) + venc + [str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.log)
        super().__init__()

    def _write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).data)

    def _finish(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        rc = self.p.wait()
        if rc != 0 and self.err is None:
            self.log.seek(0)
            self.err = RuntimeError(f"ffmpeg exited {rc}: {self.log.read().decode(errors='replace')[-300:]}")


class CV2Writer(_ThreadedWriter):
    """Fallback when ffmpeg is missing: OpenCV mp4v writer."""

    def __init__(self, path, w, h, fps, gray=False):
        import cv2
        self.cv2, self.gray = cv2, gray
        self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        super().__init__()

    def _write(self, f):
        cv = self.cv2
        self.vw.write(cv.cvtColor(f, cv.COLOR_GRAY2BGR if self.gray else cv.COLOR_RGB2BGR))

    def _finish(self):
        self.vw.release()


def _probe_encoder(venc, w, h):
    """Encode two black frames to /dev/null to check an encoder works on this machine."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", "24", "-i", "-"] + venc + ["-f", "null", "-"]
    try:
        r = subprocess.run(cmd, input=bytes(w * h * 3 * 2), capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def pick_encoder(a, w, h, fps):
    """ffmpeg video-encoder args: hardware (if --hw-encode and it works) else libx264 crf18."""
    yuv = ["-pix_fmt", "yuv420p"]
    cands = []
    if a.hw_encode:
        kbps = max(2000, int(w * h * fps * 0.6 / 1000))  # dots are noisy: be generous with bitrate
        if sys.platform == "darwin":
            cands += [("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-q:v", "65"] + yuv),
                      ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-b:v", f"{kbps}k"] + yuv)]
        if torch.cuda.is_available():
            cands += [("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
                                      "-b:v", "0"] + yuv)]
    cands += [("libx264", ["-c:v", "libx264", "-preset", a.preset, "-crf", "18"] + yuv)]
    for name, venc in cands:
        if _probe_encoder(venc, w, h):
            return name, venc
        if a.hw_encode and name != "libx264":
            print(f"{name} not usable here, trying the next encoder.")
    return None, None


# ----------------------------------------------------------------------------- video
def run_video(a, src, out, depth_fn=None):
    import cv2
    if not hasattr(a, "tag"):
        a.tag, a.dtag = param_tag(a, True), depth_tag(a, True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Can't open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if a.max_frames:
        total = min(total, a.max_frames) if total else a.max_frames
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W, H = int(w * a.scale) // 2 * 2, int(h * a.scale) // 2 * 2  # even size for H.264

    if depth_fn is None:
        print("Loading depth model (first run downloads it)...")
        depth_fn = DepthModel(a.model, a.res, a.device, a.fp32)
    if isinstance(depth_fn, DepthModel):
        dev, depth_batch = depth_fn.dev, depth_fn.batch
    else:  # legacy callable: PIL image -> numpy depth
        dev = pick_device(a.device)
        def depth_batch(frames):
            return torch.stack([torch.as_tensor(np.asarray(depth_fn(Image.fromarray(f.numpy())),
                                                           dtype=np.float32)) for f in frames]).to(dev)
    bs = a.batch or (8 if dev.type != "cpu" else 1)
    stip = make_stippler(a, w, h, device=dev)

    tmp = Path(tempfile.mkdtemp())
    silent = tmp / "silent.mp4"
    have_ffmpeg = shutil.which("ffmpeg") is not None
    enc_name, venc = pick_encoder(a, W, H, fps) if have_ffmpeg else (None, None)
    if venc:
        vw = FFmpegWriter(silent, W, H, fps, venc)
    else:
        vw = CV2Writer(silent, W, H, fps)
    dw = None
    if a.save_depth:
        dpath = src.with_name(f"{src.stem}_depth_{a.dtag}.mp4")
        if venc:
            dvenc = ["-c:v", "libx264", "-preset", a.preset, "-crf", "18", "-pix_fmt", "yuv420p"] \
                if enc_name == "libx264" else venc
            dw = FFmpegWriter(dpath, w, h, fps, dvenc, gray=True, vf="pad=ceil(iw/2)*2:ceil(ih/2)*2")
        else:
            dw = CV2Writer(dpath, w, h, fps, gray=True)
    print(f"Video {w}x{h} @ {fps:.3g} fps -> {W}x{H}, device {dev.type}, batch {bs}, "
          f"encoder {enc_name or 'opencv mp4v'}")

    reader = FrameReader(cap, a.max_frames, qsize=max(16, 2 * bs))
    reader.start()
    pin = dev.type == "cuda"
    prev = lo = hi = None
    s = a.smooth
    n = 0
    t0 = time.time()
    try:
        while True:
            frames = reader.batch(bs)
            if not frames:
                break
            cpu = torch.from_numpy(np.stack(frames))
            if pin:
                cpu = cpu.pin_memory()
            gpu = cpu.to(dev, non_blocking=True)
            while True:  # shrink the batch if the GPU runs out of memory
                try:
                    D = torch.cat([depth_batch(gpu[i:i + bs]) for i in range(0, len(frames), bs)])
                    break
                except (RuntimeError, MemoryError) as e:
                    if not _is_oom(e) or bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    elif dev.type == "mps":
                        torch.mps.empty_cache()
                    print(f"\nOut of memory, batch -> {bs}")
            with torch.inference_mode():
                for i in range(len(frames)):
                    d = D[i]
                    # smooth depth and its range over time to avoid pumping/flicker
                    prev = d if prev is None else s * prev + (1 - s) * d
                    flo, fhi = percentile(prev, [1, 99]).to(dev)
                    lo = flo if lo is None else s * lo + (1 - s) * flo
                    hi = fhi if hi is None else s * hi + (1 - s) * fhi
                    d01 = normalize(prev, lo, hi)
                    raw01 = d01
                    img = gpu[i]
                    d01 = enhance(d01, a.detail, a.detail_radius, img, a.shade)
                    canvas = render(stip, d01, (w, h), a.scale, a.size, a.dark, img if a.color else None,
                                    a.size_by_depth, a.flat_dots, canvas_size=(W, H))
                    vw.write(canvas.cpu().numpy())
                    if dw is not None:
                        dw.write((raw01 * 255).to(torch.uint8).cpu().numpy())
                    n += 1
                    el = time.time() - t0
                    rate = n / el if el > 0 else 0
                    eta = f", ETA {(total - n) / rate:.0f}s" if total and rate else ""
                    print(f"\rFrame {n}/{total or '?'}  {rate:.2f} fps{eta}   ", end="", flush=True)
    finally:
        reader.stop_flag = True
        while reader.is_alive():  # unblock a reader waiting on a full queue
            try:
                reader.q.get_nowait()
            except queue.Empty:
                pass
            reader.join(0.05)
        cap.release()
        vw.close()
        if dw is not None:
            dw.close()
    el = time.time() - t0
    print(f"\n{n} frames in {el:.1f}s ({n / max(el, 1e-9):.2f} fps)")

    if have_ffmpeg:
        base = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(src), "-map", "0:v"]
        # video is already H.264 when it came through the ffmpeg pipe: just copy it
        enc = ["-c:v", "copy"] if venc else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        # keep only the first audio track (iPhone clips also carry an "apac" spatial
        # audio track many ffmpeg builds can't decode); fall back to no audio
        attempts = [base + ["-map", "1:a:0?"] + enc + ["-c:a", "aac", "-shortest", str(out)],
                    base + enc + [str(out)]]
        for i, cmd in enumerate(attempts):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                if i == 1:
                    print("Couldn't copy the audio, saved without it.")
                break
        else:
            shutil.move(str(silent), out)
            print("ffmpeg failed (" + r.stderr.strip()[-200:] + "), saved raw video without audio.")
    else:
        shutil.move(str(silent), out)
        print("ffmpeg not found: saved without audio.")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Dots video: {out}")


def _num(v):
    return f"{v:g}"


def _safe_name(s):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-.") or "x"


def depth_tag(a, is_video=False):
    """Tag for the raw depth side output: only what affects the depth map."""
    if a.depth_in and not is_video:
        return "depthin"
    model = a.model if a.model in MODELS else a.model.rstrip("/").split("/")[-1]
    t = f"{_safe_name(model)}_r{a.res}"
    if is_video and a.smooth != 0.6:
        t += f"_sm{_num(a.smooth)}"
    return t


def param_tag(a, is_video=False):
    """Filesystem-safe tag describing every setting that changes the output."""
    model = a.model if a.model in MODELS else a.model.rstrip("/").split("/")[-1]
    t = [f"{_safe_name(model)}", f"r{a.res}", f"n{a.dots}", f"g{_num(a.gamma)}",
         f"s{_num(a.size)}", f"x{_num(a.scale)}"]
    if a.floor != 0.02:
        t.append(f"f{_num(a.floor)}")
    if is_video and a.smooth != 0.6:
        t.append(f"sm{_num(a.smooth)}")
    if a.detail > 0:
        t.append(f"det{_num(a.detail)}")
    if a.shade > 0:
        t.append(f"sh{_num(a.shade)}")
    for flag, name in [(a.fixed_dots, "fixed"), (a.dark, "dark"), (a.color, "color"), (a.flat_dots, "flat"),
                       (a.size_by_depth, "sbd"), (a.depth_in and not is_video, "depthin")]:
        if flag:
            t.append(name)
    return "_".join(_safe_name(x) for x in t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default="base",
                    help="small | base | large (best detail, slowest) or any Hugging Face depth model id")
    ap.add_argument("--res", type=int, default=518,
                    help="Depth inference resolution (short side). 518 default; 770-1036 for sharper faces")
    ap.add_argument("--detail", type=float, default=0.0,
                    help="Boost local depth relief such as facial features (try 1-2)")
    ap.add_argument("--detail-radius", type=float, default=25,
                    help="Size in pixels of the detail being boosted (smaller = finer features)")
    ap.add_argument("--shade", type=float, default=0.0,
                    help="0-1: mix in the photo's dark areas (eyes, brows, mouth) so expressions read")
    ap.add_argument("--dots", type=int, default=60000, help="Approximate number of dots per image/frame")
    ap.add_argument("--gamma", type=float, default=1.5, help=">1 pushes dots harder toward close areas")
    ap.add_argument("--floor", type=float, default=0.02, help="Minimum density for the far background (0-1)")
    ap.add_argument("--size", type=float, default=1.2, help="Dot radius in output pixels")
    ap.add_argument("--scale", type=float, default=2.0, help="Output size relative to input")
    ap.add_argument("--size-by-depth", action="store_true", help="Closer dots are also bigger")
    ap.add_argument("--color", action="store_true", help="Color each dot from the original")
    ap.add_argument("--flat-dots", action="store_true",
                    help="Solid black (or white with --dark) dots instead of depth-shaded 0-255 dots")
    ap.add_argument("--dark", action="store_true", help="White dots on black")
    ap.add_argument("--save-depth", action="store_true", help="Also save the depth map (PNG or MP4)")
    ap.add_argument("--depth-in", help="Images only: use an existing depth map instead of the model")
    ap.add_argument("--smooth", type=float, default=0.6, help="Video only: temporal smoothing 0-0.95")
    ap.add_argument("--max-frames", type=int, default=0, help="Video only: stop after N frames (for previews)")
    ap.add_argument("--fixed-dots", action="store_true",
                    help="Video: keep dots in fixed spots between frames instead of re-rolling (no flicker)")
    ap.add_argument("--seed", type=int, default=0)
    # performance
    ap.add_argument("--batch", type=int, default=0,
                    help="Video frames per depth-model batch (default 8 on GPU, 1 on CPU)")
    ap.add_argument("--fp32", action="store_true", help="Force fp32 on GPU (default fp16 on CUDA/MPS when it works)")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu) | cuda | mps | cpu")
    ap.add_argument("--hw-encode", action="store_true",
                    help="Video: hardware H.264 encoder (VideoToolbox on Mac, NVENC on CUDA), falls back to libx264")
    ap.add_argument("--preset", default="medium", help="libx264 preset (ultrafast...veryslow); faster = bigger files")
    a = ap.parse_args()

    src = Path(a.input)
    is_video = src.suffix.lower() in VIDEO_EXT
    a.tag, a.dtag = param_tag(a, is_video), depth_tag(a, is_video)
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}_dots_{a.tag}{'.mp4' if is_video else '.png'}")
    (run_video if is_video else run_image)(a, src, out)


if __name__ == "__main__":
    main()
