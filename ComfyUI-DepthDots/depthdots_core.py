"""DepthDots core: GPU depth -> stipple (dot art) logic, with no CLI / ComfyUI dependency.

Refactored from depth_dots.py. Everything runs on torch tensors on one device
(CUDA, Apple MPS or CPU): Depth Anything V2 inference with GPU preprocessing,
1/99-percentile normalization, temporal smoothing, detail/shade enhancement,
weighted random / fixed-threshold dot sampling and a vectorized antialiased
dot rasterizer.
"""
import math
import os
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops run on CPU instead of failing

import numpy as np
import torch
import torch.nn.functional as F

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CPU_ONLY = set()  # ops that failed on the accelerator and now run on CPU


def pick_device(name=None):
    if isinstance(name, torch.device):
        return name
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_oom(e):
    return isinstance(e, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def empty_cache(dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _safe(key, fn, *args):
    """Run fn on the tensors' device; if the op isn't supported there, redo it on CPU from then on."""
    dev = next((x.device for x in args if isinstance(x, torch.Tensor)), torch.device("cpu"))
    if dev.type != "cpu" and key not in _CPU_ONLY:
        try:
            return fn(*args)
        except (RuntimeError, NotImplementedError) as e:
            if is_oom(e):
                raise
            _CPU_ONLY.add(key)
            print(f"[DepthDots] {key} not supported on {dev.type} ({str(e).splitlines()[0][:120]}); using CPU for it.")
    out = fn(*[x.cpu() if isinstance(x, torch.Tensor) else x for x in args])
    return out.to(dev) if isinstance(out, torch.Tensor) else out


# ----------------------------------------------------------------------------- depth model
def model_name(model):
    return MODELS.get(model, model)


class DepthModel:
    """Depth Anything V2 with torch preprocessing (resize + ImageNet normalize) on the device.

    precision: "auto" (fp16 on CUDA/MPS when a test pass is finite, else fp32), "fp16" or "fp32".
    """

    def __init__(self, model="base", device=None, precision="auto", cache_dir=None, local_dir=None):
        from transformers import AutoModelForDepthEstimation
        self.dev = pick_device(device)
        self.name = model_name(model)
        self.short = model if model in MODELS else self.name.rstrip("/").split("/")[-1]
        src = local_dir if local_dir and os.path.isfile(os.path.join(local_dir, "config.json")) else self.name
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        try:  # only read the normalization constants; the processor isn't used per frame
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(src, cache_dir=cache_dir)
            mean = tuple(getattr(proc, "image_mean", None) or mean)
            std = tuple(getattr(proc, "image_std", None) or std)
        except Exception:
            pass
        self.mean_std = (mean, std)
        self.model = AutoModelForDepthEstimation.from_pretrained(src, cache_dir=cache_dir).eval()
        self.dtype = torch.float32
        self.to(self.dev)
        if self.dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if precision == "fp16" or (precision == "auto" and self.dev.type in ("cuda", "mps")):
            self._try_half(strict=precision == "fp16")

    def to(self, dev):
        self.dev = torch.device(dev)
        self.model.to(self.dev)
        self.mean = torch.tensor(self.mean_std[0], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(self.mean_std[1], device=self.dev).view(1, 3, 1, 1)
        return self

    def _try_half(self, strict=False):
        try:
            self.model.half()
            with torch.inference_mode():
                x = torch.randn(1, 3, 14 * 12, 14 * 16, device=self.dev, dtype=torch.float16)
                ok = bool(torch.isfinite(self.model(pixel_values=x).predicted_depth).all())
            if ok or strict:
                self.dtype = torch.float16
                return
            print("[DepthDots] fp16 produced NaNs on this device, using fp32.")
        except Exception as e:
            if strict:
                raise
            print(f"[DepthDots] fp16 not usable on {self.dev.type} ({str(e)[:100]}), using fp32.")
        self.model.float()
        self.dtype = torch.float32

    @staticmethod
    def size_for(w, h, res):
        res = int(res) // 14 * 14
        s = res / min(w, h)  # shortest side -> res, keep aspect ratio
        return max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)

    def preprocess(self, frames, res):
        """frames: (B, h, w, 3) uint8 or float 0-1 tensor on device -> normalized (B, 3, H14, W14)."""
        _, h, w, _ = frames.shape
        x = frames.permute(0, 3, 1, 2)
        x = x.float().div_(255) if x.dtype == torch.uint8 else x.float()
        size = self.size_for(w, h, res)
        if size != (h, w):
            try:
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
            except (RuntimeError, NotImplementedError):
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
            x = x.clamp_(0, 1)
        return ((x - self.mean) / self.std).to(self.dtype).contiguous()

    def batch(self, frames, res=518):
        """frames: (B, h, w, 3) -> (B, h, w) float32 raw depth on device (bigger = closer)."""
        frames = frames.to(self.dev, non_blocking=True)
        _, h, w, _ = frames.shape
        with torch.inference_mode():
            d = self.model(pixel_values=self.preprocess(frames, res)).predicted_depth
            if d.ndim == 3:
                d = d[:, None]
            return F.interpolate(d.float(), size=(h, w), mode="bicubic", align_corners=False)[:, 0]


# ----------------------------------------------------------------------------- depth processing
def percentile(x, qs):
    """np.percentile (linear) of a tensor, returned as a tensor on x.device."""
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


class DepthNormalizer:
    """1/99-percentile normalization with the script's temporal smoothing.

    smooth = 0: every frame normalized on its own. smooth > 0: EMA of the raw depth and
    of the lo/hi range across consecutive frames (video), exactly like depth_dots.py --smooth.
    """

    def __init__(self, smooth=0.0):
        self.s = float(smooth)
        self.prev = self.lo = self.hi = None

    def __call__(self, d):
        s = self.s
        self.prev = d if (self.prev is None or s <= 0) else s * self.prev + (1 - s) * d
        flo, fhi = percentile(self.prev, [1, 99]).to(d.device)
        self.lo = flo if (self.lo is None or s <= 0) else s * self.lo + (1 - s) * flo
        self.hi = fhi if (self.hi is None or s <= 0) else s * self.hi + (1 - s) * fhi
        return normalize(self.prev, self.lo, self.hi)


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
    """PIL 'L' conversion of a (h, w, 3) uint8 tensor (float 0-1 is accepted too), as 0-255 float."""
    if rgb.dtype != torch.uint8:
        rgb = (rgb.float().clamp(0, 1) * 255).round()
    c = rgb.to(torch.int32)
    return ((c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16).float()


def enhance(d01, detail, detail_radius, img=None, shade=0.0):
    """Boost local depth relief (noses, eye sockets, mouths) and optionally mix in image shading.

    img: (h, w, 3) source frame tensor (uint8 or float 0-1), only needed for shade."""
    out = d01
    if detail > 0:
        hp = d01 - blur(d01, detail_radius)
        hp = hp / (percentile(hp.abs(), [99])[0].to(hp.device) + 1e-6)
        out = out + detail * 0.25 * hp
    if shade > 0 and img is not None:
        dark = 1 - to_gray(img.to(d01.device)) / 255
        out = (1 - shade) * out + shade * dark * (d01 * 1.5).clamp(0, 1)
    return out.clamp(0, 1)


# ----------------------------------------------------------------------------- stipplers
def _sample(p, n):
    """n indices drawn with replacement, P(i) proportional to p[i]."""
    if p.numel() <= 2 ** 24 and "multinomial" not in _CPU_ONLY:
        try:
            return torch.multinomial(p, n, replacement=True)
        except (RuntimeError, NotImplementedError) as e:
            if is_oom(e):
                raise
            _CPU_ONLY.add("multinomial")
    cdf = p.double().cumsum(0) if p.device.type != "mps" else p.cumsum(0)
    u = torch.rand(n, device=p.device, dtype=cdf.dtype) * cdf[-1]
    return torch.searchsorted(cdf, u, right=True).clamp_(max=p.numel() - 1)


class RandomStippler:
    """Fresh random dots every call (re-rolled each video frame: flickering, grainy look)."""

    def __init__(self, w, h, dots, gamma, floor, seed=0, device=None):
        torch.manual_seed(seed)
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
    """Fixed candidate points with fixed thresholds: density tracks depth, dots stay put."""

    def __init__(self, w, h, dots, gamma, floor, seed=0, oversample=6, device=None):
        g = torch.Generator().manual_seed(seed)
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


def make_stippler(fixed, w, h, dots, gamma, floor, seed=0, device=None):
    cls = FixedStippler if fixed else RandomStippler
    return cls(w, h, dots, gamma, floor, seed, device=device)


# ----------------------------------------------------------------------------- rasterizer
def _footprint(cx, cy, r, ar, W, H, R):
    px = torch.ceil(cx - 0.5 - R).long()[:, None] + ar    # (n,k); pixel i spans [i, i+1)
    py = torch.ceil(cy - 0.5 - R).long()[:, None] + ar
    dx = px.float() + 0.5 - cx[:, None]
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
    """Rasterize all dots at once; each pixel takes the dot covering it most (opaque-draw look)."""
    dev = cx.device
    R = rmax + 1.0
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
    """Dot image as an (H, W, 3) uint8 tensor on depth01's device.

    rgb: (h, w, 3) source tensor (uint8 or float 0-1) to color each dot (--color)."""
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
        c = rgb.to(depth01.device)
        c = c.float() if c.dtype == torch.uint8 else (c.float().clamp(0, 1) * 255).round()
        col = c[iy, ix]
    elif flat:
        col = torch.full((xs.numel(), 3), 255.0 - bg, device=depth01.device)
    else:
        v = torch.round(dv * 255)  # dot tone = depth 0-255
        g = v if dark else 255 - v
        col = g[:, None].expand(-1, 3).contiguous()
    return _safe("render", _splat, xs * scale, ys * scale, col, r, rmax, W, H, bg)


# ----------------------------------------------------------------------------- naming
def num(v):
    return f"{v:g}"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-.") or "x"


def depth_tag(model_short, res, smooth=0.0):
    t = f"{safe_name(model_short)}_r{int(res)}"
    if smooth:
        t += f"_sm{num(smooth)}"
    return t


_KNOWN = [("r", re.compile(r"^r\d+$")), ("sm", re.compile(r"^sm[-0-9.e]+$")),
          ("det", re.compile(r"^det[-0-9.e]+$")), ("sh", re.compile(r"^sh[-0-9.e]+$")),
          ("depthin", re.compile(r"^depthin$"))]


def full_tag(upstream, dots, gamma, size, scale, floor=0.02, fixed=False, dark=False, color=False,
             flat=False, sbd=False):
    """Same tokens and order as depth_dots.py's default output name:
    model_rRES_nDOTS_gG_sS_xX[_fF][_smS][_detD][_shS][_fixed][_dark][_color][_flat][_sbd][_depthin]"""
    found, model = {}, []
    for tok in [t for t in (upstream or "").split("_") if t]:
        for key, rx in _KNOWN:
            if rx.match(tok):
                found[key] = tok
                break
        else:
            model.append(tok)
    t = (["_".join(model)] if model else []) + ([found["r"]] if "r" in found else [])
    t += [f"n{int(dots)}", f"g{num(gamma)}", f"s{num(size)}", f"x{num(scale)}"]
    if floor != 0.02:
        t.append(f"f{num(floor)}")
    t += [found[k] for k in ("sm", "det", "sh") if k in found]
    for flag, name in [(fixed, "fixed"), (dark, "dark"), (color, "color"), (flat, "flat"), (sbd, "sbd"),
                       ("depthin" in found, "depthin")]:
        if flag:
            t.append(name)
    return "_".join(safe_name(x) for x in t)


# ----------------------------------------------------------------------------- sketch outline
def luminance(rgb):
    """(B, H, W, 3) float 0-1 -> (B, 1, H, W) luminance."""
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=torch.float32)
    return (rgb[..., :3].float() * w).sum(-1)[:, None]


def _gauss_kernel(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def _gblur(x, sigma):
    if sigma <= 0:
        return x
    k, r = _gauss_kernel(float(sigma), x.device)
    C = x.shape[1]
    mode = "reflect" if min(x.shape[-2:]) > r else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of a (B, C, H, W) tensor (reflect padding)."""
    return _safe("gaussian_blur", _gblur, x, sigma)


def _robust_scale(x, q=99.0, min_scale=1e-3):
    """Per-frame q-th percentile of x (B,1,H,W), used to make thresholds contrast-independent."""
    return torch.stack([percentile(f, [q])[0].to(x.device) for f in x]).clamp_min(min_scale).view(-1, 1, 1, 1)


def xdog_lines(lum, sigma=1.0, k=1.6, phi=4.0, epsilon=0.25, denoise=1.0):
    """XDoG-style pencil outline. lum: (B,1,H,W) 0-1. Returns line strength (B,1,H,W) in 0-1.

    E = G_{k*sigma} - G_sigma is positive on the dark side of an edge; it is normalized by its
    per-frame 99.5th percentile (so contrast doesn't matter), then soft-thresholded
    at epsilon with sharpness phi: line = tanh(phi * max(E - epsilon, 0))."""
    x = gaussian_blur(lum, denoise) if denoise > 0 else lum
    e = gaussian_blur(x, sigma * k) - gaussian_blur(x, sigma)
    e = e / _robust_scale(e.clamp_min(0), 99.5, 0.01)
    return torch.tanh(phi * (e - epsilon).clamp_min(0)).clamp(0, 1)


def _sobel(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device) / 8
    k = torch.stack([kx, kx.t()])[:, None]
    g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)
    return g[:, :1], g[:, 1:]


def _thin(mag, gx, gy):
    """Non-maximum suppression along the gradient direction (4 bins), keeps edges ~1-2 px wide."""
    p = F.pad(mag, (1, 1, 1, 1), mode="replicate")
    H, W = mag.shape[-2:]
    sh = lambda dy, dx: p[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    ang = torch.atan2(gy, gx) % math.pi
    b = ((ang + math.pi / 8) // (math.pi / 4)).long() % 4  # 0: horiz grad, 1: diag, 2: vert, 3: anti-diag
    n1 = torch.stack([sh(0, 1), sh(1, 1), sh(1, 0), sh(1, -1)])
    n2 = torch.stack([sh(0, -1), sh(-1, -1), sh(-1, 0), sh(-1, 1)])
    m1 = torch.gather(n1, 0, b[None]).squeeze(0)
    m2 = torch.gather(n2, 0, b[None]).squeeze(0)
    return mag * ((mag >= m1) & (mag >= m2)).float()


def depth_edge_lines(depth, threshold=0.3, denoise=1.0, thin=True):
    """Silhouette / fold lines from a depth map (B,1,H,W) 0-1 via Sobel, 0-1 strength.

    The gradient is measured per 1/512 of the short side so the threshold doesn't depend on
    resolution, and normalized by its 99.5th percentile; threshold is relative to that."""
    x = gaussian_blur(depth.float(), denoise) if denoise > 0 else depth.float()
    gx, gy = _sobel(x)
    mag = torch.sqrt(gx ** 2 + gy ** 2) * (min(x.shape[-2:]) / 512.0)
    mag = mag / _robust_scale(mag, 99.5, 1e-3)
    if thin:
        mag = _thin(mag, gx, gy)
    return ((mag - threshold) / max(1e-3, 1 - threshold)).clamp(0, 1) ** 0.5


def dilate(x, radius):
    """Grayscale dilation (max filter, disc-ish) with a fractional radius in pixels."""
    if radius <= 0:
        return x
    r0 = int(math.floor(radius))
    out = F.max_pool2d(x, 2 * r0 + 1, stride=1, padding=r0) if r0 > 0 else x
    frac = radius - r0
    if frac > 1e-3:
        r1 = r0 + 1
        bigger = F.max_pool2d(x, 2 * r1 + 1, stride=1, padding=r1)
        out = out + frac * (bigger - out)
    return out


def remove_specks(x, clean=1.0):
    """Drop isolated specks: keep line pixels whose 5x5 neighbourhood holds enough line (lines
    passing through cover >= 5 px of it, a lone speck only 1-3). clean 0 = off, 1 = default, 2 = harder."""
    if clean <= 0:
        return x
    density = F.avg_pool2d((x > 0.25).float(), 5, stride=1, padding=2, count_include_pad=False)
    return x * (density >= 0.16 * clean).float()


def sketch_outline(rgb=None, depth=None, method="combined", sigma=1.0, k=1.6, phi=4.0, epsilon=0.25,
                   depth_threshold=0.3, denoise=1.0, depth_weight=1.0, background_fade=0.0, parts=False):
    """Line strength (B,1,H,W) 0-1 at the input resolution. rgb: (B,H,W,3); depth: (B,1,H,W) 0-1 (white = close).

    background_fade 0-1 (needs depth): fades image (xdog) lines in far areas so the subject reads.
    parts=True returns (lines, xdog_part or None, depth_part or None)."""
    lines = xd = de = None
    if depth is not None and rgb is not None and depth.shape[-2:] != rgb.shape[1:3]:
        depth = F.interpolate(depth.float(), size=rgb.shape[1:3], mode="bilinear", align_corners=False)
    if method in ("xdog", "combined"):
        if rgb is None:
            raise ValueError("xdog needs an image")
        lines = xdog_lines(luminance(rgb), sigma, k, phi, epsilon, denoise)
        if background_fade > 0 and depth is not None:
            t = ((depth.float() - 0.15) / 0.35).clamp(0, 1)
            lines = lines * ((1 - background_fade) + background_fade * t * t * (3 - 2 * t))
        xd = lines
    if method in ("depth_edges", "combined"):
        if depth is None:
            if method == "depth_edges":
                raise ValueError("method depth_edges needs a depth input")
            print("[DepthDots] sketch: no depth connected, 'combined' uses xdog only")
        else:
            de = dilate(depth_edge_lines(depth, depth_threshold, denoise), 0.5) * depth_weight
            de = de.clamp(0, 1)
            lines = de if lines is None else torch.maximum(lines, de)
    return (lines, xd, de) if parts else lines


def finish_lines(lines, out_hw, thickness=1.0, strength=1.0, clean=1.0, scale_factor=1.0):
    """Clean specks, upscale line strength to the output size, set thickness and strength.

    thickness 1 = natural width (scaled with the output); >1 dilates by (thickness-1)*scale/2 px;
    <1 thins by dropping the faint edges of each stroke."""
    lines = remove_specks(lines, clean)
    if lines.shape[-2:] != tuple(out_hw):
        lines = F.interpolate(lines, size=tuple(out_hw), mode="bilinear", align_corners=False).clamp(0, 1)
    sf = max(1.0, scale_factor)
    if thickness > 1:
        lines = dilate(lines, (thickness - 1) * sf * 0.5)
    elif thickness < 1:
        lines = lines ** (1.0 / max(thickness, 0.05))
    lines = gaussian_blur(lines, 0.3 * sf)  # soft pencil edge, antialiased
    return (lines * strength).clamp(0, 1)


# ----------------------------------------------------------------------------- combine
def parse_hex(color, default=(0.0, 0.0, 0.0)):
    s = str(color).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except Exception:
        return default


def is_dark_bg(img):
    """Guess if an image batch (B,H,W,3) 0-1 has a dark background.

    Dot renders and outlines are mostly pure background, so compare the share of pure-white
    vs pure-black pixels; fall back to the median brightness for photos."""
    x = img[..., :3].float()
    white = (x.amin(-1) >= 0.985).float().mean()
    black = (x.amax(-1) <= 0.015).float().mean()
    if max(float(white), float(black)) > 0.05:
        return bool(black > white)
    return bool(x.mean(-1).median() < 0.5)


def outline_alpha(outline, outline_dark=None):
    """Line alpha (B,H,W) from an outline IMAGE of either polarity (dark lines on light, or light on dark)."""
    lum = luminance(outline)[:, 0]
    if outline_dark is None:
        outline_dark = is_dark_bg(outline)
    return (lum if outline_dark else 1 - lum).clamp(0, 1)


def combine(dots, alpha, mode="auto", opacity=1.0, line_color="auto", dark=None):
    """Composite a line mask over a dots image.

    dots (B,H,W,3) 0-1, alpha (B,H,W) line mask 0-1 of the same size.
    auto: multiply on light backgrounds, screen on dark ones (line_color auto = black / white);
    an explicit line_color with auto paints lines in that color (lines_over)."""
    if dark is None:
        dark = is_dark_bg(dots)
    auto_color = line_color in ("", "auto", None)
    col = ((1.0, 1.0, 1.0) if dark else (0.0, 0.0, 0.0)) if auto_color else parse_hex(line_color)
    c = torch.tensor(col, device=dots.device, dtype=dots.dtype)
    a = (alpha * opacity).clamp(0, 1)[..., None]
    if mode == "auto":
        mode = ("screen" if dark else "multiply") if auto_color else "lines_over"
    if mode == "multiply":    # lines darken: line color (default black) multiplied in
        return dots * (1 - a + a * c)
    if mode == "screen":      # lines lighten: line color (default white) screened in
        return 1 - (1 - dots) * (1 - a * c)
    if mode == "lines_over":  # paint lines in line_color
        return dots * (1 - a) + c * a
    if mode == "dots_masked_by_lines":  # dots only where the lines are: lines drawn with dots
        bg = torch.full((3,), 0.0 if dark else 1.0, device=dots.device, dtype=dots.dtype)
        return bg + (dots - bg) * a
    raise ValueError(f"unknown combine mode {mode}")
