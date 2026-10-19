"""ComfyUI nodes for DepthDots (depth -> dot / stipple art), GPU accelerated."""
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import depthdots_core as core

try:  # ComfyUI helpers (optional so the module can be imported outside ComfyUI)
    import comfy.model_management as mm
except Exception:
    mm = None
try:
    import comfy.utils as cutils
except Exception:
    cutils = None
try:
    import folder_paths
except Exception:
    folder_paths = None

CATEGORY = "DepthDots"
VIDEO_EXT = ("mp4", "mov", "avi", "mkv", "webm", "m4v", "gif")

# models/depth_anything (HF cache layout inside; a plain folder with config.json also works)
MODEL_DIR = None
if folder_paths is not None:
    MODEL_DIR = os.path.join(getattr(folder_paths, "models_dir", os.path.join(os.getcwd(), "models")),
                             "depth_anything")
    try:
        folder_paths.add_model_folder_path("depth_anything", MODEL_DIR, is_default=True)
    except TypeError:  # older ComfyUI without is_default
        folder_paths.add_model_folder_path("depth_anything", MODEL_DIR)
    except Exception:
        pass


def comfy_device():
    if mm is not None:
        try:
            return mm.get_torch_device()
        except Exception:
            pass
    return core.pick_device()


def check_interrupt():
    if mm is not None:
        mm.throw_exception_if_processing_interrupted()


class _Progress:
    def __init__(self, total):
        self.bar = cutils.ProgressBar(total) if cutils is not None else None

    def update(self, n=1):
        if self.bar is not None:
            self.bar.update(n)


def image_to_depth01(img):
    """(h, w, C) IMAGE frame (float 0-1) -> (h, w) float depth 0-1. Gray or RGB depth maps both work."""
    if img.shape[-1] == 1:
        return img[..., 0].float()
    c = img[..., :3].float()
    if torch.equal(c[..., 0], c[..., 1]) and torch.equal(c[..., 0], c[..., 2]):
        return c[..., 0]
    return core.to_gray(c) / 255  # PIL "L" luminance, like the script's --depth-in


def gray_to_image(d):
    """(B, h, w) -> IMAGE (B, h, w, 3)."""
    return d[..., None].expand(*d.shape, 3).contiguous()


def match_size(src, h, w):
    """Resize a (h0, w0, C) frame to (h, w) if needed."""
    if src.shape[0] == h and src.shape[1] == w:
        return src
    x = src.permute(2, 0, 1)[None].float()
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)


def frame_of(batch, i):
    return batch[i if batch.shape[0] > 1 else 0]


# ----------------------------------------------------------------------------- loader
_MODEL_CACHE = {}


class DepthDotsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["small", "base", "large"], {"default": "base",
                      "tooltip": "Depth Anything V2 size. large = best detail (faces), slowest."}),
            "precision": (["auto", "fp16", "fp32"], {"default": "auto",
                          "tooltip": "auto = fp16 on CUDA/MPS when it works, else fp32"}),
            "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto",
                       "tooltip": "auto = ComfyUI's device (respects --cpu)"}),
            "keep_on_device": ("BOOLEAN", {"default": True,
                               "tooltip": "Keep the model on the GPU between runs (faster); off = move to CPU after use"}),
        }, "optional": {
            "hf_model_id": ("STRING", {"default": "", "tooltip": "Optional Hugging Face depth model id; overrides 'model'"}),
        }}

    RETURN_TYPES = ("DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads Depth Anything V2 (cached in models/depth_anything)."

    def load(self, model, precision, device, keep_on_device=True, hf_model_id=""):
        name = hf_model_id.strip() or model
        dev = comfy_device() if device == "auto" else torch.device(device)
        key = (core.model_name(name), str(dev), precision)
        if key not in _MODEL_CACHE:
            local = None
            if MODEL_DIR:
                os.makedirs(MODEL_DIR, exist_ok=True)
                local = os.path.join(MODEL_DIR, core.model_name(name).rstrip("/").split("/")[-1])
            m = core.DepthModel(name, dev, precision, cache_dir=MODEL_DIR, local_dir=local)
            print(f"[DepthDots] loaded {m.name} on {m.dev} ({str(m.dtype).replace('torch.', '')})")
            _MODEL_CACHE.clear()  # keep one model resident
            _MODEL_CACHE[key] = m
        return ({"model": _MODEL_CACHE[key], "device": dev, "keep": keep_on_device},)


# ----------------------------------------------------------------------------- estimate
class DepthDotsEstimateDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "depth_model": ("DEPTH_MODEL",),
            "res": ("INT", {"default": 518, "min": 14, "max": 4074, "step": 14,
                    "tooltip": "Inference short side (multiple of 14). 518 default; 770-1036 for sharper faces"}),
            "batch_size": ("INT", {"default": 0, "min": 0, "max": 256,
                           "tooltip": "Frames per model call. 0 = auto (8 on GPU, 1 on CPU)"}),
            "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                       "tooltip": "Video: temporal EMA of depth and of its 1/99% range across the batch "
                                  "(script default 0.6). 0 = off (use for unrelated images)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "DEPTH_RAW", "STRING")
    RETURN_NAMES = ("depth", "depth_mask", "raw_depth", "tag")
    FUNCTION = "estimate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Depth Anything V2 depth, normalized to 0-1 by the 1/99 percentiles (white = close)."

    def estimate(self, image, depth_model, res, batch_size, smooth):
        m = depth_model["model"]
        dev = depth_model["device"]
        if m.dev != dev:
            m.to(dev)
        B, h, w, _ = image.shape
        bs = batch_size or (8 if dev.type != "cpu" else 1)
        norm = core.DepthNormalizer(smooth)
        out = torch.empty(B, h, w)
        raw = torch.empty(B, h, w)
        bar = _Progress(B)
        i = 0
        try:
            while i < B:
                check_interrupt()
                chunk = image[i:i + bs, ..., :3]
                try:
                    D = m.batch(chunk, res)
                except (RuntimeError, MemoryError) as e:
                    if not core.is_oom(e) or bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    core.empty_cache(dev)
                    print(f"[DepthDots] out of memory, batch -> {bs}")
                    continue
                with torch.inference_mode():
                    for j in range(D.shape[0]):
                        out[i + j] = norm(D[j]).cpu()
                        raw[i + j] = D[j].cpu()
                i += D.shape[0]
                bar.update(D.shape[0])
        finally:
            if not depth_model.get("keep", True) and dev.type != "cpu":
                m.to("cpu")
                core.empty_cache(dev)
        return (gray_to_image(out), out, raw, core.depth_tag(m.short, res, smooth))


# ----------------------------------------------------------------------------- depth from image
class DepthDotsDepthFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth_image": ("IMAGE", {"tooltip": "An existing depth map (white = close), gray or RGB"}),
            "normalize": ("BOOLEAN", {"default": True, "tooltip": "Stretch by the 1/99 percentiles (like --depth-in)"}),
            "invert": ("BOOLEAN", {"default": False, "tooltip": "For maps where white = far"}),
        }, "optional": {
            "reference_image": ("IMAGE", {"tooltip": "Resize the depth map to this image's size"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("depth", "depth_mask", "tag")
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    DESCRIPTION = "Use a loaded depth map (LoadImage) instead of the model, like the script's --depth-in."

    def convert(self, depth_image, normalize, invert, reference_image=None):
        B, h, w, _ = depth_image.shape
        if reference_image is not None:
            h, w = reference_image.shape[1:3]
        out = torch.empty(B, h, w)
        for i in range(B):
            d = image_to_depth01(depth_image[i])
            if d.shape != (h, w):  # like the script: 8-bit L image resized with PIL (bicubic)
                u8 = (d.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                d = torch.from_numpy(np.asarray(Image.fromarray(u8).resize((w, h), Image.BICUBIC),
                                                dtype=np.float32) / 255)
            if invert:
                d = 1 - d
            if normalize:
                lo, hi = core.percentile(d, [1, 99])
                d = core.normalize(d, lo, hi)
            out[i] = d
        return (gray_to_image(out), out, "depthin")


# ----------------------------------------------------------------------------- enhance
class DepthDotsEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE",),
            "detail": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                       "tooltip": "Boost local depth relief such as facial features (try 1-2). 0 = off"}),
            "detail_radius": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 500.0, "step": 1.0,
                              "tooltip": "Size in pixels of the detail being boosted (smaller = finer)"}),
            "shade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                      "tooltip": "Mix in the photo's dark areas (eyes, brows, mouth); needs source_image"}),
        }, "optional": {
            "source_image": ("IMAGE",),
            "tag": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("density", "density_mask", "tag")
    FUNCTION = "enhance"
    CATEGORY = CATEGORY
    DESCRIPTION = "Detail (high-pass) boost and photo shading mixed into the depth -> dot density map."

    def enhance(self, depth, detail, detail_radius, shade, source_image=None, tag=""):
        dev = comfy_device()
        B, h, w, _ = depth.shape
        out = torch.empty(B, h, w)
        for i in range(B):
            check_interrupt()
            d = image_to_depth01(depth[i]).to(dev)
            src = None
            if source_image is not None and shade > 0:
                src = match_size(frame_of(source_image, i)[..., :3], h, w)
            out[i] = core.enhance(d, detail, detail_radius, src, shade).cpu()
        t = [tag] if tag else []
        if detail > 0:
            t.append(f"det{core.num(detail)}")
        if shade > 0 and source_image is not None:
            t.append(f"sh{core.num(shade)}")
        return (gray_to_image(out), out, "_".join(t))


# ----------------------------------------------------------------------------- render
class DepthDotsStippleRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "density": ("IMAGE", {"tooltip": "Depth or density map (white = close = more dots)"}),
            "dots": ("INT", {"default": 60000, "min": 1, "max": 20000000, "step": 1000,
                     "tooltip": "Approximate number of dots per image/frame"}),
            "gamma": ("FLOAT", {"default": 1.5, "min": 0.05, "max": 10.0, "step": 0.05,
                      "tooltip": ">1 pushes dots harder toward close areas"}),
            "floor": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                      "tooltip": "Minimum density for the far background"}),
            "size": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 50.0, "step": 0.1,
                     "tooltip": "Dot radius in output pixels"}),
            "scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1,
                      "tooltip": "Output size relative to the input"}),
            "size_by_depth": ("BOOLEAN", {"default": False, "tooltip": "Closer dots are also bigger"}),
            "color_mode": (["depth_shaded", "flat", "source_color"], {"default": "depth_shaded",
                           "tooltip": "depth_shaded: tone = depth 0-255; flat: solid ink; source_color: needs source_image"}),
            "dark": ("BOOLEAN", {"default": False, "tooltip": "White dots on black"}),
            "fixed_dots": ("BOOLEAN", {"default": False,
                           "tooltip": "Video: dots stay in fixed spots between frames (no flicker); off = re-rolled each frame"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
            "even_size": ("BOOLEAN", {"default": False,
                          "tooltip": "Round the output down to even width/height (needed for H.264 video)"}),
            "filename_prefix": ("STRING", {"default": "DepthDots/dots",
                                "tooltip": "Start of the filename_prefix output; the parameter tag is appended"}),
        }, "optional": {
            "source_image": ("IMAGE", {"tooltip": "Original image/frames for source_color"}),
            "tag": ("STRING", {"forceInput": True, "tooltip": "tag from Estimate/Enhance, for the filename"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_prefix")
    FUNCTION = "render"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Weighted random (or fixed-threshold) stippling + antialiased dot rendering on the GPU. "
                   "Batches = video frames: random dots re-rolled per frame, or fixed dots.")

    def render(self, density, dots, gamma, floor, size, scale, size_by_depth, color_mode, dark, fixed_dots,
               seed, even_size, filename_prefix, source_image=None, tag=""):
        dev = comfy_device()
        B, h, w, _ = density.shape
        W, H = int(w * scale), int(h * scale)
        if even_size:
            W, H = W // 2 * 2, H // 2 * 2
        color = color_mode == "source_color"
        if color and source_image is None:
            raise ValueError("color_mode=source_color needs source_image connected")
        stip = core.make_stippler(fixed_dots, w, h, dots, gamma, floor, seed, device=dev)
        out = torch.empty(B, H, W, 3)
        bar = _Progress(B)
        with torch.inference_mode():
            for i in range(B):
                check_interrupt()
                d = image_to_depth01(density[i]).to(dev).clamp(0, 1)
                rgb = match_size(frame_of(source_image, i)[..., :3], h, w).to(dev) if color else None
                canvas = core.render(stip, d, (w, h), scale, size, dark, rgb, size_by_depth,
                                     color_mode == "flat", canvas_size=(W, H))
                out[i] = canvas.cpu().float() / 255
                bar.update(1)
        t = core.full_tag(tag, dots, gamma, size, scale, floor, fixed_dots, dark, color,
                          color_mode == "flat", size_by_depth)
        prefix = (filename_prefix.rstrip("_") + "_" + t) if filename_prefix else t
        return (out, prefix)


# ----------------------------------------------------------------------------- audio (first track)
def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class DepthDotsLoadAudioFirstTrack:
    @classmethod
    def INPUT_TYPES(cls):
        files = []
        if folder_paths is not None:
            d = folder_paths.get_input_directory()
            files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
                           and f.rsplit(".", 1)[-1].lower() in VIDEO_EXT)
        return {"required": {
            "video": (files, {"tooltip": "Video in ComfyUI/input (same file as VHS Load Video)"}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6, "step": 0.01}),
            "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6, "step": 0.01,
                                 "tooltip": "0 = to the end"}),
        }, "optional": {
            "video_path": ("STRING", {"default": "", "tooltip": "Absolute path; overrides 'video'"}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decodes only the FIRST audio track (-map 0:a:0). Use instead of VHS's audio for iPhone "
                   "clips, whose extra 'apac' spatial-audio track many ffmpeg builds can't decode.")

    @classmethod
    def IS_CHANGED(cls, video, start_seconds, duration_seconds, video_path=""):
        p = cls._path(video, video_path)
        return f"{p}:{os.path.getmtime(p) if os.path.exists(p) else 0}"

    @staticmethod
    def _path(video, video_path=""):
        if video_path.strip():
            return video_path.strip()
        return folder_paths.get_annotated_filepath(video) if folder_paths is not None else video

    def load(self, video, start_seconds, duration_seconds, video_path=""):
        exe = _ffmpeg()
        if exe is None:
            raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg)")
        path = self._path(video, video_path)
        cmd = [exe, "-v", "error"]
        if start_seconds > 0:
            cmd += ["-ss", str(start_seconds)]
        cmd += ["-i", path]
        if duration_seconds > 0:
            cmd += ["-t", str(duration_seconds)]
        sr = 48000
        cmd += ["-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"]
        r = subprocess.run(cmd, capture_output=True)
        err = r.stderr.decode(errors="replace")
        if r.returncode != 0 and ("matches no streams" in err or "does not contain any stream" in err):
            print(f"[DepthDots] {path} has no audio track; continuing without audio")
            return (None,)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg could not read the first audio track of {path}: "
                               f"{r.stderr.decode(errors='replace')[-300:]}")
        a = torch.frombuffer(bytearray(r.stdout), dtype=torch.float32).reshape(-1, 2).t().contiguous()
        return ({"waveform": a[None], "sample_rate": sr},)


# ----------------------------------------------------------------------------- sketch outline
def _batch_pair(a, b, what):
    """Broadcast a batch of 1 against the other; refuse other length mismatches."""
    na, nb = a.shape[0], b.shape[0]
    if na != nb and 1 not in (na, nb):
        raise ValueError(f"DepthDots: {what} batch sizes differ ({na} vs {nb}); "
                         "connect matching frame counts or a single image to broadcast")
    return max(na, nb)


class DepthDotsSketchOutline:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "method": (["combined", "xdog", "depth_edges"], {"default": "combined",
                       "tooltip": "xdog: pencil lines from the image; depth_edges: silhouettes/folds from depth; "
                                  "combined: both (max)"}),
            "line_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                              "tooltip": "Line darkness/opacity gain"}),
            "line_thickness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                               "tooltip": "1 = natural pencil width; >1 dilates, <1 thins"}),
            "sigma": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 10.0, "step": 0.1,
                      "tooltip": "XDoG scale in input pixels: bigger = bolder, fewer details"}),
            "k": ("FLOAT", {"default": 1.6, "min": 1.05, "max": 5.0, "step": 0.05, "tooltip": "XDoG sigma ratio"}),
            "phi": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "XDoG sharpness: higher = crisper, more inky lines"}),
            "epsilon": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "XDoG threshold (relative to the frame's edge contrast): higher = fewer lines"}),
            "depth_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01,
                                "tooltip": "Depth edges: higher = only strong silhouettes, lower = also folds"}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                        "tooltip": "Pre-blur (sigma, px) against noise / grain"}),
            "background_fade": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "Needs depth: fade image lines in far areas so the people/subject read"}),
            "clean": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                      "tooltip": "Remove isolated specks (0 = off)"}),
            "temporal_smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                                "tooltip": "Video: EMA of the lines across frames to reduce flicker (0 = off)"}),
            "scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1,
                      "tooltip": "Output size vs input; match the Stipple Render scale (ignored with reference_image)"}),
            "dark": ("BOOLEAN", {"default": False, "tooltip": "White lines on black (for dark dot renders)"}),
            "invert": ("BOOLEAN", {"default": False, "tooltip": "Swap line/background colors of the IMAGE output"}),
        }, "optional": {
            "depth": ("IMAGE", {"tooltip": "Depth from Estimate Depth / Depth From Image (needed for depth_edges, "
                                "background_fade)"}),
            "reference_image": ("IMAGE", {"tooltip": "Resize the output to this image's size (e.g. the dots render)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("outline", "outline_mask", "tag", "xdog_lines", "depth_edge_lines")
    OUTPUT_TOOLTIPS = ("Final outline at the output size", "Line mask (1 = line)", "Settings tag for the filename",
                       "Step preview: XDoG lines only (input size; blank if unused)",
                       "Step preview: depth-edge lines only (input size; blank if unused)")
    FUNCTION = "outline"
    CATEGORY = CATEGORY
    DESCRIPTION = "Hand-drawn style sketch outline (XDoG pencil lines + depth silhouettes), batched over frames."

    def outline(self, image, method, line_strength, line_thickness, sigma, k, phi, epsilon, depth_threshold,
                denoise, background_fade, clean, temporal_smooth, scale, dark, invert, depth=None,
                reference_image=None):
        dev = comfy_device()
        B, h, w, _ = image.shape
        if depth is not None:
            B = _batch_pair(image, depth, "image/depth")
        if reference_image is not None:
            H, W = reference_image.shape[1:3]
        else:
            H, W = max(1, int(h * scale)), max(1, int(w * scale))
        sf = min(H / h, W / w)
        use_depth = depth is not None and (method != "xdog" or background_fade > 0)
        if method == "depth_edges" and depth is None:
            raise ValueError("DepthDots Sketch Outline: method depth_edges needs the depth input connected")
        out = torch.empty(B, H, W)
        xd_out = torch.zeros(B, h, w)
        de_out = torch.zeros(B, h, w)
        bar = _Progress(B)
        prev = None
        chunk = 8 if dev.type != "cpu" else 4
        with torch.inference_mode():
            for i in range(0, B, chunk):
                check_interrupt()
                idx = torch.arange(i, min(B, i + chunk))
                rgb = image[idx if image.shape[0] > 1 else torch.zeros_like(idx)][..., :3].to(dev)
                d = None
                if use_depth:
                    dsel = depth[idx if depth.shape[0] > 1 else torch.zeros_like(idx)]
                    d = torch.stack([image_to_depth01(f) for f in dsel]).to(dev)[:, None]
                L, xd, de = core.sketch_outline(rgb, d, method, sigma, k, phi, epsilon, depth_threshold, denoise,
                                                1.0, background_fade if d is not None else 0.0, parts=True)
                n = L.shape[0]
                if xd is not None:
                    xd_out[i:i + n] = (xd[:, 0] * line_strength).clamp(0, 1).cpu()
                if de is not None:
                    de_out[i:i + n] = (de[:, 0] * line_strength).clamp(0, 1).cpu()
                if temporal_smooth > 0:
                    for j in range(L.shape[0]):
                        prev = L[j] if prev is None else temporal_smooth * prev + (1 - temporal_smooth) * L[j]
                        L[j] = prev
                L = core.finish_lines(L, (H, W), line_thickness, line_strength, clean, sf)
                out[i:i + L.shape[0]] = L[:, 0].cpu()
                bar.update(L.shape[0])
        light_lines = dark != invert            # white lines on black
        show = (lambda m: m) if light_lines else (lambda m: 1 - m)
        t = f"{method}_ls{core.num(line_strength)}_lt{core.num(line_thickness)}" + ("_inv" if invert else "")
        return (gray_to_image(show(out)), out, t, gray_to_image(show(xd_out)), gray_to_image(show(de_out)))


# ----------------------------------------------------------------------------- combine
class DepthDotsCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dots": ("IMAGE",),
            "mode": (["auto", "multiply", "screen", "lines_over", "dots_masked_by_lines"], {"default": "auto",
                     "tooltip": "auto: multiply (dark lines) on light dots, screen (light lines) on dark dots; "
                                "lines_over paints line_color; dots_masked_by_lines keeps dots only on the lines"}),
            "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "line_color": ("STRING", {"default": "auto",
                           "tooltip": "auto (black on light, white on dark) or hex like #1a1a1a / #c03020"}),
            "filename_prefix": ("STRING", {"default": "DepthDots/sketch",
                                "tooltip": "Used when no dots_filename_prefix is connected"}),
        }, "optional": {
            "outline": ("IMAGE", {"tooltip": "Outline image (either polarity)"}),
            "outline_mask": ("MASK", {"tooltip": "Line mask (1 = line); preferred over outline when connected"}),
            "dots_filename_prefix": ("STRING", {"forceInput": True}),
            "outline_tag": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_prefix")
    FUNCTION = "combine"
    CATEGORY = CATEGORY
    DESCRIPTION = "Combine the dots render with the sketch outline (outline auto-resized to the dots size)."

    def combine(self, dots, mode, opacity, line_color, filename_prefix, outline=None, outline_mask=None,
                dots_filename_prefix="", outline_tag=""):
        if outline is None and outline_mask is None:
            raise ValueError("DepthDots Combine: connect outline or outline_mask")
        dev = comfy_device()
        src = outline_mask if outline_mask is not None else outline
        B = _batch_pair(dots, src, "dots/outline")
        _, H, W, _ = dots.shape
        dark = core.is_dark_bg(dots[:1])
        odark = core.is_dark_bg(outline[:1]) if outline_mask is None else None
        out = torch.empty(B, H, W, 3)
        bar = _Progress(B)
        with torch.inference_mode():
            for i in range(B):
                check_interrupt()
                d = frame_of(dots, i)[..., :3].to(dev)[None]
                if outline_mask is not None:
                    a = frame_of(outline_mask, i).to(dev)[None].float()
                else:
                    a = core.outline_alpha(frame_of(outline, i).to(dev)[None], odark)
                if a.shape[-2:] != (H, W):
                    a = F.interpolate(a[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
                out[i] = core.combine(d, a.clamp(0, 1), mode, opacity, line_color, dark)[0].clamp(0, 1).cpu()
                bar.update(1)
        base = (dots_filename_prefix or filename_prefix or "DepthDots/sketch").rstrip("_")
        extra = ([outline_tag] if outline_tag else []) + ([mode] if mode != "auto" else []) + \
            ([f"c{core.safe_name(line_color.lstrip('#'))}"] if line_color not in ("", "auto") else []) + \
            ([f"op{core.num(opacity)}"] if opacity != 1.0 else [])
        prefix = "_".join([base] + extra)
        return (out, prefix)


NODE_CLASS_MAPPINGS = {
    "DepthDotsLoader": DepthDotsLoader,
    "DepthDotsEstimateDepth": DepthDotsEstimateDepth,
    "DepthDotsDepthFromImage": DepthDotsDepthFromImage,
    "DepthDotsEnhance": DepthDotsEnhance,
    "DepthDotsStippleRender": DepthDotsStippleRender,
    "DepthDotsLoadAudioFirstTrack": DepthDotsLoadAudioFirstTrack,
    "DepthDotsSketchOutline": DepthDotsSketchOutline,
    "DepthDotsCombine": DepthDotsCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthDotsLoader": "DepthDots Depth Anything Loader",
    "DepthDotsEstimateDepth": "DepthDots Estimate Depth",
    "DepthDotsDepthFromImage": "DepthDots Depth From Image",
    "DepthDotsEnhance": "DepthDots Enhance",
    "DepthDotsStippleRender": "DepthDots Stipple Render",
    "DepthDotsLoadAudioFirstTrack": "DepthDots Load Audio (first track)",
    "DepthDotsSketchOutline": "DepthDots Sketch Outline",
    "DepthDotsCombine": "DepthDots Combine",
}
