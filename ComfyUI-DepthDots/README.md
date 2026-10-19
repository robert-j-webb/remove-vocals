# ComfyUI-DepthDots

Turn images and videos into **depth-driven dot art (stippling)**: Depth Anything V2 estimates depth,
closer areas get more dots, and the dots are drawn with an antialiased GPU rasterizer.
Everything runs on the GPU (CUDA, **Apple Silicon MPS**, or CPU): batched fp16 depth inference with
on-device preprocessing, percentile normalization, temporal smoothing, detail/shade enhancement,
weighted dot sampling and rendering. Same look as the `depth_dots.py` script.

## Install

1. Copy (or `git clone`) this folder into `ComfyUI/custom_nodes/ComfyUI-DepthDots`
   (or unzip `ComfyUI-DepthDots.zip` there).
2. Install the one extra dependency into ComfyUI's Python (torch already comes with ComfyUI):
   ```
   pip install -r ComfyUI/custom_nodes/ComfyUI-DepthDots/requirements.txt     # transformers
   ```
   (ComfyUI Desktop / portable: use its bundled python, e.g. `python_embeded\python.exe -m pip ...`.)
3. For the video workflow install **ComfyUI-VideoHelperSuite** (ComfyUI Manager -> "VideoHelperSuite")
   and have `ffmpeg` available (VHS also accepts `imageio-ffmpeg`).
4. Restart ComfyUI and drag a workflow from `workflows/` onto the canvas.

Models download automatically on first use into `ComfyUI/models/depth_anything/` (Hugging Face cache
layout). You can also put a downloaded model folder (with `config.json`) there named like the repo,
e.g. `models/depth_anything/Depth-Anything-V2-Large-hf/`. Sizes: small ~100 MB, base ~400 MB, large ~1.3 GB.

## Workflows (`workflows/`)

| file | what |
|---|---|
| `depthdots_image.json` | LoadImage -> Loader -> Estimate Depth -> Enhance -> Stipple Render -> SaveImage, plus previews of depth and density |
| `depthdots_depth_map_in.json` | Use an existing depth map: LoadImage (photo) + LoadImage (depth map) -> Depth From Image -> Enhance -> Render -> SaveImage |
| `depthdots_video.json` | VHS Load Video -> Estimate (smooth 0.6) -> Enhance -> Render (even_size) -> VHS Video Combine (H.264 crf 18, source fps via Video Info, audio passed through) |
| `depthdots_image_sketch.json` | Image: dots chain + **Sketch Outline** (fed the photo and the depth) -> **Combine** -> SaveImage |
| `depthdots_video_sketch.json` | Video: dots chain + Sketch Outline (temporal_smooth 0.5) in parallel -> Combine -> VHS Video Combine with audio passthrough |

Defaults point at `IMG_2379.png` / `falcon.mov` in `ComfyUI/input`; pick your own file in the loader.

**Step previews.** Every workflow except the depth-map one has a group **"Previews (optional — bypass with Ctrl+B)"**
with numbered previews laid out left to right: `1. Source`, `2. Depth`, `3. Density (after Enhance)`, then for the
sketch workflows `4. XDoG sketch lines`, `5. Depth-edge lines`, `6. Combined outline`, `7. Dots render`,
`8. Final (dots + outline)` (plain workflows: `4. Dots render`). Images use PreviewImage; videos use VHS Video
Combine with `save_output` off (temp H.264 crf 28). Select the group and press Ctrl+B to bypass them all;
the run and the saved output are unaffected.

Filenames: the Render node outputs a `filename_prefix` string that the SaveImage / Video Combine
prefix input is wired to, so files are named after the settings, like the script:
`DepthDots/dots_base_r518_n60000_g1.5_s1.2_x2_det1.5_sh0.3_fixed_dark_color_00001_.png`.
Disconnect that link to type your own prefix.

## Nodes (category **DepthDots**)

**DepthDots Depth Anything Loader** -> `DEPTH_MODEL`
- `model` small | base | large; `hf_model_id` (optional, overrides: any HF depth-estimation model id)
- `precision` auto (fp16 on CUDA/MPS if a test pass is finite, else fp32) | fp16 | fp32
- `device` auto (= ComfyUI's device, respects `--cpu`) | cuda | mps | cpu
- `keep_on_device` keep the model on the GPU between runs (off = move to CPU after each use)

**DepthDots Estimate Depth** (`image`, `depth_model`) -> `depth` IMAGE, `depth_mask` MASK, `raw_depth`, `tag`
- `res` inference short side (multiple of 14). 518 default, 770-1036 for sharper faces
- `batch_size` frames per model call (0 = auto: 8 on GPU, 1 on CPU; halves itself on out-of-memory)
- `smooth` 0-0.95: for video frames, EMA of the raw depth and of the 1/99 % range across the batch
  (the script's `--smooth`, default 0.6 there). **0 = off; keep 0 for batches of unrelated images.**
- Output depth is normalized by the 1/99 percentiles, white = close. `raw_depth` (type `DEPTH_RAW`) is the
  unnormalized model output. `tag` (e.g. `base_r518`) feeds the filename.

**DepthDots Depth From Image** (`depth_image`, optional `reference_image`) -> `depth`, `depth_mask`, `tag`
- Use a depth map you already have (connect `LoadImage`): grayscale or RGB both work (RGB is converted
  with the same luminance as PIL "L"). Resized to `reference_image` if given; `normalize` stretches by
  the 1/99 percentiles (like `--depth-in`); `invert` for maps where white = far.

**DepthDots Enhance** (`depth`, optional `source_image`, `tag`) -> `density` IMAGE, `density_mask`, `tag`
- `detail` high-pass boost of local relief (noses, eye sockets, mouths), `detail_radius` its size in px
- `shade` 0-1 mixes in the photo's dark areas (eyes, brows, lips), needs `source_image`
- Set detail and shade to 0 for a pass-through.

**DepthDots Stipple Render** (`density`, optional `source_image`, `tag`) -> `image`, `filename_prefix`
- `dots` (~dots per frame), `gamma` (>1 = more toward close areas), `floor` (background density)
- `size` dot radius in output px, `scale` output size vs input, `size_by_depth` (radius x (0.5 + depth))
- `color_mode` depth_shaded (tone = depth 0-255) | flat (solid ink) | source_color (dot = source pixel color)
- `dark` white dots on black; `fixed_dots` (video) dots stay put between frames, otherwise they are re-rolled
  every frame (grainy flicker); `seed`; `even_size` rounds the output to even width/height (needed for H.264)
- Batches are treated as video frames exactly like the script: one stippler per run, random dots
  re-sampled per frame, or one fixed candidate set for all frames.

**DepthDots Sketch Outline** (`image`, optional `depth`, `reference_image`) -> `outline` IMAGE, `outline_mask` MASK,
`tag`, `xdog_lines` IMAGE, `depth_edge_lines` IMAGE
- `method` combined (default) | xdog | depth_edges.
  - **xdog**: extended difference of Gaussians on the luminance (pencil look). `sigma` (scale, bigger = bolder,
    fewer details), `k` (sigma ratio), `phi` (sharpness), `epsilon` (threshold relative to the frame's edge contrast;
    higher = fewer lines).
  - **depth_edges**: Sobel on the depth map with edge thinning; catches silhouettes and folds that have little color
    contrast. `depth_threshold` higher = only strong silhouettes. Needs `depth`.
  - **combined**: max of both.
- `background_fade` (default 0.85, needs depth) fades image lines in far areas so the characters read instead of busy
  backgrounds. `denoise` pre-blur, `clean` removes isolated specks.
- `line_strength`, `line_thickness` (1 = natural; >1 dilates, <1 thins), `temporal_smooth` (EMA across frames vs
  flicker), `scale` (match Stipple Render's scale; or connect `reference_image`), `dark` (white lines on black),
  `invert`.
- `xdog_lines` / `depth_edge_lines` are the two parts before combining, at the input size, for previews.
- `tag` e.g. `combined_ls1_lt1`.

**DepthDots Combine** (`dots`, optional `outline` IMAGE or `outline_mask` MASK, `dots_filename_prefix`, `outline_tag`)
-> `image`, `filename_prefix`
- `mode` auto (multiply dark lines on light dots, screen light lines on dark renders) | multiply | screen |
  lines_over (paint `line_color`) | dots_masked_by_lines (keep dots only on the lines: lines drawn in dots).
- `opacity`, `line_color` (auto or hex like `#c03020`). The outline is resized to the dots size; a single
  image/mask is broadcast over a batch, other length mismatches raise a clear error. Either outline polarity works.
- `filename_prefix` output = dots prefix + outline tag (+ mode/color/opacity when not default), e.g.
  `DepthDots/dots_base_r518_n60000_g1.5_s1.2_x2_det1.5_sh0.3_combined_ls1_lt1`.

**DepthDots Load Audio (first track)** -> `AUDIO`
- Decodes only the first audio track (`-map 0:a:0`) of a video in `ComfyUI/input` (or `video_path`),
  as 48 kHz stereo. Use it for iPhone clips (see below). Clips without audio give no audio instead of an error.

## Tips
- **Sketch outline:** defaults are tuned for people (combined, sigma 1, epsilon 0.25, background_fade 0.85).
  Too many lines: raise `epsilon` (0.35) or `background_fade` (1.0); too few: lower `epsilon` (0.15),
  `depth_threshold` (0.1). Bolder: `sigma` 1.5 or `line_thickness` 1.5-2. Grainy video: `denoise` 1.5-2,
  `temporal_smooth` 0.5-0.7.
- **Faces / portraits:** model **large**, res **1036**, Enhance detail **1.5**, shade **0.3**.
- More texture: raise `dots` (100k-300k) and lower `size` (0.8-1.0); `scale` 3-4 for prints.
- Video without flicker: `fixed_dots` on, `smooth` 0.6-0.8.
- Speed: GPU fp16 is the default; batch 8-16 on CUDA. On Mac, if you see black/NaN output, set precision fp32.
  Long or high-res clips: set VHS `frame_load_cap` for previews; ComfyUI keeps all frames in RAM
  (render output at scale 2 is 4x the pixels).

## iPhone audio caveat (apac)
iPhone videos (.mov) can carry a second "apac" spatial-audio track that many ffmpeg builds can't decode.
VHS Load Video extracts audio with ffmpeg's default stream choice, so Video Combine may fail or come out
silent with such clips. In `depthdots_video.json` there is a ready **"iPhone clips: first audio track only"**
node: pick the same video in it and connect its `audio` to Video Combine's `audio` instead (or
disconnect audio). Alternatively remux once: `ffmpeg -i in.mov -map 0:v -map 0:a:0 -c copy out.mov`.

## Other caveats
- Dot positions for a given seed match the script on the same device, but differ between devices
  (CPU vs CUDA vs MPS random generators). Where several dots overlap a pixel the winning dot can vary
  slightly between runs (GPU scatter), which is invisible in practice.
- ComfyUI's memory manager does not track this model; on small GPUs turn `keep_on_device` off when it shares
  the GPU with big diffusion models.
- Unsupported MPS ops fall back to CPU automatically (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set if unset).

`depthdots_core.py` holds all the logic with no ComfyUI dependency and can be imported on its own.
