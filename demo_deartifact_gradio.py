import os
import gc
import numpy as np
import cv2
import gradio as gr
from PIL import Image

import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)

# -------------------------
# Import DIBR pipeline from your existing main.py
# -------------------------
from main import (
    DepthEstimator,
    normalize_depth,
    depth_to_disparity,
    edge_aware_depth_smooth,
    forward_warp_zbuffer,
    fill_holes_inpaint,
    make_anaglyph_red_cyan,
)

# -------------------------
# Utils (image)
# -------------------------
def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def np_rgb_to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")

def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def depth01_to_vis(depth01: np.ndarray) -> np.ndarray:
    vis = (np.clip(depth01, 0, 1) * 255).astype(np.uint8)
    return vis  # grayscale

def disp_to_vis(disp: np.ndarray) -> np.ndarray:
    d = disp.astype(np.float32)
    d = d - d.min()
    vis = (d / (d.max() + 1e-6) * 255).astype(np.uint8)
    return vis  # grayscale

# -------------------------
# Diffusion utils
# -------------------------
def resize_to_multiple_of_8(img: Image.Image, max_side: int | None = None) -> Image.Image:
    w, h = img.size
    if max_side is not None and max(w, h) > max_side:
        scale = max_side / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return img.resize((w, h), Image.LANCZOS)

def make_tile_condition(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.uint8)
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    sharp = cv2.addWeighted(arr, 1.15, blur, -0.15, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp)

def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------------
# Caches
# -------------------------
DEPTH_ESTIMATOR_CACHE = {}
DIFF_PIPE_CACHE = {}

def get_depth_estimator(model_type: str = "DPT_Hybrid"):
    if model_type not in DEPTH_ESTIMATOR_CACHE:
        DEPTH_ESTIMATOR_CACHE[model_type] = DepthEstimator(model_type=model_type, device="cpu")
    return DEPTH_ESTIMATOR_CACHE[model_type]

def get_diff_pipe(base_model: str, use_controlnet_tile: bool, controlnet_model: str, device: str, dtype: torch.dtype):
    key = (base_model, use_controlnet_tile, controlnet_model, device, str(dtype))
    if key in DIFF_PIPE_CACHE:
        return DIFF_PIPE_CACHE[key]

    if use_controlnet_tile:
        controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None,
        )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    DIFF_PIPE_CACHE[key] = pipe
    return pipe

def restore_with_diffusion(
    img_bgr: np.ndarray,
    base_model: str,
    use_controlnet_tile: bool,
    controlnet_model: str,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    cfg: float,
    seed: int,
    max_side: int,
    controlnet_scale: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = resize_to_multiple_of_8(pil, max_side=max_side)

    pipe = get_diff_pipe(base_model, use_controlnet_tile, controlnet_model, device, dtype)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if use_controlnet_tile:
        control = make_tile_condition(pil)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=pil,
            control_image=control,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=generator,
        ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=pil,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator,
        ).images[0]

    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out_bgr

# -------------------------
# Main: DIBR -> (warp/fill) -> diffusion restore L/R
# -------------------------
def run_dibr_plus_restore(
    img_rgb,
    # DIBR params
    model_type, k, dmax, alpha, smooth_depth, inpaint_radius, inpaint_method,
    # diffusion params
    do_restore,
    base_model, use_controlnet_tile, controlnet_model,
    prompt, negative,
    strength, steps, cfg, seed, max_side, controlnet_scale,
):
    if img_rgb is None:
        return (None,) * 12

    # 1) RGB -> BGR
    bgr = rgb_to_bgr(img_rgb)

    # 2) depth
    depth_est = get_depth_estimator(model_type)
    depth = depth_est.predict_depth(bgr)
    depth01 = normalize_depth(depth)

    # 3) optional smooth
    if smooth_depth:
        depth01_s = edge_aware_depth_smooth(depth01, sigma_s=7, sigma_r=0.1)
    else:
        depth01_s = depth01

    # 4) disparity
    disp = depth_to_disparity(depth01_s, k=float(k), dmax=float(dmax), alpha=float(alpha))

    # 5) forward warp (holes)
    right_w, right_m = forward_warp_zbuffer(bgr, disp, depth01_s, direction="R")
    left_w,  left_m  = forward_warp_zbuffer(bgr, disp, depth01_s, direction="L")

    # 6) fill holes
    right_f = fill_holes_inpaint(right_w, right_m, radius=int(inpaint_radius), method=str(inpaint_method))
    left_f  = fill_holes_inpaint(left_w,  left_m,  radius=int(inpaint_radius), method=str(inpaint_method))

    # 7) 3D outputs BEFORE restore
    sbs_before = np.concatenate([left_f, right_f], axis=1)
    ana_before = make_anaglyph_red_cyan(left_f, right_f)

    # depth/disp visual
    depth_vis = depth01_to_vis(depth01_s)
    disp_vis = disp_to_vis(disp)

    # 8) diffusion restore for L/R (optional)
    if do_restore:
        # 你也可以把 seed 分开：seed 和 seed+1，避免左右完全一致的“纹理同步”
        left_r  = restore_with_diffusion(
            left_f,
            base_model, use_controlnet_tile, controlnet_model,
            prompt, negative,
            strength, steps, cfg, seed, max_side, controlnet_scale
        )
        right_r = restore_with_diffusion(
            right_f,
            base_model, use_controlnet_tile, controlnet_model,
            prompt, negative,
            strength, steps, cfg, seed + 1, max_side, controlnet_scale
        )
    else:
        left_r, right_r = left_f, right_f

    # 9) 3D outputs AFTER restore
    sbs_after = np.concatenate([left_r, right_r], axis=1)
    ana_after = make_anaglyph_red_cyan(left_r, right_r)

    # masks visual (white=hole)
    left_m_vis  = (left_m.astype(np.uint8) * 255)
    right_m_vis = (right_m.astype(np.uint8) * 255)

    torch_gc()

    return (
        bgr_to_rgb(left_w), bgr_to_rgb(right_w),          # warp L/R
        left_m_vis, right_m_vis,                         # hole masks
        bgr_to_rgb(left_f), bgr_to_rgb(right_f),         # filled L/R
        bgr_to_rgb(left_r), bgr_to_rgb(right_r),         # restored L/R
        bgr_to_rgb(sbs_before), bgr_to_rgb(sbs_after),   # SBS before/after
        bgr_to_rgb(ana_before), bgr_to_rgb(ana_after),   # anaglyph before/after
        depth_vis, disp_vis,                             # depth/disp
    )

# -------------------------
# Gradio UI
# -------------------------
DEFAULT_PROMPT = (
    "photo, clean, natural details, sharp, high quality, "
    "no compression artifacts, no blocky artifacts, no ringing, no noise"
)
DEFAULT_NEG = (
    "worst quality, low quality, blurry, over-smoothed, plastic skin, "
    "cartoon, painting, deformed, extra fingers, text, watermark"
)

with gr.Blocks(title="DIBR + Diffusion Restore (Stereo) Demo") as demo:
    gr.Markdown(
        "## DIBR Stereo (Warp) → Inpaint → Diffusion Restore (Left & Right) Demo\n"
        "上传一张图：先生成 **Left/Right**（warp→fill），再可选用 **SD+ControlNet Tile** 分别修复左右眼图，最后输出 **SBS/Anaglyph**。\n\n"
        "提示：\n"
        "- 想更明显 3D：提高 `dmax` 或 `k`\n"
        "- 想少重画：降低 diffusion `strength`（例如 0.25~0.35）"
    )

    with gr.Row():
        inp = gr.Image(label="Input Image (RGB)", type="numpy")

        with gr.Column():
            gr.Markdown("### DIBR 参数")
            model_type = gr.Dropdown(
                choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                value="DPT_Hybrid",
                label="Depth model (MiDaS)",
            )
            k = gr.Slider(0, 80, value=32, step=1, label="k (stereo strength)")
            dmax = gr.Slider(0, 60, value=24, step=1, label="dmax (disparity budget, px)")
            alpha = gr.Slider(0.3, 3.0, value=1.0, step=0.05, label="alpha (depth compression)")
            smooth_depth = gr.Checkbox(value=True, label="Edge-aware depth smoothing")
            inpaint_radius = gr.Slider(1, 10, value=3, step=1, label="Inpaint radius")
            inpaint_method = gr.Radio(choices=["telea", "ns"], value="telea", label="Inpaint method")

            gr.Markdown("### Diffusion 修复（左右眼分别修）")
            do_restore = gr.Checkbox(value=True, label="Enable diffusion restore (Left & Right)")
            base_model = gr.Textbox(value="runwayml/stable-diffusion-v1-5", label="Base model")
            use_controlnet_tile = gr.Checkbox(value=True, label="Use ControlNet Tile")
            controlnet_model = gr.Textbox(value="lllyasviel/control_v11f1e_sd15_tile", label="ControlNet model")

            prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt", lines=2)
            negative = gr.Textbox(value=DEFAULT_NEG, label="Negative prompt", lines=2)

            with gr.Row():
                strength = gr.Slider(0.05, 0.80, value=0.35, step=0.01, label="strength")
                cfg = gr.Slider(1.0, 10.0, value=5.5, step=0.1, label="CFG")

            with gr.Row():
                steps = gr.Slider(5, 50, value=20, step=1, label="steps")
                seed = gr.Slider(0, 999999, value=23, step=1, label="seed")

            with gr.Row():
                max_side = gr.Slider(256, 1536, value=768, step=64, label="max_side")
                controlnet_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="controlnet_scale")

            btn = gr.Button("Generate (Warp → Fill → Restore)")

    gr.Markdown("### 1) Warp 结果（带洞） + Hole Mask")
    with gr.Row():
        out_lw = gr.Image(label="Left Warp (with holes)")
        out_rw = gr.Image(label="Right Warp (with holes)")
    with gr.Row():
        out_lm = gr.Image(label="Left Hole Mask (white=hole)")
        out_rm = gr.Image(label="Right Hole Mask (white=hole)")

    gr.Markdown("### 2) Inpaint 补洞后（可作为 diffusion 输入）")
    with gr.Row():
        out_lf = gr.Image(label="Left Filled")
        out_rf = gr.Image(label="Right Filled")

    gr.Markdown("### 3) Diffusion 修复后（左右眼分别输出）")
    with gr.Row():
        out_lr = gr.Image(label="Left Restored")
        out_rr = gr.Image(label="Right Restored")

    gr.Markdown("### 4) 3D 观看输出（SBS / Anaglyph：修复前 vs 修复后）")
    with gr.Row():
        out_sbs_b = gr.Image(label="SBS Before (Left | Right)")
        out_sbs_a = gr.Image(label="SBS After (Left | Right)")
    with gr.Row():
        out_ana_b = gr.Image(label="Anaglyph Before")
        out_ana_a = gr.Image(label="Anaglyph After")

    gr.Markdown("### 5) Depth / Disparity 可视化")
    with gr.Row():
        out_depth = gr.Image(label="Depth (0-255)")
        out_disp = gr.Image(label="Disparity (0-255)")

    btn.click(
        fn=run_dibr_plus_restore,
        inputs=[
            inp,
            model_type, k, dmax, alpha, smooth_depth, inpaint_radius, inpaint_method,
            do_restore,
            base_model, use_controlnet_tile, controlnet_model,
            prompt, negative,
            strength, steps, cfg, seed, max_side, controlnet_scale,
        ],
        outputs=[
            out_lw, out_rw,
            out_lm, out_rm,
            out_lf, out_rf,
            out_lr, out_rr,
            out_sbs_b, out_sbs_a,
            out_ana_b, out_ana_a,
            out_depth, out_disp,
        ],
    )

if __name__ == "__main__":
    # 如果你之前遇到 localhost 不可用，就用 share=True
    demo.launch(share=True)