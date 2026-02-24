import argparse
import os
from PIL import Image
import numpy as np
import torch
import cv2

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)

# -------------------------
# Utils
# -------------------------
def read_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

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
    """
    ControlNet Tile 通常用“原图本身”作为条件即可；
    但为了更稳一点，这里做一个轻度锐化/对比增强，让结构更清晰。
    """
    arr = np.array(img).astype(np.uint8)
    # 轻微反锐化（unsharp mask）
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    sharp = cv2.addWeighted(arr, 1.15, blur, -0.15, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径（jpg/png，尽量别用webp）")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径（png/jpg）")

    # Model options
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="基础 SD 模型（默认 SD1.5）")
    parser.add_argument("--use_controlnet_tile", action="store_true",
                        help="使用 ControlNet Tile（强烈推荐用于去伪影/修复细节）")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11f1e_sd15_tile",
                        help="ControlNet Tile 权重名（HF）")

    # Restore parameters
    parser.add_argument("--prompt", type=str, default=(
        "photo, clean, natural details, sharp, high quality, "
        "no compression artifacts, no blocky artifacts, no ringing, no noise"
    ))
    parser.add_argument("--negative", type=str, default=(
        "worst quality, low quality, blurry, over-smoothed, plastic skin, "
        "cartoon, painting, deformed, extra fingers, text, watermark"
    ))

    parser.add_argument("--strength", type=float, default=0.35,
                        help="img2img 变化强度：0.2~0.5 通常适合去伪影（越大越容易重画）")
    parser.add_argument("--steps", type=int, default=20, help="推理步数 15~30")
    parser.add_argument("--cfg", type=float, default=5.5, help="CFG 4.5~7 一般够用")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max_side", type=int, default=768,
                        help="最大边缩放（8GB 建议 512~768，越大越吃显存）")
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                        help="ControlNet 强度（tile 推荐 0.8~1.2）")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load image
    init_img = read_image(args.input)
    init_img = resize_to_multiple_of_8(init_img, max_side=args.max_side)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Build pipeline
    if args.use_controlnet_tile:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,  # 纯复原任务一般不需要安全审查（也避免一些兼容问题）
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            safety_checker=None,
        )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # 8GB 显存友好设置
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        # 有 xformers 就用（更省显存/更快）
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Control image for tile
    if args.use_controlnet_tile:
        control_img = make_tile_condition(init_img)

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image=init_img,
            control_image=control_img,
            strength=float(args.strength),
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.cfg),
            controlnet_conditioning_scale=float(args.controlnet_scale),
            generator=generator,
        )
    else:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image=init_img,
            strength=float(args.strength),
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.cfg),
            generator=generator,
        )

    out = result.images[0]
    save_image(out, args.output)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()