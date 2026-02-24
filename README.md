# DIBR-Diffusion-Stereo

Single-image 2D → 3D stereo generation with DIBR, optionally enhanced by Stable Diffusion + ControlNet Tile.

一个从单张图片生成左右眼立体视图的完整流水线，基于 DIBR（Depth-Image-Based Rendering），并可选使用 Stable Diffusion + ControlNet Tile 对左右眼结果进行去伪影和细节修复。

---

## ✨ Features

- Mono image → stereo Left / Right views
- MiDaS depth estimation
- Forward warp + Z-buffer
- Hole filling (OpenCV inpaint)
- Optional Stable Diffusion restoration for both eyes
- ControlNet Tile for structure-preserving artifact removal
- Gradio interactive demo
- Side-by-Side stereo + Red-Cyan Anaglyph output
- Full visualization of intermediate stages

---

## 🧠 Pipeline Overview
Input Image
│
▼
Depth Estimation (MiDaS)
│
▼
Disparity Generation
│
▼
Forward Warp (Left / Right)
│
▼
Hole Filling (Inpaint)
│
├───────────────► Normal Stereo Output
│
▼
Stable Diffusion (Left & Right separately)
│
▼
Restored Stereo Output

---

# 🚀 Diffusion Version Demo

## Input

![input](demo_images/diffusion/01_input.png)

---

## Stage 1 — Forward Warp (with holes)

Left Warp  
![left_warp](demo_images/diffusion/02_left_warp.png)

Right Warp  
![right_warp](demo_images/diffusion/03_right_warp.png)

---

## Stage 2 — Hole Masks

Left Mask  
![left_mask](demo_images/diffusion/04_left_mask.png)

Right Mask  
![right_mask](demo_images/diffusion/05_right_mask.png)

---

## Stage 3 — Inpaint Filled Stereo

Left Filled  
![left_filled](demo_images/diffusion/06_left_filled.png)

Right Filled  
![right_filled](demo_images/diffusion/07_right_filled.png)

---

## Stage 4 — Diffusion Restoration (Left / Right)

Left Restored  
![left_restored](demo_images/diffusion/08_left_restored.png)

Right Restored  
![right_restored](demo_images/diffusion/09_right_restored.png)

---

## Stage 5 — Stereo Outputs

### Side-by-Side (Before Diffusion)

![sbs_before](demo_images/diffusion/10_sbs_before.png)

### Side-by-Side (After Diffusion)

![sbs_after](demo_images/diffusion/11_sbs_after.png)

---

### Red-Cyan Anaglyph (Before Diffusion)

![ana_before](demo_images/diffusion/12_anaglyph_before.png)

### Red-Cyan Anaglyph (After Diffusion)

![ana_after](demo_images/diffusion/13_anaglyph_after.png)

---

## Stage 6 — Depth & Disparity Visualization

Depth Map  
![depth](demo_images/diffusion/14_depth.png)

Disparity Map  
![disparity](demo_images/diffusion/15_disparity.png)

---

# 🧪 Normal DIBR Version (Without Diffusion)

## Input

![n_input](demo_images/normal/01_input.png)

---

## Forward Warp

Left Warp  
![n_left_warp](demo_images/normal/02_left_warp.png)

Right Warp  
![n_right_warp](demo_images/normal/03_right_warp.png)

---

## Filled Stereo

Left Filled  
![n_left_filled](demo_images/normal/04_left_filled.png)

Right Filled  
![n_right_filled](demo_images/normal/05_right_filled.png)

---

## Stereo Outputs

Side-by-Side  
![n_sbs](demo_images/normal/06_sbs.png)

Red-Cyan Anaglyph  
![n_ana](demo_images/normal/07_anaglyph.png)

---

## 🛠 Installation

Tested on Windows + Conda + Python 3.10

```bash
pip install torch torchvision
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow gradio
