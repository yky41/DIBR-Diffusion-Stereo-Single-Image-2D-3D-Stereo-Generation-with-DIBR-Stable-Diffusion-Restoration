# DIBR-Diffusion-Stereo

Single-Image 2D → 3D Stereo Generation with DIBR + Stable Diffusion Restoration

---

## 📌 Overview

This project demonstrates a complete practical pipeline for generating stereo 3D views from a single monocular image.

Traditional DIBR (Depth Image Based Rendering) pipelines often suffer from:

- Warping artifacts
- Hole regions caused by occlusions
- Blocky textures
- Detail loss after inpainting

To address this, we integrate **Stable Diffusion + ControlNet Tile** as a post-processing stage to restore fine details while preserving stereo geometry.

The system produces:

- Left / Right stereo views  
- Side-by-Side stereo pairs  
- Red-Cyan Anaglyph visualization  

with optional diffusion-based enhancement.

---

## ✨ Key Features

- Single image → stereo 3D
- MiDaS depth estimation
- Forward warp + Z-buffer
- Hole mask visualization
- OpenCV inpainting
- Optional Stable Diffusion restoration
- ControlNet Tile structure guidance
- Left / Right diffusion processed separately
- SBS stereo + Anaglyph output
- Full intermediate visualization
- Gradio interactive demo

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
Hole Filling (Inpainting)
│
├────────► Normal Stereo Output
│
▼
Stable Diffusion Restoration (optional)
│
▼
Enhanced Stereo Output

---

# 🚀 Diffusion Version Demo (Full Pipeline Visualization)

All images below are generated from a single input image.

---

## 1. Input Image

![input](demo_images/diffusion/01_input.png)

---

## 2. Forward Warp (with Holes)

### Left View (Warped)

![left_warp](demo_images/diffusion/02_left_warp.png)

### Right View (Warped)

![right_warp](demo_images/diffusion/03_right_warp.png)

Forward warping introduces hole regions due to occlusions.

---

## 3. Hole Masks

### Left Hole Mask

![left_mask](demo_images/diffusion/04_left_mask.png)

### Right Hole Mask

![right_mask](demo_images/diffusion/05_right_mask.png)

White pixels indicate missing regions.

---

## 4. Inpaint Filled Views

### Left Filled

![left_filled](demo_images/diffusion/06_left_filled.png)

### Right Filled

![right_filled](demo_images/diffusion/07_right_filled.png)

OpenCV inpainting is used to fill disocclusion holes.

---

## 5. Diffusion Restoration (Left / Right Separately)

Each eye is restored independently using Stable Diffusion + ControlNet Tile.

### Left Restored

![left_restored](demo_images/diffusion/08_left_restored.png)

### Right Restored

![right_restored](demo_images/diffusion/09_right_restored.png)

This stage removes block artifacts and recovers fine textures.

---

## 6. Stereo Outputs

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

## 7. Depth & Disparity Visualization

### Depth Map

![depth](demo_images/diffusion/14_depth.png)

### Disparity Map

![disp](demo_images/diffusion/15_disparity.png)

---

## ⚙ Important Notes

- Left and Right images are restored independently.
- Right view uses seed + 1 to avoid texture locking.
- Recommended diffusion strength: 0.25 – 0.40
- Increase dmax to enhance stereo depth.

---

## 🛠 Dependencies
torch
opencv-python
diffusers
transformers
accelerate
safetensors
gradio
pillow

---

## ▶ Run Demo

### Diffusion + DIBR:

```bash
python demo_dibr_plus_diffusion.py


## 🧠 Pipeline Overview
