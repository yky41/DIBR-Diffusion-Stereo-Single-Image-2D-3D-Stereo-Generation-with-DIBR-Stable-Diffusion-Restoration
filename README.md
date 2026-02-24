# DIBR-Diffusion-Stereo  
Single-Image 2D → 3D Stereo Generation with DIBR + Stable Diffusion Restoration

This project demonstrates a complete pipeline for generating stereo 3D views from a single image using:

- MiDaS depth estimation  
- DIBR forward warping  
- Hole filling (OpenCV inpaint)  
- Optional Stable Diffusion + ControlNet Tile restoration for left/right views  

The goal is to improve visual quality of stereo pairs while preserving geometric consistency.

---

## ✨ Features

- Single image → Left / Right stereo views  
- Forward warp + Z-buffer  
- Hole mask visualization  
- Inpaint filling  
- Stable Diffusion restoration (left & right separately)  
- Side-by-Side stereo  
- Red-Cyan Anaglyph  
- Full intermediate visualization  

---

## 🧠 Pipeline
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
Hole Filling
│
├────────► Normal Stereo
│
▼
Stable Diffusion Restoration (optional)

---

# 🚀 Diffusion Version Demo

## Input

![input](01_input.png)

---

## Stage 1 — Forward Warp (with holes)

### Left Warp
![left_warp](02_left_warp.png)

### Right Warp
![right_warp](03_right_warp.png)

---

## Stage 2 — Hole Masks

### Left Mask
![left_mask](04_left_mask.png)

### Right Mask
![right_mask](05_right_mask.png)

---

## Stage 3 — Inpaint Filled Stereo

### Left Filled
![left_filled](06_left_filled.png)

### Right Filled
![right_filled](07_right_filled.png)

---

## Stage 4 — Diffusion Restoration

### Left Restored
![left_restored](08_left_restored.png)

### Right Restored
![right_restored](09_right_restored.png)

---

## Stage 5 — Stereo Outputs

### Side-by-Side (Before Diffusion)
![sbs_before](10_sbs_before.png)

### Side-by-Side (After Diffusion)
![sbs_after](11_sbs_after.png)

---

### Red-Cyan Anaglyph (Before Diffusion)
![ana_before](12_anaglyph_before.png)

### Red-Cyan Anaglyph (After Diffusion)
![ana_after](13_anaglyph_after.png)

---

## Stage 6 — Depth & Disparity

### Depth Map
![depth](14_depth.png)

### Disparity Map
![disparity](15_disparity.png)

---

## ⚙ Key Notes

- Left / Right views are restored independently  
- Right view uses `seed + 1` to avoid texture locking  
- Recommended diffusion strength: **0.25 – 0.40**  
- Increase `dmax` to enhance stereo depth  

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

Diffusion + DIBR:

```bash
python demo_dibr_plus_diffusion.py

│
▼
Enhanced Stereo Output
