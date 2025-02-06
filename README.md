# SemanticRegen
**Semantic Regenerative Attacks for Watermark Removal**

This repository contains code for **SemanticRegen**, a framework for **removing watermarks** from images using **semantic regeneration**. Our approach leverages **Visual Question Answering (VQA)**, **segmentation**, and **stable diffusion inpainting** to systematically identify and remove various types of watermarks while preserving image integrity.

## 📌 Overview
SemanticRegen is a **novel attack** on watermarking techniques that removes embedded watermarks without introducing significant distortions. Our approach:
- **Identifies prominent objects** using a **VQA captioning model**.
- **Segments** the foreground object while isolating watermarked areas.
- **Inpaints** the background using **Stable Diffusion** to remove watermarks.
- Benchmarked against existing **watermarking** and **removal** methods.

This implementation is adapted from existing frameworks, including:
- [`YuxinWenRick/tree-ring-watermark`](https://github.com/YuxinWenRick/tree-ring-watermark)
- [`XuandongZhao/WatermarkAttacker`](https://github.com/XuandongZhao/WatermarkAttacker)
- [`umd-huang-lab/WAVES`](https://github.com/umd-huang-lab/WAVES)

---

## 🛠 Installation

### **Required Models**
Download the following models and place them in the `models/` folder:

```bash
sam_vit_h_4b8939.pth
stable_signature.onnx  # from WAVES
stega_stamp.onnx       # from WAVES

Installing LangSAM
We use LangSAM, an open-source variant of Meta's Segment Anything Model (SAM):
