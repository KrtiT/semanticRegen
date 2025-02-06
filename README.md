# SemanticRegen
**Semantic Regenerative Attacks for Watermark Removal**

This repository contains code for **SemanticRegen**, a framework for **removing watermarks** from images using **semantic regeneration**. Our approach leverages **Visual Question Answering (VQA)**, **segmentation**, and **stable diffusion inpainting** to systematically identify and remove various types of watermarks while preserving image integrity.

## Overview
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

## Installation

### **Required Models**
Download the following models and place them in the `models/` folder:

```bash
sam_vit_h_4b8939.pth
stable_signature.onnx  # from WAVES
stega_stamp.onnx       # from WAVES
```
## Installing LangSAM
We use LangSAM, an open-source variant of Meta's Segment Anything Model (SAM):
```bash
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
```
If the above command fails, install GroundingDINO first:
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```
And then, install:
```bash
git clone https://github.com/luca-medeiros/lang-segment-anything
cd lang-segment-anything
pip install -e .
```
## Installing Stable Diffusion
To run experiments on the StableSig watermark, install Stable Diffusion from source:
```bash
git clone https://github.com/Stability-AI/stablediffusion
cd stablediffusion
pip install -e .
```
You may require additional dependencies:
```bash
pip install dlib ai_tools cognitive_face zprint pytorch-lightning==1.4.2 torchmetrics==0.8.2 kornia==0.6 open-clip-torch==2.7.0
```
Then, move the required model files into the correct directory:
```bash
cp txt2img.py stablediffusion/scripts/
cp models/sd2_decoder.pth stablediffusion/checkpoints/
```

## Running Experiments
To reproduce our results, run:

### Semantic Regenerative Attack
```bash
python run_Benchmark_SemanticRegen.py
```

### Baseline Attacks
```bash
python run_Benchmark_Baselines.py
```
### Example Commands
To evaluate TreeRing Watermarks, run:

```bash
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type TreeRing --w_channel 3 --w_pattern ring
```
### For StegaStamp Watermarks:
```bash
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StegaStamp
```
### For StableSig Watermarks:
```bash
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StableSig
```
### For Invisible Watermarks:
```bash
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type Invisible
```
## Analyzing Results
We provide Jupyter notebooks for analyzing the results:

| Notebook                                      | Purpose                                |
|-----------------------------------------------|----------------------------------------|
| `01-Benchmarking_Watermark_Attack_Removal.ipynb`  | Benchmarking watermark removal attacks |
| `02-Benchmarking_Image_Quality.ipynb`         | Image quality evaluation               |
| `02-Benchmarking_Image_Quality_CLIP_scores.ipynb` | CLIP score analysis                    |
| `02-Benchmarking_Image_Quality_plots.ipynb`   | Visualization of results               |
| `03-Reliability_Experiments.ipynb`           | Robustness and reliability testing     |

## Citation

If you find **SemanticRegen** useful, please cite our work:

```bibtex
@article{SemanticRegen2025,
  author = {Tallam, Krti and Cava, John and Geniesse, Caleb and Erichson, Benjamin and Mahoney, Michael W.},
  title = {Removing Watermarks with Partial Regeneration using Semantic Information},
  year = {2025},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

## References

- [WAVES Benchmark](https://github.com/umd-huang-lab/WAVES)
- [TreeRing Watermarking](https://github.com/YuxinWenRick/tree-ring-watermark)
- [StegaStamp](https://github.com/tancik/StegaStamp)
- [Stable Signature](https://github.com/facebookresearch/stable_signature)


### Developed by
👤 Krti Tallam, John Kevin Cava, Caleb Geniesse
📧 Contact: ktallam@berkeley.edu
