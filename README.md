SemanticRegen: Semantic Regenerative Attacks for Watermark Removal
SemanticRegen is a novel approach for removing watermarks using semantic inpainting and regenerative attacks. This repository provides the full implementation of our method, along with baseline comparisons and benchmarking tools.

Overview
This project builds on existing watermarking research, incorporating deep learning techniques to remove digital watermarks while maintaining image integrity. Our approach leverages:

Visual Question Answering (VQA) via BLIP2 to identify key objects
Segmentation models like LangSAM to isolate prominent elements
Stable Diffusion inpainting for seamless watermark removal
We benchmark against established attacks and watermarking schemes, including:

TreeRing Watermark (Wen et al.)
StegaStamp (Google)
Stable Signature (Meta)
Invisible Watermarks (DWT/DCT)
Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/SemanticRegen.git
cd SemanticRegen
2. Install Required Dependencies
You can install the necessary Python packages via:

bash
Copy
Edit
pip install -r requirements.txt
Key Dependencies
numpy
torch==1.13.0
torchvision
torchmetrics
transformers==4.31.0
diffusers==0.11.1
opencv-python
open_clip_torch
invisible-watermark
luca-medeiros/lang-segment-anything
Required Models
Several pre-trained models are needed for running the watermark removal experiments. Download and place them in the models/ directory:

Model Name	Source
sam_vit_h_4b8939.pth	LangSAM
stable_signature.onnx	WAVES
stega_stamp.onnx	WAVES
Installation of External Dependencies
1. Install LangSAM
LangSAM is used for segmentation. Install it via:

bash
Copy
Edit
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
If this fails, install manually:

bash
Copy
Edit
git clone https://github.com/luca-medeiros/lang-segment-anything
cd lang-segment-anything
pip install -e .
2. Install Stable Diffusion (for StableSig Experiments)
To run Stable Signature watermark experiments, install Stable Diffusion:

bash
Copy
Edit
git clone https://github.com/Stability-AI/stablediffusion
cd stablediffusion
pip install -e .
Additional dependencies:

bash
Copy
Edit
pip install dlib ai_tools cognitive_face zprint pytorch-lightning==1.4.2 torchmetrics==0.8.2 kornia==0.6 open-clip-torch==2.7.0
Usage
Running Benchmark Experiments
Our experiments can be replicated using the following scripts:

1. Run Semantic Regenerative Attack
bash
Copy
Edit
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type <WATERMARK_TYPE>
2. Run Baseline Watermark Attacks
bash
Copy
Edit
bash scripts/run_Benchmark_Baselines.sh --wm_type <WATERMARK_TYPE> --attack_type <ATTACK_METHOD>
Replace <WATERMARK_TYPE> with one of:

TreeRing
StegaStamp
StableSig
Invisible
And <ATTACK_METHOD> with:

Rinse4x
Distortion
Example Commands
TreeRing
bash
Copy
Edit
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type TreeRing --w_channel 3 --w_pattern ring
bash
Copy
Edit
bash scripts/run_Benchmark_Baselines.sh --wm_type TreeRing --w_channel 3 --w_pattern ring --attack_type Rinse4x
StegaStamp
bash
Copy
Edit
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StegaStamp
bash
Copy
Edit
bash scripts/run_Benchmark_Baselines.sh --wm_type StegaStamp --attack_type Rinse4x
StableSig
bash
Copy
Edit
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StableSig
bash
Copy
Edit
bash scripts/run_Benchmark_Baselines.sh --wm_type StableSig --attack_type Rinse4x
Analysis & Evaluation
After running experiments, results are stored as .csv files. We provide Jupyter notebooks for analysis:

01-Benchmarking_Watermark_Attack_Removal.ipynb
02-Benchmarking_Image_Quality.ipynb
02-Benchmarking_Image_Quality_CLIP_scores.ipynb
02-Benchmarking_Image_Quality_plots.ipynb
03-Reliability_Experiments.ipynb
Citation
If you use this work in your research, please cite our paper:

bibtex
Copy
Edit
@article{yourpaper2025,
  author = {Tallam, Krti and Cava, John and Geniesse, Caleb and Erichson, Benjamin and Mahoney, Michael W.},
  title = {Semantic Regenerative Attacks for Watermark Removal},
  journal = {arXiv preprint},
  year = {2025}
}
License
This project is licensed under the MIT License.

Acknowledgments
This work is adapted from multiple repositories:

YuxinWenRick/tree-ring-watermark
XuandongZhao/WatermarkAttacker
umd-huang-lab/WAVES
We extend our gratitude to the creators of these projects for their foundational work in watermarking research.

Contact
For questions, issues, or contributions, feel free to open an issue or reach out.

