# SemanticRegen
Semantic Regenerative Attacks for Watermark Removal

This code was initially adapted from [YuxinWenRick/tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark). Other attackers and watermarks used for benchmarking were adapted from other codebases, including [XuandongZhao/WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker) and [umd-huang-lab/WAVES](https://github.com/umd-huang-lab/WAVES).

To ensure that our code is runnable and reproducible, we included several files that are not ours (e.g., `optim_utils.py`, `txt2img.py`). To clarify our contributions, the code we developed is contained in the following files:


* `run_Benchmark_SemanticRegen.py` 
* `run_Benchmark_Baselines.py` 
* `scripts/run_Benchmark_SemanticRegen.sh` 
* `scripts/run_Benchmark_Baselines.sh` 
* `example_commands.sh`
* `01-Benchmarking_Watermark_Attack_Removal.ipynb`
* `02-Benchmarking_Image_Quality.ipynb`
* `02-Benchmarking_Image_Quality_CLIP_scores.ipynb`
* `02-Benchmarking_Image_Quality_plots.ipynb`
* `03-Reliability_Experiments.ipynb`
* `requirements.txt`
* `README.md`







## **Setup Notes**

### **Required Python Packages**

* [numpy](www.numpy.org)
* [torch==1.13.0]()
* [torchvision]()
* [torchmetrics]()
* [transformers==4.31.0]()
* [diffusers==0.11.1]()
* [opencv-python]()
* [open_clip_torch]()
* [invisible-watermark]()
* [luca-medeiros/lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything.git) (see below)


_For a full list of packages and required versions, see `requirements.txt`._


### **Required Models**

Note, to run our code you will need to download several models and place them in the `models` folder.
* [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* [stable_signature.onnx](https://github.com/umd-huang-lab/WAVES/blob/458274bdc39cfbf1e704e651559c206e9df19ee6/decoders/stable_signature.onnx) (from [WAVES](https://github.com/umd-huang-lab/WAVES))
* [stega_stamp.onnx](https://github.com/umd-huang-lab/WAVES/blob/458274bdc39cfbf1e704e651559c206e9df19ee6/decoders/stega_stamp.onnx) (from [WAVES](https://github.com/umd-huang-lab/WAVES))




### **Installing `diffusers`**

You will need to install the `diffusers` package. For reproducibility, we adapted a version from the following [XuandongZhao/WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker) codebase (e.g., [`src/diffusers`](https://github.com/XuandongZhao/WatermarkAttacker/tree/c0020c7a7819f39be73420403d857d705d7ffeac/src/diffusers)).


### **Installing `LangSAM`**

Note, you can try installing `LangSAM` using `pip`, e.g.,
```
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
```

If this fails, you can try installing manually. Note, you will need to install `GroundingDINO` first.

To install `GroundingDINO`, run the following
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```

Then, to install `LangSAM`, run the following
```
git clone https://github.com/luca-medeiros/lang-segment-anything
cd lang-segment-anything
pip install -e .
```



### **Installing `stablediffusion` (for `StableSig` experiments)**

To run experiments on the `StableSig` watermark, you will need to install [`stablediffusion`](https://github.com/Stability-AI/stablediffusion) from source.

To install `stablediffusion` from source, run the following:
```
git clone https://github.com/Stability-AI/stablediffusion
cd stablediffusion
pip install -e .
```

Note, you may need to install the following packages:
```
pip install dlib ai_tools cognitive_face zprint pytorch-lightning==1.4.2 torchmetrics==0.8.2 kornia==0.6 open-clip-torch==2.7.0
```

After installing `stablediffusion`, you will need to make three important changes to the codebase:

* copy the `txt2img.py` file we provide into `stablediffusion/scripts`
* copy the `models/sd2_decoder.pth` model we provide into `stablediffusion/checkpoints`
* download [`stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt`](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt) and move it into `stablediffusion/checkpoints`

For more detailed instructions, please see: https://github.com/facebookresearch/stable_signature









## **Example Usage**

Our experiments can be replicated by running the following files:

* `run_Benchmark_SemanticRegen.py` to evaluate our Semantic Regenerative Attack
* `run_Benchmark_Baselines.py` to evaluate Image Distortion and other attacks

Each file has the option to evaluate the attacks (`--attack_type`) on different types of watermarks (e.g., `--wm_type`). We provide additional scripts to wrap these commands with some default arguments (e.g., `--start 1000 --end 2000` defining the prompts to run) and another file (`example_commands.sh`) with some example command-lines using these scripts. 


### **TreeRing**

For example, to run our Semantic Regenerative Attack on Tree Ring watermarks, run the folllowing:

```
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type TreeRing --w_channel 3 --w_pattern ring 
```

To run Rinse4x on Tree Ring watermarks, run the folllowing:
```
bash scripts/run_Benchmark_Baselines.sh --wm_type TreeRing --w_channel 3 --w_pattern ring --attack_type Rinse4x 
```

### **StegaStamp**

To run our Semantic Regenerative Attack on StegaStamp watermarks, run the folllowing:

```
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StegaStamp 
```

To run Rinse4x on StegaStamp watermarks, run the folllowing:
```
bash scripts/run_Benchmark_Baselines.sh --wm_type StegaStamp  --attack_type Rinse4x 
```

### **StableSig**

To run our Semantic Regenerative Attack on StableSig watermarks, run the folllowing:

```
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StableSig 
```

To run Rinse4x on StableSig watermarks, run the folllowing:
```
bash scripts/run_Benchmark_Baselines.sh --wm_type StableSig  --attack_type Rinse4x 
```

### **Invisible**

To run our Semantic Regenerative Attack on Invisible watermarks, run the folllowing:

```
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StableSig 
```

To run Rinse4x on Invisible watermarks, run the folllowing:
```
bash scripts/run_Benchmark_Baselines.sh --wm_type Invisible  --attack_type Rinse4x 
```




## **Example Analysis**

We provide additional Jupyter notebooks showing how we analyzed the results produced by our experiments. Note, to run these notebooks, you will first need to run some of the experiments, as described above. Each experiment produces a `.csv` file of metrics computed for each prompt (e.g., p-values and bit accuracies, image quality scores). You can select which experiments to include in the analysis by changing the `experiment_names` variable.

These analysis notebooks roughly follow the results section of the paper, including:

* `01-Benchmarking_Watermark_Attack_Removal.ipynb`
* `02-Benchmarking_Image_Quality.ipynb`
* `02-Benchmarking_Image_Quality_CLIP_scores.ipynb`
* `02-Benchmarking_Image_Quality_plots.ipynb`
* `03-Reliability_Experiments.ipynb`