import matplotlib
import matplotlib.pyplot as plt

from skimage.transform import resize

import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

import torch.nn as nn
import numpy as np
import pandas as pd

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import random 
import cv2

from PIL import Image
import torchvision.transforms as T

# StegaStamp
import onnxruntime as ort
import torch.multiprocessing as mp
from PIL import Image, ImageOps




def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='SemanticRegen', name=args.run_name, tags=['semantic_regen'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric', 'no_w_p', 'w_p'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pretrained diffusion models
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id,
        subfolder='scheduler',
        cache_dir=args.cache_dir
        )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        cache_dir=args.cache_dir
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
            cache_dir=args.cache_dir
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch (THIS IS THE WATERMARK)
    gt_patch = get_watermarking_pattern(pipe, args, device)

    
    
    ################################################################################
    # Attack model imports
    ################################################################################
    
    if args.attack_type == "ImageDistortion":
        
        from optim_utils import image_distortion
    
    elif args.attack_type == "DiffWMAttacker":
        
        from diffusers import ReSDPipeline
        from diffwm_attack_utils import DiffWMAttacker
        diffwm_attacker_pipe = ReSDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            torch_dtype=torch.float16, 
            revision="fp16",
            cache_dir=args.cache_dir
        )
        diffwm_attacker_pipe.set_progress_bar_config(disable=True)
        diffwm_attacker_pipe.to(device)

    elif args.attack_type == "VAEWMAttacker":
        
        from diffusers import ReSDPipeline
        from diffwm_attack_utils import VAEWMAttacker
    
    elif args.attack_type == "Rinse4x":
        
        from diffusers import ReSDPipeline
        from waves_utils import rinse_4xDiff
        import glob
        rinse4x_pipe = ReSDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            torch_dtype=torch.float16, 
            revision="fp16",
            cache_dir=args.cache_dir
            )
        rinse4x_pipe.set_progress_bar_config(disable=True)
        rinse4x_pipe.to(device)
    
    elif args.attack_type == "Surrogate":
    
        # from surrogate_utils import *
        from surrogate_utils import resnet18, pgd_attack_classifier
        import torchattacks 
        import glob    
        
    else:
        
        print(f"Invalid args.attack_type = {args.attack_type}")
        return 
    
    ################################################################################
    
    
    ################################################################################
    # Watermarking models
    ################################################################################
        
    # StegaStamp
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.log_severity_level = 3
    stegastamp_model = ort.InferenceSession(
        "models/stega_stamp.onnx",
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": str(0)}],
        sess_options=session_options,
    )

    from diffwm_watermarker import StableSignatureWatermarker
    stablesig_watermarker = StableSignatureWatermarker(
        'stablediffusion', 
        msg_extractor='models/watermark_extractors/dec_48b_whit.torchscript.pt',
        script='txt2img.py', 
        key='111010110101000001010111010011010100010000100111'
    )
    
    from diffwm_watermarker import InvisibleWatermarker
    invisible_watermarker = InvisibleWatermarker(args.wm_secret, "dwtDct")

    ################################################################################

    
    
    ## 
    # Main Loop
    ## 
    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []
    
    no_w_p_metrics = []
    w_p_metrics = []
    
    use_prompts = range(args.start, args.end)
    if len(args.use_prompts):
        use_prompts = args.use_prompts

    # loop over each prompt
    for i in use_prompts:
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents() # this is our noise vector (X_T) + diffuse that noise with prompt to get first image
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        
        
        
        
        ################################################################################
        ### helper functions
        ################################################################################    
    
        import subprocess         
        def check_makedirs(file=None, path=None):
            # check for a folder in name (for now)
            if file is not None:
                path = os.path.dirname(file)
            if len(path) and not os.path.exists(path):
                os.makedirs(path)
            return path
        
        
        def get_reversed_latents(orig_image):
            # get image latents and reverse them (by forward diffusing)
            img = transform_img(orig_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents = pipe.get_image_latents(img, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )
            return reversed_latents
        
        ################################################################################
        
        
        
        
        ################################################################################
        ### TreeRing
        ################################################################################
      
        def encode_treering(init_latents_no_w=None, gt_patch=None, prompt='', args=None):
        
            # generation with watermarking
            if init_latents_no_w is None:
                set_random_seed(seed)
                init_latents_w = pipe.get_random_latents()
            else:
                init_latents_w = copy.deepcopy(init_latents_no_w)

            # get watermarking mask
            watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

            # inject watermark
            init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

            outputs_w = pipe(
                prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            orig_image_w = outputs_w.images[0]
        
            return orig_image_w, watermarking_mask
         
      
        def score_treering(orig_image_no_w_attack, orig_image_w_attack, watermarking_mask=None, gt_patch=None, args=None):
            
            # invert images
            reversed_latents_no_w_attack = get_reversed_latents(orig_image_no_w_attack)
            reversed_latents_w_attack = get_reversed_latents(orig_image_w_attack)

            # eval distance
            no_w_metric, w_metric = eval_watermark(
                reversed_latents_no_w_attack, reversed_latents_w_attack, 
                watermarking_mask, gt_patch, args
            )

            # eval p-value
            no_w_p, w_p = get_p_value(
                reversed_latents_no_w_attack, reversed_latents_w_attack, 
                watermarking_mask, gt_patch, args
            )

            return no_w_metric, w_metric, no_w_p, w_p
        
        ################################################################################
        
        
        
    
        ################################################################################
        ### StegaStamp
        ################################################################################
      
        def encode_stegastamp(orig_image, args=None):
            
            ##
            ##
            inputs = np.stack(
                        [
                            np.array(
                                ImageOps.fit(
                                   orig_image, (400, 400)
                                ),
                                dtype=np.float32,
                            )
                            / 255.0
                        ],
                        axis=0,
                    )

            inputs = inputs[:,:,:,:3]

            ##
            ##
            secret = args.wm_secret
            data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')

            packet_binary = ''.join(format(x, '08b') for x in data)
            secret = [int(x) for x in packet_binary]
            secret.extend([0,0,0,0])

            ##
            # Encoding?
            ##
            outputs = stegastamp_model.run(
                None,
                {
                    "image": inputs,
                    "secret": np.expand_dims(np.array(secret,dtype=np.float32),axis=0),
                },
            )
            
            # get image, key, etc.
            orig_image_encoded = outputs[0][0]
            key = outputs[2][0]

            # convert to Image and resize to 512
            orig_image_encoded = Image.fromarray(np.uint8(orig_image_encoded * 255))
            orig_image_encoded = orig_image_encoded.resize((512,512))
            
            return orig_image_encoded, key, secret
        
     
        def score_stegastamp(orig_image_attack, key=None, secret=None, args=None):
            
            # resize to 400 x 400 and convert to array
            orig_image_attack = orig_image_attack.resize((400,400))
            orig_image_attack_arr = np.array(orig_image_attack, dtype=np.float32) / 255.0
            orig_image_attack_arr = np.expand_dims(orig_image_attack_arr, axis=0)
           
            ##
            # Decoding?
            ##
            outputs = stegastamp_model.run(
                None,
                {
                    "image": orig_image_attack_arr,
                    "secret": np.expand_dims(np.array(key, dtype=np.float32), axis=0),
                },
            )
    
            # compute diff
            bool_msg = outputs[2][0]
            diff = [int(bool_msg[i] != secret[i]) for i in range(len(bool_msg))]
            
            # compute bit_acc
            bit_acc = 1 - sum(diff) / len(diff)

            # compute p_values
            from scipy.stats import binomtest
            pval = binomtest(sum(diff), len(diff), 0.5).pvalue 
            
            # return bit_acc, pval
            return bit_acc, pval
        
        ################################################################################

        
        
        
        ################################################################################
        ### StableSig
        ################################################################################

        def encode_stablesig(name=None, prompt=None, args=None):
        
            # check output folder
            output_dir = check_makedirs(path=name)
            orig_image_encoded_file = f"{output_dir}/samples/00000.png" 

            # encode watermark (if it doesn't exist)
            if not os.path.exists(orig_image_encoded_file):
                stablesig_watermarker.encode(output_dir, prompt=prompt)
            
            # load image from output_dir
            orig_image_encoded = Image.open(orig_image_encoded_file)
        
            return orig_image_encoded


        def score_stablesig(orig_image_attack, args=None):
        
            # decode the first sample
            bit_acc, pval = stablesig_watermarker.decode(orig_image_attack)
        
            # return bit_acc, pval
            return bit_acc, pval
       
        ################################################################################
    
        

        
        ################################################################################
        ### Invisible Watermark
        ################################################################################
    
        def encode_invisible(orig_image, args=None):
            
            # encode watermark
            orig_image_encoded = invisible_watermarker.encode(orig_image)
            
            # return watermarked image
            return orig_image_encoded
        
        
        def score_invisible(orig_image_attack, args=None):

            # decode the watermarked image
            decoded_wm_text = invisible_watermarker.decode(orig_image_attack)

            # get encoded text 
            encoded_wm_text = invisible_watermarker.wm_text

            # compute bit acc
            def bytearray_to_bits(x):
                """Convert bytearray to a list of bits
                """
                result = []
                for i in x:
                    bits = bin(i)[2:]
                    bits = '00000000'[len(bits):] + bits
                    result.extend([int(b) for b in bits])
                return result
            
            if type(decoded_wm_text) == bytes:
                a = bytearray_to_bits(encoded_wm_text.encode('utf-8'))
                b = bytearray_to_bits(decoded_wm_text)
            elif type(decoded_wm_text) == str:
                a = bytearray_to_bits(encoded_wm_text.encode('utf-8'))
                b = bytearray_to_bits(decoded_wm_text.encode('utf-8'))
            bit_acc = (np.array(a) == np.array(b)).mean()
        
            # return bit_acc
            return bit_acc
        
        ################################################################################
        
        
          
        
        ################################################################################
        ### Define Attackers here
        ################################################################################
      
        def diff_wm_attacker(orig_image, name=""):
            """ take an image, mask, then fill. """
            
            # check for targer output folder
            save_folder = os.path.dirname(name)
            if len(save_folder) and not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            import torchvision
            orig_image = orig_image.resize((512,512))
            cv2.imwrite(f'{name}_original.png', cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR))
            orig_image = torchvision.transforms.functional.pil_to_tensor(orig_image).float().unsqueeze(0)

            ##
            # Attack Watermark Images
            ##
            attacker = DiffWMAttacker(diffwm_attacker_pipe, batch_size=1, noise_step=60, captions={})
            
            in_paths = [f'{name}_original.png']
            out_paths = [f'{name}.png']

            attacker.attack(in_paths, out_paths)
            
            # return attacked image 
            return out_paths[0]
        
        ################################################################################
        
        def vae_wm_attacker(orig_image, name="", args=None):
            """ take an image, mask, then fill. """
            
            # check for targer output folder
            save_folder = os.path.dirname(name)
            if len(save_folder) and not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            import torchvision
            orig_image = orig_image.resize((512,512))
            cv2.imwrite(f'{name}_original.png', cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR))
            orig_image = torchvision.transforms.functional.pil_to_tensor(orig_image).float().unsqueeze(0)
            
            ##
            # Attack Watermark Images
            ##
            attacker = VAEWMAttacker(args.vae_model_name, quality=1, metric='mse', device='cuda')
            
            in_paths = [f'{name}_original.png']
            out_paths = [f'{name}.png']

            attacker.attack(in_paths, out_paths)

            # return attacked image 
            return out_paths[0]
        
        ################################################################################
        
        def rinse4x_attacker(orig_image, name=""):
            """ take an image, mask, then fill. """

            # check for targer output folder
            save_folder = os.path.dirname(name)
            if len(save_folder) and not os.path.exists(save_folder):
                os.makedirs(save_folder)

            import torchvision
            orig_image = orig_image.resize((512,512))
            cv2.imwrite(f'{name}_original.png', cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR))

            ##
            # Attack Watermark Images
            ##
            noise_step = 60    
            rinse_4x_image = rinse_4xDiff(orig_image, strength=noise_step, pipe=rinse4x_pipe, device=device)
        
            # return attacked image 
            return rinse_4x_image
        
        ################################################################################
        
        def surrogate_attacker(orig_image, name="", args=None):
            """ take an image, mask, then fill. """

            # check for targer output folder
            save_folder = os.path.dirname(name)
            if len(save_folder) and not os.path.exists(save_folder):
                os.makedirs(save_folder)

            import torchvision
            orig_image = orig_image.resize((512,512))
            cv2.imwrite(f'{name}_original.png', cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR))
            
            ##
            # Attack Watermark Images
            ##
            from torchvision.transforms.functional import pil_to_tensor
            transform = T.ToPILImage()

            EPS_FACTOR = 1 / 255
            ALPHA_FACTOR = 0.05
            N_STEPS = 200
   
            # select model type ("tree_ring", "stable_sig", "stegastamp")
            if args.wm_type.startswith("TreeRing"):
                model_path = 'surrogate_models/adv_cls_unwm_wm_tree_ring.pth'
            elif args.wm_type.startswith("StegaStamp"):
                model_path = 'surrogate_models/adv_cls_unwm_wm_stegastamp.pth'
            elif args.wm_type.startswith("StableSig"):
                model_path = 'surrogate_models/adv_cls_unwm_wm_stable_sig.pth'
            else:
                raise Exception(f"Invalid args.wm_type={args.wm_type}")
            
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 2)  
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()

            attacker = pgd_attack_classifier(model, EPS_FACTOR, ALPHA_FACTOR, N_STEPS)

            image = pil_to_tensor(orig_image).unsqueeze(0).float().to(device)
            image = image / 255
            label = torch.tensor([0]).long().to(device)
            adv_image = attacker.forward(image, label)
            adv_image = transform(adv_image.squeeze(0))
            
            # return attacked image 
            return adv_image
        
        ################################################################################
        
        
    
    
        ################################################################################
        ### Experiment Setup
        ################################################################################
          
        experiment_name = f"{args.wm_type}_{args.attack_type}"
        
        # attack specific naming conventions
        if args.attack_type == "ImageDistortion":
            if args.r_degree is not None:
                experiment_name += f"_r_degree_{args.r_degree}"
            if args.jpeg_ratio is not None:
                experiment_name += f"_jpeg_ratio_{args.jpeg_ratio}"     
            if args.crop_scale is not None and args.crop_ratio is not None:
                experiment_name += f"_crop_scale_{args.crop_scale}_crop_ratio_{crop_ratio}" 
            if args.gaussian_blur_r is not None:
                experiment_name += f"_gaussian_blur_r_{args.gaussian_blur_r}"
            if args.gaussian_std is not None:
                experiment_name += f"_gaussian_std_{args.gaussian_std}"
            if args.brightness_factor is not None:
                experiment_name += f"_brightness_factor_{args.brightness_factor}"
        elif args.attack_type == "DiffWMAttacker":
            pass
        elif args.attack_type == "VAEWMAttacker": 
            experiment_name += f"_{args.vae_model_name}"
        elif args.attack_type == "Rinse4x": 
            pass
        elif args.attack_type == "Surrogate":
            pass
        else:
            print(f"Invalid args.attack_type = {args.attack_type}")

        ################################################################################
        
        # format target filenames 
        orig_image_no_w_attack_file = f'{experiment_name}/no_watermark_attack/prompt_{i:04d}_orig_image_no_w_attack.png'
        orig_image_w_attack_file = f'{experiment_name}/watermark_attack/prompt_{i:04d}_orig_image_w_attack.png'   

        ################################################################################
        
        # make sure target folders exist
        _ = check_makedirs(file=orig_image_no_w_attack_file)
        _ = check_makedirs(file=orig_image_w_attack_file)

        ################################################################################
        
        
        
        ################################################################################
        ### Do watermarking 
        ################################################################################
        
        if args.wm_type.startswith("TreeRing"):
            orig_image_w, watermarking_mask = encode_treering(init_latents_no_w, gt_patch, current_prompt, args=args)
        elif args.wm_type.startswith("StegaStamp"):
            orig_image_w, watermark_key, watermark_secret = encode_stegastamp(orig_image_no_w, args=args)        
        elif args.wm_type.startswith("StableSig"):
            orig_image_w = encode_stablesig(
                name=f"StableSignatureImages/prompt_{i:04d}", 
                prompt=current_prompt, 
                args=args
            )       
        elif args.wm_type.startswith("Invisible"):
            orig_image_w = encode_invisible(orig_image_no_w, args=args)        
        else:
            raise Exception(f"Invalid args.wm_type={args.wm_type}")
        
        ################################################################################

    
        
        
        ################################################################################
        ### Do attacking 
        ################################################################################
            
        print(f"Attacking wm_type={args.wm_type} with attack_type={args.attack_type} ...")
        
        if args.attack_type == "ImageDistortion":
            
            # do attacking
            orig_image_no_w_attack, orig_image_w_attack = image_distortion(orig_image_no_w, orig_image_w, seed, args)

            # save attacked images
            orig_image_no_w_attack.save(orig_image_no_w_attack_file)
            orig_image_w_attack.save(orig_image_w_attack_file)

        elif args.attack_type == "DiffWMAttacker":
            
            # do attacking (attacked images will be saved to disk and read in below)
            _ = diff_wm_attacker(orig_image_no_w, name=orig_image_no_w_attack_file.split('.png')[0])
            _ = diff_wm_attacker(orig_image_w, name=orig_image_w_attack_file.split('.png')[0])
            
        elif args.attack_type == "VAEWMAttacker":
            
            # do attacking (attacked images will be saved to disk and read in below)
            _ = vae_wm_attacker(orig_image_no_w, name=orig_image_no_w_attack_file.split('.png')[0], args=args)
            _ = vae_wm_attacker(orig_image_w, name=orig_image_w_attack_file.split('.png')[0], args=args)
            
        elif args.attack_type == "Rinse4x":
            
            # do attacking
            orig_image_no_w_attack = rinse4x_attacker(orig_image_no_w, name=orig_image_no_w_attack_file.split('.png')[0])
            orig_image_w_attack = rinse4x_attacker(orig_image_w, name=orig_image_w_attack_file.split('.png')[0])
            
            # save attacked images
            orig_image_no_w_attack.save(orig_image_no_w_attack_file)
            orig_image_w_attack.save(orig_image_w_attack_file)

        elif args.attack_type == "Surrogate":
            
            # do attacking
            orig_image_no_w_attack = surrogate_attacker(orig_image_no_w, name=orig_image_no_w_attack_file.split('.png')[0], args=args)
            orig_image_w_attack = surrogate_attacker(orig_image_w, name=orig_image_w_attack_file.split('.png')[0], args=args)

            # save attacked images
            orig_image_no_w_attack.save(orig_image_no_w_attack_file)
            orig_image_w_attack.save(orig_image_w_attack_file)
            
        else:
            
            print(f"Invalid args.attack_type = {args.attack_type}")

        ################################################################################

        # load attacked files from disk
        orig_image_no_w_attack = Image.open(orig_image_no_w_attack_file)
        orig_image_w_attack = Image.open(orig_image_w_attack_file)    
        
        ################################################################################
        
        # TODO: load masks from {wm_type}_InPaint_ReplaceBG folder for now
        no_w_pct_mask, w_pct_mask = np.NaN, np.NaN
    
        # load image masks from {wm_type}_InPaint_ReplaceBG folder
        mask_image_no_w_attack_file = orig_image_no_w_attack_file.replace(experiment_name, f"{args.wm_type}_InPaint_ReplaceBG")
        mask_image_no_w_attack_file = mask_image_no_w_attack_file.replace(".png", "_background_mask.png")
        mask_image_w_attack_file = orig_image_w_attack_file.replace(experiment_name, f"{args.wm_type}_InPaint_ReplaceBG")
        mask_image_w_attack_file = mask_image_w_attack_file.replace(".png", "_background_mask.png")

        # load attacked files from disk 
        mask_image_no_w_attack = Image.open(mask_image_no_w_attack_file)
        mask_image_w_attack = Image.open(mask_image_w_attack_file)    
        
        # compute % masked
        no_w_pct_mask = (np.array(mask_image_no_w_attack) > 0).sum() / np.array(mask_image_no_w_attack).size 
        w_pct_mask = (np.array(mask_image_w_attack) > 0).sum() / np.array(mask_image_w_attack).size 

        ################################################################################
        
        
        
    
        ################################################################################
        ### Evaluate metrics
        ################################################################################
        
        no_w_metric, w_metric = np.NaN, np.NaN
        no_w_p, w_p =  np.NaN, np.NaN 
        no_w_bit_acc, w_bit_acc = np.NaN, np.NaN 
        w_no_attack_bit_acc, w_no_attack_p = np.NaN, np.NaN 
        
        ################################################################################
    
        if args.wm_type == "TreeRing":
            no_w_metric, w_metric, no_w_p, w_p = score_treering(
                orig_image_no_w_attack, orig_image_w_attack,
                watermarking_mask, gt_patch, args=args
            )
            w_no_attack_metric, _, w_no_attack_p, _ = score_treering(
                orig_image_w, orig_image_w_attack,
                watermarking_mask, gt_patch, args=args
            )
        elif args.wm_type == "StegaStamp":
            no_w_bit_acc, no_w_p = score_stegastamp(orig_image_no_w_attack, key=watermark_key, secret=watermark_secret, args=args)
            w_no_attack_bit_acc, w_no_attack_p = score_stegastamp(orig_image_w, key=watermark_key, secret=watermark_secret, args=args)
            w_bit_acc, w_p = score_stegastamp(orig_image_w_attack, key=watermark_key, secret=watermark_secret, args=args)
        elif args.wm_type.startswith("StableSig"):
            no_w_bit_acc, no_w_p = score_stablesig(orig_image_no_w_attack, args=args)
            w_no_attack_bit_acc, w_no_attack_p = score_stablesig(orig_image_w, args=args)
            w_bit_acc, w_p = score_stablesig(orig_image_w_attack, args=args)  
        elif args.wm_type.startswith("Invisible"):
            no_w_bit_acc = score_invisible(orig_image_no_w_attack, args=args)        
            w_no_attack_bit_acc = score_invisible(orig_image_w, args=args)
            w_bit_acc = score_invisible(orig_image_w_attack, args=args)        
        else:
            raise Exception(f"Invalid args.wm_type")
            
        ################################################################################
        
        # eval clip similarity
        no_w_sim, w_sim = np.NaN, np.NaN
        no_w_no_attack_sim, w_no_attack_sim = np.NaN, np.NaN
        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w_attack, orig_image_w_attack, orig_image_no_w, orig_image_w], 
                                       current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            no_w_sim = sims[0].item()
            w_sim = sims[1].item()
            no_w_no_attack_sim = sims[2].item()
            w_no_attack_sim = sims[3].item()
        
        ################################################################################

        # eval image similarity metrics    
        orig_image_no_w_tensor = T.ToTensor()(orig_image_no_w).unsqueeze(0)
        orig_image_no_w_attack_tensor = T.ToTensor()(orig_image_no_w_attack).unsqueeze(0)
        orig_image_w_tensor = T.ToTensor()(orig_image_w).unsqueeze(0)
        orig_image_w_attack_tensor = T.ToTensor()(orig_image_w_attack).unsqueeze(0)

        try: # torchmetrics > 0.6
            from torchmetrics.functional.regression import mean_squared_error as compute_mse
            from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim
            from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
        except: # torchmetrics==0.6
            from torchmetrics.functional.regression import mean_squared_error as compute_mse
            from torchmetrics.functional.image.ssim import ssim as compute_ssim
            from torchmetrics.functional.image.psnr import psnr as compute_psnr

        no_w_mse = compute_mse(orig_image_no_w_attack_tensor, orig_image_no_w_tensor).numpy() # (preds, target)
        w_mse = compute_mse(orig_image_w_attack_tensor, orig_image_w_tensor).numpy()          # (preds, target)

        no_w_ssim = compute_ssim(orig_image_no_w_attack_tensor, orig_image_no_w_tensor).numpy() # (preds, target)
        w_ssim = compute_ssim(orig_image_w_attack_tensor, orig_image_w_tensor).numpy()          # (preds, target)

        no_w_psnr = compute_psnr(orig_image_no_w_attack_tensor, orig_image_no_w_tensor).numpy() # (preds, target)
        w_psnr = compute_psnr(orig_image_w_attack_tensor, orig_image_w_tensor).numpy()          # (preds, target) 

        ################################################################################

    
        
        
        ################################################################################
        ### Evaluate (masked) metrics
        ################################################################################
        
        # load attacked files from disk
        mask_image_no_w_attack_tensor = T.ToTensor()(mask_image_no_w_attack).unsqueeze(0)
        mask_image_w_attack_tensor = T.ToTensor()(mask_image_w_attack).unsqueeze(0)
        
        # create binary mask with 3 channels
        binary_mask_image_no_w_attack_tensor = (torch.cat([mask_image_no_w_attack_tensor] * 3) == 0).long().squeeze(1).unsqueeze(0)
        binary_mask_image_w_attack_tensor = (torch.cat([mask_image_w_attack_tensor] * 3) == 0).long().squeeze(1).unsqueeze(0)

        # get masked images
        orig_image_no_w_no_bg_tensor = copy.deepcopy(orig_image_no_w_tensor)
        orig_image_no_w_no_bg_tensor[binary_mask_image_no_w_attack_tensor == 0] = 1
        
        orig_image_no_w_attack_no_bg_tensor = copy.deepcopy(orig_image_no_w_attack_tensor) 
        orig_image_no_w_attack_no_bg_tensor[binary_mask_image_no_w_attack_tensor == 0] = 1
        
        orig_image_w_no_bg_tensor = copy.deepcopy(orig_image_w_tensor)
        orig_image_w_no_bg_tensor[binary_mask_image_w_attack_tensor == 0] = 1
        
        orig_image_w_attack_no_bg_tensor = copy.deepcopy(orig_image_w_attack_tensor)
        orig_image_w_attack_no_bg_tensor[binary_mask_image_w_attack_tensor == 0] = 1
        
        # convert to images
        orig_image_no_w_no_bg = T.ToPILImage()(orig_image_no_w_no_bg_tensor.squeeze(0))    
        orig_image_no_w_attack_no_bg = T.ToPILImage()(orig_image_no_w_attack_no_bg_tensor.squeeze(0))
        
        orig_image_w_no_bg = T.ToPILImage()(orig_image_w_no_bg_tensor.squeeze(0))
        orig_image_w_attack_no_bg = T.ToPILImage()(orig_image_w_attack_no_bg_tensor.squeeze(0))
        
        # save images
        orig_image_no_w_no_bg.save(orig_image_no_w_attack_file.replace(".png", "_original_masked.png"))
        orig_image_no_w_attack_no_bg.save(orig_image_no_w_attack_file.replace(".png", "_masked.png"))
        
        orig_image_w_no_bg.save(orig_image_w_attack_file.replace(".png", "_original_masked.png"))
        orig_image_w_attack_no_bg.save(orig_image_w_attack_file.replace(".png", "_masked.png"))
        
        # for testing only
        # orig_image_no_w_attack_no_bg.save(f"orig_image_no_w_attack_no_bg.png")
        # orig_image_w_attack_no_bg.save(f"orig_image_w_attack_no_bg.png")
    
        # evaluate metrics
        no_w_no_bg_mse = compute_mse(orig_image_no_w_attack_no_bg_tensor, orig_image_no_w_no_bg_tensor).numpy() # (preds, target)
        w_no_bg_mse = compute_mse(orig_image_w_attack_no_bg_tensor, orig_image_w_no_bg_tensor).numpy()          # (preds, target)

        no_w_no_bg_ssim = compute_ssim(orig_image_no_w_attack_no_bg_tensor, orig_image_no_w_no_bg_tensor).numpy() # (preds, target)
        w_no_bg_ssim = compute_ssim(orig_image_w_attack_no_bg_tensor, orig_image_w_no_bg_tensor).numpy()          # (preds, target)

        no_w_no_bg_psnr = compute_psnr(orig_image_no_w_attack_no_bg_tensor, orig_image_no_w_no_bg_tensor).numpy() # (preds, target)
        w_no_bg_psnr = compute_psnr(orig_image_w_attack_no_bg_tensor, orig_image_w_no_bg_tensor).numpy()          # (preds, target)

        ################################################################################
    
        
        
        
        ################################################################################
        ### Log Results
        ################################################################################

        ### log stuff
        print(f"(prompt={i}, {experiment_name}) no_w_metric={no_w_metric:0.3f}, w_metric={w_metric:0.3f}, w_no_attack_p={w_no_attack_p:1.3e}, no_w_p={no_w_p:1.3e}, w_p={w_p:1.3e}, w_no_attack_bit_acc={w_no_attack_bit_acc:0.3f}, no_w_bit_acc={no_w_bit_acc:0.3f}, w_bit_acc={w_bit_acc:0.3f}, no_w_mse={no_w_mse:1.3e}, w_mse={w_mse:1.3e}, no_w_ssim={no_w_ssim:0.3f}, w_ssim={w_ssim:0.3f}, no_w_psnr={no_w_psnr:0.3f}, w_psnr={w_psnr:0.3f}, no_w_pct_mask={no_w_pct_mask:0.3f}, w_pct_mask={w_pct_mask:0.3f})")

        ################################################################################
            
        ### append all results
        results.append({
            'prompt_index': i,
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 
            'no_w_no_attack_sim': no_w_no_attack_sim, 'w_no_attack_sim': w_no_attack_sim,
            'no_w_sim': no_w_sim, 'w_sim': w_sim, 
            'w_no_attack_p': w_no_attack_p, 'no_w_p': no_w_p, 'w_p': w_p,
            'w_no_attack_bit_acc': w_no_attack_bit_acc, 'no_w_bit_acc': no_w_bit_acc, 'w_bit_acc': w_bit_acc, 
            'no_w_mse': no_w_mse, 'w_mse': w_mse,
            'no_w_ssim': no_w_ssim, 'w_ssim': w_ssim,
            'no_w_psnr': no_w_psnr, 'w_psnr': w_psnr,
            'no_w_no_bg_mse': no_w_no_bg_mse, 'w_no_bg_mse': w_no_bg_mse,
            'no_w_no_bg_ssim': no_w_no_bg_ssim, 'w_no_bg_ssim': w_no_bg_ssim,
            'no_w_no_bg_psnr': no_w_no_bg_psnr, 'w_no_bg_psnr': w_no_bg_psnr,
            'no_w_pct_mask': no_w_pct_mask, 'w_pct_mask': w_pct_mask, 
            'prompt_text': current_prompt
        })
        
        ################################################################################
        ################################################################################

        
    
    
        ################################################################################
        ### save results (on the fly)
        ################################################################################
   
        # overwrite results each prompt
        results_file = f"{experiment_name}/{experiment_name}_start_{args.start}_end_{args.end}_metrics.csv"
        
        # store results as a dataframe
        df = pd.DataFrame.from_dict(results)
        df.to_csv(results_file, index=None)
        
        ################################################################################
        ################################################################################
        
        
        
        
        
        
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark type
    parser.add_argument('--wm_type', default='StegaStamp', type=str)
    parser.add_argument('--wm_secret', default='Hello World!', type=str)
    
    # attack type
    parser.add_argument('--attack_type', default='DiffWMAttacker', type=str)
    parser.add_argument('--vae_model_name', type=str, default='bmshj2018-factorized')

    # watermark (tree ring)
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=3, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    # prompts
    parser.add_argument('--use_prompts', nargs="+", default=[], type=int)
    parser.add_argument('-v', '--verbose', default=3, type=int)    

    # cache
    parser.add_argument('--cache-dir', default=None)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)