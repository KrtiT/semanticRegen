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


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Blip2Processor, Blip2ForConditionalGeneration       
from transformers import AutoModelForCausalLM, AutoTokenizer
from conversation import get_default_conv_template
from diffusers import StableDiffusionInpaintPipeline
from lang_sam import LangSAM


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
    
    ##
    # Load in Segmentation Model and set it to evaluation
    ##
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    seg_model.eval()
    seg_model.to(device)
    
    ##
    # Inpainting models
    ##
    
    # LangSAM model
    lang_sam_model = LangSAM("vit_h", "models/sam_vit_h_4b8939.pth", cache_dir=args.cache_dir)
    
    # Blip2
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-6.7b", 
        cache_dir=args.cache_dir
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b", 
        device_map="auto",
        cache_dir=args.cache_dir
    )

    # MiniChat
    minichat_tokenizer = AutoTokenizer.from_pretrained(
        "GeneZC/MiniChat-2-3B", 
        use_fast=False,
        cache_dir=args.cache_dir
    )
    minichat_model = AutoModelForCausalLM.from_pretrained(
        "GeneZC/MiniChat-2-3B", 
        use_cache=True, 
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    ).eval()

    # Stability AI Inpainting  
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    )
    inpaint_pipe.to("cuda")

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
        ### Define SemanticRegen Attacker here
        ################################################################################
    
        def get_background_mask(image):
            """ get background mask, given an image """
            
            # do preprocess
            preprocess = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
            ])
            img = preprocess(image).unsqueeze(0).to(device)
            
            # do segmentation
            output = seg_model(img)['out'][0]

            # compute segmentation mask based on predictions
            output_predictions = output.argmax(0)

            t1 = (output_predictions > 0)
            t2 = (output_predictions == 0)

            # set background to white (255)
            output_predictions[t1] = 0
            output_predictions[t2] = 255
           
            bg_mask = output_predictions.byte().cpu().numpy().astype("uint8")
            
            return bg_mask
        
    
        def get_background_mask_sam(orig_image, name="", args=None):
            """ get background mask, given an image """
           
            ###
            # Uncondtional Image Captioning From BLIP
            ###
            
            question = "Question: What is the prominent object in this image? Answer:"
            inputs = blip_processor(orig_image, question, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs, max_new_tokens=256)
            caption_objects_1 = blip_processor.decode(out[0], skip_special_tokens=True)
            if args.verbose > 1:
                print(f"caption_objects_1 = {caption_objects_1}")
                
            # question = "Question: What is the prominent object in this image? Answer:"
            # question = "Question: What are the most prominent objects in the foreground image? Answer:"
            question = "Question: What are the most prominent subjects in this image? Answer:"
            inputs = blip_processor(orig_image, question, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs, max_new_tokens=256)
            caption_objects_2 = blip_processor.decode(out[0], skip_special_tokens=True)
            if args.verbose > 1:
                print(f"caption_objects_2 = {caption_objects_2}")

                
            ###
            # Segmentation of the original image to get the mask
            ###
            ##
            # NOTE: WGET MODELS FROM THE WEB INTO LOCAL TO CIRCUMVENT DOWNLOADING DIRECTLY TO CACHE
            # "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            # "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            # "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            ##
            masks_1, boxes_1, phrases_1, logits_1 = lang_sam_model.predict(orig_image, caption_objects_1)
            masks_2, boxes_2, phrases_2, logits_2 = lang_sam_model.predict(orig_image, caption_objects_2)
        
            
            def combine_masks(masks):
                
                # combine masks (take max)
                masks_combined = None
                pct_mask = None
                for mask in masks:

                    # get pixelwise max
                    if masks_combined is None:
                        masks_combined = mask
                    else:    
                        masks_combined = torch.maximum(masks_combined, mask)

                    # check for valid mask
                    pct_mask = (masks_combined.detach().cpu().numpy() > 0).sum() / (mask.size()[0] * mask.size()[1])
                    if pct_mask > args.mask_threshold:
                        break
                        
                # check for missing mask (redraw everything by default) 
                if masks_combined is None:
                    masks_combined = torch.tensor(np.ones(orig_image.size))
                    pct_mask = -1

                # convert to black/white
                masks_combined = masks_combined * 255
                
                # return mask, pct_combined
                return masks_combined, pct_mask
            
        
            # get combined masks
            masks_combined_1, pct_mask_1 = combine_masks(masks_1)
            masks_combined_2, pct_mask_2 = combine_masks(masks_2)
            
            # save for now
            if args.verbose > 1:
                print(f"pct_mask_1 = {pct_mask_1}")
                print(f"pct_mask_2 = {pct_mask_2}")

            mask_image_1 = masks_combined_1.reshape((512, 512))
            mask_image_1 = Image.fromarray(np.uint8(mask_image_1))
            mask_image_1.save(f'{name}_background_mask_1.png')
            
            mask_image_2 = masks_combined_2.reshape((512, 512))
            mask_image_2 = Image.fromarray(np.uint8(mask_image_2))
            mask_image_2.save(f'{name}_background_mask_2.png')
            
            
            # which mask to use
            if pct_mask_1 < pct_mask_2:
                masking = masks_combined_2
                pct_mask = pct_mask_2
            else:
                masking = masks_combined_1
                pct_mask = pct_mask_1
            
            # flip if pct is > 75% 
            # ... if object is too big, we need to flip 
            # ... so we can remove more (e.g., landscapes)
            if pct_mask < args.mask_flip_threshold:
                t1 = (masking > 0)
                t2 = (masking == 0)
            else:
                t1 = (masking == 0)
                t2 = (masking > 0)
            
            # set to black/white
            masking[t2] = 255
            masking[t1] = 0

            bg_mask = masking.byte().cpu().numpy().astype("uint8")
            return bg_mask
        
        ################################################################################
    
        def do_inpainting(orig_image, mask_image, name="", args=None):

            ###
            # Uncondtional Image Captioning From BLIP
            ###
            
            question = "Question: Where would this image taken? Answer:"
            inputs = blip_processor(orig_image, question, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs, max_new_tokens=256)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            context = 'The image is described to be taken ' + caption + '.'
            if args.verbose > 1:
                print(f"location caption = {caption}")

            question = "Question: Describe the artistic direction of the image. Answer:"
            inputs = blip_processor(orig_image, question, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs, max_new_tokens=256)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            context = context + ' The image artistic direction is described as ' + caption + '.'
            if args.verbose > 1:
                print(f"artistic caption = {caption}")

              
            ###
            # MiniCHAT for extracting information of background of the captioned image
            ###

            # text = "Given the following caption of an image, describe where this image would be in less than 10 words or less. The answer should be one sentence." + "Caption: " + caption
            # text = "Given the following caption of an image, state where this image would be taken in 10 words or less." + "Caption: " + caption
            text = 'Given the following sentences that descibe an image, write in one sentence what the background setting is and in what art style. ' + context

            conv = get_default_conv_template("minichat")
            question = text + caption

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = minichat_tokenizer([prompt]).input_ids
            output_ids = minichat_model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=0.7,
                max_new_tokens=128,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = minichat_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            new_prompt = output.split('.')[0] + '.'
            if args.verbose > 1:
                print(f"new_prompt = {new_prompt}")


            ###
            # Use Stability AI Inpainting on Mask/Image/Caption
            ###
            
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            image = inpaint_pipe(prompt=new_prompt, image=orig_image, mask_image=mask_image).images[0]
            
            return image

        ################################################################################        
        
        def mask_and_fill(orig_image, name="", args=None):
            """ take an image, mask, then fill. """
            
            # check for target output folder
            save_folder = check_makedirs(file=name)
            
            import torchvision
            cv2.imwrite(f'{name}_original.png', cv2.cvtColor(np.array(orig_image.resize((512,512))), cv2.COLOR_RGB2BGR))
            orig_image = orig_image.resize((512,512))
            input_size = orig_image.size # (256,256)
            orig_image = T.functional.pil_to_tensor(orig_image).float().unsqueeze(0)

            # load the image
            image_file = f'{name}_original.png'
            image = Image.open(image_file)

            # get segmentation mask
            bg_mask = get_background_mask_sam(image, name=name, args=args)
            bg_mask = bg_mask.reshape((512, 512))
            bg_mask = Image.fromarray(np.uint8(bg_mask))
            bg_mask.save(f'{name}_background_mask.png')
            
            
            ### TODO: Remove Background & Shift
            
            # remove background from image
            # bg_mask_arr = np.array(bg_mask)
            # image_arr = np.array(image)
            # image_arr[bg_mask_arr == 255] = 255
            # image = Image.fromarray(np.uint8(image_arr))
     
            # # resize then crop (to shift the object)
            # image = image.resize((575, 575))
            # image_tensor = T.ToTensor()(image)
            # image_tensor = T.RandomCrop((512, 512))(image_tensor)
            # image = T.ToPILImage()(image_tensor)

            # # create new mask based on image
            # image_arr = np.array(image)
            # bg_mask_arr = (image_arr == 255).astype(int) * 255
            # bg_mask = Image.fromarray(np.uint8(bg_mask_arr)) 
            
            
            # repaint on image files
            image = do_inpainting(image, bg_mask, name=f"{name}", args=args)
                
            return image
    
        ################################################################################

    
    
    
        ################################################################################
        ### Experiment Setup 
        ################################################################################
          
        experiment_name = f"{args.wm_type}_SemanticRegen"
        if args.run_id is not None:
            experiment_name += f"_run_{args.run_id}"
        if args.attack_seed is not None:
            experiment_name += f"_attack_seed_{args.attack_seed}"
        
        ################################################################################
        
        # format target filenames 
        orig_image_no_w_attack_file = f'{experiment_name}/no_watermark_attack/prompt_{i:04d}_orig_image_no_w_attack.png'
        orig_image_w_attack_file = f'{experiment_name}/watermark_attack/prompt_{i:04d}_orig_image_w_attack.png'   

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
            raise Exception(f"Invalid args.wm_type")
        
        ################################################################################
        

    
        
        ################################################################################
        ### Do attacking
        ################################################################################
       
        # set random seed before attacking
        if args.attack_seed is not None:
            set_random_seed(args.attack_seed)        
        
        ################################################################################
        
        # compute orig_image_no_w_attack (if it doesn't already exist)
        if not os.path.exists(orig_image_no_w_attack_file):                       
            orig_image_no_w_attack = mask_and_fill(
                orig_image_no_w,
                name=orig_image_no_w_attack_file.split('.png')[0], args=args
            )
            orig_image_no_w_attack.save(orig_image_no_w_attack_file)
        else:
            print(f"(prompt={i}, {experiment_name}) loading orig_image_no_w_attack from disk [remove file to recompute]")
        
        # compute orig_image_w_attack (if it doesn't already exist)
        if not os.path.exists(orig_image_w_attack_file):                       
            orig_image_w_attack = mask_and_fill(
                orig_image_w,
                name=orig_image_w_attack_file.split('.png')[0], args=args
            )
            orig_image_w_attack.save(orig_image_w_attack_file)
        else:
            print(f"(prompt={i}, {experiment_name}) loading orig_image_w_attack from disk [remove file to recompute]")
        
        ################################################################################

        # load attacked files from disk
        orig_image_no_w_attack = Image.open(orig_image_no_w_attack_file)
        orig_image_w_attack = Image.open(orig_image_w_attack_file)    
        
        ################################################################################

        # load attacked files from disk
        mask_image_no_w_attack = Image.open(orig_image_no_w_attack_file.replace(".png", "_background_mask.png"))
        mask_image_w_attack = Image.open(orig_image_w_attack_file.replace(".png", "_background_mask.png"))    
        
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
    
    # LangSAM masking
    parser.add_argument('--mask_threshold', default=0.50, type=float)
    parser.add_argument('--mask_flip_threshold', default=0.80, type=float)

    # reproducibility
    parser.add_argument('--run_id', default=None, type=int)
    parser.add_argument('--attack_seed', default=None, type=int)

    # prompts
    parser.add_argument('--use_prompts', nargs="+", default=[], type=int)
    parser.add_argument('-v', '--verbose', default=3, type=int)

    # # cache
    parser.add_argument('--cache-dir', default=None)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)