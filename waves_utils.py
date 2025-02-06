from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from torchvision import transforms
from tqdm import tqdm
import glob
import numpy as np
import argparse
# from diffusers import ReSDPipeline


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError

class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, noise_step=60, captions={}):
        self.pipe = pipe
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(
            f"Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}"
        )

    def attack(self, image, device, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor(
                [self.noise_step], dtype=torch.long, device=self.device
            )
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(
                    prompts_buf,
                    head_start_latents=latents,
                    head_start_step=50 - max(self.noise_step // 20, 1),
                    guidance_scale=7.5,
                    generator=generator,
                )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    return img

            img = np.asarray(image) / 255
            img = (img - 0.5) * 2
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(
                img.to(device=torch.device("cuda"), dtype=torch.float16)
            ).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn(
                [1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device
            )

            latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(
                torch.half
            )
            latents_buf.append(latents)
            outs_buf.append("")
            prompts_buf.append("")

            img = batched_attack(latents_buf, prompts_buf, outs_buf)
            return img

def remove_watermark(attack_method, image, strength, pipe, device):
    # create attacker
    print(f"Creating attacker {attack_method}...")

    if attack_method == "regen_diffusion":
        attacker = DiffWMAttacker(pipe, noise_step=strength, captions={})

    else:
        raise Exception(f"Unknown attacking method: {attack_method}!")

    img = attacker.attack(image, device)

    return img

def rinse_2xDiff(image, strength, pipe, device):
    first_attack = True
    for attack in ["regen_diffusion", "regen_diffusion"]:
        if first_attack:
            image = remove_watermark(attack, image, strength, pipe, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, pipe, device)
    return image


def rinse_4xDiff(image, strength, pipe, device):
    first_attack = True
    for attack in [
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
    ]:
        if first_attack:
            image = remove_watermark(attack, image, strength, pipe, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, pipe, device)
    return image