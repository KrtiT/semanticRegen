import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import os
import argparse
from torchvision.utils import save_image
from torchvision.models import resnet18
from torchattacks.attack import Attack
from torch.utils.data import Dataset, DataLoader

class WarmupPGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels, init_delta=None):
        """
        Overridden.
        """
        self.model.eval()

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def pgd_attack_classifier(model, eps, alpha, steps, random_start=True):
    # Create an instance of the attack
    attack = WarmupPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
    )

    # Set targeted mode
    attack.set_mode_targeted_by_label(quiet=True)

    return attack