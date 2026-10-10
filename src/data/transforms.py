"""
Image transforms (torchvision.transforms.v2 — no albumentations version pinning needed).

v2 fix (AUDIT B6): v1 trained on Resize(256)+RandomCrop(224) but evaluated on Resize(224),
so lesions appeared at a different scale at test time. Here training uses RandomResizedCrop
whose scale range includes the full image, and evaluation uses the full image.
"""

import random

import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRot90:
    """Lossless rotation by a random multiple of 90° (dermoscopy has no canonical orientation)."""

    def __call__(self, img):
        k = random.randint(0, 3)
        return torch.rot90(img, k, dims=(-2, -1)) if k else img


def get_train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRot90(),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_inference_transforms(img_size: int = 224) -> T.Compose:
    return get_val_transforms(img_size)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
