"""Preprocessing shared by training and inference: resize -> center crop -> ImageNet normalization."""
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_preprocessing(image: Image.Image, img_size: int, crop_size: int, is_mask: bool = False) -> torch.Tensor:
    """Single image pipeline for training AND inference.

    :param image: PIL image to transform (RGB image or grayscale mask).
    :param img_size: edge of the square resize applied before the crop.
    :param crop_size: edge of the center crop that reaches the model.
    :param is_mask: if True, keep the geometry but stay binary: nearest-neighbor
        resize, no normalization, binarize to {0.0, 1.0}.
    :return: image tensor normalized with ImageNet stats, or binary mask tensor.
    """
    if is_mask:
        pipeline = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(crop_size), T.PILToTensor(),
        ])
        return (pipeline(image) > 127).float()
    pipeline = T.Compose([
        T.Resize(img_size), T.CenterCrop(crop_size), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return pipeline(image)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Inverse of the ImageNet normalization, for display only.

    :param image: normalized image tensor (C, H, W).
    :return: image tensor clamped to [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)
