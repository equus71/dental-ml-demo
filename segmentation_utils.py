import torch
import random
from torchvision import transforms as T
from torchvision.transforms import Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def gauss_noise_aug(
        img: torch.Tensor,
        mean: float = 0.01,
        sigma: float = 0.09,
) -> torch.Tensor:
    """
    Training image augmentation with gaussian noise.

    Parameters
    ----------
    img: torch.Tensor
    mean: float
    sigma: float

    Returns
    -------
    out: torch.Tensor
    """
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = random.random() * sigma + mean

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


def get_transformations(
        image_processor,
        aug: bool,
        augmentation_list=("random_hf", "gaussian_blur", "gaussian_noise"),
):
    """
    Get the transformations to apply to the images
    Parameters
    ----------
    image_processor
        ImageProcessor object matching used model
    aug: bool
        Whether to apply augmentation or not

    Returns
    -------
    transformations: torchvision.transforms.Compose
        Composed transformations to apply to the images
    """
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    patch_height, patch_width = (
        image_processor.size["height"],
        image_processor.size["width"],
    )
    resize_transformation = T.Resize((patch_height, patch_width))
    transformations = [T.ToTensor(), resize_transformation, normalize]
    if aug:
        transformations = []
        if "random_hf" in augmentation_list:
            transformations.append(T.RandomHorizontalFlip())
        if "gaussian_blur" in augmentation_list:
            transformations.append(
                T.RandomApply(
                    p=0.5, transforms=[T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))]
                )
            )
        transformations.append(
            T.ToTensor(),
        )
        if "gaussian_noise" in augmentation_list:
            transformations.append(gauss_noise_aug)
        transformations.extend([resize_transformation, normalize])
        if "random_erasing" in augmentation_list:
            transformations.append(T.RandomErasing())
    return T.Compose(transformations)


def get_transformation_with_albumentations(
        image_processor,
        aug: bool,
):
    train_transform = A.Compose(
        [
            A.Resize(width=448, height=448),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )

    val_transform = A.Compose(
        [
            A.Resize(width=448, height=448),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )

    return train_transform if aug else val_transform


class SegmentationDataCollator:
    def __call__(self, features):
        batch = {}

        # Stack all images
        batch["pixel_values"] = torch.stack([feature["pixel_values"] for feature in features])

        # Stack all labels/masks and ensure they're in the right format
        batch["labels"] = torch.stack([
            # Take only first channel if mask is multi-channel and ensure long dtype
            (feature["label"][0] if feature["label"].ndim == 3 else feature["label"]).long()
            for feature in features
        ])

        return batch


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        # First, apply sigmoid to get probabilities
        probs = torch.sigmoid(torch.from_numpy(logits))
        # Get predictions by taking argmax over the class dimension
        predictions = probs.argmax(dim=1).numpy()  # This will give us shape (batch, height, width)

        # Calculate metrics
        accuracy = (predictions == labels).mean()

        # Calculate IoU for the foreground class (class 1)
        intersection = np.logical_and(predictions == 1, labels == 1).sum()
        union = np.logical_or(predictions == 1, labels == 1).sum()
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

        # Calculate Dice score
        dice = (2 * intersection) / (predictions.sum() + labels.sum() + 1e-6)

        return {
            "accuracy": accuracy,
            "iou": iou,
            "dice": dice
        }
