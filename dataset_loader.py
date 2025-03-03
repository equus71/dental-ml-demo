import os
import random

import albumentations as A
import imageio.v3
import numpy as np
import torch
from PIL import Image


class DentalDataset(torch.utils.data.Dataset):
    """
    Dataset loaded for dental images.

    NOTE: This keeps raw images in memory.
    """

    def __init__(self, path, transform, size, segmentation_folder="Segmentation1",
                 lazy_mode=False, pil_mode=False):
        """
        Parameters
        ----------
        path
            Image path
        transform
            Transform to apply to images
        size: int
            Desired size of images
        lazy_mode: bool
            If True, then images will load into the memory one by one,
            instead loading the whole dataset to the memory upfront.
        pil_mode: bool
            If True, load images as PIL Images. If False, load as numpy arrays.
        """
        super(DentalDataset).__init__()
        self.size = size
        self.lazy_mode = lazy_mode
        self.pil_mode = pil_mode
        self.dataset = load_images_from_path(path,
                                             segmentation_folder=segmentation_folder,
                                             lazy_mode=self.lazy_mode,
                                             pil_mode=self.pil_mode)
        self.transform = transform

    def __iter__(self):
        raise NotImplementedError("DentalDataset does not support iteration")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_id, (image, label) = self.dataset[idx]

        if self.lazy_mode:
            image = load_image(image, self.pil_mode, self.size)

        if self.transform:
            # If using Albumentations
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
            else:
                # For regular torchvision transforms
                image = self.transform(image)

        return {"pixel_values": image, "label": label}


def load_images_from_path(image_path,
                          segmentation_folder="segmentation",
                          pil_mode=False,
                          lazy_mode=False,
                          ):
    img_list = []
    sample_count = 0
    segmentation_path = os.path.join(os.path.dirname(image_path), segmentation_folder)

    for f in os.listdir(image_path):
        if f.endswith(".png") or f.endswith(".jpg"):
            img_path = os.path.join(image_path, f)
            seg_path = os.path.join(segmentation_path, f)

            # Skip if segmentation mask doesn't exist
            if not os.path.exists(seg_path):
                continue

            # Load images to check sizes
            img = load_image(img_path, pil_mode, is_mask=False)
            seg = load_image(seg_path, pil_mode, is_mask=True)

            # Verify sizes match
            if img.shape[:2] != seg.shape[:2]:
                print(f"Warning: Skipping {f} due to size mismatch. Image: {img.shape}, Mask: {seg.shape}")
                continue

            name = os.path.splitext(f)[0]
            if lazy_mode:
                img_list.append((name, (img_path, seg_path)))
            else:
                img_list.append((name, (img, seg)))
            sample_count += 1

    random.shuffle(img_list)
    return img_list


def load_image(img_path: str, pil_mode: bool, is_mask=False):
    if pil_mode:
        with Image.open(img_path) as img:
            img.load()
            img = np.array(img)
    else:
        img = imageio.v3.imread(img_path)

    # If it's a mask, ensure it's binary (0 and 1 only)
    if is_mask:
        if len(img.shape) == 3:
            img = img[..., 0]  # Take first channel
        # Ensure binary mask (0 and 1 only)
        img = (img > 0).astype(np.int64)  # Convert any positive values to 1, everything else to 0
    else:
        # If it's an image, ensure it's RGB
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)

    return img
