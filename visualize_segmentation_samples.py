import os

import cv2
import numpy as np
import torch

from tooth_segmentation_dataset_loader import ToothSegmentationDataset


def convert_image(img):
    """
    Convert an image (tensor or numpy array) to uint8 numpy array in HxWxC format.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # If channel-first, convert to HxWxC.
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        # Scale float images if needed
        if img.dtype != np.uint8:
            # Assume float in range [0,1]
            img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def convert_mask(mask):
    """
    Convert a mask (tensor or numpy array) to a 3-channel uint8 image.
    If the mask is binary (values 0 or 1) then it is scaled to 0 or 255.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # If mask is float and likely in range [0,1], scale to 0-255.
    if mask.dtype != np.uint8:
        # If maximum value is <=1 it is assumed to be binary.
        if mask.max() <= 1:
            mask = (mask * 255).clip(0, 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    # If mask is 2D, convert to 3 channels
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def main():
    # Set the dataset root directory (adjust if necessary)
    dataset_root = "./datasets/tooth_ds_val"
    # Initialize the segmentation dataset WITHOUT additional transforms
    dataset = ToothSegmentationDataset(path=dataset_root, transform=None)

    # Create output directory for visualizations if it doesn't exist.
    output_dir = "./segmentation_patches_visualization"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(dataset)} samples for segmentation.")

    # Loop over each sample in the dataset.
    for i in range(len(dataset)):
        sample = dataset[i]
        image_patch = sample["image_patch"]
        mask = sample["mask"]

        # Convert to proper numpy image format.
        img_np = convert_image(image_patch)
        mask_np = convert_mask(mask)

        # Ensure that image_patch and mask have the same height.
        if img_np.shape[0] != mask_np.shape[0]:
            # Resize the mask to match the height of the image patch.
            mask_np = cv2.resize(mask_np, (mask_np.shape[1], img_np.shape[0]))

        # Glue the image patch and mask horizontally.
        glued = np.concatenate([img_np, mask_np], axis=1)

        out_path = os.path.join(output_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(out_path, glued)
        print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
