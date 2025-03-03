"""
Dataset loader for tooth segmentation.

For every annotated object in the original dataset, this loader:
    - Loads the full image.
    - Decodes the object's bitmap annotation to obtain a binary mask.
    - Computes a tight bounding box based on the nonzero region of the mask.
    - Applies the origin offset (from the annotation) to map the mask coordinates to the full image.
    - Crops the image patch (from the full image) and also crops the object mask accordingly.
    
Each sample is a dictionary:
    {
        "image_patch": image patch as a tensor or augmented output,
        "mask": binary mask corresponding to the cropped patch.
    }
The optional transforms (typically from Albumentations) should take keys "image" and "mask".
"""

import base64
import io
import json
import os
import zlib

import cv2
import numpy as np
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import Dataset


def decode_bitmap(data_str):
    """
    Decode a bitmap string from a JSON annotation.
    The string is expected to be a base64-encoded zlib-compressed image.
    Returns a numpy array (grayscale) of the decoded mask.
    """
    try:
        decoded = base64.b64decode(data_str)
    except Exception as e:
        raise ValueError(f"Error decoding base64: {e}")
    try:
        decompressed = zlib.decompress(decoded)
        mask_img = Image.open(io.BytesIO(decompressed)).convert("L")
    except Exception:
        mask_img = Image.open(io.BytesIO(decoded)).convert("L")
    mask = np.array(mask_img)
    return mask


class ToothSegmentationDataset(Dataset):
    def __init__(self, path, image_folder="ds/img", ann_folder="ds/ann", transform=None, padding: int = 32):
        """
        Arguments:
            path: Root directory of the dataset. Assumes the following structure:
                  path/
                    ds/img/   -- image files
                    ds/ann/   -- corresponding JSON annotations (named as "<image_filename>.json")
            transforms: Optional Albumentations transforms that take keys "image" and "mask".
            padding: Extra padding (in pixels) to add around the object's bounding box.
        """
        self.root = path
        self.image_dir = os.path.join(path, image_folder)
        self.ann_dir = os.path.join(path, ann_folder)
        self.transform = transform
        self.padding = padding

        # Build samples: each sample corresponds to one annotated object.
        self.samples = []  # each sample is a dict with keys: "image_path", "mask_data", "origin", "abs_bbox", "mask_bbox"
        image_files = sorted(os.listdir(self.image_dir))
        for img_file in image_files:
            image_path = os.path.join(self.image_dir, img_file)
            ann_file = img_file + ".json"  # assuming annotation filename is image file + ".json"
            ann_path = os.path.join(self.ann_dir, ann_file)
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path, "r") as f:
                    ann = json.load(f)
            except Exception as e:
                print(f"Failed to load annotation for {img_file}: {e}")
                continue

            # Process each object in the JSON annotation.
            for obj in ann.get("objects", []):
                if obj.get("geometryType") != "bitmap":
                    continue
                bitmap_info = obj.get("bitmap", {})
                data_str = bitmap_info.get("data", "")
                if not data_str:
                    continue
                origin = bitmap_info.get("origin", [0, 0])
                # Decode the mask once to validate and compute bounding boxes.
                try:
                    mask = decode_bitmap(data_str)
                except Exception as e:
                    print(f"Skipping object in {img_file} due to decode error: {e}")
                    continue

                # Convert to binary mask
                bin_mask = (mask > 128).astype(np.uint8)
                ys, xs = np.where(bin_mask)
                if ys.size == 0 or xs.size == 0:
                    print(f"Warning: Empty mask found in {img_file}; skipping object.")
                    continue

                # Compute bbox in annotation mask coordinates
                x0_ann = int(np.min(xs))
                y0_ann = int(np.min(ys))
                x1_ann = int(np.max(xs)) + 1  # +1 so that cropping includes the max index
                y1_ann = int(np.max(ys)) + 1
                mask_bbox = (x0_ann, y0_ann, x1_ann, y1_ann)

                # Absolute bbox for the full image: add the origin offset.
                abs_bbox = (x0_ann + int(origin[0]),
                            y0_ann + int(origin[1]),
                            x1_ann + int(origin[0]),
                            y1_ann + int(origin[1]))

                self.samples.append({
                    "image_path": image_path,
                    "mask_data": data_str,
                    "origin": origin,
                    "abs_bbox": abs_bbox,  # (left, top, right, bottom) in full image coordinates
                    "mask_bbox": mask_bbox  # bounding box in annotation mask coordinates
                })

        print(f"Loaded {len(self.samples)} annotated objects from {len(image_files)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load full image
        image = Image.open(sample["image_path"]).convert("RGB")
        # Crop image patch using the absolute bounding box with extra padding.
        abs_bbox = sample["abs_bbox"]  # (left, top, right, bottom)
        pad = self.padding
        image_width, image_height = image.size
        padded_left = max(0, abs_bbox[0] - pad)
        padded_top = max(0, abs_bbox[1] - pad)
        padded_right = min(image_width, abs_bbox[2] + pad)
        padded_bottom = min(image_height, abs_bbox[3] + pad)
        padded_bbox = (padded_left, padded_top, padded_right, padded_bottom)
        image_patch = image.crop(padded_bbox)

        # Decode the annotation mask from stored data.
        mask = decode_bitmap(sample["mask_data"])
        bin_mask = (mask > 128).astype(np.uint8)
        # Crop the mask based on the bounding box (with extra padding) within the annotation mask.
        x0_ann, y0_ann, x1_ann, y1_ann = sample["mask_bbox"]
        pad = self.padding
        mask_height, mask_width = bin_mask.shape

        # Desired crop coordinates with padding (may extend out-of-bound)
        desired_x0 = x0_ann - pad
        desired_y0 = y0_ann - pad
        desired_x1 = x1_ann + pad
        desired_y1 = y1_ann + pad

        # Compute required padding if the desired crop extends beyond the mask boundaries.
        pad_left = max(0, 0 - desired_x0)
        pad_top = max(0, 0 - desired_y0)
        pad_right = max(0, desired_x1 - mask_width)
        pad_bottom = max(0, desired_y1 - mask_height)

        # Compute crop indices (clamped to mask boundaries)
        crop_x0_mask = max(0, desired_x0)
        crop_y0_mask = max(0, desired_y0)
        crop_x1_mask = min(mask_width, desired_x1)
        crop_y1_mask = min(mask_height, desired_y1)

        # Crop the available region from the mask.
        cropped_mask = bin_mask[crop_y0_mask:crop_y1_mask, crop_x0_mask:crop_x1_mask]

        # Pad the cropped mask with zeros for areas that may be missing.
        mask_crop = np.pad(cropped_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode="constant", constant_values=0)

        # Optionally, apply transforms (e.g. resizing, normalization, augmentation) to both image_patch and mask_crop.
        if self.transform is not None:
            # Note: Albumentations expects numpy array for "image" and "mask"
            image_np = np.array(image_patch)
            # The issue was caused by inconsistent mask and image sizes after cropping and padding.
            # We need to ensure the mask is resized to match the *final* image patch size
            # *before* passing them to Albumentations.
            h, w = image_np.shape[:2]
            mask_crop_resized = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            # Now, pass the correctly sized image and mask to Albumentations.
            augmented = self.transform(image=image_np, mask=mask_crop_resized)
            # if augmented['mask'].shape[0] != augmented['image'].shape[0]:
            #     print(f"Warning: mask and image shapes are different after transform: {augmented['mask'].shape} != {augmented['image'].shape}")

            image_patch = augmented["image"]
            mask_crop = augmented["mask"]
        else:
            # Otherwise convert image_patch to tensor and mask to tensor.
            to_tensor = tvt.ToTensor()
            image_patch = to_tensor(image_patch)
            mask_crop = torch.as_tensor(mask_crop, dtype=torch.long)

        return {"pixel_values": image_patch, "label": mask_crop}
