import os
import json
import base64
import zlib
import io
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

def decode_bitmap(data_str):
    """
    Decode a bitmap string from a JSON annotation.
    The string is expected to be a base64-encoded zlib-compressed image.
    Returns a numpy array (grayscale) of the decoded mask.
    """
    try:
        # Decode base64 string
        decoded = base64.b64decode(data_str)
    except Exception as e:
        raise ValueError(f"Error decoding base64: {e}")
    try:
        # Attempt to decompress the data (if it was compressed)
        decompressed = zlib.decompress(decoded)
        mask_img = Image.open(io.BytesIO(decompressed)).convert("L")
    except Exception:
        # If decompression fails, assume the decoded bytes form a valid image
        mask_img = Image.open(io.BytesIO(decoded)).convert("L")
    mask = np.array(mask_img)
    # print(f"Decoded mask shape: {mask.shape}, dtype: {mask.dtype}, max value: {mask.max()}, min value: {mask.min()}") # DEBUG
    return mask

class ToothDetectionJsonDataset(Dataset):
    """
    A dataset class for tooth detection using JSON annotations.

    Assumes the following folder structure:

        root/
            Images/
                1.jpg
                2.jpg
                ...
            ann/
                1.jpg.json
                2.jpg.json
                ...

    Each JSON file should have a structure similar to:
    {
        "description": "",
        "tags": [],
        "size": {"height": H, "width": W},
        "objects": [
            {
                "id": ...,
                "classTitle": "15",
                "geometryType": "bitmap",
                "bitmap": {
                    "data": "....",       // base64 encoded zlib-compressed image
                    "origin": [x_origin, y_origin]
                }
            },
            ... other objects ...
        ]
    }

    For each object, a bounding box is computed by decoding the bitmap,
    thresholding to get a binary mask, finding the (min, max) x and y coordinates,
    applying the origin offset, and then normalizing by the image size.
    """
    def __init__(self, root, image_folder="ds/img", ann_folder="ds/ann", transforms=None):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset (e.g., "datasets/tooth_ds/ds")
        image_folder : str
            Subfolder name for images (default "Images")
        ann_folder : str
            Subfolder name for annotations (default "ann")
        transforms : callable, optional
            A function (or Albumentations Compose) to apply to the image and boxes.
            It should accept keys "image", "bboxes", and "labels" and return a dict.
        """
        self.image_dir = os.path.join(root, image_folder)
        self.ann_dir = os.path.join(root, ann_folder)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filename and corresponding annotation filename.
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        ann_filename = image_filename + ".json"  # Annotation file assumed to be "1.jpg.json"
        ann_path = os.path.join(self.ann_dir, ann_filename)

        # Load image.
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Load JSON annotation.
        with open(ann_path, "r") as f:
            ann = json.load(f)

        boxes = []
        labels = []
        # Process each object in the annotation.
        for obj in ann.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue
            bitmap_info = obj.get("bitmap", {})
            data_str = bitmap_info.get("data", "")
            origin = bitmap_info.get("origin", [0, 0])
            if not data_str:
                continue
            # Decode the bitmap to obtain a binary mask.
            mask = decode_bitmap(data_str)
            bin_mask = (mask > 128).astype(np.uint8)
            ys, xs = np.where(bin_mask)
            # print(f"Mask shape: {mask.shape}, Binary mask shape: {bin_mask.shape}") # DEBUG
            # print(f"Non-zero indices (ys, xs) shapes: ys.shape={ys.shape}, xs.shape={xs.shape}") # DEBUG
            if ys.size == 0 or xs.size == 0:
                print("Warning: Empty mask found, skipping object.") # DEBUG
                continue
            # Compute bounding box coordinates and apply origin offset.
            x_min = int(np.min(xs)) + origin[0]
            x_max = int(np.max(xs)) + origin[0]
            y_min = int(np.min(ys)) + origin[1]
            y_max = int(np.max(ys)) + origin[1]

            # Clip boxes to image boundaries (optional, but good practice)
            x_min = max(0, x_min) # Ensure x_min is not negative
            y_min = max(0, y_min) # Ensure y_min is not negative
            x_max = min(w, x_max) # Ensure x_max is within image width
            y_max = min(h, y_max) # Ensure y_max is within image height

            if y_max <= y_min or x_max <= x_min: # Check for invalid boxes
                print(f"Warning: Invalid bbox found (y_max <= y_min or x_max <= x_min) after clipping: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}. Skipping object.") # DEBUG
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            # For simplicity, all teeth are assigned the same class label (1).
            labels.append(1)

        # First, work with absolute coordinates.
        # (boxes is a list of [xmin, ymin, xmax, ymax] in absolute pixel coordinates.)
        abs_boxes = np.array(boxes, dtype=np.float32)

        # Convert absolute boxes to COCO format in absolute coordinates
        # (i.e. [x_min, y_min, width, height])
        coco_boxes_abs = abs_boxes.copy()
        if coco_boxes_abs.size > 0:
            coco_boxes_abs[:, 2] = abs_boxes[:, 2] - abs_boxes[:, 0]  # width = x_max - x_min
            coco_boxes_abs[:, 3] = abs_boxes[:, 3] - abs_boxes[:, 1]  # height = y_max - y_min
        else:
            print(f"Warning: No bounding boxes found for image {idx} at {image_path}")
        
        target = {
            "boxes": coco_boxes_abs,  # COCO format boxes: [x, y, w, h] normalized
            "labels": np.array(labels, dtype=np.int64),  # temporary key; will be renamed to "class_labels"
            "image_id": torch.as_tensor([idx])  # converted to torch tensor
        }
        
        # Apply transforms if provided.
        if self.transforms is not None:
            # Provide Albumentations with absolute COCO boxes.
            transformed = self.transforms(image=np.array(image), bboxes=coco_boxes_abs.tolist(), labels=labels)
            image = transformed["image"]
            # Get transformed image dimensions
            h_transformed, w_transformed = image.shape[:2]
            # Normalize the transformed absolute boxes by dividing by transformed image dimensions
            transformed_boxes = np.array(transformed["bboxes"], dtype=np.float32)
            norm_boxes = transformed_boxes.copy()
            norm_boxes[:, 0] /= w_transformed  # x_min normalized
            norm_boxes[:, 1] /= h_transformed  # y_min normalized
            norm_boxes[:, 2] /= w_transformed  # width normalized
            norm_boxes[:, 3] /= h_transformed  # height normalized
            target["boxes"] = torch.as_tensor(norm_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            image = T.ToTensor()(image)
            # If no transform is applied, normalize the absolute boxes by the original image size.
            norm_boxes = coco_boxes_abs.copy()
            norm_boxes[:, 0] /= w
            norm_boxes[:, 1] /= h
            norm_boxes[:, 2] /= w
            norm_boxes[:, 3] /= h
            target["boxes"] = torch.as_tensor(norm_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        
        # Rename "labels" to "class_labels" as expected by RT-DETR
        target["class_labels"] = target.pop("labels")
        return {"pixel_values": image, "labels": target, "original_image_path": image_path} 