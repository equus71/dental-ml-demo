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

class ToothDetectionJsonDatasetTiled(Dataset):
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
    def __init__(self, root, image_folder="ds/img", ann_folder="ds/ann", transforms=None, tile_size=640, skip_empty_tiles=True):
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
        tile_size : int
            Size of the tiles to be created from the images
        skip_empty_tiles : bool
            Whether to skip tiles that do not contain any bounding boxes
        """
        self.image_dir = os.path.join(root, image_folder)
        self.ann_dir = os.path.join(root, ann_folder)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transforms = transforms
        self.tile_size = tile_size
        self.skip_empty_tiles = skip_empty_tiles
        self.tiles = []  # list of dicts with tile metadata
        for filename in self.image_files:
            image_path = os.path.join(self.image_dir, filename)
            ann_filename = filename + ".json"
            ann_path = os.path.join(self.ann_dir, ann_filename)
            if not os.path.exists(ann_path):
                continue  # Skip if no annotation
            try:
                with open(ann_path, "r") as f:
                    ann = json.load(f)
            except Exception as e:
                print(f"Failed to load annotation for {filename}: {e}")
                continue
            with Image.open(image_path) as img:
                w, h = img.size
            # Break the image into tiles with tile_size; tiles at the border may be smaller.
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    tile_box = [x, y, x + tile_size, y + tile_size]  # [x0, y0, x1, y1]
                    if self.skip_empty_tiles:
                        # Check if this tile contains any bounding boxes.
                        has_bboxes = False
                        for obj in ann.get("objects", []):
                            if obj.get("geometryType") != "bitmap":
                                continue
                            bitmap_info = obj.get("bitmap", {})
                            data_str = bitmap_info.get("data", "")
                            origin = bitmap_info.get("origin", [0, 0])
                            if not data_str:
                                continue
                            try:
                                mask = decode_bitmap(data_str)
                            except Exception:
                                continue
                            bin_mask = (mask > 128).astype(np.uint8)
                            ys, xs = np.where(bin_mask)
                            if ys.size == 0 or xs.size == 0:
                                continue
                            # Compute object bbox in full image coordinates
                            obj_x_min = int(np.min(xs)) + origin[0]
                            obj_x_max = int(np.max(xs)) + origin[0]
                            obj_y_min = int(np.min(ys)) + origin[1]
                            obj_y_max = int(np.max(ys)) + origin[1]
                            # Clip bbox to full image boundaries
                            obj_x_min = max(0, obj_x_min)
                            obj_y_min = max(0, obj_y_min)
                            obj_x_max = min(w, obj_x_max)
                            obj_y_max = min(h, obj_y_max)
                            if obj_x_max <= obj_x_min or obj_y_max <= obj_y_min:
                                continue
                            # Compute intersection of bbox with tile region.
                            inter_x_min = max(obj_x_min, x)
                            inter_y_min = max(obj_y_min, y)
                            inter_x_max = min(obj_x_max, x + tile_size)
                            inter_y_max = min(obj_y_max, y + tile_size)
                            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                                has_bboxes = True
                                break  # Found at least one bbox in this tile
                        if not has_bboxes:
                            continue  # Skip this tile if no bboxes
                    self.tiles.append({
                        "image_filename": filename,
                        "tile_bbox": tile_box,
                        "full_size": (w, h)
                    })

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # New implementation for tiled dataset.
        tile_info = self.tiles[idx]
        image_filename = tile_info["image_filename"]
        image_path = os.path.join(self.image_dir, image_filename)
        tile_box = tile_info["tile_bbox"]  # [x0, y0, x1, y1]
        x0, y0, x1, y1 = tile_box
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            # Crop the tile region (clip if tile extends beyond image boundaries)
            crop_x0, crop_y0 = x0, y0
            crop_x1, crop_y1 = min(x1, w), min(y1, h)
            tile = np.array(img)[crop_y0:crop_y1, crop_x0:crop_x1, :]
        # Zero-pad tile if it's smaller than tile_size x tile_size
        tile_h, tile_w = tile.shape[:2]
        pad_bottom = self.tile_size - tile_h
        pad_right = self.tile_size - tile_w
        if pad_bottom > 0 or pad_right > 0:
            tile = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant", constant_values=0)

        # Load annotation for this image.
        ann_filename = image_filename + ".json"
        ann_path = os.path.join(self.ann_dir, ann_filename)
        with open(ann_path, "r") as f:
            ann = json.load(f)
        boxes = []
        labels = []
        for obj in ann.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue
            bitmap_info = obj.get("bitmap", {})
            data_str = bitmap_info.get("data", "")
            origin = bitmap_info.get("origin", [0, 0])
            if not data_str:
                continue
            try:
                mask = decode_bitmap(data_str)
            except Exception:
                continue
            bin_mask = (mask > 128).astype(np.uint8)
            ys, xs = np.where(bin_mask)
            if ys.size == 0 or xs.size == 0:
                continue
            # Compute object bbox in full image coordinates
            obj_x_min = int(np.min(xs)) + origin[0]
            obj_x_max = int(np.max(xs)) + origin[0]
            obj_y_min = int(np.min(ys)) + origin[1]
            obj_y_max = int(np.max(ys)) + origin[1]
            # Clip bbox to full image boundaries
            obj_x_min = max(0, obj_x_min)
            obj_y_min = max(0, obj_y_min)
            obj_x_max = min(w, obj_x_max)
            obj_y_max = min(h, obj_y_max)
            if obj_x_max <= obj_x_min or obj_y_max <= obj_y_min:
                continue
            # Compute intersection of bbox with tile region.
            inter_x_min = max(obj_x_min, x0)
            inter_y_min = max(obj_y_min, y0)
            inter_x_max = min(obj_x_max, x1)
            inter_y_max = min(obj_y_max, y1)
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                continue
            # Convert intersection to tile coordinate system.
            tile_bbox_coords = [inter_x_min - x0, inter_y_min - y0, inter_x_max - x0, inter_y_max - y0]
            boxes.append(tile_bbox_coords)
            labels.append(1)

        # Convert boxes to COCO format ([x,y,w,h]) normalized by tile size.
        abs_boxes = np.array(boxes, dtype=np.float32)
        if abs_boxes.size > 0:
            coco_boxes = abs_boxes.copy()
            coco_boxes[:, 2] = abs_boxes[:, 2] - abs_boxes[:, 0]
            coco_boxes[:, 3] = abs_boxes[:, 3] - abs_boxes[:, 1]
            coco_boxes[:, 0] /= self.tile_size
            coco_boxes[:, 1] /= self.tile_size
            coco_boxes[:, 2] /= self.tile_size
            coco_boxes[:, 3] /= self.tile_size
        else:
            coco_boxes = abs_boxes

        target = {
            "boxes": coco_boxes,  # normalized COCO boxes for the tile
            "labels": np.array(labels, dtype=np.int64),
            "image_id": torch.as_tensor([idx])
        }

        if self.transforms is not None:
            transformed = self.transforms(image=tile, bboxes=coco_boxes.tolist(), labels=labels)
            tile = transformed["image"]
            h_trans, w_trans = tile.shape[:2]
            transformed_boxes = np.array(transformed["bboxes"], dtype=np.float32)
            norm_boxes = transformed_boxes.copy()
            norm_boxes[:, 0] /= w_trans
            norm_boxes[:, 1] /= h_trans
            norm_boxes[:, 2] /= w_trans
            norm_boxes[:, 3] /= h_trans
            target["boxes"] = torch.as_tensor(norm_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            tile = T.ToTensor()(tile)
            # Ensure norm_boxes is assigned even when no transforms are applied.
            norm_boxes = coco_boxes.copy() if coco_boxes.size > 0 else np.empty((0, 4), dtype=np.float32)
            target["boxes"] = torch.as_tensor(norm_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        target["class_labels"] = target.pop("labels")
        return {"pixel_values": tile, "labels": target, "original_image_path": f"{image_filename}_tile_{x0}_{y0}"} 