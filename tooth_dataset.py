import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ToothDetectionDataset(Dataset):
    """
    A dataset class for tooth detection.

    Assumes the following folder structure:

        root/
            Images/
                image1.png
                image2.png
                ...
            Masks/
                image1.png
                image2.png
                ...

    Each mask is a grayscale image where 0 is background and each tooth is represented
    by a unique positive integer.
    """

    def __init__(self, root, image_folder="Images", mask_folder="Masks", transforms=None):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset (e.g. "datasets/tooth_ds")
        image_folder : str
            Subfolder name for images. Default "Images"
        mask_folder : str
            Subfolder name for masks. Default "Masks"
        transforms : callable, optional
            A function (or Albumentations Compose) to apply to the image and boxes.
            (It should accept keys "image", "bboxes", and "labels" and return a dict.)
        """
        self.image_dir = os.path.join(root, image_folder)
        self.mask_dir = os.path.join(root, mask_folder)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and corresponding mask
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to numpy array
        mask_np = np.array(mask)

        # Get unique instance ids (ignoring background = 0)
        instance_ids = np.unique(mask_np)
        instance_ids = instance_ids[instance_ids != 0]

        boxes = []
        labels = []
        # For each tooth instance, compute the bounding box in pixel coordinates
        for instance in instance_ids:
            ys, xs = np.where(mask_np == instance)
            if ys.size == 0 or xs.size == 0:
                continue
            x_min = np.min(xs)
            x_max = np.max(xs)
            y_min = np.min(ys)
            y_max = np.max(ys)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # All teeth are assigned label 1

        boxes = np.array(boxes, dtype=np.float32)
        w, h = image.size

        # Normalize bounding boxes to [0,1] in (xmin, ymin, xmax, ymax) format
        if len(boxes) > 0:
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h

        target = {
            "boxes": boxes,  # shape: (num_instances, 4)
            "labels": np.array(labels, dtype=np.int64),  # shape: (num_instances,)
            "image_id": np.array([idx])
        }

        # Apply transforms if provided.
        # If using Albumentations, ensure bbox_params are specified accordingly.
        if self.transforms is not None:
            # Example for Albumentations: bboxes in pascal_voc format and labels list.
            transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed["image"]
            # Update boxes and labels if required.
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            # If no transforms are provided, convert image to tensor using torchvision
            image = T.ToTensor()(image)
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return {"pixel_values": image, "labels": target}
