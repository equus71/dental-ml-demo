import os
import cv2
import torch
import numpy as np
from tooth_detection_json_dataset import ToothDetectionJsonDataset
import torchvision.transforms as T
import supervision as sv  # ensure you have installed supervision: pip install supervision
from PIL import Image

def denormalize_boxes(boxes, width, height):
    """
    Convert normalized COCO bounding boxes (range [0,1]) to absolute pixel coordinates in xyxy format.
    Returns:
        numpy array of shape (N, 4) where each box is [x, y, x+w, y+h]
    """
    if boxes.size == 0:
        return boxes

    # First convert normalized values back to absolute values
    abs_boxes = boxes.copy()
    abs_boxes[:, 0] *= width   # x coordinate (absolute)
    abs_boxes[:, 1] *= height  # y coordinate (absolute)
    abs_boxes[:, 2] *= width   # width (absolute)
    abs_boxes[:, 3] *= height  # height (absolute)

    # Now convert from [x, y, w, h] to [x, y, x+w, y+h] (i.e. xyxy format)
    xyxy = abs_boxes.copy()
    xyxy[:, 2] = abs_boxes[:, 0] + abs_boxes[:, 2]
    xyxy[:, 3] = abs_boxes[:, 1] + abs_boxes[:, 3]
    return xyxy

def main():
    dataset_root = "./datasets/tooth_ds_train"  # Example dataset root
    dataset = ToothDetectionJsonDataset(root=dataset_root, transforms=None)
    
    output_dir = "./visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a BoxAnnotator instance from the supervision package.
    # Removed 'text_thickness' because it is not supported.
    box_annotator = sv.BoxAnnotator(thickness=2)
    
    # Class mapping for display names (if desired)
    class_names = {1: "tooth"}
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")
            continue

        # Get image and target
        # sample["pixel_values"] is a tensor of shape [C, H, W] with values in [0,1]
        image_tensor = sample["pixel_values"]
        target = sample["labels"]  # this contains keys: "boxes" and "class_labels"
        
        # Convert image tensor to a NumPy array (H, W, 3) in uint8 format for visualization.
        image_np = image_tensor.cpu().numpy()
        image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Get image dimensions.
        height, width, _ = image_np.shape

        # Get bounding boxes (normalized)
        boxes_tensor = target["boxes"]  # shape (N,4)
        boxes = boxes_tensor.cpu().numpy()
        if len(boxes) == 0:
            print(f"Warning: No bounding boxes found for image {idx}")
            continue
        
        # Convert normalized boxes to absolute pixel coordinates.
        boxes_abs = denormalize_boxes(boxes, width, height)
        
        # Get class labels (from "class_labels")
        class_labels = target["class_labels"].cpu().numpy()  # shape (N,)
        
        # Create a detections instance from supervision.
        # We set a default confidence of 1.0 for all GT boxes.
        detections = sv.Detections(
            xyxy=boxes_abs,
            confidence=np.ones(len(boxes_abs)),
            class_id=class_labels
        )
        
        # Draw annotated bounding boxes on the image.
        annotated_image = box_annotator.annotate(
            scene=image_np.copy(), detections=detections,
        )
        
        # Save or display the annotated image.
        output_path = os.path.join(output_dir, f"image_{idx:04d}.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image: {output_path}")

if __name__ == "__main__":
    main()