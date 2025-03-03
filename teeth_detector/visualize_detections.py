import os
import torch
from transformers import AutoModelForObjectDetection
from tooth_detection_json_dataset import ToothDetectionJsonDataset
from train_teeth_detector import get_transform
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T  # Import torchvision transforms
from transformers.models.auto.image_processing_auto import AutoImageProcessor

# Add denormalize_boxes function here (copy from render_gt_bboxes.py)
def denormalize_boxes(boxes, width, height):
    """
    Convert normalized bounding boxes (range [0,1]) to absolute pixel coordinates.
    boxes: numpy array of shape (N, 4) where each box is [x_min, y_min, x_max, y_max]
    """
    if boxes.size == 0:  # Handle empty boxes array
        return boxes  # Return empty array as is
    boxes_abs = boxes.copy()
    boxes_abs[:, [0, 2]] *= width
    boxes_abs[:, [1, 3]] *= height
    return boxes_abs


# Function to perform inference
def predict(image, model, device):
    # Prepare image for the model
    image = image.to(device)
    image = image.unsqueeze(0) # Add batch dimension
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(image)

    # RT-DETR-V2 produces normalized COCO boxes in [x, y, w, h]
    # Extract boxes (shape: [N,4]) from the model output.
    coco_boxes = outputs.pred_boxes[0].cpu().numpy()  # COCO-format boxes (normalized)

    # Convert from COCO [x, y, w, h] format to xyxy (normalized).
    boxes_xyxy = coco_boxes.copy()
    boxes_xyxy[:, 2] = coco_boxes[:, 0] + coco_boxes[:, 2]  # x_max = x_min + width
    boxes_xyxy[:, 3] = coco_boxes[:, 1] + coco_boxes[:, 3]  # y_max = y_min + height

    # Process logits and compute scores and class ids
    logits = outputs.logits[0]  # shape: (N, num_classes)
    scores = torch.sigmoid(logits.max(-1).values)  # shape: (N,)
    class_ids = torch.argmax(logits, dim=1)

    detections = sv.Detections(
        xyxy=boxes_xyxy,
        confidence=scores.cpu().numpy(),
        class_id=class_ids.cpu().numpy()
    )
    return detections

def main(
    val_dataset_path: str = "./datasets/tooth_ds_val/", # Path to validation dataset
    model_path: str = "tooth_detector_deta50/stage1/checkpoint-70", # Path to your trained model
    output_vis_dir: str = "./predicted_tooths_visualizations", # Directory to save visualizations
    conf_threshold: float = 0.1 # Confidence threshold for detections
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load validation dataset
    val_dataset = ToothDetectionJsonDataset(
        root=val_dataset_path,
        transforms=get_transform(train=False) # Use validation transforms
    )
    # 2. Load trained model and image processor
    model = AutoModelForObjectDetection.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-resnet-50")
    model.to(device)

    # 3. Create supervision annotators 
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1)

    os.makedirs(output_vis_dir, exist_ok=True)

    with torch.no_grad():
        # 4. Inference and visualization loop
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            image_tensor = sample["pixel_values"] # Tensor [C, H, W]
            original_image_path = sample["original_image_path"]
            original_image_pil = Image.open(original_image_path).convert("RGB")
            image_np = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
            height, width, _ = image_np.shape

            # Perform prediction
            outputs = model(image_tensor.unsqueeze(0).to(device))
            
            # Post-process predictions using image processor
            processed_outputs = image_processor.post_process_object_detection(
                outputs,
                threshold=conf_threshold,
                target_sizes=[(height, width)]
            )[0]

            # Convert to supervision Detections format
            detections = sv.Detections(
                xyxy=processed_outputs["boxes"].cpu().numpy(),
                confidence=processed_outputs["scores"].cpu().numpy(),
                class_id=processed_outputs["labels"].cpu().numpy()
            )

            # Format labels for visualization
            labels = [
                f"tooth {confidence:.2f}"
                for confidence in detections.confidence
            ]

            # Annotate frame (on original image)
            annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # Save visualization
            output_path = os.path.join(output_vis_dir, f"predicted_image_{os.path.basename(original_image_path)}") # Use original filename
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved predicted image: {output_path}")

    print(f"Visualizations saved to {output_vis_dir}")


if __name__ == "__main__":
    main() 