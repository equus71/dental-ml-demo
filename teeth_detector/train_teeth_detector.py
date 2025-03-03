import os
import torch
from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer
from tooth_detection_json_dataset import (
    ToothDetectionJsonDataset,
)  # Assuming this is in the same directory
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import RTDetrV2ForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from functools import partial
from transformers import RTDetrImageProcessor
from dataclasses import dataclass


# Define transformations using Albumentations
def get_transform(train, size=640):
    if train:
        return A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.RandomBrightnessContrast(p=0.2),
                A.AdditiveNoise(    
                    noise_type="gaussian",
                    spatial_mode="shared",
                    noise_params={"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)}
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
        )


def compute_metrics(eval_pred):
    """
    Compute mAP and mAP@50 using Torchmetrics.
    Assumes that predictions is a list of dictionaries, each with keys:
        "boxes" (a list/array of boxes in COCO normalized format [x,y,w,h]),
        "scores" (the confidence scores for each box),
        "labels" (predicted class labels).
    And that each ground-truth dictionary (in labels) has keys:
        "boxes": ground truth COCO boxes [x,y,w,h] normalized, and
        "class_labels": ground truth class labels.
    """ 
    predictions, references = eval_pred
    preds_list = []
    targets_list = []
    img_size = 640.0  # All images are resized to 640x640 in our transforms
    for pred, target in zip(predictions, references):
        # Ensure prediction dict has the required keys.
        if "boxes" not in pred:
            if "pred_boxes" in pred and "logits" in pred:
                # Extract data from model outputs.
                pred_boxes = pred["pred_boxes"]
                logits = pred["logits"]
                scores = torch.sigmoid(logits.max(-1).values)
                labels = torch.argmax(logits, dim=1)
                # Convert tensors to numpy arrays.
                pred["boxes"] = pred_boxes.cpu().numpy() if isinstance(pred_boxes, torch.Tensor) else pred_boxes
                pred["scores"] = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
                pred["labels"] = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            else:
                # If missing expected keys, treat as no detections.
                pred["boxes"] = np.empty((0, 4), dtype=np.float32)
                pred["scores"] = np.empty((0,), dtype=np.float32)
                pred["labels"] = np.empty((0,), dtype=np.int64)

        # Convert predicted boxes from normalized COCO ([x, y, w, h]) to absolute xyxy format
        if len(pred["boxes"]) == 0:
            pred_boxes = torch.empty((0, 4))
        else:
            boxes = torch.tensor(pred["boxes"], dtype=torch.float32)  # (N,4) in [x, y, w, h]
            boxes_abs = boxes * img_size
            x1 = boxes_abs[:, 0]
            y1 = boxes_abs[:, 1]
            x2 = boxes_abs[:, 0] + boxes_abs[:, 2]
            y2 = boxes_abs[:, 1] + boxes_abs[:, 3]
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # Convert ground truth boxes similarly
        if len(target["boxes"]) == 0:
            gt_boxes = torch.empty((0, 4))
        else:
            boxes = torch.tensor(target["boxes"])  # (N,4) in [x, y, w, h]
            boxes_abs = boxes * img_size
            x1 = boxes_abs[:, 0]
            y1 = boxes_abs[:, 1]
            x2 = boxes_abs[:, 0] + boxes_abs[:, 2]
            y2 = boxes_abs[:, 1] + boxes_abs[:, 3]
            gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)


        preds_list.append({
            "boxes": pred_boxes,
            "scores": torch.tensor(pred["scores"]),
            "labels": torch.tensor(pred["labels"]),
        })
        targets_list.append({
            "boxes": torch.tensor(target["boxes"], dtype=torch.float32),
            "labels": torch.tensor(target["class_labels"]),
        })
    metric = MeanAveragePrecision()
    metric.update(preds_list, targets_list)
    result = metric.compute()
    # Return the mAP and mAP50 as Python scalars
    return {"mAP": result["map"].item(), "mAP50": result["map_50"].item()}

from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)

    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

@dataclass
class ModelOutput:

    logits: torch.Tensor

    pred_boxes: torch.Tensor

@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    post_processed_targets = []
    post_processed_predictions = []
    image_size = 640

    # Collect targets in the required format for metric computation
    for single_target in targets:
        boxes = torch.tensor(single_target["boxes"])
        boxes = convert_bbox_yolo_to_pascal(boxes, (image_size, image_size))
        labels = torch.tensor(single_target["class_labels"])
        post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    logits, boxes = predictions[1], predictions[2]
    # for single_im_logits, single_im_boxes in zip(logits, boxes):
    output = ModelOutput(
        logits=torch.tensor(logits), pred_boxes=torch.tensor(boxes)
    )
    post_processed_output = image_processor.post_process_object_detection(
        output, threshold=threshold, target_sizes=(image_size, image_size)
    )
    post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    for class_id, class_map, class_mar in zip(
        classes, map_per_class, mar_100_per_class
    ):
        class_name = (
            id2label[class_id.item()] if id2label is not None else class_id.item()
        )
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
    return metrics


def main(
    train_dataset_path: str = "./datasets/tooth_ds_train/",
    val_dataset_path: str = "./datasets/tooth_ds_val/",
    model_checkpoint: str = "PekingU/rtdetr_v2_r101vd",
    output_dir: str = "tooth_detector_rt_detr_v2",
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 15,
    gradient_accumulation_steps: int = 4,
):

    network_size = 960
    # 1. Create Datasets
    train_dataset = ToothDetectionJsonDataset(
        root=train_dataset_path, transforms=get_transform(train=True, size=network_size)
    )

    val_dataset = ToothDetectionJsonDataset(
        root=val_dataset_path, transforms=get_transform(train=False, size=network_size)
    )

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    # 2. Load Pre-trained Model
    model = RTDetrV2ForObjectDetection.from_pretrained(
        model_checkpoint,
        num_labels=2,  # 1 class (tooth) + no-class (background)
        ignore_mismatched_sizes=True,
    )
    id2label = {0: "background", 1: "tooth"}
    image_processor = RTDetrImageProcessor.from_pretrained(model_checkpoint)

    # # 3. Define Training Arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     learning_rate=learning_rate,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     warmup_ratio=0.1,
    #     metric_for_best_model="eval_loss",
    #     logging_steps=10,
    #     logging_dir=os.path.join(output_dir, "logs"),
    #     save_total_limit=2,  # Only keep the last 2 checkpoints
    #     remove_unused_columns=False,  # Important for object detection tasks
    #     dataloader_num_workers=4,  # Adjust based on your CPU cores and memory
    #     fp16=True,  # Enable mixed precision training for faster training and less memory usage (if supported by your GPU)
    # )

    # 4. Create Data Collator: a custom collate function that stacks images and leaves labels as a list
    def collate_fn(batch):
        pixel_values = torch.stack([sample["pixel_values"] for sample in batch])
        labels = [sample["labels"] for sample in batch]
        return {"pixel_values": pixel_values, "labels": labels}

    data_collator = collate_fn

    # eval_compute_metrics_fn = partial(
    #     compute_metrics,
    #     image_processor=image_processor,
    #     id2label=id2label,
    #     threshold=0.0,
    # )

    # # 5. Create Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     data_collator=data_collator,
    #     # compute_metrics=eval_compute_metrics_fn,
    # )

    # ---------------------------
    # Stage 1: Train with backbone frozen.
    # ---------------------------
    stage1_epochs = 3
    print("Starting Stage 1 training (backbone frozen)...")
    # Freeze backbone parameters
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # Create new training arguments for stage 1.
    training_args_stage1 = TrainingArguments(
        output_dir=os.path.join(output_dir, "stage1"),
        num_train_epochs=stage1_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs_stage1"),
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=True,
    )

    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # compute_metrics=eval_compute_metrics_fn,
    )

    trainer_stage1.train()

    # ---------------------------
    # Stage 2: Unfreeze backbone and train entire model.
    # ---------------------------
    stage2_epochs = 15
    print("Starting Stage 2 training (unfreezing backbone)...")
    # Unfreeze all parameters.
    for name, param in model.named_parameters():
        param.requires_grad = True

    training_args_stage2 = TrainingArguments(
        output_dir=os.path.join(output_dir, "stage2"),
        num_train_epochs=stage2_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs_stage2"),
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=True,
    )

    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # compute_metrics=eval_compute_metrics_fn,
    )

    trainer_stage2.train()

    # Optionally, evaluate the model after training
    eval_results = trainer_stage2.evaluate()
    print(f"Final Evaluation results: {eval_results}")

    # Optionally, save the trained model (from stage 2)
    trainer_stage2.save_model(os.path.join(output_dir, "best_model"))


if __name__ == "__main__":
    main()
