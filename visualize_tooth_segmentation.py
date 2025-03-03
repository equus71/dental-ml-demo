import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import fire
from tqdm import tqdm

from transformers import AutoImageProcessor
from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation
from tooth_segmentation_dataset_loader import ToothSegmentationDataset
from segmentation_utils import get_transformation_with_albumentations


def visualize_segmentation(
        model_path,
        validation_dataset="./datasets/tooth_ds_val",
        output_folder="./segmentation_results",
        base_model_name="StanfordAIMI/dinov2-base-xray-224",
        batch_size=4,
        num_samples=None,
        mask_alpha=0.5,
        mask_color=[1.0, 0.2, 0.2],  # Red color for the mask
        device=None,
):
    """
    Visualize tooth segmentation results by overlaying colorful semi-transparent masks on input images.
    
    Args:
        model_path: Path to the trained model
        validation_dataset: Path to the validation dataset
        output_folder: Folder to save visualization results
        base_model_name: Base model name used for training
        batch_size: Batch size for inference
        num_samples: Number of samples to visualize (None for all)
        mask_alpha: Transparency of the mask overlay (0-1)
        mask_color: RGB color for the mask overlay
        device: Device to run inference on (None for auto-detection)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load model and image processor
    print(f"Loading model from {model_path}...")
    image_processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = Dinov2ForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load validation dataset
    print(f"Loading validation dataset from {validation_dataset}...")
    transformations = get_transformation_with_albumentations(
        image_processor=image_processor, aug=False
    )
    val_dataset = ToothSegmentationDataset(
        path=validation_dataset,
        transform=transformations,
        padding=32,
    )

    # Limit number of samples if specified
    if num_samples is not None:
        num_samples = min(num_samples, len(val_dataset))
        indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    else:
        indices = range(len(val_dataset))
        num_samples = len(val_dataset)

    print(f"Visualizing {num_samples} samples...")

    # Process each sample
    for i, idx in enumerate(tqdm(indices)):
        sample = val_dataset[idx]

        # Get the original image path from the dataset
        original_image_path = val_dataset.samples[idx]["image_path"]
        image_filename = os.path.basename(original_image_path)

        # Get the input image and ground truth mask
        input_image = sample["pixel_values"].unsqueeze(0).to(device)  # Add batch dimension
        gt_mask = sample["label"]

        # Run inference
        with torch.no_grad():
            outputs = model(input_image)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Get the predicted mask (class 1 for tooth)
            pred_mask = probs[0, 1].cpu().numpy()

            # Convert to binary mask with threshold 0.5
            binary_pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Convert input tensor back to numpy for visualization
        if isinstance(input_image, torch.Tensor):
            # If using PyTorch tensors, denormalize and convert to numpy
            img = input_image[0].cpu().permute(1, 2, 0).numpy()

            # Denormalize if needed (adjust based on your normalization)
            img = (img * image_processor.image_std) + image_processor.image_mean
            img = np.clip(img, 0, 1)
        else:
            img = input_image[0]

        # Create visualization with mask overlay
        plt.figure(figsize=(12, 6))

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        # Plot ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        mask_overlay = np.zeros((*gt_mask.shape, 4))
        mask_overlay[gt_mask == 1] = [*mask_color, mask_alpha]
        plt.imshow(mask_overlay, alpha=mask_alpha)
        plt.title("Ground Truth")
        plt.axis("off")

        # Plot predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        pred_overlay = np.zeros((*binary_pred_mask.shape, 4))
        pred_overlay[binary_pred_mask == 1] = [*mask_color, mask_alpha]
        plt.imshow(pred_overlay, alpha=mask_alpha)
        plt.title("Prediction")
        plt.axis("off")

        # Save the visualization
        output_path = os.path.join(output_folder, f"{i}_{image_filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    print(f"Visualizations saved to {output_folder}")


if __name__ == "__main__":
    fire.Fire(visualize_segmentation)
