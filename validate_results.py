import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from dataset_loader import DentalDataset
from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation
from train_segmenation import get_transformation_with_albumentations


def save_segmentation_results(
        model_path="segout2/checkpoint-180",
        validation_dataset="datasets/dental_val/Images",
        output_dir="validation_results2",
        base_model_name="StanfordAIMI/dinov2-base-xray-224",
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = Dinov2ForSemanticSegmentation.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Create validation dataset
    transformations = get_transformation_with_albumentations(image_processor, aug=False)
    val_dataset = DentalDataset(
        path=validation_dataset,
        transform=transformations,
        size=image_processor.size["height"],
    )

    # Process each image
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            # Get sample
            sample = val_dataset[idx]
            image_id = val_dataset.dataset[idx][0]  # Get original image name

            # Get predictions
            print(sample["pixel_values"].shape)
            inputs = {
                "pixel_values": sample["pixel_values"].unsqueeze(0).to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            # Convert to image
            pred_mask = predictions[0, 0].cpu().numpy()  # Take first batch and channel
            true_mask = sample["label"].numpy()

            # Save results
            pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            true_img = Image.fromarray((true_mask * 255).astype(np.uint8))

            # Save original image, prediction and ground truth
            pred_img.save(os.path.join(output_dir, f"{image_id}_pred.png"))
            true_img.save(os.path.join(output_dir, f"{image_id}_true.png"))

            # Create a side-by-side comparison
            comparison = Image.new('RGB', (pred_img.width * 2, pred_img.height))
            comparison.paste(true_img.convert('RGB'), (0, 0))
            comparison.paste(pred_img.convert('RGB'), (pred_img.width, 0))
            comparison.save(os.path.join(output_dir, f"{image_id}_comparison.png"))

            print(f"Processed image {image_id}")


if __name__ == "__main__":
    import fire

    fire.Fire(save_segmentation_results)
