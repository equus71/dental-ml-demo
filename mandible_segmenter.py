import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor
from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation
import fire


class MandibleSegmenter:
    """
    A class for dental X-ray segmentation using a fine-tuned Dinov2 model.
    
    This class handles:
    - Loading and maintaining the model in memory
    - Image preprocessing
    - Model inference
    - Postprocessing of segmentation masks
    """

    def __init__(
            self,
            model_path: str = "segout2/checkpoint-best",
            base_model_name: str = "StanfordAIMI/dinov2-base-xray-224",
            device: str = None
    ):
        """
        Initialize the segmenter with a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model checkpoint
        base_model_name : str
            Name of the base model (used for image processor)
        device : str, optional
            Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        """
        # Set up device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load image processor and model
        self.processor = AutoImageProcessor.from_pretrained(base_model_name)
        self.model = Dinov2ForSemanticSegmentation.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create transformation pipeline matching validation
        self.transform = A.Compose(
            [
                A.Resize(width=448, height=448),
                A.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
                ToTensorV2(),
            ]
        )

    def preprocess(self, image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Preprocess an input image for segmentation.
        
        Parameters
        ----------
        image_path : str
            Path to the input image
            
        Returns
        -------
        tuple[torch.Tensor, tuple[int, int]]
            Preprocessed image tensor and original image size
        """
        # Load and convert to RGB
        image = np.array(Image.open(image_path).convert("RGB"))
        original_size = (image.shape[1], image.shape[0])  # width, height

        # Apply transformations
        transformed = self.transform(image=image)
        inputs = transformed["image"].unsqueeze(0)  # Add batch dimension
        inputs = inputs.to(self.device)

        return inputs, original_size

    def predict(self, preprocessed_image: torch.Tensor) -> np.ndarray:
        """
        Run model inference on preprocessed image.
        
        Parameters
        ----------
        preprocessed_image : torch.Tensor
            Preprocessed input image tensor
            
        Returns
        -------
        np.ndarray
            Raw prediction mask
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=preprocessed_image)
            logits = outputs.logits
            # Apply sigmoid and threshold at 0.5 to match validation
            probs = torch.sigmoid(logits)
            predictions = (probs < 0.5).float()
            predictions = predictions[0, 0].cpu().numpy()  # Take first batch and channel
        return predictions

    def postprocess(self,
                    raw_predictions: np.ndarray,
                    original_size: tuple[int, int]
                    ) -> Image.Image:
        """
        Postprocess raw predictions into final segmentation mask.
        
        Parameters
        ----------
        raw_predictions : np.ndarray
            Raw prediction mask from model
        original_size : tuple[int, int]
            Original image size to scale mask to
            
        Returns
        -------
        Image.Image
            Final processed segmentation mask
        """
        # Convert to PIL Image and resize to original size
        pred_mask = Image.fromarray((raw_predictions * 255).astype(np.uint8))
        pred_mask = pred_mask.resize(original_size, resample=Image.NEAREST)
        return pred_mask

    def segment_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Perform full segmentation pipeline on an image.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        output_path : str, optional
            Path to save the segmentation mask. If None, mask is only returned
            
        Returns
        -------
        np.ndarray
            Segmentation mask as a numpy array
        """
        # Run the full pipeline
        inputs, original_size = self.preprocess(image_path)
        raw_predictions = self.predict(inputs)
        mask = self.postprocess(raw_predictions, original_size)

        # Save if output path is provided
        if output_path:
            mask.save(output_path)
            print(f"Saved segmentation mask to {output_path}")

        return np.array(mask)


def main(
        image_path: str,
        output_path: str,
        model_path: str = "segout2/checkpoint-180",
        base_model_name: str = "StanfordAIMI/dinov2-base-xray-224"
):
    """CLI interface for image segmentation"""
    segmenter = MandibleSegmenter(model_path, base_model_name)
    segmenter.segment_image(image_path, output_path)


if __name__ == "__main__":
    fire.Fire(main)
