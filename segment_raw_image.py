import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation
import fire


class DentalSegmenter:
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

        # Get model input size
        if "width" in self.processor.size and "height" in self.processor.size:
            self.target_size = (self.processor.size["width"], self.processor.size["height"])
        else:
            self.target_size = (self.processor.size["height"], self.processor.size["height"])

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
        raw_image = Image.open(image_path).convert("RGB")
        original_size = raw_image.size

        # Resize to model's target size
        resized_image = raw_image.resize(self.target_size, resample=Image.BILINEAR)
        np_image = np.array(resized_image)

        # Process through image processor
        inputs = self.processor(images=np_image, return_tensors="pt").pixel_values
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
            predictions = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
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
        pred_mask = Image.fromarray(raw_predictions * 255)
        pred_mask = pred_mask.resize(original_size, resample=Image.NEAREST)
        return pred_mask

    def segment_image(self, image_path: str, output_path: str = None) -> Image.Image:
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
        Image.Image
            Segmentation mask
        """
        # Run the full pipeline
        inputs, original_size = self.preprocess(image_path)
        raw_predictions = self.predict(inputs)
        mask = self.postprocess(raw_predictions, original_size)

        # Save if output path is provided
        if output_path:
            mask.save(output_path)
            print(f"Saved segmentation mask to {output_path}")

        return mask


def main(
        image_path: str,
        output_path: str,
        model_path: str = "segout2/checkpoint-best",
        base_model_name: str = "StanfordAIMI/dinov2-base-xray-224"
):
    """CLI interface for image segmentation"""
    segmenter = DentalSegmenter(model_path, base_model_name)
    segmenter.segment_image(image_path, output_path)


if __name__ == "__main__":
    fire.Fire(main)
