import os

import albumentations as A
import cv2
import fire
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor

from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation


class ToothSegmenter:
    """
    A class for dental tooth segmentation using a fine-tuned Dinov2 model.
    
    This class handles:
    - Loading and maintaining the model in memory
    - Cropping teeth from dental X-rays based on detection boxes
    - Running inference on cropped teeth
    - Projecting segmentation masks back to the original image space
    """

    def __init__(
            self,
            model_path: str = "./models/tooth-segmentator-dinov2",
            base_model_name: str = "StanfordAIMI/dinov2-base-xray-224",
            device: str = None,
            crop_margin: int = 32
    ):
        """
        Initialize the tooth segmenter with a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model checkpoint
        base_model_name : str
            Name of the base model (used for image processor)
        device : str, optional
            Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        crop_margin : int
            Margin in pixels to add around tooth bounding boxes when cropping
        """
        # Set up device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load image processor and model
        print(f"Loading tooth segmentation model from {model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(base_model_name)
        self.model = Dinov2ForSemanticSegmentation.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Save crop margin
        self.crop_margin = crop_margin

        # Create transformation pipeline for inference
        self.transform = A.Compose(
            [
                A.Resize(width=448, height=448),
                A.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
                ToTensorV2(),
            ]
        )

    def crop_teeth(self, image, teeth_boxes, confidence_threshold=0.5):
        """
        Crop tooth regions from a dental X-ray based on detection boxes.
        
        Parameters
        ----------
        image : np.ndarray
            Input image as a NumPy array (BGR or RGB)
        teeth_boxes : list
            List of detected teeth boxes, each in the format [class_id, score, x1, y1, x2, y2]
        confidence_threshold : float
            Minimum confidence threshold for considering a tooth detection
            
        Returns
        -------
        list
            List of dictionaries containing cropped teeth information
            Each dictionary has:
            - 'crop': cropped image
            - 'box': original detection box
            - 'bbox': [x1, y1, x2, y2] with margin
            - 'index': tooth index
            - 'score': detection confidence score
        """
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                # Assuming BGR format from OpenCV
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assuming already in RGB format
                image_rgb = image
        else:
            # Convert grayscale to RGB if needed
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image_rgb.shape[:2]
        cropped_teeth = []

        for i, box in enumerate(teeth_boxes):
            # Parse box information
            class_id, score, x1, y1, x2, y2 = box

            # Skip low confidence detections
            if score < confidence_threshold:
                continue

            # Add margin to bounding box
            x1_margin = max(0, int(x1) - self.crop_margin)
            y1_margin = max(0, int(y1) - self.crop_margin)
            x2_margin = min(w, int(x2) + self.crop_margin)
            y2_margin = min(h, int(y2) + self.crop_margin)

            # Ensure the crop has valid dimensions
            if x2_margin <= x1_margin or y2_margin <= y1_margin:
                print(f"Warning: Invalid crop dimensions for tooth {i + 1}. Skipping.")
                continue

            # Crop the tooth region
            tooth_crop = image_rgb[y1_margin:y2_margin, x1_margin:x2_margin]

            # Store crop and metadata
            cropped_teeth.append({
                'crop': tooth_crop,
                'box': box,
                'bbox': [x1_margin, y1_margin, x2_margin, y2_margin],
                'index': i,
                'score': score
            })

        return cropped_teeth

    def preprocess_crop(self, crop):
        """
        Preprocess a cropped tooth image for model inference.
        
        Parameters
        ----------
        crop : np.ndarray
            Cropped tooth image
            
        Returns
        -------
        torch.Tensor
            Preprocessed image tensor ready for model inference
        """
        # Apply transformations
        transformed = self.transform(image=crop)
        inputs = transformed["image"].unsqueeze(0)  # Add batch dimension
        inputs = inputs.to(self.device)

        return inputs

    def predict_crop(self, preprocessed_crop):
        """
        Run model inference on a preprocessed tooth crop.
        
        Parameters
        ----------
        preprocessed_crop : torch.Tensor
            Preprocessed tooth crop tensor
            
        Returns
        -------
        np.ndarray
            Binary segmentation mask for the tooth
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=preprocessed_crop)
            logits = outputs.logits

            # Apply sigmoid and threshold at 0.5
            probs = torch.sigmoid(logits)

            # Get the mask for the tooth class (index 1)
            if probs.shape[1] > 1:
                mask = probs[0, 1].cpu().numpy()  # Assuming class 1 is tooth
            else:
                mask = probs[0, 0].cpu().numpy()  # Single channel case

            # Convert to binary mask
            binary_mask = (mask > 0.5).astype(np.float32)

        return binary_mask

    def project_mask_to_original(self, binary_mask, crop_info):
        """
        Project a tooth segmentation mask back to the original image space.
        
        Parameters
        ----------
        binary_mask : np.ndarray
            Binary segmentation mask from model prediction
        crop_info : dict
            Crop metadata containing bbox information
            
        Returns
        -------
        tuple
            (x_offset, y_offset, resized_mask) where:
            - x_offset, y_offset: Coordinates in the original image
            - resized_mask: Mask resized to match the crop dimensions
        """
        # Get crop dimensions
        x1, y1, x2, y2 = crop_info['bbox']
        crop_width = x2 - x1
        crop_height = y2 - y1

        # Resize mask to match crop dimensions
        resized_mask = cv2.resize(
            binary_mask,
            (crop_width, crop_height),
            interpolation=cv2.INTER_NEAREST
        )

        return x1, y1, resized_mask

    def segment_teeth(self, image, teeth_boxes, output_dir=None,
                      create_visualization=True, mask_color=[1.0, 0.2, 0.2], mask_alpha=0.5):
        """
        Perform full segmentation pipeline on all detected teeth in an image.
        
        Parameters
        ----------
        image : np.ndarray or str
            Input image as a NumPy array or path to an image file
        teeth_boxes : list
            List of detected teeth boxes, each in the format [class_id, score, x1, y1, x2, y2]
        output_dir : str, optional
            Directory to save individual tooth segmentations and full segmentation mask
        create_visualization : bool
            Whether to create and save visualizations
        mask_color : list
            RGB color for the tooth mask overlay [r, g, b], values from 0 to 1
        mask_alpha : float
            Transparency of the mask overlay (0-1)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'full_mask': Full segmentation mask for all teeth
            - 'teeth_masks': List of individual tooth masks with position information
            - 'original_shape': Original image shape
        """
        # Load image if a path is provided
        if isinstance(image, str):
            loaded_image = cv2.imread(image)
            if loaded_image is None:
                raise ValueError(f"Could not load image from path: {image}")
            # Convert to RGB
            image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
        else:
            # Make a copy to avoid modifying the original
            if image.dtype == np.uint8 and image.shape[2] == 3:
                # Assuming BGR format from OpenCV
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assuming already in RGB format or grayscale
                image_rgb = image.copy()

        # Get image dimensions
        h, w = image_rgb.shape[:2]

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Crop teeth from the image
        cropped_teeth = self.crop_teeth(image_rgb, teeth_boxes)

        if not cropped_teeth:
            print("No valid teeth detected with confidence >= 0.5")
            return {
                'full_mask': np.zeros((h, w), dtype=np.uint8),
                'teeth_masks': [],
                'original_shape': (h, w)
            }

        # Create a full mask for all teeth
        full_mask = np.zeros((h, w), dtype=np.uint8)
        teeth_masks = []

        # Process each cropped tooth
        for tooth_idx, tooth_info in enumerate(cropped_teeth):
            # Preprocess the crop
            preprocessed_crop = self.preprocess_crop(tooth_info['crop'])

            # Run inference
            binary_mask = self.predict_crop(preprocessed_crop)

            # Project mask back to original image space
            x_offset, y_offset, resized_mask = self.project_mask_to_original(binary_mask, tooth_info)

            # Add to the full mask
            crop_height, crop_width = resized_mask.shape
            full_mask[
            y_offset:y_offset + crop_height,
            x_offset:x_offset + crop_width
            ] = np.logical_or(
                full_mask[
                y_offset:y_offset + crop_height,
                x_offset:x_offset + crop_width
                ],
                resized_mask > 0
            )

            # Store individual tooth mask
            teeth_masks.append({
                'mask': resized_mask,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'width': crop_width,
                'height': crop_height,
                'tooth_index': tooth_info['index'],
                'score': tooth_info['score']
            })

            # Save individual tooth results if output_dir is provided
            if output_dir:
                # Save cropped tooth
                crop_path = os.path.join(output_dir, f"tooth_{tooth_idx + 1}_crop.png")
                cv2.imwrite(crop_path, cv2.cvtColor(tooth_info['crop'], cv2.COLOR_RGB2BGR))

                # Save tooth mask
                mask_path = os.path.join(output_dir, f"tooth_{tooth_idx + 1}_mask.png")
                cv2.imwrite(mask_path, (resized_mask * 255).astype(np.uint8))

                # Create and save overlay visualization
                if create_visualization:
                    overlay_img = self.create_tooth_overlay(
                        tooth_info['crop'],
                        binary_mask,
                        alpha=mask_alpha,
                        color=mask_color
                    )
                    overlay_path = os.path.join(output_dir, f"tooth_{tooth_idx + 1}_overlay.png")
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

        # Save full mask if output_dir is provided
        if output_dir:
            full_mask_path = os.path.join(output_dir, "full_teeth_mask.png")
            cv2.imwrite(full_mask_path, (full_mask * 255).astype(np.uint8))

            # Create and save full visualization
            if create_visualization:
                full_overlay = self.create_full_visualization(
                    image_rgb,
                    full_mask,
                    teeth_boxes,
                    alpha=mask_alpha,
                    color=mask_color
                )
                full_overlay_path = os.path.join(output_dir, "full_teeth_segmentation.png")
                cv2.imwrite(full_overlay_path, cv2.cvtColor(full_overlay, cv2.COLOR_RGB2BGR))

        return {
            'full_mask': full_mask,
            'teeth_masks': teeth_masks,
            'original_shape': (h, w)
        }

    def create_tooth_overlay(self, tooth_image, tooth_mask, alpha=0.5, color=[1.0, 0.2, 0.2]):
        """
        Create an overlay visualization of the tooth segmentation.
        
        Parameters
        ----------
        tooth_image : np.ndarray
            Cropped tooth image
        tooth_mask : np.ndarray
            Binary segmentation mask
        alpha : float
            Transparency of the overlay
        color : list
            RGB color for the overlay
            
        Returns
        -------
        np.ndarray
            Overlay visualization image
        """
        # Resize mask to match the input image size if needed
        if tooth_image.shape[:2] != tooth_mask.shape:
            tooth_mask = cv2.resize(
                tooth_mask,
                (tooth_image.shape[1], tooth_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Create a colored mask
        colored_mask = np.zeros((*tooth_mask.shape, 3), dtype=np.float32)
        colored_mask[tooth_mask > 0] = color

        # Create the overlay
        overlay = tooth_image.astype(np.float32) / 255.0
        overlay = overlay * (1 - alpha * tooth_mask[:, :, np.newaxis]) + colored_mask * alpha * tooth_mask[:, :,
                                                                                                np.newaxis]
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        return overlay

    def create_full_visualization(self, image, full_mask, teeth_boxes, alpha=0.5, color=[1.0, 0.2, 0.2]):
        """
        Create a full visualization with teeth detection boxes and segmentation mask.
        
        Parameters
        ----------
        image : np.ndarray
            Original image
        full_mask : np.ndarray
            Full segmentation mask for all teeth
        teeth_boxes : list
            List of detected teeth boxes
        alpha : float
            Transparency of the overlay
        color : list
            RGB color for the overlay
            
        Returns
        -------
        np.ndarray
            Visualization image with detection boxes and segmentation mask
        """
        # Make a copy of the input image
        vis_image = image.copy()

        # Create the segmentation overlay
        colored_mask = np.zeros((*full_mask.shape, 3), dtype=np.float32)
        colored_mask[full_mask > 0] = color

        # Apply the mask overlay
        vis_image = vis_image.astype(np.float32) / 255.0
        vis_image = vis_image * (1 - alpha * full_mask[:, :, np.newaxis]) + colored_mask * alpha * full_mask[:, :,
                                                                                                   np.newaxis]
        vis_image = np.clip(vis_image * 255, 0, 255).astype(np.uint8)

        # Draw detection boxes
        for box in teeth_boxes:
            class_id, score, x1, y1, x2, y2 = box
            if score >= 0.5:  # Only show confident detections
                # Draw rectangle
                cv2.rectangle(
                    vis_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),  # Green color for boxes
                    2
                )

                # Display confidence score
                cv2.putText(
                    vis_image,
                    f"{score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        return vis_image


def main(
        image_path: str = "./datasets/tooth_ds_val/ds/img/574.jpg",
        output_dir: str = "./tooth_segmentation_results",
        model_path: str = "./models/tooth-segmentator-dinov2",
        base_model_name: str = "StanfordAIMI/dinov2-base-xray-224",
        teeth_json: str = None,  # Path to a JSON file with teeth detections
        confidence_threshold: float = 0.5,
        crop_margin: int = 32
):
    """CLI interface for tooth segmentation"""
    from teeth_detector_paddle.teeth_detector_paddle import TeethDetector
    import json

    # Initialize the tooth segmenter
    segmenter = ToothSegmenter(
        model_path=model_path,
        base_model_name=base_model_name,
        crop_margin=crop_margin
    )

    # Get teeth detections
    if teeth_json and os.path.exists(teeth_json):
        # Load teeth detections from JSON file
        with open(teeth_json, 'r') as f:
            detection_results = json.load(f)
        teeth_boxes = detection_results.get('boxes', [])
    else:
        # Detect teeth using the detector
        print("Detecting teeth in the image...")
        detector = TeethDetector(threshold=confidence_threshold)
        detection_results = detector.predict_from_file(image_path, visual=False, save_results=False)
        teeth_boxes = detection_results['boxes']

    # Perform tooth segmentation
    segmentation_results = segmenter.segment_teeth(
        image_path,
        teeth_boxes,
        output_dir=output_dir,
        create_visualization=True
    )

    print(f"Segmentation completed. Results saved to {output_dir}")
    print(f"Found and processed {len(segmentation_results['teeth_masks'])} teeth")


if __name__ == "__main__":
    fire.Fire(main)
