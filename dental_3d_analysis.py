#!/usr/bin/env python3
# dental_3d_analysis.py

import json
import os
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import fire

from dental_3d_renderer import Dental3DRenderer
from dental_image_analysis import DentalImageAnalyzer
from image_utils import create_blended_visualization, enhanced_image_diff
from mandible_segmenter import MandibleSegmenter
from tooth_line_extractor import extract_tooth_line_from_mask, visualize_tooth_line
from tooth_segmenter import ToothSegmenter


class DentalPairAnalysis:
    """
    A class for analyzing and visualizing a pair of dental X-ray images.

    This class encapsulates the complete workflow for dental X-ray analysis,
    including teeth detection, segmentation, alignment, and visualization.

    Parameters
    ----------
    img1_path : str, optional
        Path to the first image
    img2_path : str, optional
        Path to the second image
    output_dir : str, optional
        Directory to save results, by default "./pair_analysis_results"
    tooth_model_path : str, optional
        Path to the tooth segmentation model, by default "./models/tooth-segmentator-dinov2"
    mandible_model_path : str, optional
        Path to the mandible segmentation model, by default "./models/mandible-segmentator-dinov2"
    base_model_name : str, optional
        Base model name for the segmentation models, by default "StanfordAIMI/dinov2-base-xray-224"
    tooth_crop_margin : int, optional
        Margin to add around tooth bounding boxes when cropping, by default 32
    tooth_line_offset : int, optional
        Maximum vertical distance to consider for tooth line, by default 100
    z_scale : float, optional
        Scale factor for z-axis in 3D visualization, by default 1.0
    curvature_scale : float, optional
        Scale factor for curvature in 3D visualization, by default 4.0
    curvature_factor : float, optional
        Factor affecting the curvature shape in 3D visualization, by default 12.0

    Notes
    -----
    The analysis workflow includes:
    - Teeth detection and matching
    - Affine transformation estimation
    - Mandible segmentation
    - Tooth segmentation
    - Tooth line extraction
    - Various visualizations of results
    - 3D visualization using VTK
    """

    def __init__(
            self,
            img1_path=None,
            img2_path=None,
            output_dir="./pair_analysis_results",
            tooth_model_path="./models/tooth-segmentator-dinov2",
            mandible_model_path="./models/mandible-segmentator-dinov2",
            base_model_name="StanfordAIMI/dinov2-base-xray-224",
            tooth_crop_margin=32,
            tooth_line_offset=100,
            z_scale=1.0,
            curvature_scale=4.0,
            curvature_factor=12.0
    ):
        """
        Initialize the dental pair analysis.
    
    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        output_dir: Directory to save results
        tooth_model_path: Path to the tooth segmentation model
        mandible_model_path: Path to the mandible segmentation model
        base_model_name: Base model name for the segmentation models
        tooth_crop_margin: Margin to add around tooth bounding boxes when cropping
        tooth_line_offset: Maximum vertical distance to consider for tooth line
        z_scale: Scale factor for z-axis in 3D visualization
        curvature_scale: Scale factor for curvature in 3D visualization
        curvature_factor: Factor affecting the curvature shape in 3D visualization
        """
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.output_dir = output_dir
        self.tooth_model_path = tooth_model_path
        self.mandible_model_path = mandible_model_path
        self.base_model_name = base_model_name
        self.tooth_crop_margin = tooth_crop_margin
        self.tooth_line_offset = tooth_line_offset

        # 3D visualization parameters
        self.z_scale = z_scale
        self.curvature_scale = curvature_scale
        self.curvature_factor = curvature_factor

        # Results storage
        self.results = None
        self.img1 = None
        self.img2 = None
        self.img1_rgb = None
        self.img2_rgb = None
        self.warped_img1 = None
        self.affine_matrix = None

        # Segmentation results
        self.img1_teeth_results = None
        self.img2_teeth_results = None
        self.img1_mandible_mask = None
        self.img2_mandible_mask = None

        # Bottom teeth masks
        self.img1_bottom_teeth_mask = None
        self.img2_bottom_teeth_mask = None
        self.img1_bottom_teeth_indices = None
        self.img2_bottom_teeth_indices = None

        # Tooth line results
        self.img1_tooth_line_results = None
        self.img2_tooth_line_results = None

    def setup_output_directories(self):
        """
        Create the output directory structure.

        Creates directories for:
        - Segmentation results (mandible and teeth)
        - Individual teeth images
        - Tooth line analysis
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Create segmentation directories
        self.segmentation_dir = os.path.join(self.output_dir, "segmentation")
        os.makedirs(self.segmentation_dir, exist_ok=True)

        # Create mandible directory
        self.mandible_dir = os.path.join(self.segmentation_dir, "mandible")
        os.makedirs(self.mandible_dir, exist_ok=True)

        # Create teeth directory
        self.teeth_dir = os.path.join(self.segmentation_dir, "teeth")
        os.makedirs(self.teeth_dir, exist_ok=True)

        # Create directories for individual image teeth
        self.img1_teeth_dir = os.path.join(self.teeth_dir, "image1_teeth")
        os.makedirs(self.img1_teeth_dir, exist_ok=True)

        self.img2_teeth_dir = os.path.join(self.teeth_dir, "image2_teeth")
        os.makedirs(self.img2_teeth_dir, exist_ok=True)

        # Create tooth line directory
        self.tooth_line_dir = os.path.join(self.output_dir, "tooth_line")
        os.makedirs(self.tooth_line_dir, exist_ok=True)

    def load_images(self):
        """
        Load the input images and convert to RGB.

        Raises
        ------
        ValueError
            If image paths are not specified or images cannot be loaded
        """
        if not self.img1_path or not self.img2_path:
            raise ValueError("Image paths must be specified")

        self.img1 = cv2.imread(self.img1_path)
        self.img2 = cv2.imread(self.img2_path)

        if self.img1 is None:
            raise ValueError(f"Could not load image from {self.img1_path}")
        if self.img2 is None:
            raise ValueError(f"Could not load image from {self.img2_path}")

        # Convert to RGB for visualization
        self.img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

    def analyze_image_pair(self):
        """
        Perform analysis on the image pair using DentalImageAnalyzer.
        This includes teeth detection, feature matching, and affine transformation.
        """
        print(f"Analyzing image pair: {os.path.basename(self.img1_path)} and {os.path.basename(self.img2_path)}")

        # Initialize the dental image analyzer
        analyzer = DentalImageAnalyzer(output_dir=self.output_dir)

        # Process the image pair
        self.results = analyzer.process_image_pair(self.img1_path, self.img2_path)

        # Check if analysis was successful
        if 'error' in self.results:
            print(f"Error during analysis: {self.results['error']}")
            return False

        # Print analysis results
        print("\nAnalysis Results:")
        print(f"Number of teeth detected in image 1: {self.results['num_teeth_img1']}")
        print(f"Number of teeth detected in image 2: {self.results['num_teeth_img2']}")
        print(f"Number of tooth pairs identified: {self.results['num_tooth_pairs']}")
        print(f"Total matches within teeth regions: {self.results['combined_teeth_matches']}")

        # Check if affine transformation was calculated
        if self.results['affine_matrix'] is None:
            print("Could not calculate affine transformation - not enough matching points.")
            return False

        # Print transformation error metrics
        print("\nTransformation Error Metrics:")
        print(f"Mean error: {self.results['affine_mean_error']:.2f} pixels")
        print(f"Median error: {self.results['affine_median_error']:.2f} pixels")
        print(f"Min error: {self.results['affine_min_error']:.2f} pixels")
        print(f"Max error: {self.results['affine_max_error']:.2f} pixels")
        print(f"Standard deviation: {self.results['affine_std_error']:.2f} pixels")

        # Store the affine matrix
        self.affine_matrix = np.array(self.results['affine_matrix'])

        # Apply the affine transformation to align img1 with img2
        h, w = self.img2.shape[:2]
        self.warped_img1 = cv2.warpAffine(self.img1_rgb, self.affine_matrix, (w, h))

        return True

    def visualize_teeth_detection(self):
        """Create visualization of teeth detection in both images."""
        plt.figure(figsize=(12, 6))

        # Image 1 with teeth boxes
        plt.subplot(1, 2, 1)
        plt.imshow(self.img1_rgb)
        plt.title(f"Image 1: {os.path.basename(self.img1_path)}\n{self.results['num_teeth_img1']} teeth detected")

        # Draw teeth boxes on image 1
        for box in self.results['teeth_img1']:
            class_id, score, x1, y1, x2, y2 = box
            if score >= 0.5:  # Only show confident detections
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                  fill=False, edgecolor='lime', linewidth=2))
                plt.text(x1, y1 - 5, f"{score:.2f}", color='lime',
                         backgroundcolor='black', fontsize=8)

        plt.axis('off')

        # Image 2 with teeth boxes
        plt.subplot(1, 2, 2)
        plt.imshow(self.img2_rgb)
        plt.title(f"Image 2: {os.path.basename(self.img2_path)}\n{self.results['num_teeth_img2']} teeth detected")

        # Draw teeth boxes on image 2
        for box in self.results['teeth_img2']:
            class_id, score, x1, y1, x2, y2 = box
            if score >= 0.5:  # Only show confident detections
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                  fill=False, edgecolor='lime', linewidth=2))
                plt.text(x1, y1 - 5, f"{score:.2f}", color='lime',
                         backgroundcolor='black', fontsize=8)

        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "teeth_detection.png"), dpi=300)
        plt.close()

    def create_overlay_visualizations(self):
        """Create and save overlay visualizations of the images."""
        # Create blended visualization (50% opacity)
        blended_img = create_blended_visualization(self.warped_img1, self.img2_rgb, alpha=0.5)

        # Create checkerboard visualization
        checkerboard_img = self.create_checkerboard_visualization(self.warped_img1, self.img2_rgb, size=50)

        # Save blended visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(blended_img)
        plt.title("Blended Overlay (Warped Image 1 + Image 2)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "blended_overlay.png"), dpi=300)
        plt.close()

        # Save checkerboard visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(checkerboard_img)
        plt.title("Checkerboard Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "checkerboard_overlay.png"), dpi=300)
        plt.close()

        # Create a comprehensive visualization with all images
        plt.figure(figsize=(20, 15))

        plt.subplot(2, 2, 1)
        plt.imshow(self.img1_rgb)
        plt.title("Original Image 1")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(self.img2_rgb)
        plt.title("Original Image 2 (Target)")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(self.warped_img1)
        plt.title("Warped Image 1 (Aligned to Image 2)")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(blended_img)
        plt.title("Blended Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "alignment_comparison.png"), dpi=300)
        plt.close()

    def create_checkerboard_visualization(self, img1, img2, size=50):
        """
        Create a checkerboard visualization of two images.

        Parameters
        ----------
        img1 : np.ndarray
            First image
        img2 : np.ndarray
            Second image
        size : int, optional
            Size of checkerboard squares in pixels, by default 50

        Returns
        -------
        np.ndarray
            Checkerboard visualization image

        Notes
        -----
        The checkerboard pattern alternates between showing regions from img1 and img2,
        making it easier to visually compare alignment between the images.
        """
        # Ensure images have the same shape
        assert img1.shape == img2.shape, "Images must have the same dimensions"

        h, w = img1.shape[:2]
        result = np.zeros_like(img1)

        # Create checkerboard pattern
        for i in range(0, h, size):
            for j in range(0, w, size):
                # Determine if this is a "white" or "black" square
                if ((i // size) + (j // size)) % 2 == 0:
                    # Use img1 for this square
                    result[i:min(i + size, h), j:min(j + size, w)] = img1[i:min(i + size, h), j:min(j + size, w)]
                else:
                    # Use img2 for this square
                    result[i:min(i + size, h), j:min(j + size, w)] = img2[i:min(i + size, h), j:min(j + size, w)]

        return result

    def create_difference_visualization(self):
        """Create and save difference visualization between warped image and target image."""
        # Convert images to grayscale for difference calculation
        warped_img1_gray = cv2.cvtColor(self.warped_img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        # Create enhanced difference visualization
        diff_img = enhanced_image_diff(warped_img1_gray, img2_gray)

        # Save difference visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(diff_img)
        plt.title("Enhanced Difference Visualization")
        plt.colorbar(label="Difference Intensity")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "difference_visualization.png"), dpi=300)
        plt.close()

    def save_transformation_matrix(self):
        """Save the affine transformation matrix to a text file."""
        matrix_path = os.path.join(self.output_dir, "affine_transformation_matrix.txt")
        with open(matrix_path, 'w') as f:
            f.write("Affine Transformation Matrix (2x3):\n")
            f.write(f"{self.affine_matrix[0, 0]:.6f} {self.affine_matrix[0, 1]:.6f} {self.affine_matrix[0, 2]:.6f}\n")
            f.write(f"{self.affine_matrix[1, 0]:.6f} {self.affine_matrix[1, 1]:.6f} {self.affine_matrix[1, 2]:.6f}\n")

            # Add interpretation
            f.write("\nMatrix Interpretation:\n")
            scale_x = np.sqrt(self.affine_matrix[0, 0] ** 2 + self.affine_matrix[1, 0] ** 2)
            scale_y = np.sqrt(self.affine_matrix[0, 1] ** 2 + self.affine_matrix[1, 1] ** 2)
            rotation = np.arctan2(self.affine_matrix[1, 0], self.affine_matrix[0, 0]) * 180 / np.pi
            translation_x = self.affine_matrix[0, 2]
            translation_y = self.affine_matrix[1, 2]

            f.write(f"Scale X: {scale_x:.4f}\n")
            f.write(f"Scale Y: {scale_y:.4f}\n")
            f.write(f"Rotation: {rotation:.2f} degrees\n")
            f.write(f"Translation X: {translation_x:.2f} pixels\n")
            f.write(f"Translation Y: {translation_y:.2f} pixels\n")

    def save_analysis_results(self):
        """Save the analysis results to a JSON file with type information for non-serializable objects."""

        def process_value(value):
            """
            Process a value to make it JSON serializable and add type information.

            Parameters
            ----------
            value : any
                The value to process

            Returns
            -------
            dict or any
                A JSON-serializable representation of the value with type information

            Notes
            -----
            Recursively processes dictionaries and lists. Special handling is provided for:
            - NumPy arrays
            - NumPy scalar types
            - Lists and tuples
            - Dictionaries
            - Basic Python types (int, float, str, bool)
            """
            if value is None:
                return {"__type__": "None", "value": None}

            # Handle basic types that are JSON serializable
            if isinstance(value, (int, float, str, bool)):
                return value

            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                return {
                    "__type__": "ndarray",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "value": value.tolist()
                }

            # Handle numpy scalar types
            if np.isscalar(value) and isinstance(value, np.generic):
                return {
                    "__type__": str(type(value).__name__),
                    "value": value.item()
                }

            # Handle lists and tuples
            if isinstance(value, (list, tuple)):
                processed_items = [process_value(item) for item in value]
                return {
                    "__type__": "list" if isinstance(value, list) else "tuple",
                    "value": processed_items
                }

            # Handle dictionaries recursively
            if isinstance(value, dict):
                processed_dict = {}
                for k, v in value.items():
                    processed_dict[k] = process_value(v)
                return processed_dict

            # For other types, convert to string representation
            return {
                "__type__": str(type(value).__name__),
                "value": str(value)
            }

        # Process the entire results dictionary
        processed_results = process_value(self.results)

        # Add metadata
        processed_results["__metadata__"] = {
            "creation_time": str(np.datetime64('now')),
            "description": "Dental image analysis results with type information"
        }

        # Save to JSON file
        json_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(json_path, 'w') as f:
            json.dump(processed_results, f, indent=4)

        print(f"Analysis results saved to {json_path}")

    def perform_mandible_segmentation(self):
        """Perform mandible segmentation on both input images."""
        print("\nPerforming mandible segmentation...")

        # Initialize mandible segmenter
        segmenter = MandibleSegmenter(model_path=self.mandible_model_path, base_model_name=self.base_model_name)

        # Process first image
        img1_basename = os.path.basename(self.img1_path)
        img1_mask_path = os.path.join(self.mandible_dir, f"mandible_mask_{img1_basename}")
        self.img1_mandible_mask = segmenter.segment_image(self.img1_path, img1_mask_path)

        # Process second image
        img2_basename = os.path.basename(self.img2_path)
        img2_mask_path = os.path.join(self.mandible_dir, f"mandible_mask_{img2_basename}")
        self.img2_mandible_mask = segmenter.segment_image(self.img2_path, img2_mask_path)

        # Create visualization of mandible segmentation
        self.visualize_mandible_segmentation()

        print(f"Mandible segmentation completed. Results saved to {self.mandible_dir}")

    def visualize_mandible_segmentation(self):
        """Create visualization of mandible segmentation for both images."""
        # Create figure for mandible visualization
        plt.figure(figsize=(12, 6))

        # Image 1 with mandible mask
        plt.subplot(1, 2, 1)
        plt.imshow(self.img1_rgb)
        plt.imshow(self.img1_mandible_mask, alpha=0.5, cmap='cool')
        plt.title(f"Image 1: {os.path.basename(self.img1_path)}\nMandible Segmentation")
        plt.axis('off')

        # Image 2 with mandible mask
        plt.subplot(1, 2, 2)
        plt.imshow(self.img2_rgb)
        plt.imshow(self.img2_mandible_mask, alpha=0.5, cmap='cool')
        plt.title(f"Image 2: {os.path.basename(self.img2_path)}\nMandible Segmentation")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.mandible_dir, "mandible_segmentation.png"), dpi=300)
        plt.close()

    def perform_tooth_segmentation(self):
        """Perform tooth segmentation on teeth from both images using the ToothSegmenter."""
        print("\nPerforming tooth segmentation...")

        # Initialize the tooth segmenter
        segmenter = ToothSegmenter(
            model_path=self.tooth_model_path,
            base_model_name=self.base_model_name,
            crop_margin=self.tooth_crop_margin
        )

        # Process teeth in first image
        print(f"Processing teeth in image 1...")
        self.img1_teeth_results = segmenter.segment_teeth(
            self.img1_path,
            self.results['teeth_img1'],
            output_dir=self.img1_teeth_dir,
            create_visualization=True
        )

        # Process teeth in second image
        print(f"Processing teeth in image 2...")
        self.img2_teeth_results = segmenter.segment_teeth(
            self.img2_path,
            self.results['teeth_img2'],
            output_dir=self.img2_teeth_dir,
            create_visualization=True
        )

        # Create a combined visualization showing teeth segmentation in both images
        self.create_combined_teeth_visualization()

        print(f"Tooth segmentation completed. Results saved to {self.teeth_dir}")
        print(f"Found and processed {len(self.img1_teeth_results['teeth_masks'])} teeth in image 1")
        print(f"Found and processed {len(self.img2_teeth_results['teeth_masks'])} teeth in image 2")

    def create_combined_teeth_visualization(self):
        """Create a combined visualization of tooth segmentation for both images."""
        # Create figure for combined tooth visualization
        plt.figure(figsize=(12, 6))

        # Get the full masks from segmentation results
        mask1 = self.img1_teeth_results['full_mask']
        mask2 = self.img2_teeth_results['full_mask']

        # Image 1 with teeth mask
        plt.subplot(1, 2, 1)
        plt.imshow(self.img1_rgb)

        # Create colored overlay for teeth mask
        colored_mask1 = np.zeros((*mask1.shape, 4))  # RGBA
        colored_mask1[mask1 > 0] = [1.0, 0.2, 0.2, 0.5]  # Red with 50% transparency

        plt.imshow(colored_mask1)
        plt.title(f"Image 1: {os.path.basename(self.img1_path)}\nTooth Segmentation")
        plt.axis('off')

        # Image 2 with teeth mask
        plt.subplot(1, 2, 2)
        plt.imshow(self.img2_rgb)

        # Create colored overlay for teeth mask
        colored_mask2 = np.zeros((*mask2.shape, 4))  # RGBA
        colored_mask2[mask2 > 0] = [1.0, 0.2, 0.2, 0.5]  # Red with 50% transparency

        plt.imshow(colored_mask2)
        plt.title(f"Image 2: {os.path.basename(self.img2_path)}\nTooth Segmentation")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.teeth_dir, "combined_teeth_segmentation.png"), dpi=300)
        plt.close()

    def perform_tooth_line_extraction(self):
        """Extract tooth lines from mandible segmentation masks for both images."""
        print("\nPerforming tooth line extraction...")

        # Ensure mandible segmentation has been performed
        if self.img1_mandible_mask is None or self.img2_mandible_mask is None:
            print("Mandible segmentation must be performed before tooth line extraction.")
            return False

        # Extract tooth line from first image
        img1_basename = os.path.basename(self.img1_path)
        try:
            print(f"Extracting tooth line from image 1...")
            self.img1_tooth_line_results = {}
            (
                self.img1_tooth_line_results['points'],
                self.img1_tooth_line_results['quad_func'],
                self.img1_tooth_line_results['coeffs'],
                self.img1_tooth_line_results['center_x'],
                self.img1_tooth_line_results['min_x'],
                self.img1_tooth_line_results['max_x']
            ) = extract_tooth_line_from_mask(
                self.img1_mandible_mask,
                self.tooth_line_offset
            )

            # Save tooth line data for image 1
            np.savez(
                os.path.join(self.tooth_line_dir, f"img1_{img1_basename}_tooth_line.npz"),
                points=self.img1_tooth_line_results['points'],
                coefficients=self.img1_tooth_line_results['coeffs'],
                center_x=self.img1_tooth_line_results['center_x'],
                min_x=self.img1_tooth_line_results['min_x'],
                max_x=self.img1_tooth_line_results['max_x']
            )

            # Create visualization for image 1
            vis_path = os.path.join(self.tooth_line_dir, f"img1_{img1_basename}_tooth_line.png")
            visualize_tooth_line(
                self.img1_rgb,
                self.img1_mandible_mask,
                self.img1_tooth_line_results['points'],
                self.img1_tooth_line_results['quad_func'],
                vis_path
            )

        except Exception as e:
            print(f"Error extracting tooth line from image 1: {str(e)}")
            traceback.print_exc()

        # Extract tooth line from second image
        img2_basename = os.path.basename(self.img2_path)
        try:
            print(f"Extracting tooth line from image 2...")
            self.img2_tooth_line_results = {}
            (
                self.img2_tooth_line_results['points'],
                self.img2_tooth_line_results['quad_func'],
                self.img2_tooth_line_results['coeffs'],
                self.img2_tooth_line_results['center_x'],
                self.img2_tooth_line_results['min_x'],
                self.img2_tooth_line_results['max_x']
            ) = extract_tooth_line_from_mask(
                self.img2_mandible_mask,
                self.tooth_line_offset
            )

            # Save tooth line data for image 2
            np.savez(
                os.path.join(self.tooth_line_dir, f"img2_{img2_basename}_tooth_line.npz"),
                points=self.img2_tooth_line_results['points'],
                coefficients=self.img2_tooth_line_results['coeffs'],
                center_x=self.img2_tooth_line_results['center_x'],
                min_x=self.img2_tooth_line_results['min_x'],
                max_x=self.img2_tooth_line_results['max_x']
            )

            # Create visualization for image 2
            vis_path = os.path.join(self.tooth_line_dir, f"img2_{img2_basename}_tooth_line.png")
            visualize_tooth_line(
                self.img2_rgb,
                self.img2_mandible_mask,
                self.img2_tooth_line_results['points'],
                self.img2_tooth_line_results['quad_func'],
                vis_path
            )

        except Exception as e:
            print(f"Error extracting tooth line from image 2: {str(e)}")
            traceback.print_exc()

        # Create a combined visualization showing tooth lines in both images
        self.create_combined_tooth_line_visualization()

        print(f"Tooth line extraction completed. Results saved to {self.tooth_line_dir}")
        return True

    def create_combined_tooth_line_visualization(self):
        """Create a combined visualization of tooth lines for both images."""
        # Skip if tooth line extraction failed for either image
        if self.img1_tooth_line_results is None or self.img2_tooth_line_results is None:
            print("Cannot create combined visualization - tooth line extraction failed for one or both images.")
            return

        # Create figure for combined tooth line visualization
        plt.figure(figsize=(12, 6))

        # Image 1 with tooth line
        plt.subplot(1, 2, 1)
        plt.imshow(self.img1_rgb)

        # Plot tooth line points and curve for image 1
        tooth_line_x = [p[0] for p in self.img1_tooth_line_results['points']]
        tooth_line_y = [p[1] for p in self.img1_tooth_line_results['points']]
        plt.scatter(tooth_line_x, tooth_line_y, color='green', s=5, label='Extracted Points')

        x_range = np.arange(self.img1_tooth_line_results['min_x'], self.img1_tooth_line_results['max_x'] + 1)
        y_range = self.img1_tooth_line_results['quad_func'](x_range)
        plt.plot(x_range, y_range, color='red', linewidth=2, label='Fitted Curve')

        plt.title(f"Image 1: {os.path.basename(self.img1_path)}\nTooth Line")
        plt.legend()
        plt.axis('off')

        # Image 2 with tooth line
        plt.subplot(1, 2, 2)
        plt.imshow(self.img2_rgb)

        # Plot tooth line points and curve for image 2
        tooth_line_x = [p[0] for p in self.img2_tooth_line_results['points']]
        tooth_line_y = [p[1] for p in self.img2_tooth_line_results['points']]
        plt.scatter(tooth_line_x, tooth_line_y, color='green', s=5, label='Extracted Points')

        x_range = np.arange(self.img2_tooth_line_results['min_x'], self.img2_tooth_line_results['max_x'] + 1)
        y_range = self.img2_tooth_line_results['quad_func'](x_range)
        plt.plot(x_range, y_range, color='red', linewidth=2, label='Fitted Curve')

        plt.title(f"Image 2: {os.path.basename(self.img2_path)}\nTooth Line")
        plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.tooth_line_dir, "combined_tooth_line_visualization.png"), dpi=300)
        plt.close()

        # Create a visualization comparing the tooth line coefficients
        self.create_tooth_line_comparison_visualization()

    def create_tooth_line_comparison_visualization(self):
        """Create a visualization comparing the tooth line curves from both images."""
        # Create figure for tooth line comparison
        plt.figure(figsize=(10, 8))

        # Get coefficients for both images
        coeffs1 = self.img1_tooth_line_results['coeffs']
        coeffs2 = self.img2_tooth_line_results['coeffs']

        # Create x range that covers both tooth lines
        min_x = min(self.img1_tooth_line_results['min_x'], self.img2_tooth_line_results['min_x'])
        max_x = max(self.img1_tooth_line_results['max_x'], self.img2_tooth_line_results['max_x'])
        x_range = np.arange(min_x, max_x + 1)

        # Calculate y values for both curves
        y1 = self.img1_tooth_line_results['quad_func'](x_range)
        y2 = self.img2_tooth_line_results['quad_func'](x_range)

        # Plot both curves
        plt.plot(x_range, y1, 'r-', linewidth=2, label=f'Image 1: {os.path.basename(self.img1_path)}')
        plt.plot(x_range, y2, 'b-', linewidth=2, label=f'Image 2: {os.path.basename(self.img2_path)}')

        # Add coefficient information to the plot
        plt.text(0.05, 0.95, f"Image 1 coefficients: a={coeffs1[0]:.6f}, b={coeffs1[1]:.6f}, c={coeffs1[2]:.6f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.text(0.05, 0.90, f"Image 2 coefficients: a={coeffs2[0]:.6f}, b={coeffs2[1]:.6f}, c={coeffs2[2]:.6f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Calculate and display curvature difference
        curvature_diff = abs(coeffs1[0] - coeffs2[0])
        plt.text(0.05, 0.85, f"Curvature difference: {curvature_diff:.6f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title("Tooth Line Comparison")
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.tooth_line_dir, "tooth_line_comparison.png"), dpi=300)
        plt.close()

    def create_3d_visualization(self):
        """Create and display 3D visualization of the dental analysis results."""
        print("\nCreating 3D visualization...")

        # Check that we have all the necessary data
        if (self.img1_rgb is None or self.img2_rgb is None or
                self.img1_mandible_mask is None or self.img2_mandible_mask is None or
                self.img1_tooth_line_results is None or self.img2_tooth_line_results is None):
            print("Cannot create 3D visualization - missing required data.")
            return False

        # Extract teeth masks from teeth segmentation results
        img1_teeth_mask = None
        img2_teeth_mask = None

        if self.img1_teeth_results is not None and 'full_mask' in self.img1_teeth_results:
            img1_teeth_mask = self.img1_teeth_results['full_mask']

        if self.img2_teeth_results is not None and 'full_mask' in self.img2_teeth_results:
            img2_teeth_mask = self.img2_teeth_results['full_mask']

        # Create the 3D renderer
        renderer = Dental3DRenderer(
            img1_rgb=self.img1_rgb,
            img2_rgb=self.img2_rgb,
            img1_mandible_mask=self.img1_mandible_mask,
            img2_mandible_mask=self.img2_mandible_mask,
            img1_bottom_teeth_mask=self.img1_bottom_teeth_mask,
            img2_bottom_teeth_mask=self.img2_bottom_teeth_mask,
            affine_matrix=self.affine_matrix,
            img1_tooth_line_results=self.img1_tooth_line_results,
            img2_tooth_line_results=self.img2_tooth_line_results,
            z_scale=self.z_scale,
            curvature_scale=self.curvature_scale,
            curvature_factor=self.curvature_factor
        )

        # Create and display the visualization
        success = renderer.create_visualization()

        if success:
            print("3D visualization completed successfully.")
        else:
            print("Error creating 3D visualization.")

        return success

    def identify_bottom_teeth(self):
        """
        Identify bottom teeth by checking if they are below the tooth line.
        Creates a new mask containing only the bottom teeth.
        """
        print("\nIdentifying bottom teeth...")

        # Ensure teeth segmentation and tooth line extraction have been performed
        if (self.img1_teeth_results is None or self.img2_teeth_results is None or
                self.img1_tooth_line_results is None or self.img2_tooth_line_results is None):
            print("Teeth segmentation and tooth line extraction must be performed before identifying bottom teeth.")
            return False

        # Process image 1
        print("Processing bottom teeth in image 1...")
        # Initialize with zeros of the same shape as the image
        h1, w1 = self.img1_rgb.shape[:2]
        self.img1_bottom_teeth_mask = np.zeros((h1, w1), dtype=np.uint8)
        self.img1_bottom_teeth_indices = []

        # Get the tooth line function for image 1
        tooth_line_func1 = self.img1_tooth_line_results['quad_func']

        # Check each tooth mask to see if it's below the tooth line
        for i, tooth_info in enumerate(self.img1_teeth_results['teeth_masks']):
            # Extract the actual mask from the tooth info dictionary
            tooth_mask = np.zeros((h1, w1), dtype=np.uint8)

            # Get mask and position information
            mask = tooth_info['mask']
            x_offset = tooth_info['x_offset']
            y_offset = tooth_info['y_offset']
            height, width = mask.shape

            # Place the mask in the correct position
            tooth_mask[
            y_offset:y_offset + height,
            x_offset:x_offset + width
            ] = mask

            # Find all non-zero points in the tooth mask
            y_coords, x_coords = np.where(tooth_mask > 0)

            if len(y_coords) == 0:
                continue  # Skip empty masks

            # Calculate the expected y-value on the tooth line for each x-coordinate
            tooth_line_y_values = tooth_line_func1(x_coords)

            # Count how many points are below the tooth line (y > tooth_line_y)
            points_below_line = np.sum(y_coords > tooth_line_y_values)

            # If a significant portion of the tooth is below the line, consider it a bottom tooth
            if points_below_line / len(y_coords) > 0.01:
                # Add to bottom teeth mask
                self.img1_bottom_teeth_mask = np.logical_or(
                    self.img1_bottom_teeth_mask,
                    tooth_mask > 0
                ).astype(np.uint8)
                self.img1_bottom_teeth_indices.append(i)
                print(
                    f"  Tooth {i + 1} identified as bottom tooth in image 1 ({points_below_line / len(y_coords) * 100:.1f}% below line)")

        # Process image 2
        print("Processing bottom teeth in image 2...")
        # Initialize with zeros of the same shape as the image
        h2, w2 = self.img2_rgb.shape[:2]
        self.img2_bottom_teeth_mask = np.zeros((h2, w2), dtype=np.uint8)
        self.img2_bottom_teeth_indices = []

        # Get the tooth line function for image 2
        tooth_line_func2 = self.img2_tooth_line_results['quad_func']

        # Check each tooth mask to see if it's below the tooth line
        for i, tooth_info in enumerate(self.img2_teeth_results['teeth_masks']):
            # Extract the actual mask from the tooth info dictionary
            tooth_mask = np.zeros((h2, w2), dtype=np.uint8)

            # Get mask and position information
            mask = tooth_info['mask']
            x_offset = tooth_info['x_offset']
            y_offset = tooth_info['y_offset']
            height, width = mask.shape

            # Place the mask in the correct position
            tooth_mask[
            y_offset:y_offset + height,
            x_offset:x_offset + width
            ] = mask

            # Find all non-zero points in the tooth mask
            y_coords, x_coords = np.where(tooth_mask > 0)

            if len(y_coords) == 0:
                continue  # Skip empty masks

            # Calculate the expected y-value on the tooth line for each x-coordinate
            tooth_line_y_values = tooth_line_func2(x_coords)

            # Count how many points are below the tooth line (y > tooth_line_y)
            points_below_line = np.sum(y_coords > tooth_line_y_values)

            # If a significant portion of the tooth is below the line, consider it a bottom tooth
            # Using 30% as a threshold - can be adjusted
            if points_below_line / len(y_coords) > 0.3:
                # Add to bottom teeth mask
                self.img2_bottom_teeth_mask = np.logical_or(
                    self.img2_bottom_teeth_mask,
                    tooth_mask > 0
                ).astype(np.uint8)
                self.img2_bottom_teeth_indices.append(i)
                print(
                    f"  Tooth {i + 1} identified as bottom tooth in image 2 ({points_below_line / len(y_coords) * 100:.1f}% below line)")

        # Create visualization of bottom teeth
        self.visualize_bottom_teeth()

        # Save bottom teeth masks
        img1_basename = os.path.basename(self.img1_path)
        img2_basename = os.path.basename(self.img2_path)

        # Create directory for bottom teeth
        bottom_teeth_dir = os.path.join(self.teeth_dir, "bottom_teeth")
        os.makedirs(bottom_teeth_dir, exist_ok=True)

        # Save masks as images
        cv2.imwrite(
            os.path.join(bottom_teeth_dir, f"bottom_teeth_mask_{img1_basename}"),
            (self.img1_bottom_teeth_mask * 255).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(bottom_teeth_dir, f"bottom_teeth_mask_{img2_basename}"),
            (self.img2_bottom_teeth_mask * 255).astype(np.uint8)
        )

        print(
            f"Bottom teeth identification completed. Found {len(self.img1_bottom_teeth_indices)} bottom teeth in image 1 and {len(self.img2_bottom_teeth_indices)} bottom teeth in image 2.")
        return True

    def visualize_bottom_teeth(self):
        """Create visualization of bottom teeth for both images."""
        # Create figure for bottom teeth visualization
        plt.figure(figsize=(12, 6))

        # Image 1 with bottom teeth mask
        plt.subplot(1, 2, 1)
        plt.imshow(self.img1_rgb)

        # Ensure mask is binary
        binary_mask1 = self.img1_bottom_teeth_mask.astype(bool)

        # Create colored overlay for bottom teeth mask
        colored_mask1 = np.zeros((*binary_mask1.shape, 4))  # RGBA
        colored_mask1[binary_mask1] = [0.2, 0.8, 0.2, 0.5]  # Green with 50% transparency

        plt.imshow(colored_mask1)
        plt.title(f"Image 1: {os.path.basename(self.img1_path)}\nBottom Teeth")
        plt.axis('off')

        # Image 2 with bottom teeth mask
        plt.subplot(1, 2, 2)
        plt.imshow(self.img2_rgb)

        # Ensure mask is binary
        binary_mask2 = self.img2_bottom_teeth_mask.astype(bool)

        # Create colored overlay for bottom teeth mask
        colored_mask2 = np.zeros((*binary_mask2.shape, 4))  # RGBA
        colored_mask2[binary_mask2] = [0.2, 0.8, 0.2, 0.5]  # Green with 50% transparency

        plt.imshow(colored_mask2)
        plt.title(f"Image 2: {os.path.basename(self.img2_path)}\nBottom Teeth")
        plt.axis('off')

        plt.tight_layout()

        # Create directory for bottom teeth
        bottom_teeth_dir = os.path.join(self.teeth_dir, "bottom_teeth")
        os.makedirs(bottom_teeth_dir, exist_ok=True)

        plt.savefig(os.path.join(bottom_teeth_dir, "bottom_teeth_visualization.png"), dpi=300)
        plt.close()

    def run_analysis(self):
        """
        Run the complete analysis workflow.

        This method executes all analysis steps in sequence:
        1. Setup output directories
        2. Load and analyze image pair
        3. Create visualizations
        4. Perform segmentation
        5. Extract tooth lines
        6. Identify bottom teeth
        7. Save results
        8. Create 3D visualization

        Returns
        -------
        bool
            True if analysis was successful, False otherwise

        Notes
        -----
        Results are saved to the specified output directory. If any step fails,
        the method returns False and prints error information.
        """
        try:
            # Setup directories
            self.setup_output_directories()

            # Load images
            self.load_images()

            # Analyze image pair
            success = self.analyze_image_pair()
            if not success:
                return False

            # Create visualizations
            self.visualize_teeth_detection()
            self.create_overlay_visualizations()
            self.create_difference_visualization()

            # Perform segmentation
            self.perform_mandible_segmentation()
            self.perform_tooth_segmentation()

            # Perform tooth line extraction
            self.perform_tooth_line_extraction()

            # Identify bottom teeth
            self.identify_bottom_teeth()

            # Save results
            self.save_transformation_matrix()
            self.save_analysis_results()

            # Create 3D visualization
            self.create_3d_visualization()

            print(f"\nResults saved to {self.output_dir}")
            return True

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            traceback.print_exc()
            return False


def main(
        img1_path: str = "./datasets/tooth_ds_val/ds/img/574.jpg",
        img2_path: str = "./datasets/tooth_ds_val/ds/img/594.jpg",
        output_dir: str = "./pair_analysis_results",
        tooth_model_path: str = "./models/tooth-segmentator-dinov2",
        mandible_model_path: str = "./models/mandible-segmentator-dinov2",
        base_model_name: str = "StanfordAIMI/dinov2-base-xray-224",
        tooth_crop_margin: int = 32,
        tooth_line_offset: int = 100,
        z_scale: float = 1.0,
        curvature_scale: float = 4.0,
        curvature_factor: float = 12.0
) -> int:
    """
    Run dental pair analysis from command line.

    Parameters
    ----------
    img1_path : str, optional
        Path to first dental X-ray image
    img2_path : str, optional
        Path to second dental X-ray image
    output_dir : str, optional
        Directory to save results
    tooth_model_path : str, optional
        Path to tooth segmentation model
    mandible_model_path : str, optional
        Path to mandible segmentation model
    base_model_name : str, optional
        Base model name for segmentation
    tooth_crop_margin : int, optional
        Margin around tooth bounding boxes
    tooth_line_offset : int, optional
        Maximum vertical distance for tooth line
    z_scale : float, optional
        Scale factor for z-axis
    curvature_scale : float, optional
        Scale factor for curvature
    curvature_factor : float, optional
        Factor affecting curvature shape

    Returns
    -------
    int
        0 if analysis was successful, 1 otherwise
    """
    analyzer = DentalPairAnalysis(
        img1_path=img1_path,
        img2_path=img2_path,
        output_dir=output_dir,
        tooth_model_path=tooth_model_path,
        mandible_model_path=mandible_model_path,
        base_model_name=base_model_name,
        tooth_crop_margin=tooth_crop_margin,
        tooth_line_offset=tooth_line_offset,
        z_scale=z_scale,
        curvature_scale=curvature_scale,
        curvature_factor=curvature_factor
    )

    success = analyzer.run_analysis()
    return 0 if success else 1


if __name__ == "__main__":
    fire.Fire(main)
