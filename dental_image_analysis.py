import glob
import os
import random
import time
import argparse
from itertools import combinations
from typing import Any, Dict, Tuple, Union, List

import cv2
import numpy as np
import pandas as pd
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from tqdm import tqdm
import fire

from teeth_detector_paddle.teeth_detector_paddle import TeethDetector


class DentalImageAnalyzer:
    """
    Class for analyzing dental X-ray images.

    This class provides functionality for loading, preprocessing, and analyzing
    dental X-ray images. It includes methods for segmentation, feature extraction,
    and visualization.

    Parameters
    ----------
    images_dir : str, optional
        Directory containing dental images (optional for single pair mode)
    output_dir : str, optional
        Directory to save results, by default "./analysis_results"
    save_interval : int, optional
        How often to save the dataframe (every N pairs), by default 10

    Attributes
    ----------
    images_dir : str
        Directory containing dental images
    output_dir : str
        Directory to save results
    save_interval : int
        How often to save the dataframe (every N pairs)
    device : torch.device
        Device to run computations on (CPU/GPU)
    extractor_superpoint : SuperPoint
        Feature extractor for SuperPoint
    matcher_superpoint : LightGlue
        Matcher for SuperPoint
    extractor_disk : DISK
        Feature extractor for DISK
    matcher_disk : LightGlue
        Matcher for DISK
    teeth_detector : TeethDetector
        Teeth detector for teeth detection
    results_df : pd.DataFrame
        Dataframe to store results
    image_paths : list
        List of image paths
    """

    def __init__(self, images_dir=None, output_dir="./analysis_results", save_interval=10):
        """
        Initialize the dental image analyzer.
        
        Args:
            images_dir: Directory containing dental images (optional for single pair mode)
            output_dir: Directory to save results
            save_interval: How often to save the dataframe (every N pairs)
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.save_interval = save_interval

        # Create output directory if it doesn't exist
        self.setup_output_directories()

        # Initialize feature extractors and matchers
        print("Initializing feature extractors and matchers...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # SuperPoint + LightGlue
        self.extractor_superpoint = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher_superpoint = LightGlue(features='superpoint').eval().to(self.device)

        # DISK + LightGlue
        self.extractor_disk = DISK(max_num_keypoints=4096).eval().to(self.device)
        self.matcher_disk = LightGlue(features='disk').eval().to(self.device)

        # Initialize teeth detector
        print("Initializing teeth detector...")
        self.teeth_detector = TeethDetector(threshold=0.5)

        # Initialize results dataframe
        self.results_df = pd.DataFrame(columns=[
            'pair_id', 'image1', 'image2',
            'num_teeth_img1', 'num_teeth_img2',
            'superpoint_matches', 'disk_matches',
            'superpoint_teeth_matches', 'disk_teeth_matches',
            'combined_teeth_matches', 'num_tooth_pairs',
            'affine_mean_error', 'affine_median_error',
            'affine_min_error', 'affine_max_error', 'affine_std_error',
            'affine_matrix', 'processing_time'
        ])

        # Get all image paths if directory is provided
        if images_dir:
            self.image_paths = self._get_image_paths()
            print(f"Found {len(self.image_paths)} images in {images_dir}")
        else:
            self.image_paths = []

    def setup_output_directories(self) -> None:
        """
        Set up the output directory structure.

        Creates the main output directory and subdirectories for storing
        different types of analysis results.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "matches"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "teeth"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkerboard"), exist_ok=True)

    def _get_image_paths(self) -> List[str]:
        """
        Get all image paths from the images directory.

        Returns
        -------
        list
            List of paths to image files in the images directory
        """
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
        return sorted(image_paths)

    def create_teeth_mask(self, image_path, teeth_boxes, threshold=0.5):
        """Create a binary mask where teeth regions are 1 and background is 0."""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in teeth_boxes:
            # box format: [class_id, score, x1, y1, x2, y2]
            score = box[1]
            if score < threshold:
                continue
            x1, y1, x2, y2 = map(int, box[2:6])
            # Add some padding around teeth
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            mask[y1:y2, x1:x2] = 1

        return mask

    def filter_matches_by_teeth(self, points0, points1, mask0, mask1):
        """Filter matches to only include those within teeth regions in both images."""
        valid_indices = []

        for i, (pt0, pt1) in enumerate(zip(points0, points1)):
            x0, y0 = int(pt0[0]), int(pt0[1])
            x1, y1 = int(pt1[0]), int(pt1[1])

            # Check if both points are within image bounds
            if (0 <= x0 < mask0.shape[1] and 0 <= y0 < mask0.shape[0] and
                    0 <= x1 < mask1.shape[1] and 0 <= y1 < mask1.shape[0]):

                # Check if both points are within teeth regions
                if mask0[y0, x0] == 1 and mask1[y1, x1] == 1:
                    valid_indices.append(i)

        return valid_indices

    def pair_teeth_between_images(self, src_points, dst_points, teeth_boxes_src, teeth_boxes_dst,
                                  confidence_threshold=0.5):
        """
        Pair teeth between two images based on feature point matches.
        
        Args:
            src_points: Nx2 array of source feature points (x, y)
            dst_points: Nx2 array of destination feature points (x, y)
            teeth_boxes_src: List of source tooth bounding boxes [class_id, score, x1, y1, x2, y2]
            teeth_boxes_dst: List of destination tooth bounding boxes [class_id, score, x1, y1, x2, y2]
            confidence_threshold: Minimum confidence score to consider a tooth detection
            
        Returns:
            List of tuples (src_box, dst_box) representing paired teeth between images
        """
        # Filter teeth boxes by confidence threshold
        valid_teeth_src = [box for box in teeth_boxes_src if box[1] >= confidence_threshold]
        valid_teeth_dst = [box for box in teeth_boxes_dst if box[1] >= confidence_threshold]

        # Initialize connection matrix: rows=src teeth, cols=dst teeth
        connection_matrix = np.zeros((len(valid_teeth_src), len(valid_teeth_dst)), dtype=int)

        # For each point pair, check which teeth they connect
        for (x_src, y_src), (x_dst, y_dst) in zip(src_points, dst_points):
            src_tooth_idx = -1
            dst_tooth_idx = -1

            # Find which source tooth contains this point
            for i, (_, _, x1, y1, x2, y2) in enumerate(valid_teeth_src):
                if x1 <= x_src <= x2 and y1 <= y_src <= y2:
                    src_tooth_idx = i
                    break

            # Find which destination tooth contains this point
            for j, (_, _, x1, y1, x2, y2) in enumerate(valid_teeth_dst):
                if x1 <= x_dst <= x2 and y1 <= y_dst <= y2:
                    dst_tooth_idx = j
                    break

            # If both points belong to teeth, increment the connection count
            if src_tooth_idx >= 0 and dst_tooth_idx >= 0:
                connection_matrix[src_tooth_idx, dst_tooth_idx] += 1

        # For each source tooth, find the destination tooth with the most connections
        tooth_pairs = []
        for src_idx in range(len(valid_teeth_src)):
            # Skip if this source tooth has no connections
            if np.sum(connection_matrix[src_idx]) == 0:
                continue

            # Find the destination tooth with the most connections
            dst_idx = np.argmax(connection_matrix[src_idx])

            # Only include pairs with at least one connection
            if connection_matrix[src_idx, dst_idx] > 0:
                tooth_pairs.append((valid_teeth_src[src_idx], valid_teeth_dst[dst_idx]))

        return tooth_pairs

    def collect_inliers_from_tooth_pairs(self, filtered_points0, filtered_points1, tooth_pairs):
        """
        Collect inlier points from tooth pairs by estimating per-tooth homographies.
        
        Args:
            filtered_points0: Source points filtered by teeth regions
            filtered_points1: Destination points filtered by teeth regions
            tooth_pairs: List of paired teeth between images
            
        Returns:
            Numpy arrays of inlier source and destination points
        """
        # Group points into buckets based on tooth pairs
        point_buckets = []

        for i, (src_tooth, dst_tooth) in enumerate(tooth_pairs):
            # Extract coordinates from tooth boxes
            src_class, src_score, src_x1, src_y1, src_x2, src_y2 = src_tooth
            dst_class, dst_score, dst_x1, dst_y1, dst_x2, dst_y2 = dst_tooth

            # Find points that belong to this tooth pair
            bucket_points = []
            for j, ((x_src, y_src), (x_dst, y_dst)) in enumerate(zip(filtered_points0, filtered_points1)):
                # Check if source point is inside source tooth
                src_match = (src_x1 <= x_src <= src_x2 and src_y1 <= y_src <= src_y2)
                # Check if destination point is inside destination tooth
                dst_match = (dst_x1 <= x_dst <= dst_x2 and dst_y1 <= y_dst <= dst_y2)

                # If both points match their respective teeth, add to bucket
                if src_match and dst_match:
                    bucket_points.append((j, (x_src, y_src), (x_dst, y_dst)))

            # Add this bucket to our collection
            point_buckets.append({
                'src_tooth': src_tooth,
                'dst_tooth': dst_tooth,
                'points': bucket_points,
                'count': len(bucket_points)
            })

        # Estimate per-tooth homographies for buckets with enough points
        all_inlier_src_points = []
        all_inlier_dst_points = []

        for bucket in point_buckets:
            points = bucket['points']
            count = bucket['count']

            # Skip if we don't have enough points for homography estimation
            if count < 4:
                continue

            # Extract point coordinates for homography estimation
            src_points = np.array([p[1] for p in points])
            dst_points = np.array([p[2] for p in points])

            try:
                # Estimate homography for this tooth pair
                H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3.0)

                # Collect inliers
                for i, (_, src_point, dst_point) in enumerate(points):
                    if i < len(status) and status[i] == 1:
                        all_inlier_src_points.append(src_point)
                        all_inlier_dst_points.append(dst_point)
            except Exception:
                # Skip this tooth pair if homography estimation fails
                continue

        return np.array(all_inlier_src_points), np.array(all_inlier_dst_points)

    def calculate_affine_errors(self, src_points, dst_points, affine_matrix):
        """
        Calculate projection errors for points transformed by an affine matrix.
        
        Args:
            src_points: Source points
            dst_points: Destination points
            affine_matrix: 2x3 affine transformation matrix
            
        Returns:
            Dictionary of error statistics
        """
        # Calculate the error for each point
        projection_errors = []
        for i in range(len(src_points)):
            # Get the original point from image 1
            pt1 = src_points[i]

            # Get the corresponding point in image 2
            pt2 = dst_points[i]

            # Apply the affine transformation to the point from image 1
            pt1_transformed = np.array([[pt1[0]], [pt1[1]], [1.0]])
            pt1_transformed = np.dot(affine_matrix, pt1_transformed)
            pt1_transformed = (pt1_transformed[0][0], pt1_transformed[1][0])

            # Calculate the Euclidean distance between the transformed point and the target point
            error = np.sqrt((pt1_transformed[0] - pt2[0]) ** 2 + (pt1_transformed[1] - pt2[1]) ** 2)
            projection_errors.append(error)

        # Calculate statistics
        error_stats = {
            'mean': np.mean(projection_errors),
            'median': np.median(projection_errors),
            'min': np.min(projection_errors),
            'max': np.max(projection_errors),
            'std': np.std(projection_errors)
        }

        return error_stats

    def process_image_pair(self, img1_path, img2_path, pair_id=None):
        """
        Process a pair of dental images and calculate transformation metrics.
        
        Args:
            img1_path: Path to the first image
            img2_path: Path to the second image
            pair_id: Identifier for this image pair (auto-generated if None)
            
        Returns:
            Dictionary of results for this pair
        """
        if pair_id is None:
            pair_id = f"pair_{os.path.basename(img1_path)}_{os.path.basename(img2_path)}"

        start_time = time.time()

        # Initialize results dictionary
        results = {
            'pair_id': pair_id,
            'image1': os.path.basename(img1_path),
            'image2': os.path.basename(img2_path)
        }

        try:
            # Detect teeth in both images
            teeth_results_1 = self.teeth_detector.predict_from_file(img1_path, visual=False, save_results=False)
            teeth_results_2 = self.teeth_detector.predict_from_file(img2_path, visual=False, save_results=False)

            # Store teeth detection results
            results['teeth_img1'] = teeth_results_1['boxes']
            results['teeth_img2'] = teeth_results_2['boxes']

            # Create teeth masks
            teeth_mask_1 = self.create_teeth_mask(img1_path, teeth_results_1['boxes'], threshold=0.5)
            teeth_mask_2 = self.create_teeth_mask(img2_path, teeth_results_2['boxes'], threshold=0.5)

            # Record number of teeth detected
            results['num_teeth_img1'] = len(teeth_results_1['boxes'])
            results['num_teeth_img2'] = len(teeth_results_2['boxes'])

            # Load images for feature matching
            image0 = load_image(img1_path).to(self.device)
            image1 = load_image(img2_path).to(self.device)

            # Extract features using SuperPoint
            feats0_superpoint = self.extractor_superpoint.extract(image0)
            feats1_superpoint = self.extractor_superpoint.extract(image1)

            # Match features using SuperPoint + LightGlue
            matches01_superpoint = self.matcher_superpoint({'image0': feats0_superpoint, 'image1': feats1_superpoint})
            feats0_superpoint, feats1_superpoint, matches01_superpoint = [
                rbd(x) for x in [feats0_superpoint, feats1_superpoint, matches01_superpoint]
            ]
            matches_superpoint = matches01_superpoint['matches']
            points0_superpoint = feats0_superpoint['keypoints'][matches_superpoint[..., 0]]
            points1_superpoint = feats1_superpoint['keypoints'][matches_superpoint[..., 1]]

            # Extract features using DISK
            feats0_disk = self.extractor_disk.extract(image0)
            feats1_disk = self.extractor_disk.extract(image1)

            # Match features using DISK + LightGlue
            matches01_disk = self.matcher_disk({'image0': feats0_disk, 'image1': feats1_disk})
            feats0_disk, feats1_disk, matches01_disk = [
                rbd(x) for x in [feats0_disk, feats1_disk, matches01_disk]
            ]
            matches_disk = matches01_disk['matches']
            points0_disk = feats0_disk['keypoints'][matches_disk[..., 0]]
            points1_disk = feats1_disk['keypoints'][matches_disk[..., 1]]

            # Record match counts before filtering
            results['superpoint_matches'] = len(matches_superpoint)
            results['disk_matches'] = len(matches_disk)

            # Convert PyTorch tensors to numpy arrays
            points0_np_superpoint = points0_superpoint.cpu().numpy()
            points1_np_superpoint = points1_superpoint.cpu().numpy()
            points0_np_disk = points0_disk.cpu().numpy()
            points1_np_disk = points1_disk.cpu().numpy()

            # Filter matches to only include those within teeth regions
            valid_indices_superpoint = self.filter_matches_by_teeth(
                points0_np_superpoint, points1_np_superpoint, teeth_mask_1, teeth_mask_2
            )
            valid_indices_disk = self.filter_matches_by_teeth(
                points0_np_disk, points1_np_disk, teeth_mask_1, teeth_mask_2
            )

            # Apply filtering
            filtered_points0_superpoint = points0_np_superpoint[valid_indices_superpoint]
            filtered_points1_superpoint = points1_np_superpoint[valid_indices_superpoint]
            filtered_points0_disk = points0_np_disk[valid_indices_disk]
            filtered_points1_disk = points1_np_disk[valid_indices_disk]

            # Record match counts after filtering
            results['superpoint_teeth_matches'] = len(filtered_points0_superpoint)
            results['disk_teeth_matches'] = len(filtered_points0_disk)

            # Combine filtered matches from both strategies
            filtered_combined_points0 = np.vstack([filtered_points0_superpoint, filtered_points0_disk])
            filtered_combined_points1 = np.vstack([filtered_points1_superpoint, filtered_points1_disk])

            results['combined_teeth_matches'] = len(filtered_combined_points0)

            # Pair teeth between images
            tooth_pairs = self.pair_teeth_between_images(
                filtered_combined_points0,
                filtered_combined_points1,
                teeth_results_1['boxes'],
                teeth_results_2['boxes'],
                confidence_threshold=0.5
            )

            results['num_tooth_pairs'] = len(tooth_pairs)
            results['tooth_pairs'] = tooth_pairs

            # Collect inliers from tooth pairs
            inlier_src_points, inlier_dst_points = self.collect_inliers_from_tooth_pairs(
                filtered_combined_points0,
                filtered_combined_points1,
                tooth_pairs
            )

            # If we have enough inliers, calculate affine transformation
            if len(inlier_src_points) >= 3:
                # Calculate affine transformation matrix
                affine_matrix, _ = cv2.estimateAffinePartial2D(
                    inlier_src_points,
                    inlier_dst_points,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0
                )

                # Calculate projection errors
                error_stats = self.calculate_affine_errors(
                    inlier_src_points,
                    inlier_dst_points,
                    affine_matrix
                )

                # Record affine transformation results
                results['affine_mean_error'] = error_stats['mean']
                results['affine_median_error'] = error_stats['median']
                results['affine_min_error'] = error_stats['min']
                results['affine_max_error'] = error_stats['max']
                results['affine_std_error'] = error_stats['std']
                results['affine_matrix'] = affine_matrix.tolist()

            else:
                # Not enough inliers for affine transformation
                results['affine_mean_error'] = np.nan
                results['affine_median_error'] = np.nan
                results['affine_min_error'] = np.nan
                results['affine_max_error'] = np.nan
                results['affine_std_error'] = np.nan
                results['affine_matrix'] = None

            # Record processing time
            results['processing_time'] = time.time() - start_time

            return results

        except Exception as e:
            # Record error and processing time
            print(f"Error processing pair {pair_id} ({img1_path}, {img2_path}): {str(e)}")
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time

            return results

    def run_analysis(self, max_pairs=None):
        """
        Run analysis on all image pairs.
        
        Args:
            max_pairs: Maximum number of pairs to process (None for all)
        """
        # Generate all possible image pairs
        image_pairs = list(combinations(self.image_paths, 2))

        # Randomly shuffle the image pairs
        random.shuffle(image_pairs)

        if max_pairs is not None:
            image_pairs = image_pairs[:max_pairs]

        print(f"Processing {len(image_pairs)} image pairs based on {len(self.image_paths)} images...")

        # Process each pair
        for i, (img1_path, img2_path) in enumerate(tqdm(image_pairs)):
            pair_id = f"pair_{i + 1}"

            # Process this pair
            results = self.process_image_pair(img1_path, img2_path, pair_id)

            # Add results to dataframe
            self.results_df = pd.concat([self.results_df, pd.DataFrame([results])], ignore_index=True)

            # Save dataframe every save_interval pairs
            if (i + 1) % self.save_interval == 0:
                self.save_results(f"dental_analysis_interim_{i + 1}.csv")

        # Save final results
        self.save_results("dental_analysis_final.csv")

        print("Analysis complete!")

    def save_results(self, filename):
        """Save results dataframe to CSV."""
        output_path = os.path.join(self.output_dir, filename)
        self.results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Also save a pickle version for easier loading with complex data types
        pickle_path = output_path.replace('.csv', '.pkl')
        self.results_df.to_pickle(pickle_path)
        print(f"Pickle results saved to {pickle_path}")

    def match_features_superpoint(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """
        Extract and match features using SuperPoint and LightGlue.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image

        Returns
        -------
        dict
            Dictionary containing feature matches and transformation information
        """
        # Extract features
        feats0 = self.extractor_superpoint.extract(img1)
        feats1 = self.extractor_superpoint.extract(img2)

        # Match features
        matches01 = self.matcher_superpoint({'image0': feats0, 'image1': feats1})
        
        return {
            'keypoints0': feats0['keypoints'],
            'keypoints1': feats1['keypoints'],
            'matches': matches01['matches'],
            'confidence': matches01['confidence']
        }

    def match_features_disk(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """
        Extract and match features using DISK and LightGlue.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image

        Returns
        -------
        dict
            Dictionary containing feature matches and transformation information
        """
        # Extract features
        feats0 = self.extractor_disk.extract(img1)
        feats1 = self.extractor_disk.extract(img2)

        # Match features
        matches01 = self.matcher_disk({'image0': feats0, 'image1': feats1})
        
        return {
            'keypoints0': feats0['keypoints'],
            'keypoints1': feats1['keypoints'],
            'matches': matches01['matches'],
            'confidence': matches01['confidence']
        }

    def detect_teeth(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """
        Detect teeth in both images using the teeth detector.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image

        Returns
        -------
        dict
            Dictionary containing teeth detection results for both images
        """
        # Detect teeth in both images
        teeth1 = self.teeth_detector.detect(img1)
        teeth2 = self.teeth_detector.detect(img2)
        
        return {
            'teeth1': teeth1,
            'teeth2': teeth2
        }

    def create_visualizations(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            matches_superpoint: Dict[str, Any],
            matches_disk: Dict[str, Any],
            teeth_results: Dict[str, Any]
    ) -> None:
        """
        Create and save visualizations of the analysis results.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image
        matches_superpoint : dict
            SuperPoint feature matching results
        matches_disk : dict
            DISK feature matching results
        teeth_results : dict
            Teeth detection results
        """
        # Create matches visualization
        matches_viz = self.visualize_matches(img1, img2, matches_superpoint, matches_disk)
        cv2.imwrite(os.path.join(self.output_dir, "matches", "matches.png"), matches_viz)

        # Create teeth visualization
        teeth_viz = self.visualize_teeth(img1, img2, teeth_results)
        cv2.imwrite(os.path.join(self.output_dir, "teeth", "teeth.png"), teeth_viz)

        # Create checkerboard visualization
        checkerboard = self.create_checkerboard_visualization(img1, img2)
        cv2.imwrite(os.path.join(self.output_dir, "checkerboard", "checkerboard.png"), checkerboard)

    def visualize_matches(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            matches_superpoint: Dict[str, Any],
            matches_disk: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create a visualization of feature matches.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image
        matches_superpoint : dict
            SuperPoint feature matching results
        matches_disk : dict
            DISK feature matching results

        Returns
        -------
        np.ndarray
            Visualization image showing feature matches
        """
        # Create visualization
        h, w = img1.shape[:2]
        viz = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Draw matches
        viz[:h, :w] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        viz[h:, w:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Draw SuperPoint matches
        for kp1, kp2, conf in zip(matches_superpoint['keypoints0'],
                                 matches_superpoint['keypoints1'],
                                 matches_superpoint['confidence']):
            color = (0, int(255 * conf), 0)
            cv2.line(viz, (int(kp1[0]), int(kp1[1])),
                    (int(kp2[0]) + w, int(kp2[1]) + h), color, 1)
        
        return viz

    def visualize_teeth(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            teeth_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create a visualization of teeth detection results.

        Parameters
        ----------
        img1 : np.ndarray
            First input image
        img2 : np.ndarray
            Second input image
        teeth_results : dict
            Teeth detection results

        Returns
        -------
        np.ndarray
            Visualization image showing detected teeth
        """
        # Create visualization
        h, w = img1.shape[:2]
        viz = np.zeros((h * 2, w, 3), dtype=np.uint8)
        
        # Draw original images
        viz[:h] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        viz[h:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Draw teeth detections
        for tooth in teeth_results['teeth1']:
            cv2.rectangle(viz, (tooth['x1'], tooth['y1']),
                        (tooth['x2'], tooth['y2']), (0, 255, 0), 2)
        
        for tooth in teeth_results['teeth2']:
            cv2.rectangle(viz, (tooth['x1'], tooth['y1'] + h),
                        (tooth['x2'], tooth['y2'] + h), (0, 255, 0), 2)
        
        return viz

    def analyze_all_pairs(self) -> None:
        """
        Analyze all possible pairs of images in the dataset.

        This method:
        1. Gets all possible pairs of images
        2. Runs the analysis pipeline on each pair
        3. Saves results periodically
        """
        if not self.image_paths:
            self._get_image_paths()

        total_pairs = len(list(combinations(self.image_paths, 2)))
        print(f"Total number of pairs to analyze: {total_pairs}")

        for i, (img1_path, img2_path) in enumerate(combinations(self.image_paths, 2)):
            print(f"\nAnalyzing pair {i+1}/{total_pairs}")
            print(f"Image 1: {img1_path}")
            print(f"Image 2: {img2_path}")

            try:
                results = self.run_analysis(img1_path, img2_path, i)
                self.save_results(results, i)

                if (i + 1) % self.save_interval == 0:
                    self.save_dataframe()

            except Exception as e:
                print(f"Error analyzing pair {i+1}: {str(e)}")
                continue

        self.save_dataframe()

    def save_results(self, results: Dict[str, Any], pair_idx: int) -> None:
        """
        Save analysis results to the results dataframe.

        Parameters
        ----------
        results : dict
            Dictionary containing analysis results
        pair_idx : int
            Index of the image pair being processed
        """
        # Process results to make them JSON serializable
        processed_results = {k: self.process_value(v) for k, v in results.items()}
        
        # Add to dataframe
        self.results_df.loc[pair_idx] = processed_results

    def save_dataframe(self) -> None:
        """
        Save the results dataframe to CSV and pickle files.
        """
        # Save as CSV
        csv_path = os.path.join(self.output_dir, "results.csv")
        self.results_df.to_csv(csv_path)
        print(f"CSV results saved to {csv_path}")

        # Save as pickle
        pickle_path = os.path.join(self.output_dir, "results.pkl")
        self.results_df.to_pickle(pickle_path)
        print(f"Pickle results saved to {pickle_path}")


if __name__ == "__main__":
    """
    Main entry point for running the dental image analysis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze dental X-ray image pairs")
    parser.add_argument("--images_dir", type=str, required=True,
                      help="Directory containing dental X-ray images")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                      help="Directory to save analysis results")
    parser.add_argument("--save_interval", type=int, default=10,
                      help="How often to save results (every N pairs)")
    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = DentalImageAnalyzer(args.images_dir, args.output_dir, args.save_interval)
    analyzer.analyze_all_pairs()
