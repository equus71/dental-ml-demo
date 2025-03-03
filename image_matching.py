from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os

from teeth_detector_paddle.teeth_detector_paddle import TeethDetector
from image_utils import enhanced_image_diff, create_checkerboard_visualization, create_blended_visualization, \
    normalize_image, match_image_sizes

# Define image paths
# image_1_path = './datasets/dental_val/Images/107.png'
# image_2_path = './datasets/dental_val/Images/109.png'
image_1_path = './datasets/tooth_ds_val/ds/img/574.jpg'
image_2_path = './datasets/tooth_ds_val/ds/img/594.jpg'

# Initialize the teeth detector
print("Initializing teeth detector...")
teeth_detector = TeethDetector(threshold=0.5)

# Detect teeth in both images
print(f"Detecting teeth in image 1: {image_1_path}")
teeth_results_1 = teeth_detector.predict_from_file(image_1_path, visual=True, save_results=False)
print(f"Detecting teeth in image 2: {image_2_path}")
teeth_results_2 = teeth_detector.predict_from_file(image_2_path, visual=True, save_results=False)


# Create teeth masks for both images
def create_teeth_mask(image_path, teeth_boxes, threshold=0.5):
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


# Create teeth masks
teeth_mask_1 = create_teeth_mask(image_1_path, teeth_results_1['boxes'], threshold=0.5)
teeth_mask_2 = create_teeth_mask(image_2_path, teeth_results_2['boxes'], threshold=0.5)

# Visualize teeth masks
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(image_1_path)[..., ::-1])  # Convert BGR to RGB
plt.imshow(teeth_mask_1, alpha=0.3, cmap='cool')
plt.title("Image 1 with Teeth Mask")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.imread(image_2_path)[..., ::-1])  # Convert BGR to RGB
plt.imshow(teeth_mask_2, alpha=0.3, cmap='cool')
plt.title("Image 2 with Teeth Mask")
plt.axis('off')

plt.tight_layout()
plt.savefig("teeth_masks.png", dpi=300)
plt.close()

print(f"Number of teeth detected in image 1: {len(teeth_results_1['boxes'])}")
print(f"Number of teeth detected in image 2: {len(teeth_results_2['boxes'])}")

# Load images for feature matching
image0 = load_image(image_1_path).cuda()
image1 = load_image(image_2_path).cuda()

# Strategy 1: SuperPoint + LightGlue
extractor1 = SuperPoint(max_num_keypoints=4096).eval().cuda()
matcher1 = LightGlue(features='superpoint').eval().cuda()

# Strategy 2: DISK + LightGlue
extractor2 = DISK(max_num_keypoints=4096).eval().cuda()
matcher2 = LightGlue(features='disk').eval().cuda()

# Extract features using both strategies
feats0_strategy1 = extractor1.extract(image0)
feats1_strategy1 = extractor1.extract(image1)

feats0_strategy2 = extractor2.extract(image0)
feats1_strategy2 = extractor2.extract(image1)

# Match features using both strategies
matches01_strategy1 = matcher1({'image0': feats0_strategy1, 'image1': feats1_strategy1})
feats0_strategy1, feats1_strategy1, matches01_strategy1 = [rbd(x) for x in
                                                           [feats0_strategy1, feats1_strategy1, matches01_strategy1]]
matches_strategy1 = matches01_strategy1['matches']
points0_strategy1 = feats0_strategy1['keypoints'][matches_strategy1[..., 0]]
points1_strategy1 = feats1_strategy1['keypoints'][matches_strategy1[..., 1]]

matches01_strategy2 = matcher2({'image0': feats0_strategy2, 'image1': feats1_strategy2})
feats0_strategy2, feats1_strategy2, matches01_strategy2 = [rbd(x) for x in
                                                           [feats0_strategy2, feats1_strategy2, matches01_strategy2]]
matches_strategy2 = matches01_strategy2['matches']
points0_strategy2 = feats0_strategy2['keypoints'][matches_strategy2[..., 0]]
points1_strategy2 = feats1_strategy2['keypoints'][matches_strategy2[..., 1]]

# Print match counts before filtering
print(f"Strategy 1 (SuperPoint) before filtering: {len(matches_strategy1)} matches")
print(f"Strategy 2 (DISK) before filtering: {len(matches_strategy2)} matches")

# Convert PyTorch tensors to numpy arrays
points0_np_strategy1 = points0_strategy1.cpu().numpy()
points1_np_strategy1 = points1_strategy1.cpu().numpy()
points0_np_strategy2 = points0_strategy2.cpu().numpy()
points1_np_strategy2 = points1_strategy2.cpu().numpy()


# Filter matches to only include those within teeth regions in both images
def filter_matches_by_teeth(points0, points1, mask0, mask1):
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


# Filter matches
valid_indices_strategy1 = filter_matches_by_teeth(
    points0_np_strategy1, points1_np_strategy1, teeth_mask_1, teeth_mask_2)
valid_indices_strategy2 = filter_matches_by_teeth(
    points0_np_strategy2, points1_np_strategy2, teeth_mask_1, teeth_mask_2)

# Apply filtering
filtered_points0_strategy1 = points0_np_strategy1[valid_indices_strategy1]
filtered_points1_strategy1 = points1_np_strategy1[valid_indices_strategy1]
filtered_points0_strategy2 = points0_np_strategy2[valid_indices_strategy2]
filtered_points1_strategy2 = points1_np_strategy2[valid_indices_strategy2]

# Print match counts after filtering
print(f"Strategy 1 (SuperPoint) after teeth filtering: {len(filtered_points0_strategy1)} matches")
print(f"Strategy 2 (DISK) after teeth filtering: {len(filtered_points0_strategy2)} matches")


# Visualization function for both strategies
def visualize_dual_matches(img1_path, img2_path,
                           kpts1_strategy1, kpts2_strategy1,
                           kpts1_strategy2, kpts2_strategy2,
                           output_path=None):
    """
    Visualize matches from two different strategies with different colors.
    
    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        kpts1_strategy1, kpts2_strategy1: Keypoints from strategy 1
        kpts1_strategy2, kpts2_strategy2: Keypoints from strategy 2
        output_path: Path to save the visualization
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a new image with both images stacked vertically
    h_total = h1 + h2
    w_max = max(w1, w2)
    vis_img = np.zeros((h_total, w_max, 3), dtype=np.uint8)

    # Place images in the visualization
    vis_img[:h1, :w1] = img1_rgb
    vis_img[h1:h1 + h2, :w2] = img2_rgb

    # Create figure and axis
    plt.figure(figsize=(12, 12))
    plt.imshow(vis_img)

    # Draw matches from strategy 1 with green lines
    for i in range(len(kpts1_strategy1)):
        x1, y1 = kpts1_strategy1[i]
        x2, y2 = kpts2_strategy1[i]
        y2 += h1  # Adjust y-coordinate for the second image

        # Draw green line for strategy 1
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=0.8, alpha=0.7)

    # Draw matches from strategy 2 with blue lines
    for i in range(len(kpts1_strategy2)):
        x1, y1 = kpts1_strategy2[i]
        x2, y2 = kpts2_strategy2[i]
        y2 += h1  # Adjust y-coordinate for the second image

        # Draw blue line for strategy 2
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=0.8, alpha=0.7)

    # Add legend
    plt.plot([], [], 'g-', linewidth=2, label=f'SuperPoint: {len(kpts1_strategy1)} matches')
    plt.plot([], [], 'b-', linewidth=2, label=f'DISK: {len(kpts1_strategy2)} matches')
    plt.legend(loc='upper right')

    plt.title('Comparison of Feature Matching Strategies')
    plt.axis('off')

    # Save or show the visualization
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Dual strategy visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


# OpenCV visualization for both strategies
def visualize_dual_matches_cv2(img1_path, img2_path,
                               kpts1_strategy1, kpts2_strategy1,
                               kpts1_strategy2, kpts2_strategy2,
                               output_path=None):
    """
    Visualize matches from two different strategies with different colors using OpenCV.
    
    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        kpts1_strategy1, kpts2_strategy1: Keypoints from strategy 1
        kpts1_strategy2, kpts2_strategy2: Keypoints from strategy 2
        output_path: Path to save the visualization
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a new image with both images stacked vertically
    h_total = h1 + h2
    w_max = max(w1, w2)
    vis_img = np.zeros((h_total, w_max, 3), dtype=np.uint8)

    # Place images in the visualization
    vis_img[:h1, :w1] = img1
    vis_img[h1:h1 + h2, :w2] = img2

    # Draw matches from strategy 1 with green lines
    for i in range(len(kpts1_strategy1)):
        x1, y1 = int(kpts1_strategy1[i][0]), int(kpts1_strategy1[i][1])
        x2, y2 = int(kpts2_strategy1[i][0]), int(kpts2_strategy1[i][1] + h1)

        # Draw green line for strategy 1
        cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw keypoints (red color)
        cv2.circle(vis_img, (x1, y1), 2, (0, 0, 255), -1)
        cv2.circle(vis_img, (x2, y2), 2, (0, 0, 255), -1)

    # Draw matches from strategy 2 with blue lines
    for i in range(len(kpts1_strategy2)):
        x1, y1 = int(kpts1_strategy2[i][0]), int(kpts1_strategy2[i][1])
        x2, y2 = int(kpts2_strategy2[i][0]), int(kpts2_strategy2[i][1] + h1)

        # Draw blue line for strategy 2
        cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw keypoints (yellow color)
        cv2.circle(vis_img, (x1, y1), 2, (0, 255, 255), -1)
        cv2.circle(vis_img, (x2, y2), 2, (0, 255, 255), -1)

    # Add text with match counts
    cv2.putText(vis_img, f"SuperPoint: {len(kpts1_strategy1)} matches", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_img, f"DISK: {len(kpts1_strategy2)} matches", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"OpenCV dual strategy visualization saved to {output_path}")
    else:
        # Resize for display if too large
        scale = min(1.0, 1200 / w_max)
        if scale < 1.0:
            vis_img = cv2.resize(vis_img, None, fx=scale, fy=scale)

        cv2.imshow("Dual Strategy Matches", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Visualize both strategies with filtered matches
visualize_dual_matches(
    image_1_path,
    image_2_path,
    filtered_points0_strategy1,
    filtered_points1_strategy1,
    filtered_points0_strategy2,
    filtered_points1_strategy2,
    "teeth_filtered_matches.png"
)

# Also create OpenCV visualization
visualize_dual_matches_cv2(
    image_1_path,
    image_2_path,
    filtered_points0_strategy1,
    filtered_points1_strategy1,
    filtered_points0_strategy2,
    filtered_points1_strategy2,
    "teeth_filtered_matches_cv2.png"
)

# Combine filtered matches from both strategies
filtered_combined_points0 = np.vstack([filtered_points0_strategy1, filtered_points0_strategy2])
filtered_combined_points1 = np.vstack([filtered_points1_strategy1, filtered_points1_strategy2])

print(f"Total matches after teeth filtering: {len(filtered_combined_points0)}")


def spatial_monte_carlo_ransac(src_points, dst_points, num_iterations=1000, threshold=3.0, influence_radius=50,
                               min_points=20):
    """
    RANSAC variant that uses Monte Carlo sampling based on spatial distribution.
    
    Args:
        src_points: Source points (Nx2 array)
        dst_points: Destination points (Nx2 array)
        num_iterations: Number of RANSAC iterations
        threshold: Inlier threshold
        influence_radius: Radius of influence for density calculation
        
    Returns:
        best_model: Best transformation model
        inliers: Indices of inlier points
    """
    best_model = None
    best_inliers = []

    assert min_points >= 4, "Minimum points needed for model estimation is 4"

    # Calculate local density for each point
    local_densities = np.zeros(len(src_points))
    for i in range(len(src_points)):
        # Count points within influence_radius
        distances = np.sqrt(np.sum((src_points - src_points[i]) ** 2, axis=1))
        local_densities[i] = np.sum(distances < influence_radius)

    # Inverse density to get probability (lower density = higher probability)
    # Add small constant to avoid division by zero
    inverse_densities = 1.0 / (local_densities + 1e-6)

    # Normalize to create probability distribution
    probs = inverse_densities / np.sum(inverse_densities)
    # print(f"min/max/sum of probs: {np.min(probs)}, {np.max(probs)}, {np.sum(probs)}")

    for _ in range(num_iterations):
        # Sample points based on inverse density
        sample_indices = []
        # Generate random values for all points
        random_values = np.random.random(len(src_points)) / min_points

        # Accept points where random value is less than probability
        for i in range(len(src_points)):
            if random_values[i] < probs[i]:
                sample_indices.append(i)

        if len(sample_indices) < min_points:
            continue

        # Estimate model using sampled points
        src_samples = src_points[sample_indices]
        dst_samples = dst_points[sample_indices]

        try:
            # For example, estimate a homography (replace with your transformation)
            model = cv2.findHomography(src_samples, dst_samples, 0)[0]

            # Count inliers
            src_points_h = np.hstack((src_points, np.ones((len(src_points), 1))))
            transformed = np.dot(src_points_h, model.T)
            transformed = transformed[:, :2] / transformed[:, 2:]

            distances = np.sqrt(np.sum((transformed - dst_points) ** 2, axis=1))
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_model = model
                best_inliers = inliers
        except:
            continue

    return best_model, best_inliers


# Apply spatial density filtering
H, inliers = spatial_monte_carlo_ransac(
    filtered_combined_points0,
    filtered_combined_points1,
)
print(f"Homography matrix:\n{H} ")
print(f"Number of inliers: {len(inliers)}")

# Load the original images
img1 = cv2.imread(image_1_path)
img2 = cv2.imread(image_2_path)

# Convert to RGB for better visualization
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Get image dimensions
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Apply the homography to transform img1 to align with img2
warped_img1 = cv2.warpPerspective(img1_rgb, H, (w2, h2))

# Calculate the difference between the warped image and the target image
# Convert to grayscale for better difference visualization
warped_img1_gray = cv2.cvtColor(warped_img1, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

# Normalize images to 0-255 range
warped_img1_gray = cv2.normalize(warped_img1_gray, None, 0, 255, cv2.NORM_MINMAX)
img2_gray = cv2.normalize(img2_gray, None, 0, 255, cv2.NORM_MINMAX)

# Calculate absolute difference
diff_color = enhanced_image_diff(warped_img1_gray, img2_gray)

# Create a visualization of the results
plt.figure(figsize=(20, 15))

# Original image 1
plt.subplot(2, 2, 1)
plt.imshow(img1_rgb)
plt.title("Original Image 1")
plt.axis('off')

# Original image 2
plt.subplot(2, 2, 2)
plt.imshow(img2_rgb)
plt.title("Original Image 2 (Target)")
plt.axis('off')

# Warped image 1
plt.subplot(2, 2, 3)
plt.imshow(warped_img1)
plt.title("Warped Image 1 (Aligned to Image 2)")
plt.axis('off')

# Difference image
plt.subplot(2, 2, 4)
plt.imshow(diff_color)
plt.title("Difference (Warped Image 1 - Image 2)")
plt.axis('off')

plt.tight_layout()
plt.savefig("teeth_filtered_alignment_results.png", dpi=300)
plt.close()

# Create a blended visualization to better see the alignment
alpha = 0.5  # Blending factor
blended = create_blended_visualization(warped_img1, img2_rgb, alpha=0.5)

plt.figure(figsize=(15, 10))
plt.imshow(blended)
plt.title("Blended Result (Warped Image 1 + Image 2)")
plt.axis('off')
plt.tight_layout()
plt.savefig("teeth_filtered_blended_result.png", dpi=300)
plt.close()

# Create a checkerboard pattern visualization
checkerboard = create_checkerboard_visualization(img1, img2, pattern_size=50)

print("Teeth-filtered alignment complete. Results saved to:")
print("- teeth_filtered_alignment_results.png")
print("- teeth_filtered_blended_result.png")

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def piecewise_dental_transform(src_img, src_points, dst_points, teeth_mask_src, teeth_mask_dst):
    """
    Apply a specialized dental transformation that:
    1. Uses a global model for Y-axis alignment
    2. Uses local models for X-axis alignment per tooth region
    3. Ensures smooth transitions between regions
    
    Args:
        src_img: Source image to transform
        src_points: Source keypoints (Nx2 array)
        dst_points: Destination keypoints (Nx2 array)
        teeth_mask_src: Binary mask of teeth in source image
        teeth_mask_dst: Binary mask of teeth in destination image
    """
    h, w = src_img.shape[:2]

    # 1. Global Y-axis transformation (simpler model)
    # Use a low-degree polynomial for vertical alignment
    poly_y = PolynomialFeatures(degree=2)
    src_points_poly_y = poly_y.fit_transform(src_points)

    ransac_y = RANSACRegressor(min_samples=20, residual_threshold=5.0)
    ransac_y.fit(src_points_poly_y, dst_points[:, 1])

    # 2. Find connected teeth regions for local X transformations
    from scipy import ndimage
    labeled_teeth_src, num_teeth_src = ndimage.label(teeth_mask_src)

    # Create a grid of coordinates
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    grid_coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Initialize output arrays for the transformed coordinates
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)

    # 3. Apply global Y transformation to all points
    grid_poly_y = poly_y.transform(grid_coords)
    y_map = ransac_y.predict(grid_poly_y).reshape(h, w)

    # 4. Apply local X transformations per tooth region
    # First, create a default global X transformation as fallback
    poly_x_global = PolynomialFeatures(degree=2)
    src_points_poly_x = poly_x_global.fit_transform(src_points)

    ransac_x_global = RANSACRegressor(min_samples=20, residual_threshold=5.0)
    ransac_x_global.fit(src_points_poly_x, dst_points[:, 0])

    # Apply global X transformation to all points initially
    grid_poly_x = poly_x_global.transform(grid_coords)
    x_map = ransac_x_global.predict(grid_poly_x).reshape(h, w)

    # 5. For each tooth region, create and apply a local X transformation
    local_x_models = {}

    for tooth_id in range(1, num_teeth_src + 1):
        # Get points in this tooth region
        tooth_mask = (labeled_teeth_src == tooth_id)
        tooth_indices = []

        for i, (x, y) in enumerate(src_points):
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < w and 0 <= y_int < h and tooth_mask[y_int, x_int]:
                tooth_indices.append(i)

        # If we have enough points in this tooth, create a local model
        if len(tooth_indices) >= 10:
            tooth_src_points = src_points[tooth_indices]
            tooth_dst_points = dst_points[tooth_indices]

            # Create a local X transformation for this tooth
            poly_x_local = PolynomialFeatures(degree=2)
            tooth_src_poly = poly_x_local.fit_transform(tooth_src_points)

            ransac_x_local = RANSACRegressor(min_samples=min(8, len(tooth_indices)),
                                             residual_threshold=3.0)
            ransac_x_local.fit(tooth_src_poly, tooth_dst_points[:, 0])

            local_x_models[tooth_id] = {
                'poly': poly_x_local,
                'model': ransac_x_local,
                'center': np.mean(tooth_src_points, axis=0)
            }

    # 6. Apply local X transformations with smooth blending
    # For each pixel, find the nearest tooth models and blend their predictions
    for tooth_id, model_info in local_x_models.items():
        # Get the region of this tooth with some padding for smooth transition
        tooth_mask = (labeled_teeth_src == tooth_id)
        tooth_mask_dilated = ndimage.binary_dilation(tooth_mask, iterations=10)

        # Get coordinates in the dilated region
        y_tooth, x_tooth = np.where(tooth_mask_dilated)
        tooth_coords = np.column_stack((x_tooth, y_tooth))

        if len(tooth_coords) == 0:
            continue

        # Transform these coordinates using the local model
        tooth_poly = model_info['poly'].transform(tooth_coords)
        x_local = model_info['model'].predict(tooth_poly)

        # Calculate distance-based weights for smooth blending
        center = model_info['center']
        distances = np.sqrt(np.sum((tooth_coords - center) ** 2, axis=1))
        max_dist = np.max(distances) + 1e-5
        weights = np.maximum(0, 1 - distances / max_dist)

        # Update the x_map with weighted local transformation
        for i, (x, y) in enumerate(tooth_coords):
            # Blend with existing value based on weight and distance from tooth center
            current_val = x_map[y, x]
            new_val = x_local[i]
            weight = weights[i]

            # Higher weight for points inside the actual tooth
            if tooth_mask[y, x]:
                weight = max(weight, 0.8)

            x_map[y, x] = weight * new_val + (1 - weight) * current_val

    # 7. Apply the transformation to the image
    from scipy.ndimage import map_coordinates
    output = np.zeros_like(src_img)

    # Ensure coordinates are within bounds
    x_map = np.clip(x_map, 0, w - 1)
    y_map = np.clip(y_map, 0, h - 1)

    # Remap each channel
    for i in range(src_img.shape[2]):
        output[:, :, i] = map_coordinates(src_img[:, :, i], [y_map, x_map], order=1, mode='nearest')

    return output


def pair_teeth_between_images(src_points, dst_points, teeth_boxes_src, teeth_boxes_dst, confidence_threshold=0.5):
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


tooth_pairs = pair_teeth_between_images(
    filtered_combined_points0,
    filtered_combined_points1,
    teeth_results_1['boxes'],
    teeth_results_2['boxes'],
    confidence_threshold=0.5
)

# Group points into buckets based on tooth pairs
print("\nGrouping points by tooth pairs...")
point_buckets = []

for i, (src_tooth, dst_tooth) in enumerate(tooth_pairs):
    # Extract coordinates from tooth boxes
    src_class, src_score, src_x1, src_y1, src_x2, src_y2 = src_tooth
    dst_class, dst_score, dst_x1, dst_y1, dst_x2, dst_y2 = dst_tooth

    # Find points that belong to this tooth pair
    bucket_points = []
    for j, ((x_src, y_src), (x_dst, y_dst)) in enumerate(zip(filtered_combined_points0, filtered_combined_points1)):
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

# Print statistics about point distribution
print(f"\nFound {len(tooth_pairs)} tooth pairs with matching points")
total_points = 0

for i, bucket in enumerate(point_buckets):
    src_tooth = bucket['src_tooth']
    dst_tooth = bucket['dst_tooth']
    count = bucket['count']
    total_points += count

    # Print information about this tooth pair
    print(f"Tooth Pair {i + 1}: Source tooth (class {int(src_tooth[0])}, score {src_tooth[1]:.2f}) → "
          f"Destination tooth (class {int(dst_tooth[0])}, score {dst_tooth[1]:.2f}): {count} points")

# Estimate per-tooth homographies for buckets with enough points
print("\nEstimating per-tooth homographies...")
tooth_homographies = []

for i, bucket in enumerate(point_buckets):
    points = bucket['points']
    count = bucket['count']
    src_tooth = bucket['src_tooth']
    dst_tooth = bucket['dst_tooth']

    # Skip if we don't have enough points for homography estimation
    if count < 4:
        print(f"Skipping Tooth Pair {i + 1} (only {count} points, need at least 4)")
        tooth_homographies.append(None)
        continue

    # Extract point coordinates for homography estimation
    src_points = np.array([p[1] for p in points])
    dst_points = np.array([p[2] for p in points])

    try:
        # Estimate homography for this tooth pair
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3.0)
        inliers = np.sum(status)

        print(f"Tooth Pair {i + 1}: Estimated homography with {inliers}/{count} inliers")

        # Store the homography along with tooth information
        tooth_homographies.append({
            'H': H,
            'src_tooth': src_tooth,
            'dst_tooth': dst_tooth,
            'inliers': inliers,
            'points': points,
            'status': status,
            'total_points': count,
            'inlier_ratio': inliers / count if count > 0 else 0
        })
    except Exception as e:
        print(f"Error estimating homography for Tooth Pair {i + 1}: {e}")
        tooth_homographies.append(None)

# Print summary of tooth homographies
print("\nTooth homography summary:")
valid_homographies = [h for h in tooth_homographies if h is not None]
print(f"Successfully estimated {len(valid_homographies)} homographies out of {len(tooth_pairs)} tooth pairs")

# Optional: Filter out low-quality homographies
good_homographies = [h for h in valid_homographies if h['inlier_ratio'] > 0.5]
print(f"Good quality homographies (>50% inliers): {len(good_homographies)}")
# Collect all inliers from good homographies to calculate a global homography
print("\nCollecting inliers from good homographies for global homography estimation...")

all_inlier_src_points = []
all_inlier_dst_points = []

for homography in good_homographies:
    # Get the points for this tooth pair
    status = homography['status'].flatten()
    points = homography['points']

    for i, (_, src_point, dst_point) in enumerate(points):
        if i < len(status) and status[i] == 1:
            all_inlier_src_points.append(src_point)
            all_inlier_dst_points.append(dst_point)

# Convert to numpy arrays
all_inlier_src_points = np.array(all_inlier_src_points)
all_inlier_dst_points = np.array(all_inlier_dst_points)

# Calculate global homography using all inliers
if len(all_inlier_src_points) >= 4:
    try:
        global_H, global_status = cv2.findHomography(
            all_inlier_src_points,
            all_inlier_dst_points,
            0,
            3.0
        )

        global_inliers = np.sum(global_status)
        global_inlier_ratio = global_inliers / len(all_inlier_src_points)

        print(f"Global homography estimated with {global_inliers}/{len(all_inlier_src_points)} inliers")
        print(f"Global inlier ratio: {global_inlier_ratio:.2f}")
        print(f"Global homography matrix:\n{global_H}")
    except Exception as e:
        print(f"Error estimating global homography: {e}")
        global_H = None
else:
    print(f"Not enough inliers ({len(all_inlier_src_points)}) to estimate global homography")
    global_H = None

# If we have a global homography, create a comparison with the homography-based warping
if global_H is not None:
    print("\nCreating comparison with homography-based warping...")

    # Load images in RGB format
    img1_rgb = cv2.cvtColor(cv2.imread(image_1_path), cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(cv2.imread(image_2_path), cv2.COLOR_BGR2RGB)

    # Get image dimensions
    h2, w2 = img2_rgb.shape[:2]

    # Warp image 1 to image 2 using the global homography
    warped_img1_homography = cv2.warpPerspective(img1_rgb, global_H, (w2, h2))

    # Convert to grayscale for difference calculation
    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    warped_img1_homography_gray = cv2.cvtColor(warped_img1_homography, cv2.COLOR_RGB2GRAY)

    # Calculate absolute difference between warped image 1 and image 2
    diff_image = cv2.absdiff(warped_img1_homography_gray, img2_gray)

    # Normalize and colorize the difference image for better visualization
    diff_image_normalized = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
    diff_image_color = cv2.applyColorMap(diff_image_normalized, cv2.COLORMAP_JET)
    diff_image_color = cv2.cvtColor(diff_image_color, cv2.COLOR_BGR2RGB)

    # Create a figure to display the comparison
    plt.figure(figsize=(20, 15))

    # Original image 1
    plt.subplot(2, 2, 1)
    plt.imshow(img1_rgb)
    plt.title("Original Image 1")
    plt.axis('off')

    # Original image 2
    plt.subplot(2, 2, 2)
    plt.imshow(img2_rgb)
    plt.title("Original Image 2 (Target)")
    plt.axis('off')

    # Homography-warped image 1
    plt.subplot(2, 2, 3)
    plt.imshow(warped_img1_homography)
    plt.title("Homography-Warped Image 1")
    plt.axis('off')

    # Difference image
    plt.subplot(2, 2, 4)
    plt.imshow(diff_image_color)
    plt.title("Difference (Warped Image 1 - Image 2)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("homography_comparison.png", dpi=300)
    plt.close()

    print("Homography comparison saved to homography_comparison.png")

    # Create a blended visualization
    alpha = 0.5
    blended = cv2.addWeighted(warped_img1_homography, alpha, img2_rgb, 1 - alpha, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.title("Blended Result (Warped Image 1 + Image 2)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("homography_blended_result.png", dpi=300)
    plt.close()

    print("Blended result saved to homography_blended_result.png")

# Apply affine transformation using all inlier points
print("Applying affine transformation using all inlier points...")

# Calculate affine transformation matrix using all inlier points
affine_matrix, _ = cv2.estimateAffinePartial2D(
    filtered_combined_points0,
    filtered_combined_points1,
    method=cv2.LMEDS,
)

# Apply the affine transformation to the first image
h, w = img2_rgb.shape[:2]
warped_img1_affine = cv2.warpAffine(img1_rgb, affine_matrix, (w, h))

# Calculate difference image
warped_affine_gray = cv2.cvtColor(warped_img1_affine, cv2.COLOR_RGB2GRAY)
warped_affine_gray = cv2.normalize(warped_affine_gray, None, 0, 255, cv2.NORM_MINMAX)

warped_affine_gray, img2_gray = match_image_sizes(warped_affine_gray, img2_gray)
diff_image_color = enhanced_image_diff(warped_affine_gray, img2_gray)

# Create a figure to display the comparison
plt.figure(figsize=(20, 15))

# Original image 1
plt.subplot(2, 2, 1)
plt.imshow(img1_rgb)
plt.title("Original Image 1")
plt.axis('off')

# Original image 2
plt.subplot(2, 2, 2)
plt.imshow(img2_rgb)
plt.title("Original Image 2 (Target)")
plt.axis('off')

# Affine-warped image 1
plt.subplot(2, 2, 3)
plt.imshow(warped_img1_affine)
plt.title("Affine-Warped Image 1")
plt.axis('off')

# Difference image
plt.subplot(2, 2, 4)
plt.imshow(diff_image_color)
plt.title("Difference (Affine-Warped Image 1 - Image 2)")
plt.axis('off')

plt.tight_layout()
plt.savefig("affine_comparison.png", dpi=300)
plt.close()

print("Affine comparison saved to affine_comparison.png")

# Create a blended visualization for affine transformation
alpha = 0.5
blended_affine = cv2.addWeighted(warped_img1_affine, alpha, img2_rgb, 1 - alpha, 0)

plt.figure(figsize=(10, 8))
plt.imshow(blended_affine)
plt.title("Blended Result (Affine-Warped Image 1 + Image 2)")
plt.axis('off')
plt.tight_layout()
plt.savefig("affine_blended_result.png", dpi=300)
plt.close()

print("Affine blended result saved to affine_blended_result.png")

# Calculate the projection error for every point used in the affine transformation
print("Calculating affine transformation projection errors...")

# Calculate the error for each point
projection_errors = []
for i in range(len(filtered_combined_points0)):
    # Get the original point from image 1
    pt1 = filtered_combined_points0[i]

    # Get the corresponding point in image 2
    pt2 = filtered_combined_points1[i]

    # Apply the affine transformation to the point from image 1
    pt1_transformed = np.array([[pt1[0]], [pt1[1]], [1.0]])
    pt1_transformed = np.dot(affine_matrix, pt1_transformed)
    pt1_transformed = (pt1_transformed[0][0], pt1_transformed[1][0])

    # Calculate the Euclidean distance between the transformed point and the target point
    error = np.sqrt((pt1_transformed[0] - pt2[0]) ** 2 + (pt1_transformed[1] - pt2[1]) ** 2)
    projection_errors.append(error)

# Calculate statistics
mean_error = np.mean(projection_errors)
median_error = np.median(projection_errors)
max_error = np.max(projection_errors)
min_error = np.min(projection_errors)
std_error = np.std(projection_errors)

print(f"Affine Transformation Projection Errors (in pixels):")
print(f"  Mean Error: {mean_error:.2f}")
print(f"  Median Error: {median_error:.2f}")
print(f"  Min Error: {min_error:.2f}")
print(f"  Max Error: {max_error:.2f}")
print(f"  Standard Deviation: {std_error:.2f}")

# Visualize the projection errors
plt.figure(figsize=(10, 6))
plt.hist(projection_errors, bins=20, alpha=0.7, color='blue')
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}')
plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_error:.2f}')
plt.title('Affine Transformation Projection Errors')
plt.xlabel('Error (pixels)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("affine_projection_errors.png", dpi=300)
plt.close()

print("Affine projection errors visualization saved to affine_projection_errors.png")

# Apply the specialized dental transformation
print("Applying piecewise dental transformation...")
warped_img1_dental = piecewise_dental_transform(
    img1_rgb,
    filtered_combined_points0,
    filtered_combined_points1,
    teeth_mask_1,
    teeth_mask_2
)

# Visualize the results
plt.figure(figsize=(20, 15))

# Original image 1
plt.subplot(2, 2, 1)
plt.imshow(img1_rgb)
plt.title("Original Image 1")
plt.axis('off')

# Original image 2
plt.subplot(2, 2, 2)
plt.imshow(img2_rgb)
plt.title("Original Image 2 (Target)")
plt.axis('off')

# Dental transformed image
plt.subplot(2, 2, 3)
plt.imshow(warped_img1_dental)
plt.title("Dental Transformed Image 1")
plt.axis('off')

# Difference image
warped_dental_gray = cv2.cvtColor(warped_img1_dental, cv2.COLOR_RGB2GRAY)
warped_dental_gray = cv2.normalize(warped_dental_gray, None, 0, 255, cv2.NORM_MINMAX)

warped_dental_gray, img2_gray = match_image_sizes(warped_dental_gray, img2_gray)
diff_dental = enhanced_image_diff(warped_dental_gray, img2_gray)
# diff_dental = cv2.absdiff(warped_dental_gray, img2_gray)


plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(diff_dental, cv2.COLOR_BGR2RGB))
plt.title("Difference (Dental Transform)")
plt.axis('off')

plt.tight_layout()
plt.savefig("dental_transformation_results.png", dpi=300)
plt.close()

# Create a comparison of all methods
plt.figure(figsize=(20, 10))

# Homography result
plt.subplot(1, 3, 1)
plt.imshow(warped_img1)
plt.title("Homography Transform")
plt.axis('off')

# Polynomial result
plt.subplot(1, 3, 2)
plt.imshow(warped_img1_gray)
plt.title("Polynomial Transform")
plt.axis('off')

# Dental transform result
plt.subplot(1, 3, 3)
plt.imshow(warped_img1_dental)
plt.title("Dental Transform")
plt.axis('off')

plt.tight_layout()
plt.savefig("transformation_comparison.png", dpi=300)
plt.close()

print("Dental transformation complete. Results saved to:")
print("- dental_transformation_results.png")
print("- transformation_comparison.png")
