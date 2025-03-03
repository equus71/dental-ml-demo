#!/usr/bin/env python3
# tooth_line_extractor.py - Extract tooth line from mandible segmentation

import numpy as np
import matplotlib.pyplot as plt
from mandible_segmenter import MandibleSegmenter
from PIL import Image


def get_top_contour(mask: np.ndarray) -> np.ndarray:
    """
    For each column in the mask, find the first (topmost) pixel that belongs to
    the mask (assumed nonzero). Returns an array of shape (width,) where each
    element is the y-coordinate of the topmost mask pixel for that column
    (or -1 if none found).
    """
    height, width = mask.shape
    top_contour = np.full(width, fill_value=-1, dtype=int)
    for x in range(width):
        for y in range(height):
            if mask[y, x] > 0:
                top_contour[x] = y
                break
    return top_contour


def extract_tooth_line(top_contour: np.ndarray, center_x: int, tooth_line_offset: int, center_y: int, width: int):
    """
    Starting from the center column (center_x) with y-coordinate center_y,
    scan the top_contour leftwards and rightwards, collecting points until the
    y coordinate of the point is at least `tooth_line_offset` pixels above center_y.
    
    Args:
        top_contour: Array of y-coordinates representing the top edge of the mask
        center_x: X-coordinate of the center point to start from
        tooth_line_offset: Maximum vertical distance from center_y to consider
        center_y: Y-coordinate of the center point
        width: Width of the image
        
    Returns:
        Array of (x, y) points representing the tooth line
    """
    left_points = []
    # Scan to the left (decreasing x)
    for x in range(center_x, -1, -1):
        y = top_contour[x]
        if y == -1:
            continue
        # Stop if point is more than tooth_line_offset above the center point
        if y < center_y - tooth_line_offset:
            break
        left_points.append((x, y))

    right_points = []
    # Scan to the right (increasing x)
    for x in range(center_x + 1, width):
        y = top_contour[x]
        if y == -1:
            continue
        if y < center_y - tooth_line_offset:
            break
        right_points.append((x, y))

    # Combine points ensuring the x order is increasing
    points = left_points[::-1] + [(center_x, center_y)] + right_points
    points = np.array(points)
    if points.shape[0] < 3:
        raise ValueError("Not enough points to fit a quadratic curve.")
    return points


def fit_quadratic(points: np.ndarray):
    """
    Fit a quadratic function to the given points (x,y) using np.polyfit.
    Returns:
      - quad_func: a callable function f(x) = ax^2 + bx + c
      - coeffs: the coefficients [a, b, c]
    """
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    coeffs = np.polyfit(x_vals, y_vals, 2)
    quad_func = np.poly1d(coeffs)
    return quad_func, coeffs


def extract_tooth_line_from_mask(mask: np.ndarray, tooth_line_offset: int = 100):
    """
    Extract the tooth line from a mandible segmentation mask.
    
    Args:
        mask: Binary mask of the mandible segmentation
        tooth_line_offset: Maximum vertical distance to consider for tooth line
        
    Returns:
        tooth_line_pts: Array of (x, y) points representing the tooth line
        quad_func: Fitted quadratic function 
        coeffs: Coefficients of the quadratic function
        center_x: X-coordinate of the center point
        min_x: Minimum X-coordinate of the tooth line
        max_x: Maximum X-coordinate of the tooth line
    """
    # Ensure mask is binary
    if mask.max() > 1:
        binary_mask = (mask > 128).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    height, width = binary_mask.shape

    # Compute top contour for each column
    top_contour = get_top_contour(binary_mask)

    # Get the center point in x and its corresponding top (y) value
    center_x = width // 2
    center_y = top_contour[center_x]
    if center_y == -1:
        raise ValueError("No segmentation found in the center column!")

    # Extract points along the top edge
    tooth_line_pts = extract_tooth_line(top_contour, center_x, tooth_line_offset, center_y, width)

    # Fit a quadratic curve to the extracted points
    quad_func, coeffs = fit_quadratic(tooth_line_pts)

    # Find min/max x for projection
    x_coords = [p[0] for p in tooth_line_pts]
    min_x = min(x_coords)
    max_x = max(x_coords)

    return tooth_line_pts, quad_func, coeffs, center_x, min_x, max_x


def visualize_tooth_line(image, mask, tooth_line_pts, quad_func, output_path=None):
    """
    Visualize the extracted tooth line on the original image and segmentation mask.
    
    Args:
        image: Original image
        mask: Segmentation mask
        tooth_line_pts: Points representing the tooth line
        quad_func: Fitted quadratic function
        output_path: Path to save the visualization (if None, displays it instead)
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Display original image with tooth line overlay
    axes[0].imshow(image)
    axes[0].set_title("Original Image with Tooth Line")

    # Display mask with tooth line overlay
    if len(mask.shape) == 2:
        # Create RGB mask for visualization
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb_mask[mask > 0] = [0, 0, 255]  # Blue mask
        axes[1].imshow(rgb_mask)
    else:
        axes[1].imshow(mask)
    axes[1].set_title("Segmentation Mask with Tooth Line")

    # Plot tooth line points on both subplots
    for ax in axes:
        # Plot extracted points
        tooth_line_x = [p[0] for p in tooth_line_pts]
        tooth_line_y = [p[1] for p in tooth_line_pts]
        ax.scatter(tooth_line_x, tooth_line_y, color='green', s=5, label='Extracted Points')

        # Plot fitted curve
        x_range = np.arange(min(tooth_line_x), max(tooth_line_x) + 1)
        y_range = quad_func(x_range)
        ax.plot(x_range, y_range, color='red', linewidth=2, label='Fitted Curve')

        ax.legend()
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main(image_path, output_dir=None, tooth_line_offset=100, visualize=True):
    """
    Extract tooth line from mandible segmentation and visualize the results.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs (if None, current directory is used)
        tooth_line_offset: Maximum vertical distance to consider for tooth line
        visualize: Whether to generate visualization
    """
    import os

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(image_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    original_image = np.array(Image.open(image_path).convert('RGB'))

    # Initialize the mandible segmenter
    print("Initializing mandible segmenter...")
    segmenter = MandibleSegmenter(
        model_path="./models/mandible-segmentator-dinov2",
        base_model_name="StanfordAIMI/dinov2-base-xray-224"
    )

    # Generate mandible segmentation
    print("Performing mandible segmentation...")
    mask = segmenter.segment_image(image_path)

    # Extract tooth line
    print("Extracting tooth line...")
    tooth_line_pts, quad_func, coeffs, center_x, min_x, max_x = extract_tooth_line_from_mask(
        mask, tooth_line_offset
    )

    print(f"Tooth line extracted with {len(tooth_line_pts)} points")
    print(f"Quadratic coefficients: {coeffs}")
    print(f"Tooth line spans from x={min_x} to x={max_x}")

    # Save tooth line data
    output_filename = os.path.splitext(os.path.basename(image_path))[0]
    np.savez(
        os.path.join(output_dir, f"{output_filename}_tooth_line.npz"),
        points=tooth_line_pts,
        coefficients=coeffs,
        center_x=center_x,
        min_x=min_x,
        max_x=max_x
    )

    # Visualize results if requested
    if visualize:
        print("Generating visualization...")
        vis_path = os.path.join(output_dir, f"{output_filename}_tooth_line_visualization.png")
        visualize_tooth_line(original_image, mask, tooth_line_pts, quad_func, vis_path)

    print("Tooth line extraction complete!")
    return {
        "tooth_line_points": tooth_line_pts,
        "quadratic_function": quad_func,
        "coefficients": coeffs,
        "center_x": center_x,
        "min_x": min_x,
        "max_x": max_x
    }


if __name__ == "__main__":
    import fire

    fire.Fire(main)
