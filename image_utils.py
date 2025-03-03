import cv2
import numpy as np


def enhanced_image_diff(img1, img2):
    """
    Enhanced difference visualization .
    
    Args:
        img1: First grayscale image (normalized to 0-255)
        img2: Second grayscale image (normalized to 0-255)
        
    Returns:
        RGB difference visualization
    """
    # Create masks for pixels with value 0 in either image
    mask1 = (img1 == 0)
    mask2 = (img2 == 0)

    # Logical OR of the masks
    combined_mask = np.logical_or(mask1, mask2)

    # the diff
    diff_image = cv2.absdiff(img1, img2)

    # Apply the mask to the difference image
    diff_image[combined_mask] = 0

    # Normalize and colorize the difference image for better visualization
    diff_image_normalized = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
    diff_image_color = cv2.applyColorMap(diff_image_normalized, cv2.COLORMAP_JET)
    diff_image_color = cv2.cvtColor(diff_image_color, cv2.COLOR_BGR2RGB)
    return diff_image_color


def create_checkerboard_visualization(img1, img2, pattern_size=50):
    """
    Create a checkerboard visualization of two images.
    
    Args:
        img1: First image
        img2: Second image
        pattern_size: Size of checkerboard squares in pixels
        
    Returns:
        Checkerboard visualization
    """
    # Ensure images are the same size
    if img1.shape != img2.shape:
        # Resize to match the smaller dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        min_h, min_w = min(h1, h2), min(w1, w2)
        img1 = cv2.resize(img1, (min_w, min_h))
        img2 = cv2.resize(img2, (min_w, min_h))

    h, w = img1.shape[:2]
    result = np.zeros_like(img1)

    for i in range(0, h, pattern_size):
        for j in range(0, w, pattern_size):
            # Determine if this is an "even" or "odd" checkerboard square
            is_even = ((i // pattern_size) + (j // pattern_size)) % 2 == 0

            # Set the region in the result image
            y_end = min(i + pattern_size, h)
            x_end = min(j + pattern_size, w)

            if is_even:
                result[i:y_end, j:x_end] = img1[i:y_end, j:x_end]
            else:
                result[i:y_end, j:x_end] = img2[i:y_end, j:x_end]

    return result


def create_blended_visualization(img1, img2, alpha=0.5):
    """
    Create a blended visualization of two images.
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blending factor (0.0-1.0) - weight of the first image
        
    Returns:
        Blended visualization
    """
    # Ensure images are the same size
    if img1.shape != img2.shape:
        # Resize to match the smaller dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        min_h, min_w = min(h1, h2), min(w1, w2)
        img1 = cv2.resize(img1, (min_w, min_h))
        img2 = cv2.resize(img2, (min_w, min_h))

    # Blend the images
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    return blended


def normalize_image(img):
    """
    Normalize an image to 0-255 range.
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def match_image_sizes(img1, img2, border_type=cv2.BORDER_CONSTANT, border_value=0):
    """
    Ensure two images have the same dimensions by padding the smaller one.
    
    Args:
        img1: First image
        img2: Second image
        border_type: Type of border to add (default: cv2.BORDER_CONSTANT)
        border_value: Value to use for padding if using constant border
        
    Returns:
        Tuple of (img1_matched, img2_matched) with the same dimensions
    """
    # Get original dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Make copies to avoid modifying originals
    img1_matched = img1.copy()
    img2_matched = img2.copy()

    # If first image is smaller in any dimension, pad it
    if h1 < h2 or w1 < w2:
        pad_h = max(0, h2 - h1)
        pad_w = max(0, w2 - w1)
        img1_matched = cv2.copyMakeBorder(
            img1_matched, 0, pad_h, 0, pad_w,
            border_type, value=border_value
        )

    # Update dimensions after first padding
    h1, w1 = img1_matched.shape[:2]

    # If second image is smaller in any dimension, pad it
    if h2 < h1 or w2 < w1:
        pad_h = max(0, h1 - h2)
        pad_w = max(0, w1 - w2)
        img2_matched = cv2.copyMakeBorder(
            img2_matched, 0, pad_h, 0, pad_w,
            border_type, value=border_value
        )

    return img1_matched, img2_matched
