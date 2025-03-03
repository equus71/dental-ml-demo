#!/usr/bin/env python3
# dental_3d_renderer.py - 3D visualization of dental data using VTK

import cv2
import numpy as np
import vtk
from PIL import Image
from mandible_segmenter import MandibleSegmenter
from tooth_line_extractor import extract_tooth_line_from_mask


class Dental3DRenderer:
    """
    A class for 3D visualization of dental X-ray images and their analysis results.
    
    This class provides interactive visualization of:
    - 3D mandible projection based on tooth line
    - Teeth segmentation overlaid on mandible
    - Comparison between two images with proper alignment
    - Difference visualization with various colormaps
    
    The difference visualization is calculated using pixel-wise differences between
    the reference image and the warped version of the transformed image, using the
    union of masks from both images to determine the regions of interest.

    Keyboard Controls:
    -----------------
    1: Show only Image 2 (reference)
    2: Show only Image 1 (transformed)
    D: Show difference (grayscale)
    C: Show difference (colormap)
    5: Show both images aligned
    T: Toggle teeth visibility
    L: Toggle tooth line visibility
    H: Show this help
    """

    def __init__(
            self,
            img1_rgb: np.ndarray = None,
            img2_rgb: np.ndarray = None,
            img1_mandible_mask: np.ndarray = None,
            img2_mandible_mask: np.ndarray = None,
            img1_bottom_teeth_mask: np.ndarray = None,
            img2_bottom_teeth_mask: np.ndarray = None,
            affine_matrix: np.ndarray = None,
            img1_tooth_line_results: dict = None,
            img2_tooth_line_results: dict = None,
            z_scale: float = 1.0,
            curvature_scale: float = 4.0,
            curvature_factor: float = 12.0
    ) -> None:
        """
        Initialize the 3D dental renderer.
        
        Args:
            img1_rgb: RGB image 1
            img2_rgb: RGB image 2
            img1_mandible_mask: Mandible segmentation mask for image 1
            img2_mandible_mask: Mandible segmentation mask for image 2
            img1_bottom_teeth_mask: Bottom teeth segmentation mask for image 1
            img2_bottom_teeth_mask: Bottom teeth segmentation mask for image 2
            affine_matrix: Affine transformation matrix from image 1 to image 2
            img1_tooth_line_results: Tooth line extraction results for image 1
            img2_tooth_line_results: Tooth line extraction results for image 2
            z_scale: Scale factor for z-axis
            curvature_scale: Scale factor for curvature
            curvature_factor: Factor affecting the curvature shape
        """
        # Store input data
        self.img1_rgb = img1_rgb
        self.img2_rgb = img2_rgb
        self.img1_mandible_mask = img1_mandible_mask
        self.img2_mandible_mask = img2_mandible_mask
        self.img1_bottom_teeth_mask = img1_bottom_teeth_mask
        self.img2_bottom_teeth_mask = img2_bottom_teeth_mask
        self.affine_matrix = affine_matrix
        self.img1_tooth_line_results = img1_tooth_line_results
        self.img2_tooth_line_results = img2_tooth_line_results

        # Visualization parameters
        self.z_scale = z_scale
        self.curvature_scale = curvature_scale
        self.curvature_factor = curvature_factor

        # Display state
        self.display_mode = 1  # Default: show reference image (image 2)
        self.show_teeth = True
        self.show_tooth_line = False  # Default: hide tooth line

        # Processed data
        self.img1_gray = None
        self.img2_gray = None
        self.img1_warped = None
        self.img1_warped_gray = None

        # 3D point clouds
        self.img1_mandible_points_3d = None
        self.img1_mandible_colors = None
        self.img2_mandible_points_3d = None
        self.img2_mandible_colors = None

        # Bottom teeth point clouds
        self.img1_bottom_teeth_points_3d = None
        self.img1_bottom_teeth_colors = None
        self.img2_bottom_teeth_points_3d = None
        self.img2_bottom_teeth_colors = None

        # Precomputed difference point clouds
        self.mandible_diff_points = None
        self.mandible_diff_colors = None
        self.mandible_diff_colored_points = None
        self.mandible_diff_colored_colors = None

        # Precomputed teeth difference point clouds
        self.teeth_diff_points = None
        self.teeth_diff_colors = None
        self.teeth_diff_colored_points = None
        self.teeth_diff_colored_colors = None

        # Store VTK actors
        self.actors = {}

        # VTK objects
        self.renderer = None
        self.render_window = None
        self.interactor = None

        # Visualization state
        self.current_mode = 0  # 0=img2 only, 1=img1 only, 2=diff, 3=diff with colormap, 4=both aligned

        # Extract grayscale versions for 3D rendering
        if self.img1_rgb is not None:
            self.img1_gray = cv2.cvtColor(self.img1_rgb, cv2.COLOR_RGB2GRAY)
        if self.img2_rgb is not None:
            self.img2_gray = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        # Data storage for 3D point clouds
        self.img1_points_3d = None
        self.img1_colors = None
        self.img2_points_3d = None
        self.img2_colors = None
        self.img1_teeth_points_3d = None
        self.img1_teeth_colors = None
        self.img2_teeth_points_3d = None
        self.img2_teeth_colors = None

    def project_to_3d(
            self,
            x: float,
            y: float,
            z: float,
            center_x: float,
            min_x: float,
            max_x: float,
            width: int,
            curvature_scale: float,
            curvature_factor: float
    ) -> tuple[float, float, float]:
        """
        Projects a 2D point (x, y, z) to 3D space with U-shaped curvature.
        
        Args:
            x: X-coordinate in 2D space
            y: Y-coordinate in 2D space (usually 0 for the projection plane)
            z: Z-coordinate in 2D space (height from tooth line)
            center_x: X-coordinate of the center of the tooth line
            min_x: Minimum X-coordinate of the tooth line
            max_x: Maximum X-coordinate of the tooth line
            width: Width of the image
            curvature_scale: Scale factor for curvature
            curvature_factor: Factor affecting the curvature shape
            
        Returns:
            Tuple (x, y, z) of 3D coordinates
        """
        if x <= center_x:
            x_mapped = (center_x - x) / (center_x - min_x) if (center_x - min_x) > 0 else 0
        else:
            x_mapped = (x - center_x) / (max_x - center_x) if (max_x - center_x) > 0 else 0

        curved_y = curvature_scale * (curvature_factor * x_mapped) ** 2
        return x, curved_y, z

    def create_point_cloud_from_mask(
            self,
            mask: np.ndarray,
            quad_func: callable,
            image: np.ndarray,
            center_x: float,
            min_x: float,
            max_x: float,
            width: int,
            teeth_mask: np.ndarray = None,
            filter_bottom_teeth: bool = True,
            use_image_values_for_teeth: bool = True
    ) -> tuple[list, list, list, list]:
        """
        Create a 3D point cloud from a mask, curving it into a U-shape based on the tooth line.

        Parameters
        ----------
        mask : np.ndarray
            Binary segmentation mask (mandible)
        quad_func : callable
            Quadratic function fitting the tooth line
        image : np.ndarray
            Original grayscale image for colors
        center_x : float
            X-coordinate of the center of the tooth line
        min_x : float
            Minimum X-coordinate of the tooth line
        max_x : float
            Maximum X-coordinate of the tooth line
        width : int
            Width of the image
        teeth_mask : np.ndarray, optional
            Binary mask of teeth segmentation
        filter_bottom_teeth : bool, optional
            If True, only include teeth overlapping with mandible, by default True
        use_image_values_for_teeth : bool, optional
            If True, use image values for teeth coloring, by default True

        Returns
        -------
        tuple[list, list, list, list]
            - mandible_points: List of 3D points for mandible
            - mandible_colors: List of colors for mandible points
            - teeth_points: List of 3D points for teeth (or empty list)
            - teeth_colors: List of colors for teeth points (or empty list)
        """
        height, width = mask.shape
        mandible_points = []
        mandible_colors = []
        teeth_points = []
        teeth_colors = []

        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Process mandible points
        for y in range(height):
            for x in range(width):
                if binary_mask[y, x] > 0:
                    # Vertical displacement (Z-axis)
                    z = self.z_scale * (quad_func(x) - y)

                    # Project to 3D using the function
                    curved_x, curved_y, curved_z = self.project_to_3d(
                        x, 0, z, center_x, min_x, max_x, width,
                        self.curvature_scale, self.curvature_factor
                    )
                    mandible_points.append((curved_x, curved_y, curved_z))

                    # Use grayscale value for color
                    intensity = image[y, x] / 255.0
                    mandible_colors.append((intensity, intensity, intensity))

        # Process teeth points if provided
        if teeth_mask is not None:
            binary_teeth_mask = (teeth_mask > 0).astype(np.uint8)

            for y in range(height):
                for x in range(width):
                    # Only include teeth that overlap with mandible if filter_bottom_teeth is True
                    if binary_teeth_mask[y, x] > 0 and (not filter_bottom_teeth or binary_mask[y, x] > 0):
                        # Vertical displacement (Z-axis) - slightly above mandible
                        z = self.z_scale * (quad_func(x) - y) + 0.5  # Small offset to be above mandible

                        # Project to 3D using the function
                        curved_x, curved_y, curved_z = self.project_to_3d(
                            x, 0, z, center_x, min_x, max_x, width,
                            self.curvature_scale, self.curvature_factor
                        )
                        teeth_points.append((curved_x, curved_y, curved_z))

                        if use_image_values_for_teeth:
                            # Use grayscale value for teeth color
                            intensity = image[y, x] / 255.0
                            teeth_colors.append((intensity, intensity, intensity))
                        else:
                            # Use red color for teeth
                            teeth_colors.append((1.0, 0.2, 0.2))

        return mandible_points, mandible_colors, teeth_points, teeth_colors

    def apply_affine_to_points(self, points_3d, affine_matrix):
        """
        Apply the affine transformation to a list of 3D points.
        
        Args:
            points_3d: List of (x, y, z) tuples
            affine_matrix: 2x3 affine transformation matrix
            
        Returns:
            List of transformed (x, y, z) tuples
        """
        if points_3d is None or len(points_3d) == 0:
            return []

        transformed_points = []
        for x, y, z in points_3d:
            # Apply affine transformation to x, z (2D coordinates in the original image)
            # y is the projection depth, which stays constant
            pt = np.array([x, z, 1.0])
            transformed = np.dot(affine_matrix, pt)
            transformed_points.append((transformed[0], y, transformed[1]))

        return transformed_points

    def create_vtk_point_cloud(
            self,
            points: list,
            colors: list
    ) -> vtk.vtkActor:
        """
        Create a VTK point cloud from points and colors.

        Parameters
        ----------
        points : list
            List of (x, y, z) coordinates
        colors : list
            List of (r, g, b) colors in range [0, 1]

        Returns
        -------
        vtk.vtkActor or None
            VTK actor for the point cloud, or None if points list is empty

        Notes
        -----
        The points and colors lists must have the same length. Each color tuple
        should contain RGB values in the range [0, 1].
        """
        if not points:
            return None

        vtk_points = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")

        for point, color in zip(points, colors):
            vtk_points.InsertNextPoint(point)
            # Convert color from 0-1 range to 0-255 range
            vtk_colors.InsertNextTuple3(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Create vertices for the point cloud
        vertices = vtk.vtkCellArray()
        for i in range(len(points)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)

        # Add colors to polydata
        polydata.GetPointData().SetScalars(vtk_colors)

        # Setup mapper and actor for the point cloud
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)

        return actor

    def create_vtk_tooth_line(
            self,
            tooth_line_pts: list,
            tooth_line_results: dict = None,
            transformed: bool = False
    ) -> vtk.vtkActor:
        """
        Create a VTK actor for visualizing the tooth line.

        Parameters
        ----------
        tooth_line_pts : list
            List of (x, y) tooth line points
        tooth_line_results : dict, optional
            Tooth line extraction results containing parameters
        transformed : bool, optional
            Whether to transform points using the affine matrix, by default False

        Returns
        -------
        vtk.vtkActor or None
            VTK actor for the tooth line, or None if points list is empty

        Notes
        -----
        The tooth line is visualized as a green line in 3D space, positioned at
        the level of the detected tooth line.
        """
        if tooth_line_pts is None or len(tooth_line_pts) == 0:
            return None

        # Project tooth line points to 3D
        quad_func = tooth_line_results.get('quad_func') if tooth_line_results else None
        center_x = tooth_line_results.get('center_x') if tooth_line_results else (self.img2_rgb.shape[1] // 2)
        min_x = tooth_line_results.get('min_x') if tooth_line_results else min([p[0] for p in tooth_line_pts])
        max_x = tooth_line_results.get('max_x') if tooth_line_results else max([p[0] for p in tooth_line_pts])
        width = self.img2_rgb.shape[1]

        # Create 3D points for tooth line
        tooth_line_points_3d = []
        for x, y in tooth_line_pts:
            # Set z=0 for tooth line visualization (at the level of the tooth line)
            z = 0
            x3d, y3d, z3d = self.project_to_3d(
                x, 0, z, center_x, min_x, max_x, width,
                self.curvature_scale, self.curvature_factor
            )
            tooth_line_points_3d.append((x3d, y3d, z3d))

        # Apply affine transformation if needed
        if transformed and self.affine_matrix is not None:
            tooth_line_points_3d = self.apply_affine_to_points(tooth_line_points_3d, self.affine_matrix)

        # Create VTK objects for the tooth line
        vtk_line_points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        n = len(tooth_line_points_3d)
        line.GetPointIds().SetNumberOfIds(n)

        for i, (x, y, z) in enumerate(tooth_line_points_3d):
            vtk_line_points.InsertNextPoint(x, y, z)
            line.GetPointIds().SetId(i, i)

        linePolyData = vtk.vtkPolyData()
        linePolyData.SetPoints(vtk_line_points)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        linePolyData.SetLines(cells)

        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(linePolyData)
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetLineWidth(3)
        lineActor.GetProperty().SetColor(0, 1, 0)  # green

        return lineActor

    def create_difference_color_map(self, points1, points2, colors1=None, colors2=None, use_colormap=False):
        """
        Create a point cloud showing the difference between two point clouds.
        
        Args:
            points1: First set of 3D points
            points2: Second set of 3D points
            colors1: Colors for first set of points (optional)
            colors2: Colors for second set of points (optional)
            use_colormap: Whether to use a colormap for differences
            
        Returns:
            Tuple of (points, colors) for the difference visualization
        """
        # This method is now deprecated in favor of create_difference_from_images
        # We keep it for backward compatibility but it's not used anymore
        if not points1 or not points2:
            return [], []

        # Create KD-Tree for nearest neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(points2)

        diff_points = []
        diff_colors = []

        # Find nearest neighbors and calculate distances
        max_dist = 0
        distances = []

        for pt in points1:
            distance, idx = tree.query(pt)
            distances.append(distance)
            max_dist = max(max_dist, distance)

        # Normalize distances and create colors
        for i, pt in enumerate(points1):
            normalized_dist = distances[i] / max_dist if max_dist > 0 else 0

            if use_colormap:
                # Use a color map (blue to red) to visualize differences
                r = min(1.0, normalized_dist * 2.0)
                g = 0.0
                b = min(1.0, 2.0 - normalized_dist * 2.0)
            else:
                # Use brightness to visualize differences
                brightness = min(1.0, normalized_dist * 3.0)
                r = brightness
                g = brightness
                b = brightness

            diff_points.append(pt)
            diff_colors.append((r, g, b))

        return diff_points, diff_colors

    def create_difference_from_images(self, use_colormap: bool = False) -> tuple[list, list]:
        """
        Create a difference point cloud based on pixel-wise difference between images.

        Parameters
        ----------
        use_colormap : bool, optional
            Whether to use a colormap for differences, by default False

        Returns
        -------
        tuple[list, list]
            - points: List of 3D points for the difference visualization
            - colors: List of colors for the difference points

        Notes
        -----
        The difference is calculated between image2 (reference) and the warped version
        of image1. The union of masks from both images is used to determine the
        regions of interest.
        """
        if self.img2_gray is None or self.img1_warped_gray is None:
            return [], []

        # Create union of masks for mandible
        if self.img2_mandible_mask is not None and self.img1_mandible_mask_warped is not None:
            mandible_union_mask = cv2.bitwise_or(self.img2_mandible_mask, self.img1_mandible_mask_warped)
        else:
            return [], []

        # Create union of masks for teeth if available
        if self.img2_bottom_teeth_mask is not None and self.img1_bottom_teeth_mask_warped is not None:
            teeth_union_mask = cv2.bitwise_or(self.img2_bottom_teeth_mask, self.img1_bottom_teeth_mask_warped)
        else:
            teeth_union_mask = None

        # Calculate pixel-wise difference
        diff_image = cv2.absdiff(self.img2_gray, self.img1_warped_gray)

        # Normalize difference for better visualization
        max_diff = np.max(diff_image)
        if max_diff > 0:
            normalized_diff = diff_image.astype(np.float32) / max_diff
        else:
            normalized_diff = diff_image.astype(np.float32)

        # Apply colormap if requested
        if use_colormap:
            # Create a colormap (blue to red)
            colormap = np.zeros((diff_image.shape[0], diff_image.shape[1], 3), dtype=np.float32)
            colormap[:, :, 0] = np.minimum(1.0, normalized_diff * 2.0)  # Red
            colormap[:, :, 1] = 0.0  # Green
            colormap[:, :, 2] = np.minimum(1.0, 2.0 - normalized_diff * 2.0)  # Blue
        else:
            # Use grayscale for difference
            colormap = np.zeros((diff_image.shape[0], diff_image.shape[1], 3), dtype=np.float32)
            colormap[:, :, 0] = normalized_diff  # Red
            colormap[:, :, 1] = normalized_diff  # Green
            colormap[:, :, 2] = normalized_diff  # Blue

        # Extract tooth line parameters from image2
        if self.img2_tooth_line_results is None:
            return [], []

        center_x = self.img2_tooth_line_results['center_x']
        min_x = self.img2_tooth_line_results['min_x']
        max_x = self.img2_tooth_line_results['max_x']
        quad_func = self.img2_tooth_line_results['quad_func']

        # Create 3D points for mandible difference
        height, width = mandible_union_mask.shape
        diff_points = []
        diff_colors = []

        # Process mandible points
        for y in range(height):
            for x in range(width):
                if mandible_union_mask[y, x] > 0:
                    # Vertical displacement (Z-axis)
                    z = self.z_scale * (quad_func(x) - y)

                    # Project to 3D using the function
                    curved_x, curved_y, curved_z = self.project_to_3d(
                        x, 0, z, center_x, min_x, max_x, width,
                        self.curvature_scale, self.curvature_factor
                    )
                    diff_points.append((curved_x, curved_y, curved_z))

                    # Use color from colormap
                    diff_colors.append((
                        colormap[y, x, 0],
                        colormap[y, x, 1],
                        colormap[y, x, 2]
                    ))

        return diff_points, diff_colors

    def create_teeth_difference_from_images(self, use_colormap: bool = False) -> tuple[list, list]:
        """
        Create a difference point cloud for teeth based on pixel-wise difference between images.

        Parameters
        ----------
        use_colormap : bool, optional
            Whether to use a colormap for differences, by default False

        Returns
        -------
        tuple[list, list]
            - points: List of 3D points for the teeth difference visualization
            - colors: List of colors for the teeth difference points

        Notes
        -----
        The difference is calculated between image2 (reference) and the warped version
        of image1. The union of teeth masks from both images is used to determine the
        regions of interest.
        """
        if (self.img2_gray is None or self.img1_warped_gray is None or
                self.img2_bottom_teeth_mask is None or self.img1_bottom_teeth_mask_warped is None):
            return [], []

        # Create union of masks for teeth
        teeth_union_mask = cv2.bitwise_or(self.img2_bottom_teeth_mask, self.img1_bottom_teeth_mask_warped)

        # Calculate pixel-wise difference
        diff_image = cv2.absdiff(self.img2_gray, self.img1_warped_gray)

        # Normalize difference for better visualization
        max_diff = np.max(diff_image)
        if max_diff > 0:
            normalized_diff = diff_image.astype(np.float32) / max_diff
        else:
            normalized_diff = diff_image.astype(np.float32)

        # Apply colormap if requested
        if use_colormap:
            # Create a colormap (blue to red)
            colormap = np.zeros((diff_image.shape[0], diff_image.shape[1], 3), dtype=np.float32)
            colormap[:, :, 0] = np.minimum(1.0, normalized_diff * 2.0)  # Red
            colormap[:, :, 1] = 0.0  # Green
            colormap[:, :, 2] = np.minimum(1.0, 2.0 - normalized_diff * 2.0)  # Blue
        else:
            # Use grayscale for difference
            colormap = np.zeros((diff_image.shape[0], diff_image.shape[1], 3), dtype=np.float32)
            colormap[:, :, 0] = normalized_diff  # Red
            colormap[:, :, 1] = normalized_diff  # Green
            colormap[:, :, 2] = normalized_diff  # Blue

        # Extract tooth line parameters from image2
        if self.img2_tooth_line_results is None:
            return [], []

        center_x = self.img2_tooth_line_results['center_x']
        min_x = self.img2_tooth_line_results['min_x']
        max_x = self.img2_tooth_line_results['max_x']
        quad_func = self.img2_tooth_line_results['quad_func']

        # Create 3D points for teeth difference
        height, width = teeth_union_mask.shape
        teeth_diff_points = []
        teeth_diff_colors = []

        # Process teeth points
        for y in range(height):
            for x in range(width):
                if teeth_union_mask[y, x] > 0:
                    # Vertical displacement (Z-axis) - slightly above mandible
                    z = self.z_scale * (quad_func(x) - y) + 0.5  # Small offset to be above mandible

                    # Project to 3D using the function
                    curved_x, curved_y, curved_z = self.project_to_3d(
                        x, 0, z, center_x, min_x, max_x, width,
                        self.curvature_scale, self.curvature_factor
                    )
                    teeth_diff_points.append((curved_x, curved_y, curved_z))

                    # Use color from colormap
                    teeth_diff_colors.append((
                        colormap[y, x, 0],
                        colormap[y, x, 1],
                        colormap[y, x, 2]
                    ))

        return teeth_diff_points, teeth_diff_colors

    def prepare_data(self) -> bool:
        """
        Prepare data for visualization by processing images and creating point clouds.

        This method performs several key operations:
        1. Converts RGB images to grayscale
        2. Applies affine transformation to align image1 with image2
        3. Creates 3D point clouds for mandibles and teeth
        4. Precomputes difference point clouds for faster visualization

        Returns
        -------
        bool
            True if data preparation was successful, False otherwise
        """
        # Convert RGB images to grayscale
        if self.img1_rgb is not None:
            self.img1_gray = cv2.cvtColor(self.img1_rgb, cv2.COLOR_RGB2GRAY)
        if self.img2_rgb is not None:
            self.img2_gray = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        # Apply affine transformation to align img1 with img2
        if self.img1_rgb is not None and self.affine_matrix is not None:
            h, w = self.img2_rgb.shape[:2]
            self.img1_warped = cv2.warpAffine(self.img1_rgb, self.affine_matrix, (w, h))
            self.img1_warped_gray = cv2.cvtColor(self.img1_warped, cv2.COLOR_RGB2GRAY)

            # Also warp the masks to align with image2
            if self.img1_mandible_mask is not None:
                self.img1_mandible_mask_warped = cv2.warpAffine(self.img1_mandible_mask, self.affine_matrix, (w, h))
            else:
                self.img1_mandible_mask_warped = None

            if self.img1_bottom_teeth_mask is not None:
                self.img1_bottom_teeth_mask_warped = cv2.warpAffine(self.img1_bottom_teeth_mask, self.affine_matrix,
                                                                    (w, h))
            else:
                self.img1_bottom_teeth_mask_warped = None

        # Create 3D point clouds for image 2 (reference)
        if (self.img2_mandible_mask is not None and
                self.img2_tooth_line_results is not None and
                self.img2_gray is not None):

            # Extract tooth line parameters
            center_x = self.img2_tooth_line_results['center_x']
            min_x = self.img2_tooth_line_results['min_x']
            max_x = self.img2_tooth_line_results['max_x']
            quad_func = self.img2_tooth_line_results['quad_func']

            # Create point cloud for mandible
            (
                self.img2_mandible_points_3d,
                self.img2_mandible_colors,
                _,  # Unused teeth points
                _  # Unused teeth colors
            ) = self.create_point_cloud_from_mask(
                self.img2_mandible_mask,
                quad_func,
                self.img2_gray,
                center_x,
                min_x,
                max_x,
                self.img2_gray.shape[1],
                teeth_mask=None,
                filter_bottom_teeth=False
            )

            # Create point cloud for bottom teeth if available
            if self.img2_bottom_teeth_mask is not None:
                (
                    _,  # Unused mandible points
                    _,  # Unused mandible colors
                    self.img2_bottom_teeth_points_3d,
                    self.img2_bottom_teeth_colors
                ) = self.create_point_cloud_from_mask(
                    self.img2_mandible_mask,
                    quad_func,
                    self.img2_gray,
                    center_x,
                    min_x,
                    max_x,
                    self.img2_gray.shape[1],
                    teeth_mask=self.img2_bottom_teeth_mask,
                    filter_bottom_teeth=False,  # Already filtered
                    use_image_values_for_teeth=True  # Use image values for teeth coloring
                )

        # Create 3D point clouds for image 1 (transformed)
        # Use the warped masks and image2's tooth line parameters for proper alignment
        if self.img1_warped_gray is not None and self.img1_mandible_mask_warped is not None and self.img2_tooth_line_results is not None:
            # Use image2's tooth line parameters for 3D projection
            center_x = self.img2_tooth_line_results['center_x']
            min_x = self.img2_tooth_line_results['min_x']
            max_x = self.img2_tooth_line_results['max_x']
            quad_func = self.img2_tooth_line_results['quad_func']

            # Create point cloud for mandible
            (
                self.img1_mandible_points_3d,
                self.img1_mandible_colors,
                _,  # Unused teeth points
                _  # Unused teeth colors
            ) = self.create_point_cloud_from_mask(
                self.img1_mandible_mask_warped,
                quad_func,
                self.img1_warped_gray,
                center_x,
                min_x,
                max_x,
                self.img1_warped_gray.shape[1],
                teeth_mask=None,
                filter_bottom_teeth=False
            )

            # Create point cloud for bottom teeth if available
            if self.img1_bottom_teeth_mask_warped is not None:
                (
                    _,  # Unused mandible points
                    _,  # Unused mandible colors
                    self.img1_bottom_teeth_points_3d,
                    self.img1_bottom_teeth_colors
                ) = self.create_point_cloud_from_mask(
                    self.img1_mandible_mask_warped,
                    quad_func,
                    self.img1_warped_gray,
                    center_x,
                    min_x,
                    max_x,
                    self.img1_warped_gray.shape[1],
                    teeth_mask=self.img1_bottom_teeth_mask_warped,
                    filter_bottom_teeth=False,  # Already filtered
                    use_image_values_for_teeth=True  # Use image values for teeth coloring
                )

        # Precompute difference point clouds for faster switching
        # Use the new image-based difference calculation methods

        # Grayscale difference for mandible
        self.mandible_diff_points, self.mandible_diff_colors = self.create_difference_from_images(
            use_colormap=False
        )

        # Colored difference for mandible
        self.mandible_diff_colored_points, self.mandible_diff_colored_colors = self.create_difference_from_images(
            use_colormap=True
        )

        # Grayscale difference for teeth
        self.teeth_diff_points, self.teeth_diff_colors = self.create_teeth_difference_from_images(
            use_colormap=False
        )

        # Colored difference for teeth
        self.teeth_diff_colored_points, self.teeth_diff_colored_colors = self.create_teeth_difference_from_images(
            use_colormap=True
        )

        return True

    def keypress_callback(self, obj: vtk.vtkRenderWindowInteractor, event: str) -> None:
        """
        Handle keyboard events for changing display modes.

        Parameters
        ----------
        obj : vtk.vtkRenderWindowInteractor
            The VTK interactor object that triggered the event
        event : str
            The event type (should be "KeyPressEvent")

        Notes
        -----
        Keyboard controls:
        - 1: Show only image 2 (reference)
        - 2: Show only image 1 (transformed)
        - D: Show difference (grayscale)
        - C: Show difference (colormap)
        - 5: Show both images aligned
        - T: Toggle teeth visibility
        - L: Toggle tooth line visibility
        - H: Show help
        """
        key = obj.GetKeySym()

        if key == "1":
            # Mode 1: Show only image 2 (reference)
            self.display_mode = 1
            self.update_display()
            print("Mode: Showing Image 2 (reference)")

        elif key == "2":
            # Mode 2: Show only image 1 (transformed)
            self.display_mode = 2
            self.update_display()
            print("Mode: Showing Image 1 (transformed)")

        elif key == "d" or key == "D":
            # Difference mode (grayscale)
            self.display_mode = 3
            self.update_display()
            print("Mode: Showing Difference (grayscale)")

        elif key == "c" or key == "C":
            # Colored difference mode (colormap)
            self.display_mode = 4
            self.update_display()
            print("Mode: Showing Difference (colormap)")

        elif key == "5":
            # Mode 5: Show both images aligned
            self.display_mode = 5
            self.update_display()
            print("Mode: Showing Both Images Aligned")

        elif key == "t" or key == "T":
            # Toggle teeth visibility
            self.show_teeth = not self.show_teeth
            self.update_display()
            if self.show_teeth:
                print("Teeth visible")
            else:
                print("Teeth hidden")

        elif key == "l" or key == "L":
            # Toggle tooth line visibility
            self.show_tooth_line = not self.show_tooth_line
            self.update_display()
            if self.show_tooth_line:
                print("Tooth line visible")
            else:
                print("Tooth line hidden")

        elif key == "h" or key == "H":
            # Show help
            self.show_help()

    def show_help(self) -> None:
        """
        Display help information about keyboard controls and visualization colors.

        This method prints a formatted help text to the console that includes:
        - Available keyboard controls
        - Description of visualization colors in different modes
        """
        help_text = """
        Keyboard Controls:
        -----------------
        1: Show only Image 2 (reference)
        2: Show only Image 1 (transformed)
        D: Show difference (grayscale)
        C: Show difference (colormap)
        5: Show both images aligned
        T: Toggle teeth visibility
        L: Toggle tooth line visibility
        H: Show this help
        
        Visualization Colors:
        -------------------
        - Teeth are shown with their actual grayscale values from the X-ray
        - In difference mode (D), brighter areas indicate larger differences
        - In colored difference mode (C), blue to red spectrum shows increasing differences
        - In aligned mode (5), Image 1 is shown in red tones, Image 2 in blue tones
        """
        print(help_text)

    def update_display(self) -> None:
        """
        Update the display based on current mode and settings.

        This method handles the visualization update based on the current display mode:
        - Mode 1: Show only Image 2 (reference)
        - Mode 2: Show only Image 1 (transformed)
        - Mode 3: Show difference (grayscale)
        - Mode 4: Show difference (colormap)
        - Mode 5: Show both images aligned

        The method also respects the current state of teeth and tooth line visibility
        toggles.
        """
        # Remove all actors
        for actor in self.renderer.GetActors():
            self.renderer.RemoveActor(actor)

        if self.display_mode == 1:
            # Show only image 2 (reference)
            if self.img2_mandible_points_3d:
                actor = self.create_vtk_point_cloud(self.img2_mandible_points_3d, self.img2_mandible_colors)
                self.renderer.AddActor(actor)

                # Add tooth line for reference if enabled
                if self.show_tooth_line and self.img2_tooth_line_results:
                    line_actor = self.create_vtk_tooth_line(
                        self.img2_tooth_line_results['points'],
                        self.img2_tooth_line_results
                    )
                    self.renderer.AddActor(line_actor)

            # Add teeth if enabled
            if self.show_teeth:
                # Add bottom teeth
                if self.img2_bottom_teeth_points_3d:
                    bottom_teeth_actor = self.create_vtk_point_cloud(
                        self.img2_bottom_teeth_points_3d,
                        self.img2_bottom_teeth_colors
                    )
                    self.renderer.AddActor(bottom_teeth_actor)

        elif self.display_mode == 2:
            # Show only image 1 (transformed)
            if self.img1_mandible_points_3d:
                actor = self.create_vtk_point_cloud(self.img1_mandible_points_3d, self.img1_mandible_colors)
                self.renderer.AddActor(actor)

                # Add tooth line for reference if enabled (using image2's tooth line since we're in aligned space)
                if self.show_tooth_line and self.img2_tooth_line_results:
                    line_actor = self.create_vtk_tooth_line(
                        self.img2_tooth_line_results['points'],
                        self.img2_tooth_line_results
                    )
                    self.renderer.AddActor(line_actor)

            # Add teeth if enabled
            if self.show_teeth:
                # Add bottom teeth
                if self.img1_bottom_teeth_points_3d:
                    bottom_teeth_actor = self.create_vtk_point_cloud(
                        self.img1_bottom_teeth_points_3d,
                        self.img1_bottom_teeth_colors
                    )
                    self.renderer.AddActor(bottom_teeth_actor)

        elif self.display_mode == 3:
            # Show difference (grayscale) - using precomputed difference
            if self.mandible_diff_points:
                actor = self.create_vtk_point_cloud(self.mandible_diff_points, self.mandible_diff_colors)
                self.renderer.AddActor(actor)

            # Add tooth line for reference if enabled (using image2's tooth line since we're in aligned space)
            if self.show_tooth_line and self.img2_tooth_line_results:
                line_actor = self.create_vtk_tooth_line(
                    self.img2_tooth_line_results['points'],
                    self.img2_tooth_line_results
                )
                self.renderer.AddActor(line_actor)

            # Add teeth difference if enabled
            if self.show_teeth and self.teeth_diff_points:
                bottom_teeth_actor = self.create_vtk_point_cloud(
                    self.teeth_diff_points,
                    self.teeth_diff_colors
                )
                self.renderer.AddActor(bottom_teeth_actor)

        elif self.display_mode == 4:
            # Show difference (colormap) - using precomputed colored difference
            if self.mandible_diff_colored_points:
                actor = self.create_vtk_point_cloud(self.mandible_diff_colored_points,
                                                    self.mandible_diff_colored_colors)
                self.renderer.AddActor(actor)

            # Add tooth line for reference if enabled (using image2's tooth line since we're in aligned space)
            if self.show_tooth_line and self.img2_tooth_line_results:
                line_actor = self.create_vtk_tooth_line(
                    self.img2_tooth_line_results['points'],
                    self.img2_tooth_line_results
                )
                self.renderer.AddActor(line_actor)

            # Add teeth difference if enabled
            if self.show_teeth and self.teeth_diff_colored_points:
                bottom_teeth_actor = self.create_vtk_point_cloud(
                    self.teeth_diff_colored_points,
                    self.teeth_diff_colored_colors
                )
                self.renderer.AddActor(bottom_teeth_actor)

        elif self.display_mode == 5:
            # Show both images aligned
            # Add image 2 (reference) with blue tint
            if self.img2_mandible_points_3d:
                # Create blue-tinted colors
                blue_colors = [(c[0] * 0.5, c[1] * 0.5, c[2] * 0.8) for c in self.img2_mandible_colors]
                actor = self.create_vtk_point_cloud(self.img2_mandible_points_3d, blue_colors)
                self.renderer.AddActor(actor)

            # Add image 1 (transformed) with red tint
            if self.img1_mandible_points_3d:
                # Create red-tinted colors
                red_colors = [(c[0] * 0.8, c[1] * 0.5, c[2] * 0.5) for c in self.img1_mandible_colors]
                actor = self.create_vtk_point_cloud(self.img1_mandible_points_3d, red_colors)
                self.renderer.AddActor(actor)

            # Add tooth line for reference if enabled (using image2's tooth line since we're in aligned space)
            if self.show_tooth_line and self.img2_tooth_line_results:
                line_actor = self.create_vtk_tooth_line(
                    self.img2_tooth_line_results['points'],
                    self.img2_tooth_line_results
                )
                self.renderer.AddActor(line_actor)

            # Add teeth if enabled
            if self.show_teeth:
                # Add image 2 bottom teeth with blue tint
                if self.img2_bottom_teeth_points_3d:
                    blue_bottom_teeth_colors = [(c[0] * 0.5, c[1] * 0.5, c[2] * 0.8) for c in
                                                self.img2_bottom_teeth_colors]
                    bottom_teeth_actor = self.create_vtk_point_cloud(
                        self.img2_bottom_teeth_points_3d,
                        blue_bottom_teeth_colors
                    )
                    self.renderer.AddActor(bottom_teeth_actor)

                # Add image 1 bottom teeth with red tint
                if self.img1_bottom_teeth_points_3d:
                    red_bottom_teeth_colors = [(c[0] * 0.8, c[1] * 0.5, c[2] * 0.5) for c in
                                               self.img1_bottom_teeth_colors]
                    bottom_teeth_actor = self.create_vtk_point_cloud(
                        self.img1_bottom_teeth_points_3d,
                        red_bottom_teeth_colors
                    )
                    self.renderer.AddActor(bottom_teeth_actor)

        # Render the scene
        self.render_window.Render()

    def create_visualization(self) -> bool:
        """
        Create and display the 3D visualization.

        This method initializes the VTK visualization pipeline and starts the
        interactive display. It performs the following steps:
        1. Prepares the data for visualization
        2. Creates VTK renderer and render window
        3. Sets up the interaction callbacks
        4. Displays initial help text
        5. Starts the interactive visualization

        Returns
        -------
        bool
            True if visualization was created successfully, False otherwise
        """
        # Prepare data
        if not self.prepare_data():
            print("Failed to prepare data for visualization.")
            return False

        # Create VTK renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1024, 768)
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # Dark blue background

        # Create interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # Add keyboard callback
        self.interactor.AddObserver("KeyPressEvent", self.keypress_callback)

        # Initial display mode
        self.update_display()

        # Add text with instructions
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("Press 'H' for help on keyboard controls")
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(text_actor)

        # Display help at startup
        self.show_help()

        # Initialize and start the interactor
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

        return True


def main(
        img1_path: str,
        img2_path: str,
        mandible_model_path: str = "./models/mandible-segmentator-dinov2",
        base_model_name: str = "StanfordAIMI/dinov2-base-xray-224",
        tooth_line_offset: int = 100,
        z_scale: float = 1.0,
        curvature_scale: float = 4.0,
        curvature_factor: float = 12.0
) -> None:
    """
    Run the 3D dental visualization directly from command line.

    This function loads dental X-ray images, performs segmentation and analysis,
    and creates an interactive 3D visualization.

    Parameters
    ----------
    img1_path : str
        Path to first dental X-ray image
    img2_path : str
        Path to second dental X-ray image (reference)
    mandible_model_path : str, optional
        Path to mandible segmentation model, by default "./models/mandible-segmentator-dinov2"
    base_model_name : str, optional
        Base model name for segmentation, by default "StanfordAIMI/dinov2-base-xray-224"
    tooth_line_offset : int, optional
        Maximum vertical distance to consider for tooth line, by default 100
    z_scale : float, optional
        Scale factor for z-axis (vertical displacement), by default 1.0
    curvature_scale : float, optional
        Scale factor for curvature, by default 4.0
    curvature_factor : float, optional
        Factor affecting the curvature shape, by default 12.0

    Notes
    -----
    The function performs the following steps:
    1. Loads and processes the input X-ray images
    2. Segments mandibles using the provided model
    3. Extracts tooth lines and creates teeth masks
    4. Calculates the affine transformation between images
    5. Creates and displays the interactive 3D visualization
    """
    # Load images
    img1_rgb = np.array(Image.open(img1_path).convert('RGB'))
    img2_rgb = np.array(Image.open(img2_path).convert('RGB'))

    # Initialize mandible segmenter
    segmenter = MandibleSegmenter(
        model_path=mandible_model_path,
        base_model_name=base_model_name
    )

    # Segment mandibles
    img1_mandible_mask = segmenter.segment_image(img1_path)
    img2_mandible_mask = segmenter.segment_image(img2_path)

    # Extract tooth lines
    img1_tooth_line_results = {}
    (
        img1_tooth_line_results['points'],
        img1_tooth_line_results['quad_func'],
        img1_tooth_line_results['coeffs'],
        img1_tooth_line_results['center_x'],
        img1_tooth_line_results['min_x'],
        img1_tooth_line_results['max_x']
    ) = extract_tooth_line_from_mask(img1_mandible_mask, tooth_line_offset)

    img2_tooth_line_results = {}
    (
        img2_tooth_line_results['points'],
        img2_tooth_line_results['quad_func'],
        img2_tooth_line_results['coeffs'],
        img2_tooth_line_results['center_x'],
        img2_tooth_line_results['min_x'],
        img2_tooth_line_results['max_x']
    ) = extract_tooth_line_from_mask(img2_mandible_mask, tooth_line_offset)

    # Extract bottom teeth masks using adaptive thresholding
    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    img1_teeth_binary = cv2.adaptiveThreshold(
        img1_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
    )
    img1_bottom_teeth_mask = cv2.bitwise_and(img1_teeth_binary, img1_teeth_binary, mask=img1_mandible_mask)

    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    img2_teeth_binary = cv2.adaptiveThreshold(
        img2_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
    )
    img2_bottom_teeth_mask = cv2.bitwise_and(img2_teeth_binary, img2_teeth_binary, mask=img2_mandible_mask)

    # Calculate affine transformation from image1 to image2
    if len(img1_tooth_line_results['points']) >= 3 and len(img2_tooth_line_results['points']) >= 3:
        # Use three points from each tooth line for affine transformation
        src_pts = np.float32([img1_tooth_line_results['points'][0],
                              img1_tooth_line_results['points'][len(img1_tooth_line_results['points']) // 2],
                              img1_tooth_line_results['points'][-1]])
        dst_pts = np.float32([img2_tooth_line_results['points'][0],
                              img2_tooth_line_results['points'][len(img2_tooth_line_results['points']) // 2],
                              img2_tooth_line_results['points'][-1]])

        affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    else:
        # Use identity transformation if not enough points
        affine_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    # Create and run the visualization
    renderer = Dental3DRenderer(
        img1_rgb=img1_rgb,
        img2_rgb=img2_rgb,
        img1_mandible_mask=img1_mandible_mask,
        img2_mandible_mask=img2_mandible_mask,
        img1_bottom_teeth_mask=img1_bottom_teeth_mask,
        img2_bottom_teeth_mask=img2_bottom_teeth_mask,
        img1_tooth_line_results=img1_tooth_line_results,
        img2_tooth_line_results=img2_tooth_line_results,
        affine_matrix=affine_matrix,
        z_scale=z_scale,
        curvature_scale=curvature_scale,
        curvature_factor=curvature_factor
    )

    renderer.create_visualization()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
