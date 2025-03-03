import argparse
import numpy as np
import vtk
from mandible_segmenter import MandibleSegmenter
from PIL import Image
import math


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


def project_to_3d(x, y, z, center_x, min_x, max_x, width, curvature_scale, curvature_factor):
    """
    Projects a 2D point (x, y, z) to 3D space with U-shaped curvature.
    """
    if x <= center_x:
        x_mapped = (center_x - x) / (center_x - min_x) if (center_x - min_x) > 0 else 0
    else:
        x_mapped = (x - center_x) / (max_x - center_x)

    curved_y = curvature_scale * (curvature_factor * x_mapped) ** 2
    return x, curved_y, z


def create_point_cloud_from_mask(mask: np.ndarray, quad_func, original_image: np.ndarray, center_x, min_x, max_x, width,
                                 z_scale: float = 1.0, curvature_scale: float = 1.0, curvature_factor: float = 1.0):
    """
    Creates a 3D point cloud, curving it into a U-shape based on the tooth line.
    Now using provided center_x, min_x, max_x for projection.
    """
    height, width = mask.shape
    points = []
    colors = []

    if len(original_image.shape) == 3:
        gray_image = np.mean(original_image, axis=2).astype(np.uint8)
    else:
        gray_image = original_image

    # Calculate the "U-shape" transformation
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                # Vertical displacement (Z-axis)
                z = z_scale * (quad_func(x) - y)

                # Project to 3D using the new function, using provided projection parameters
                curved_x, curved_y, curved_z = project_to_3d(
                    x, 0, z, center_x, min_x, max_x, width, curvature_scale, curvature_factor
                )
                points.append((curved_x, curved_y, curved_z))
                intensity = gray_image[y, x] / 255.0
                colors.append((intensity, intensity, intensity))

    # Calculate min/max for each dimension
    if points:  # Check if points list is not empty
        points_array = np.array(points)
        x_min_pc, y_min_pc, z_min_pc = np.min(points_array, axis=0)
        x_max_pc, y_max_pc, z_max_pc = np.max(points_array, axis=0)

        print("Point cloud bounds:")
        print(f"X: min={x_min_pc:.2f}, max={x_max_pc:.2f}")
        print(f"Y: min={y_min_pc:.2f}, max={y_max_pc:.2f}")
        print(f"Z: min={z_min_pc:.2f}, max={z_max_pc:.2f}")

    return points, colors


def visualize_point_cloud(points, colors, tooth_line_points_3d=None):
    """
    Create a VTK visualization. Points are rendered as a point cloud with grayscale colors.
    If tooth_line_points_3d is provided, an overlay polyline (green) is added in 3D.
    """
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

    # Create renderer and add point cloud actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    # Optionally add the tooth-line as an overlay (green polyline) in 3D
    if tooth_line_points_3d is not None:
        vtk_line_points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        n = len(tooth_line_points_3d)
        line.GetPointIds().SetNumberOfIds(n)
        for i, (x, y, z) in enumerate(tooth_line_points_3d):  # Expecting 3D points now
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
        renderer.AddActor(lineActor)

    # Setup render window and interactor
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)
    renderer.SetBackground(0.1, 0.2, 0.4)  # background color

    renderWindow.Render()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Start()


def main(image_path: str, tooth_line_offset: int = 100, z_scale: float = 1.0, curvature_scale: float = 4.0,
         curvature_factor: float = 12.0):
    # Initialize the segmenter
    segmenter = MandibleSegmenter(
        model_path="segout2/checkpoint-180",
        base_model_name="StanfordAIMI/dinov2-base-xray-224"
    )

    # Load original image for colors
    original_image = np.array(Image.open(image_path).convert('RGB'))

    # Run the segmentation pipeline
    mask = segmenter.segment_image(image_path)

    # Depending on preprocessing, ensure mask is binary 0 or 255
    if mask.max() > 1:
        binary_mask = (mask > 128).astype(np.uint8) * 255
    else:
        binary_mask = (mask > 0).astype(np.uint8) * 255

    height, width = binary_mask.shape

    # Compute top contour for each column (the first y where mask is positive)
    top_contour = get_top_contour(binary_mask)

    # Get the center point in x and its corresponding top (y) value
    center_x = width // 2
    center_y = top_contour[center_x]
    if center_y == -1:
        print("No segmentation found in the center column!")
        return

    # Extract points along the top edge (tooth-line candidates) by scanning left and right until y < center_y - offset
    tooth_line_pts = extract_tooth_line(top_contour, center_x, tooth_line_offset, center_y, width)

    # Fit a quadratic curve to the extracted points
    quad_func, coeffs = fit_quadratic(tooth_line_pts)
    print("Fitted quadratic coefficients:", coeffs)

    # Find min/max x for projection - using tooth line points
    x_coords = [p[0] for p in tooth_line_pts]
    min_x = min(x_coords)
    max_x = max(x_coords)
    center_x_tooth_line = width // 2  # or np.mean(x_coords) if tooth_line is not centered

    # Project tooth line points to 3D - using tooth line derived projection params
    tooth_line_points_3d = [
        project_to_3d(x, 0, 0, center_x_tooth_line, min_x, max_x, width, curvature_scale, curvature_factor)
        for x, y in tooth_line_pts
    ]

    # Create point cloud with curvature - using tooth line derived projection params
    points_3d, colors = create_point_cloud_from_mask(
        binary_mask,
        quad_func,
        original_image,
        center_x=center_x_tooth_line,  # Use tooth line center_x
        min_x=min_x,  # Use tooth line min_x
        max_x=max_x,  # Use tooth line max_x
        width=width,
        z_scale=z_scale,
        curvature_scale=curvature_scale,
        curvature_factor=curvature_factor,
    )

    # Visualize the points and overlay the tooth line in 3D
    visualize_point_cloud(points_3d, colors, tooth_line_points_3d=tooth_line_points_3d)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
