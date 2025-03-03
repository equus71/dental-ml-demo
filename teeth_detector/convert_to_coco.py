import os
import json
import numpy as np
from PIL import Image
from tooth_detection_json_dataset import decode_bitmap  # Assuming this is in the same directory


def convert_to_coco(root, image_folder="ds/img", ann_folder="ds/ann", output_json="coco_annotations.json"):
    """
    Converts teeth annotations from the custom JSON format to COCO format.

    Args:
        root: Root directory of the dataset.
        image_folder: Subfolder name for images.
        ann_folder: Subfolder name for annotations.
        output_json: Output JSON file path for COCO annotations.
    """

    image_dir = os.path.join(root, image_folder)
    ann_dir = os.path.join(root, ann_folder)
    image_files = sorted(
        [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    )

    # COCO data structure
    coco_data = {
        "info": {
            "description": "Teeth Detection Dataset (COCO Format)",
            "version": "1.0",
            "year": 2024,  # Update as needed
            "contributor": "Your Name/Organization",  # Update as needed
            "date_created": "2024/01/26",  # Update as needed
        },
        "licenses": [{"url": "", "id": 1, "name": "Unknown"}],  # Update as needed
        "categories": [{"id": 1, "name": "tooth", "supercategory": "none"}],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1  # Initialize annotation ID

    for image_id, image_filename in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, image_filename)
        ann_filename = image_filename + ".json"
        ann_path = os.path.join(ann_dir, ann_filename)

        # Load image to get dimensions
        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Add image information to COCO data
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
                "date_captured": "",  # You might want to add this if available
                "license": 1,  # Update as needed
                "coco_url": "",  # Not applicable
                "flickr_url": "",  # Not applicable
            }
        )

        # Load and process annotations
        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)
        except Exception as e:
            print(f"Error loading or parsing annotation {ann_path}: {e}")
            continue

        for obj in ann.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue  # Skip non-bitmap objects

            bitmap_info = obj.get("bitmap", {})
            data_str = bitmap_info.get("data", "")
            origin = bitmap_info.get("origin", [0, 0])

            if not data_str:
                continue

            try:
                mask = decode_bitmap(data_str)
            except Exception as e:
                print(f"Error decoding bitmap for {image_filename}: {e}")
                continue

            bin_mask = (mask > 128).astype(np.uint8)
            ys, xs = np.where(bin_mask)

            if ys.size == 0 or xs.size == 0:
                continue  # Skip empty masks

            x_min = int(np.min(xs)) + origin[0]
            x_max = int(np.max(xs)) + origin[0]
            y_min = int(np.min(ys)) + origin[1]
            y_max = int(np.max(ys)) + origin[1]

            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            if x_max <= x_min or y_max <= y_min:
                continue  # Skip invalid bboxes

            # COCO bbox format: [x_min, y_min, width, height]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = float(bbox[2] * bbox[3])

            # Create COCO annotation
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming "tooth" is category 1
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],  # Not used for bounding box detection
                    "iscrowd": 0,  # Assuming no crowd instances
                }
            )
            annotation_id += 1

    # Save COCO annotations to JSON file
    with open(os.path.join(root, output_json), "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO annotations saved to: {os.path.join(root, output_json)}")


if __name__ == "__main__":
    # Example usage:
    dataset_root = "./datasets/tooth_ds_train/"  # Replace with your dataset root
    convert_to_coco(
        root=dataset_root,
        image_folder="ds/img",
        ann_folder="ds/ann",
        output_json="coco_annotations_train.json",
    )

    dataset_root = "./datasets/tooth_ds_val/"  # Replace with your dataset root
    convert_to_coco(
        root=dataset_root,
        image_folder="ds/img",
        ann_folder="ds/ann",
        output_json="coco_annotations_val.json",
    ) 