import onnxruntime
import numpy as np
from PIL import Image
import cv2


class ObjectDetectorONNX:
    """
    A wrapper for object detection using an ONNX model exported from PaddleDetection.
    """

    RT_DETR_INPUT_SIZE = 640
    RT_DETR_INPUT_IMAGE_SIZE = "im_shape"
    RT_DETR_INPUT_IMAGE = "image"
    RT_DETR_INPUT_SCALE_FACTOR = "scale_factor"

    def __init__(self, model_path, use_gpu=False):
        """
        Initializes the ONNX runtime session.

        Args:
            model_path (str): Path to the ONNX model file.
            use_gpu (bool): Whether to use GPU for inference (if available).
        """
        self.model_path = model_path

        providers = ["CPUExecutionProvider"]  # Default to CPU
        if use_gpu:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]  # Try CUDA first, then CPU

        self.session = onnxruntime.InferenceSession(
            self.model_path, providers=providers
        )

        # Get input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Assert expected inputs are present
        expected_inputs = [
            self.RT_DETR_INPUT_IMAGE,
            self.RT_DETR_INPUT_IMAGE_SIZE,
            self.RT_DETR_INPUT_SCALE_FACTOR,
        ]
        for input_name in expected_inputs:
            assert (
                input_name in self.input_names
            ), f"Expected input '{input_name}' not found in model inputs"

    def preprocess(self, image_np):
        """
        Preprocesses the input image (numpy array).

        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, C) in BGR format.

        Returns:
            tuple: (preprocessed_image, scale_factor, original_size)
                preprocessed_image: The image data ready for the ONNX model (numpy array).
                scale_factor: Scaling factor used during resizing (for post-processing).
                original_size: (width, height) of the original image.
        """

        if not isinstance(image_np, np.ndarray):
            raise TypeError("image_np must be a NumPy array")

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("image_np must be a 3-channel image (H, W, C)")

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

        original_size = np.array([[image_np.shape[1], image_np.shape[0]]], dtype=np.float32)  # (width, height)

        resized_img = cv2.resize(
            image_np, (self.RT_DETR_INPUT_SIZE, self.RT_DETR_INPUT_SIZE)
        )
        scale_factor = np.array([[
            self.RT_DETR_INPUT_SIZE / image_np.shape[1],
            self.RT_DETR_INPUT_SIZE / image_np.shape[0],
        ]], dtype=np.float32)

        # Convert to RGB and normalize (assuming model expects RGB input in [0, 1])
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0

        # Add batch dimension and change to channel-first (NCHW)
        input_tensor = np.transpose(normalized_img, (2, 0, 1))[np.newaxis, :]

        return input_tensor, scale_factor, original_size

    def inference(self, input_tensor, scale_factor, original_size):
        """
        Runs inference on the preprocessed image.

        Args:
            input_tensor (numpy.ndarray): Preprocessed image data.

        Returns:
            list: Raw output from the ONNX model.
        """
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                self.RT_DETR_INPUT_IMAGE: input_tensor,
                self.RT_DETR_INPUT_IMAGE_SIZE: original_size.astype(np.float32),
                self.RT_DETR_INPUT_SCALE_FACTOR: scale_factor.astype(np.float32),
            },
        )
        return outputs

    def postprocess(
        self,
        outputs,
        scale_factor,
        original_size,
        score_threshold=0.5,
        iou_threshold=0.45,
    ):
        """
        Post-processes the raw output of the ONNX model.

        Args:
            outputs (list): Raw output from the ONNX model.
            scale_factor (tuple): Scaling factor used during resizing.
            original_size (tuple): (width, height) of the original image.
            score_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).

        Returns:
            list: A list of dictionaries, each representing a detected object.
                  Each dictionary contains: 'bbox' (list), 'label' (int), 'score' (float).
        """
        # Adapt this part based on your specific model's output format.
        # This example assumes the output is a list containing:
        #  - boxes: [num_boxes, 4] (x1, y1, x2, y2) in normalized coordinates
        #  - scores: [num_boxes]
        #  - labels: [num_boxes]

        # Check if outputs is empty or not
        if not outputs or len(outputs[0]) == 0:
            return []

        # Extracting the bounding boxes, scores, and labels from the output
        boxes = outputs[0]
        if len(outputs) == 3:
            # Assume format: [boxes, scores, labels]
            scores = outputs[1]
            labels = outputs[2]
        elif len(outputs) == 1 and boxes.shape[1] == 6:
            # Assume format: [x1, y1, x2, y2, score, label]
            scores = boxes[:, 4]
            labels = boxes[:, 5].astype(int)
            boxes = boxes[:, :4]
        elif len(outputs) == 2:
            scores = np.ones(len(boxes))  # dummy scores
            labels = outputs[1].flatten().astype(int)
        else:
            raise ValueError("Unexpected output format from ONNX model.")

        # Scale boxes back to original image size
        boxes[:, 0] /= scale_factor[0]
        boxes[:, 1] /= scale_factor[1]
        boxes[:, 2] /= scale_factor[0]
        boxes[:, 3] /= scale_factor[1]

        # Clip boxes to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[0])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[1])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[1])

        # Filter by score threshold
        mask = scores >= score_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]

        # Non-Maximum Suppression (NMS)
        keep_indices = self.nms(filtered_boxes, filtered_scores, iou_threshold)
        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_labels = filtered_labels[keep_indices]

        # Format results
        results = []
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            results.append(
                {
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "label": int(label),
                    "score": float(score),
                }
            )

        return results

    def nms(self, boxes, scores, iou_threshold):
        """
        Performs Non-Maximum Suppression (NMS).

        Args:
            boxes: (numpy.ndarray) [N, 4] in format [x1, y1, x2, y2]
            scores: (numpy.ndarray) [N]
            iou_threshold: (float)

        Returns:
            list: Indices of the boxes to keep after NMS.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # Sort by score in descending order

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 because we excluded order[0]

        return keep

    def detect(self, image_np, score_threshold=0.5, iou_threshold=0.45):
        """
        Combines preprocessing, inference, and postprocessing for a single image.

        Args:
            image_np (np.ndarray): Input image as a NumPy array (H, W, C) in BGR format.
            score_threshold (float): Score threshold for filtering detections.
            iou_threshold (float): IoU threshold for NMS.

        Returns:
            list: Detections (see postprocess() for format).
        """
        input_tensor, scale_factor, original_size = self.preprocess(image_np)
        outputs = self.inference(input_tensor, scale_factor, original_size)
        results = self.postprocess(
            outputs, scale_factor, original_size, score_threshold, iou_threshold
        )
        return results


# Example Usage (assuming you have an ONNX model file 'model.onnx'):
if __name__ == "__main__":
    # Replace 'path/to/your/model.onnx' with the actual path to your ONNX model
    detector = ObjectDetectorONNX(
        "./teeth_detector_onnx/rtdetrv2__r50vd_6x_teeth_sim.onnx", use_gpu=True
    )

    # Replace 'path/to/your/image.jpg' with the path to an image you want to test
    image_path = (
        "/home/gbiziel/PycharmProjects/sandbox/datasets/dental_org/Images/6.png"
    )
    image = cv2.imread(image_path)  # Load image using cv2

    # Run detection
    detections = detector.detect(image, score_threshold=0.3, iou_threshold=0.5)

    # Print the detections
    print("Detections:")
    for detection in detections:
        print(detection)

    # --- Visualization (optional) ---
    # You can use OpenCV to draw the bounding boxes on the image
    if detections:
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["label"]
            score = detection["score"]

            x1, y1, x2, y2 = [int(c) for c in bbox]  # Ensure integer coordinates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            text = f"Label: {label}, Score: {score:.2f}"
            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
