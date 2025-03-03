import os
import yaml
import numpy as np
import cv2
from PIL import Image
from teeth_detector_paddle.ppdetdeploy.infer import Detector, create_inputs, visualize_box_mask


class TeethDetector:
    def __init__(
        self,
        model_dir="./teeth_detector_paddle/rtdetrv2_r50vd_6x_teeths_exp",
        device="CPU",
        threshold=0.5,
        output_dir="test_output/",
    ):
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold
        self.output_dir = output_dir
        
        # Initialize detector
        self.detector = Detector(
            model_dir,
            device=device,
            run_mode="paddle",
            batch_size=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            cpu_threads=1,
            enable_mkldnn=False,
            enable_mkldnn_bfloat16=False,
            threshold=threshold,
            output_dir=output_dir,
            use_fd_format=False,
        )
        
        # Load labels from config
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
            self.labels = yml_conf['label_list']
    
    def predict_from_file(self, image_path, visual=True, save_results=False):
        """
        Predict teeth from an image file
        
        Args:
            image_path (str): Path to the image file
            visual (bool): Whether to visualize the results
            save_results (bool): Whether to save the results
            
        Returns:
            dict: Detection results
        """
        results = self.detector.predict_image(
            [image_path],
            run_benchmark=False,
            repeats=1,
            visual=visual,
            save_results=save_results,
        )
        return results
    
    def predict_from_numpy(self, image_array, visual=False, save_path=None):
        """
        Predict teeth from a numpy array
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array (BGR format, same as OpenCV)
            visual (bool): Whether to visualize the results
            save_path (str): Path to save the visualization (if visual=True)
            
        Returns:
            dict: Detection results with 'boxes' and 'boxes_num' keys
                  boxes: shape [N, 6] with each row being [class_id, score, x1, y1, x2, y2]
        """
        # Make sure image is in BGR format (OpenCV format)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif image_array.shape[2] == 3 and np.array_equal(image_array[0,0], image_array[0,0,::-1]):
            # Simple check if image might be in RGB format
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Create a temporary file path for the image
        temp_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_image.jpg")
        
        # Save the numpy array as an image
        cv2.imwrite(temp_path, image_array)
        
        # Process the image using the detector's preprocess method
        inputs = self.detector.preprocess([temp_path])
        
        # Run prediction
        self.detector.predictor.run()
        output_names = self.detector.predictor.get_output_names()
        boxes_tensor = self.detector.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        
        if len(output_names) == 1:
            # Some exported models cannot get tensor 'bbox_num'
            np_boxes_num = np.array([len(np_boxes)])
        else:
            boxes_num = self.detector.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
        
        # Create result dictionary
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        
        # Filter boxes by threshold
        result = self.detector.filter_box(result, self.threshold)
        
        # Visualize if requested
        if visual:
            # Convert back to PIL Image for visualization
            pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            
            # Create visualization
            vis_result = {}
            vis_result['boxes'] = result['boxes']
            vis_result['boxes_num'] = np.array([len(result['boxes'])])
            
            # Visualize boxes on image
            output_image = visualize_box_mask(
                pil_image, 
                vis_result, 
                self.labels, 
                threshold=self.threshold
            )
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                output_image.save(save_path, quality=95)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = TeethDetector(device="CPU")
    
    # Test with file
    image_path = "./datasets/dental_org/Images/111.png"
    results = detector.predict_from_file(image_path)
    print("File prediction results:", results)
    
    # Test with numpy array
    image_array = cv2.imread(image_path)
    results = detector.predict_from_numpy(
        image_array, 
        visual=True,
        save_path="./test_output/numpy_prediction.jpg"
    )
    print("Numpy prediction results:", results)

# model_dir = "./teeth_detector_paddle/rtdetrv2_r50vd_6x_teeths_exp"
# deploy_file = os.path.join(model_dir, "infer_cfg.yml")
# with open(deploy_file) as f:
#     yml_conf = yaml.safe_load(f)

#     detector = Detector(
#         model_dir,
#         device=-1,
#         run_mode="paddle",
#         batch_size=1,
#         trt_min_shape=1,
#         trt_max_shape=1280,
#         trt_opt_shape=640,
#         trt_calib_mode=False,
#         cpu_threads=1,
#         enable_mkldnn=False,
#         enable_mkldnn_bfloat16=False,
#         threshold=0.5,
#         output_dir="test_output/",
#         use_fd_format=False,
#     )

#     results = detector.predict_image(
#         ["./datasets/dental_org/Images/111.png"],
#         run_benchmark=False,
#         repeats=1,
#         visual=True,
#         save_results=False,
#     )

#     print(results)
