import logging
import os
import time
import random
import numpy as np
import openvino as ov
import openvino.properties as props
from pydantic import Field
from typing_extensions import Literal
import pandas as pd
#from frigate.track.centroid_tracker import CentroidTracker  # Import the CentroidTracker

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.util.model import (
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

# def unclip_cv2(box, unclip_ratio):
#     """Unclips the bounding box using cv2 dilation."""
#     distance = cv2.contourArea(box) * unclip_ratio / cv2.arcLength(box, True)
    
#     # Create a mask
#     mask = np.zeros((640, 640), dtype=np.uint8) # adjust image size as needed.
#     cv2.fillPoly(mask, [box.astype(np.int32)], 255)

#     # Dilate the mask
#     kernel_size = int(distance)
#     if kernel_size < 1:
#         kernel_size = 1
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     dilated_mask = cv2.dilate(mask, kernel, iterations=1)

#     # Find contours
#     contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Convert to NumPy array
#     if contours:
#         expanded = contours[0].reshape(-1, 2)
#         return expanded
#     else:
#         return box #return the original box if no contour found.

# def save_cropped_images_and_write_csv(crop, detected_labels, confidence_intervals, bounding_boxes, frame_number, frame_time, output_dir="/media/frigate/cropped_images", output_file="human_attributes.csv"):
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#     counter2= random.randint(1,20)
#     # Convert the image to BGR format
#     crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

#     # Create a filename with both timestamp and frame number
#     timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
#     cropped_image_filename = f"frame_{frame_number}_time_{timestamp}_{counter2}.jpg"
#     cropped_image_path = os.path.join(output_dir, cropped_image_filename)

#     # Save the cropped image in BGR format
#     cv2.imwrite(cropped_image_path, crop_bgr)

#     # Prepare the data for the DataFrame
#     data = {
#         "Image Name": cropped_image_filename,
#         "Frame Number": frame_number,
#         "Frame Time": frame_time,
#         "Detected Labels": [detected_labels],
#         "Confidence Intervals": [confidence_intervals],
#         "Bounding Box": [bounding_boxes]
#     }

#     # Create a DataFrame from the collected data
#     df = pd.DataFrame(data)

#     # Write the DataFrame to a CSV file
#     csv_output_path = os.path.join("/media/frigate", output_file)

#     # Check if the file exists to determine if we need to write the header
#     if not os.path.isfile(csv_output_path):
#         df.to_csv(csv_output_path, index=False)  # Write header if file does not exist
#     else:
#         df.to_csv(csv_output_path, mode='a', header=False, index=False)  # Append without header

#     #print(f"Attributes written to {csv_output_path}")

# def load_labels(labelmap_path):
#     encodings = ['utf-8', 'latin-1', 'cp1252']  # List of encodings to try
    
#     for encoding in encodings:
#         try:
#             with open(labelmap_path, 'r', encoding=encoding) as f:
#                 labels = f.read().strip().splitlines()
#             return labels
#         except UnicodeDecodeError:
#             continue
    
#     # If none of the encodings work, try binary mode
#     try:
#         with open(labelmap_path, 'rb') as f:
#             labels = f.read().decode('utf-8', errors='ignore').strip().splitlines()
#         return labels
#     except Exception as e:
#         logger.error(f"Failed to load labels from {labelmap_path}: {str(e)}")
#         return []
    
# def unclip(box, unclip_ratio):
#         """Unclips the bounding box using pyclipper."""
#         poly = box.tolist()
#         distance = cv2.contourArea(box) * unclip_ratio / cv2.arcLength(box, True)
#         offset = pyclipper.PyclipperOffset()
#         offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
#         expanded = np.array(offset.Execute(distance))
#         return expanded.reshape(-1, 2)

# def post_process_detections(feature_map, thresh=0.5, box_thresh=0.2, unclip_ratio=2.0):


#     bitmap = (feature_map > thresh).astype(np.uint8)
    
#     dest_width, dest_height = 640, 640  
    
#     # Initialize scores list
#     scores = []
#     contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     boxes = []
#     confidences = []

#     for contour in contours:
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)

#         sside = max(cv2.contourArea(box), 1)
#         # print(sside)
#         if sside < 3:
#             continue
        
#         score = cv2.contourArea(box)
#         # print('score', score)

#         if score < box_thresh:
#             continue

#         unclipped_box = unclip_cv2(box, unclip_ratio)
        
#         # Ensure the box has exactly 4 points
#         if len(unclipped_box) > 4:
#             # Get the bounding rectangle of the unclipped polygon
#             rect = cv2.minAreaRect(unclipped_box)
#             unclipped_box = cv2.boxPoints(rect)

#         #resize to original size
#         height, width = bitmap.shape
#         unclipped_box[:, 0] = np.clip(np.round(unclipped_box[:, 0] / width * dest_width), 0, 640)
#         unclipped_box[:, 1] = np.clip(np.round(unclipped_box[:, 1] / height * dest_height), 0, 640)

#         boxes.append(unclipped_box.astype(np.int16))
#         scores.append(score)

#     if not boxes:  # If no boxes were found
#         return np.array([], dtype=np.int16), []
        
#     # Ensure all boxes have the same shape before creating array
#     boxes = [box[:4] if len(box) > 4 else box for box in boxes]  # Take only first 4 points if more exist
#     return np.array(boxes, dtype=np.int16), scores

# def order_points(pts):
#     """Orders the corner points of a rectangle in clockwise order."""
#     rect = np.zeros((4, 2), dtype="float32")

#     # The top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     # Now, compute the difference between the points,
#     # the top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     return rect.astype("int")

# def decode_license_plate_ctc(rec_result, label_file):
#     """Decodes the recognition result using CTC principles."""

#     print('In decode License plate module')
#     # print('rec_result size', rec_result.shape)
#     #print(rec_result)
#     predicted_indices = np.argmax(rec_result, axis=2)  # Get predicted indices

#     # Load the label file
#     with open(label_file, 'r') as f:
#         labels = f.read().splitlines()

#     # Add the '<blank>' character to the labels (crucial for CTC)
#     labels = ['<blank>'] + labels
#     # print('label',len(labels))

#     decoded_text = []
#     # print('Predicted_indices', predicted_indices )

#     for batch_idx in range(predicted_indices.shape[0]):  # Iterate through batch (1)
#         current_text = ""
#         previous_char_index = -1  # Initialize to an invalid index

#         #for feature_map_idx in range(predicted_indices.shape[1]): # Iterate through the extra dimension(1)
#         for i in range(predicted_indices.shape[1]):  # Iterate through sequence length (40)
#             char_index = predicted_indices[batch_idx, i].item()

#             if char_index != 0 and char_index != previous_char_index:  # Not blank and not a repeat
#                 current_text += labels[char_index]

#             previous_char_index = char_index

#         decoded_text.append(current_text)

#     print('decoded text',decoded_text)
#     return decoded_text

# def load_character_dict(file_path):
#     with open(file_path, 'r') as f:
#             # Read all lines and strip whitespace
#         characters = [line.strip() for line in f.readlines()]
#     return characters

# def min_max_scale(feature_map):
#     """Scales the feature map to the range [0, 1] using Min-Max scaling."""
#     min_val = np.min(feature_map)
#     max_val = np.max(feature_map)

#     if max_val == min_val:
#         # Handle the case where all values are the same
#         return np.zeros_like(feature_map)

#     scaled_feature_map = (feature_map - min_val) / (max_val - min_val)
#     return scaled_feature_map


class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [
        ModelTypeEnum.dfine,
        ModelTypeEnum.rfdetr,
        ModelTypeEnum.ssd,
        ModelTypeEnum.yolonas,
        ModelTypeEnum.yologeneric,
        ModelTypeEnum.yolox,
        ModelTypeEnum.yolov8,
        ModelTypeEnum.yolov11,
    ]

    def __init__(self, detector_config: OvDetectorConfig):
        super().__init__(detector_config)
        self.ov_core = ov.Core()
        self.ov_model_type = detector_config.model.model_type
        self.next_id = 0 
        self.detector_config = detector_config
        self.frame_counter = 0
        
        # Initialize human attribute model parameters
        # self.human_attr_enabled = detector_config.model.human_attr
        # if self.human_attr_enabled:
        #     if not detector_config.model.human_attr_model_path:
        #         logger.error("Human attribute model path not specified")
        #         raise ValueError("human_attr_model_path is required when human_attr is enabled")
        #     if not detector_config.model.human_attr_labelmap_path:
        #         logger.error("Human attribute labelmap path not specified")
        #         raise ValueError("human_attr_labelmap_path is required when human_attr is enabled")
        #     self.human_attr_model = None  # Will be initialized when needed
        
        self.h = detector_config.model.height
        self.w = detector_config.model.width

        # #self.tracker = CentroidTracker(detector_config)  # Initialize the tracker
        # self.tracked_objects = {}  # This can be managed by the tracker
        # self.processed_object_ids = set()

        # #self.tracker = CentroidTracker(detector_config)  # Initialize the tracker
        # self.tracked_objects = {}  # This can be managed by the tracker
        # self.processed_object_ids = set()
        # self.frame_buffer = []

        # # Initialize vehicle attribute model parameters
        # self.vehicle_attr_enabled = detector_config.model.vehicle_attr
        # if self.vehicle_attr_enabled:
        #     if not detector_config.model.vehicle_attr_model_path:
        #         logger.error("Vehicle attribute model path not specified")
        #         raise ValueError("vehicle_attr_model_path is required when vehicle_attr is enabled")
        #     if not detector_config.model.vehicle_attr_labelmap_path:
        #         logger.error("Vehicle attribute labelmap path not specified")
        #         raise ValueError("vehicle_attr_labelmap_path is required when vehicle_attr is enabled")
        #     self.vehicle_attr_model = None  # Will be initialized when needed
        #     self.processed_vehicle_ids = set()

        # self.vehicle_alpr_enabled = detector_config.model.vehicle_alpr
        # if self.vehicle_alpr_enabled:
        #     if not detector_config.model.vehicle_alpr_det_model_path:
        #         logger.error("Vehicle alpr det model path not specified")
        #         raise ValueError("vehicle_alpr_Det_model_path is required when vehicle_attr is enabled")
        #     if not detector_config.model.vehicle_alpr_rec_model_path:
        #         logger.error("Vehicle alpr rec model path not specified")
        #         raise ValueError("vehicle_alpr_rec_model_path is required when vehicle_attr is enabled")
        #     self.vehicle_alpr_model = None 
            
        # self.human_falling_enabled = detector_config.model.human_falling
        # if self.human_falling_enabled:
        #     if not detector_config.model.human_falling_model_path:
        #         logger.error("Human fall detection model path not specified")
        #         raise ValueError("human_falling_model_path is needed to detect falling")
        #     self.human_falling_model = None 

        # self.human_fighting_enabled = detector_config.model.human_fighting
        # if self.human_fighting_enabled:
        #     if not detector_config.model.human_fighting_model_path:
        #         logger.error("Human fight detection model path not specified")
        #         raise ValueError("human_fighting_model_path is needed to detect fighting")
        #     self.human_fighting_model = None 

        # self.human_calling_enabled = detector_config.model.human_calling
        # if self.human_calling_enabled:
        #     if not detector_config.model.human_calling_model_path:
        #         logger.error("Human call detection model path not specified")
        #         raise ValueError("human_calling_model_path is needed to detect calling")
        #     self.human_calling_model = None 

        # self.human_smoking_enabled = detector_config.model.human_smoking
        # if self.human_smoking_enabled:
        #     if not detector_config.model.human_smoking_model_path:
        #         logger.error("Human smoking detection model path not specified")
        #         raise ValueError("human_smoking_model_path is needed to detect smoking")
        #     self.human_smoking_model = None 
         
        if not os.path.isfile(detector_config.model.path):
            logger.error(f"OpenVino model file {detector_config.model.path} not found.")
            raise FileNotFoundError

        os.makedirs(os.path.join(MODEL_CACHE_DIR, "openvino"), exist_ok=True)
        self.ov_core.set_property(
            {props.cache_dir: os.path.join(MODEL_CACHE_DIR, "openvino")}
        )
        self.interpreter = self.ov_core.compile_model(
            model=detector_config.model.path, device_name=detector_config.device
        )

        self.model_invalid = False

        if self.ov_model_type not in self.supported_models:
            logger.error(
                f"OpenVino detector does not support {self.ov_model_type} models."
            )
            self.model_invalid = True

        # Ensure the SSD model has the right input and output shapes
        if self.ov_model_type == ModelTypeEnum.ssd:
            model_inputs = self.interpreter.inputs
            model_outputs = self.interpreter.outputs

            if len(model_inputs) != 1:
                logger.error(
                    f"SSD models must only have 1 input. Found {len(model_inputs)}."
                )
                self.model_invalid = True
            if len(model_outputs) != 1:
                logger.error(
                    f"SSD models must only have 1 output. Found {len(model_outputs)}."
                )
                self.model_invalid = True

            if model_inputs[0].get_shape() != ov.Shape([1, self.w, self.h, 3]):
                logger.error(
                    f"SSD model input doesn't match. Found {model_inputs[0].get_shape()}."
                )
                self.model_invalid = True

            output_shape = model_outputs[0].get_shape()
            if output_shape[0] != 1 or output_shape[1] != 1 or output_shape[3] != 7:
                logger.error(f"SSD model output doesn't match. Found {output_shape}.")
                self.model_invalid = True

        if self.ov_model_type == ModelTypeEnum.yolonas:
            model_inputs = self.interpreter.inputs
            model_outputs = self.interpreter.outputs

            if len(model_inputs) != 1:
                logger.error(
                    f"YoloNAS models must only have 1 input. Found {len(model_inputs)}."
                )
                self.model_invalid = True
            if len(model_outputs) != 1:
                logger.error(
                    f"YoloNAS models must be exported in flat format and only have 1 output. Found {len(model_outputs)}."
                )
                self.model_invalid = True

            if model_inputs[0].get_shape() != ov.Shape([1, 3, self.w, self.h]):
                logger.error(
                    f"YoloNAS model input doesn't match. Found {model_inputs[0].get_shape()}, but expected {[1, 3, self.w, self.h]}."
                )
                self.model_invalid = True

            output_shape = model_outputs[0].partial_shape
            if output_shape[-1] != 7:
                logger.error(
                    f"YoloNAS models must be exported in flat format. Model output doesn't match. Found {output_shape}."
                )
                self.model_invalid = True

        if self.ov_model_type == ModelTypeEnum.yolox:
            self.output_indexes = 0
            while True:
                try:
                    tensor_shape = self.interpreter.output(self.output_indexes).shape
                    logger.info(
                        f"Model Output-{self.output_indexes} Shape: {tensor_shape}"
                    )
                    self.output_indexes += 1
                except Exception:
                    logger.info(f"Model has {self.output_indexes} Output Tensors")
                    break
            self.num_classes = tensor_shape[2] - 5
            logger.info(f"YOLOX model has {self.num_classes} classes")
            self.calculate_grids_strides()

    ## Takes in class ID, confidence score, and array of [x, y, w, h] that describes detection position,
    ## returns an array that's easily passable back to Frigate.
    def process_yolo(self, class_id, conf, pos):
        return [
            class_id,  # class ID
            conf,  # confidence score
            (pos[1] - (pos[3] / 2)) / self.h,  # y_min
            (pos[0] - (pos[2] / 2)) / self.w,  # x_min
            (pos[1] + (pos[3] / 2)) / self.h,  # y_max
            (pos[0] + (pos[2] / 2)) / self.w,  # x_max
        ]

    # def assign_tracking_ids(self, formatted_detections):
    #     current_frame_ids = {}
        
    #     for detection in formatted_detections:
    #         box = detection["box"]
    #         center_x = (box[1] + box[3]) / 2
    #         center_y = (box[0] + box[2]) / 2
    #         detection_id = None

    #         for obj_id, obj in self.tracked_objects.items():
    #             obj_box = obj["box"]
    #             obj_center_x = (obj_box[1] + obj_box[3]) / 2
    #             obj_center_y = (obj_box[0] + obj_box[2]) / 2
    #             distance = np.sqrt((center_x - obj_center_x) ** 2 + (center_y - obj_center_y) ** 2)

    #             # Increase distance threshold to better match same person
    #             if distance < 0.1:  # Changed from 0.003 to 0.1
    #                 detection_id = obj_id
    #                 self.tracked_objects[obj_id]["box"] = box
    #                 break

    #         if detection_id is None:
    #             detection_id = self.next_id
    #             self.next_id += 1

    #         current_frame_ids[detection_id] = {
    #             "box": box,
    #             "label": detection["label"],
    #             "score": detection["score"],
    #             # "frame_time": detection["frame_time"]
    #         }

    #     return current_frame_ids

    def detect_raw(self, tensor_input):
        infer_request = self.interpreter.create_infer_request()
        #print("Size of tensor_input:", tensor_input.shape)  # Add this line to print the size

        # TODO: see if we can use shared_memory=True
        input_tensor = ov.Tensor(array=tensor_input)

        if self.ov_model_type == ModelTypeEnum.dfine:
            infer_request.set_tensor("images", input_tensor)
            target_sizes_tensor = ov.Tensor(
                np.array([[self.h, self.w]], dtype=np.int64)
            )
            infer_request.set_tensor("orig_target_sizes", target_sizes_tensor)
            infer_request.infer()
            tensor_output = (
                infer_request.get_output_tensor(0).data,
                infer_request.get_output_tensor(1).data,
                infer_request.get_output_tensor(2).data,
            )
            return post_process_dfine(tensor_output, self.w, self.h)

        infer_request.infer(input_tensor)

        detections = np.zeros((20, 6), np.float32)

        if self.model_invalid:
            return detections
        elif self.ov_model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr(
                [
                    infer_request.get_output_tensor(0).data,
                    infer_request.get_output_tensor(1).data,
                ]
            )
        elif self.ov_model_type == ModelTypeEnum.ssd:
            results = infer_request.get_output_tensor(0).data[0][0]

            for i, (_, class_id, score, xmin, ymin, xmax, ymax) in enumerate(results):
                if i == 20:
                    break
                detections[i] = [
                    class_id,
                    float(score),
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                ]

            # Print the shape of the detections array
            #print("Detections shape:", detections.shape) #for debug

            return detections
        elif self.ov_model_type == ModelTypeEnum.yolonas:
            predictions = infer_request.get_output_tensor(0).data

            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                # when running in GPU mode, empty predictions in the output have class_id of -1
                if class_id < 0:
                    break
                detections[i] = [
                    class_id,
                    confidence,
                    y_min / self.h,
                    x_min / self.w,
                    y_max / self.h,
                    x_max / self.w,
                ]
            return detections
        elif self.ov_model_type == ModelTypeEnum.yologeneric:
            out_tensor = []

            for item in infer_request.output_tensors:
                out_tensor.append(item.data)

            return post_process_yolo(out_tensor, self.w, self.h)
        elif self.ov_model_type == ModelTypeEnum.yolox:
            out_tensor = infer_request.get_output_tensor()
            # [x, y, h, w, box_score, class_no_1, ..., class_no_80],
            results = out_tensor.data
            results[..., :2] = (results[..., :2] + self.grids) * self.expanded_strides
            results[..., 2:4] = np.exp(results[..., 2:4]) * self.expanded_strides
            image_pred = results[0, ...]

            class_conf = np.max(
                image_pred[:, 5 : 5 + self.num_classes], axis=1, keepdims=True
            )
            class_pred = np.argmax(image_pred[:, 5 : 5 + self.num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= 0.3).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = np.concatenate(
                (image_pred[:, :5], class_conf, class_pred), axis=1
            )
            detections = detections[conf_mask]

            ordered = detections[detections[:, 5].argsort()[::-1]][:20]

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    object_detected[6], object_detected[5], object_detected[:4]
                )
            return detections

        # Add the YOLOv8/v11 output processing here
        if self.ov_model_type in (ModelTypeEnum.yolov8, ModelTypeEnum.yolov11):
            print('In yolov8 openvino')
            # Increment frame counter
            self.frame_counter += 1
            print('frame number', self.frame_counter)
            current_time = time.time()

            out_tensor = infer_request.get_output_tensor()
            results = out_tensor.data[0]
            output_data = np.transpose(results)
            scores = np.max(output_data[:, 4:], axis=1)
            if len(scores) == 0:
                return np.zeros((20, 6), np.float32)
            scores = np.expand_dims(scores, axis=1)
            dets = np.concatenate((output_data, scores), axis=1)
            dets = dets[dets[:, -1] > 0.5, :]
            ordered = dets[dets[:, -1].argsort()[::-1]][:20]
            detections = np.zeros((20, 6), np.float32)

            formatted_detections = []
            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )
                formatted_detections.append({
                    "label": detections[i][0],
                    "score": float(detections[i][1]),
                    "box": [
                        float(detections[i][2]),
                        float(detections[i][3]),
                        float(detections[i][4]),
                        float(detections[i][5])
                    ],
                })
            # print('Formatted_detections', formatted_detections)

            # print('Detections', detections)

            
            return detections
        
        elif self.ov_model_type == ModelTypeEnum.yolov5:
            out_tensor = infer_request.get_output_tensor()
            output_data = out_tensor.data[0]
            # filter out lines with scores below threshold
            conf_mask = (output_data[:, 4] >= 0.5).squeeze()
            output_data = output_data[conf_mask]
            # limit to top 20 scores, descending order
            ordered = output_data[output_data[:, 4].argsort()[::-1]][:20]

            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[5:]),
                    object_detected[4],
                    object_detected[:4],
                )
            return detections
    
    