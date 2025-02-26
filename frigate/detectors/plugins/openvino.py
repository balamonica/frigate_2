import logging
import os
import time
import time

import numpy as np
import openvino as ov
import openvino.properties as props
from pydantic import Field
from typing_extensions import Literal
import pandas as pd
#from frigate.track.centroid_tracker import CentroidTracker  # Import the CentroidTracker

#from frigate.track.centroid_tracker import CentroidTracker  # Import the CentroidTracker

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.detectors.util import preprocess, yolov8_postprocess
from frigate.util.image import draw_box_with_label
import cv2
#from frigate.track import ObjectTracker
#from frigate.track import ObjectTracker
#from bytetrack import BYTETracker
#from frigate.track.centroid_tracker import CentroidTracker
#from frigate.track.centroid_tracker import CentroidTracker
image_counter = 0
#tracked_objects = {}
#next_id = 0
#tracked_objects = {}
#next_id = 0
logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"

def save_cropped_images_and_write_csv(crop, detected_labels, confidence_intervals, bounding_boxes, frame_number, frame_time, output_dir="/media/frigate/cropped_images", output_file="human_attributes.csv"):
def save_cropped_images_and_write_csv(crop, detected_labels, confidence_intervals, bounding_boxes, frame_number, frame_time, output_dir="/media/frigate/cropped_images", output_file="human_attributes.csv"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the image to BGR format
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    # Convert the image to BGR format
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    # Create a filename with both timestamp and frame number
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    cropped_image_filename = f"frame_{frame_number}_time_{timestamp}.jpg"
    # Create a filename with both timestamp and frame number
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    cropped_image_filename = f"frame_{frame_number}_time_{timestamp}.jpg"
    cropped_image_path = os.path.join(output_dir, cropped_image_filename)

    # Save the cropped image in BGR format
    cv2.imwrite(cropped_image_path, crop_bgr)
    # Save the cropped image in BGR format
    cv2.imwrite(cropped_image_path, crop_bgr)

    # Prepare the data for the DataFrame
    # Prepare the data for the DataFrame
    data = {
        "Image Name": cropped_image_filename,
        "Frame Number": frame_number,  # Add frame number
        "Frame Time": frame_time,  # Add frame time - added comma
        "Frame Number": frame_number,  # Add frame number
        "Frame Time": frame_time,  # Add frame time - added comma
        "Detected Labels": [detected_labels],  # Store as a list
        "Confidence Intervals": [confidence_intervals],  # Store as a list
        "Bounding Box": [bounding_boxes]  # Store as a list
    }

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    csv_output_path = os.path.join("/media/frigate", output_file)

    # Check if the file exists to determine if we need to write the header
    if not os.path.isfile(csv_output_path):
        df.to_csv(csv_output_path, index=False)  # Write header if file does not exist
    else:
        df.to_csv(csv_output_path, mode='a', header=False, index=False)  # Append without header

    #print(f"Attributes written to {csv_output_path}")
    #print(f"Attributes written to {csv_output_path}")

def load_labels(labelmap_path):
    encodings = ['utf-8', 'latin-1', 'cp1252']  # List of encodings to try
    
    for encoding in encodings:
        try:
            with open(labelmap_path, 'r', encoding=encoding) as f:
                labels = f.read().strip().splitlines()
            return labels
        except UnicodeDecodeError:
            continue
    
    # If none of the encodings work, try binary mode
    try:
        with open(labelmap_path, 'rb') as f:
            labels = f.read().decode('utf-8', errors='ignore').strip().splitlines()
        return labels
    except Exception as e:
        logger.error(f"Failed to load labels from {labelmap_path}: {str(e)}")
        return []
    encodings = ['utf-8', 'latin-1', 'cp1252']  # List of encodings to try
    
    for encoding in encodings:
        try:
            with open(labelmap_path, 'r', encoding=encoding) as f:
                labels = f.read().strip().splitlines()
            return labels
        except UnicodeDecodeError:
            continue
    
    # If none of the encodings work, try binary mode
    try:
        with open(labelmap_path, 'rb') as f:
            labels = f.read().decode('utf-8', errors='ignore').strip().splitlines()
        return labels
    except Exception as e:
        logger.error(f"Failed to load labels from {labelmap_path}: {str(e)}")
        return []

class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [ModelTypeEnum.ssd, ModelTypeEnum.yolonas, ModelTypeEnum.yolox, ModelTypeEnum.yolov8, ModelTypeEnum.yolov11,  ModelTypeEnum.yolov11_humanattr]

    def __init__(self, detector_config: OvDetectorConfig):
        self.ov_core = ov.Core()
        self.ov_model_type = detector_config.model.model_type
        self.next_id = 0 
        self.detector_config = detector_config
        self.frame_counter = 0
        
        # Initialize human attribute model parameters
        self.human_attr_enabled = detector_config.model.human_attr
        if self.human_attr_enabled:
            if not detector_config.model.human_attr_model_path:
                logger.error("Human attribute model path not specified")
                raise ValueError("human_attr_model_path is required when human_attr is enabled")
            if not detector_config.model.human_attr_labelmap_path:
                logger.error("Human attribute labelmap path not specified")
                raise ValueError("human_attr_labelmap_path is required when human_attr is enabled")
            self.human_attr_model = None  # Will be initialized when needed
        
        self.h = detector_config.model.height
        self.w = detector_config.model.width

        #self.tracker = CentroidTracker(detector_config)  # Initialize the tracker
        self.tracked_objects = {}  # This can be managed by the tracker
        self.processed_object_ids = set()

        #self.tracker = CentroidTracker(detector_config)  # Initialize the tracker
        self.tracked_objects = {}  # This can be managed by the tracker
        self.processed_object_ids = set()

        # Initialize vehicle attribute model parameters
        self.vehicle_attr_enabled = detector_config.model.vehicle_attr
        if self.vehicle_attr_enabled:
            if not detector_config.model.vehicle_attr_model_path:
                logger.error("Vehicle attribute model path not specified")
                raise ValueError("vehicle_attr_model_path is required when vehicle_attr is enabled")
            if not detector_config.model.vehicle_attr_labelmap_path:
                logger.error("Vehicle attribute labelmap path not specified")
                raise ValueError("vehicle_attr_labelmap_path is required when vehicle_attr is enabled")
            self.vehicle_attr_model = None  # Will be initialized when needed
            self.processed_vehicle_ids = set()

        # Initialize vehicle attribute model parameters
        self.vehicle_attr_enabled = detector_config.model.vehicle_attr
        if self.vehicle_attr_enabled:
            if not detector_config.model.vehicle_attr_model_path:
                logger.error("Vehicle attribute model path not specified")
                raise ValueError("vehicle_attr_model_path is required when vehicle_attr is enabled")
            if not detector_config.model.vehicle_attr_labelmap_path:
                logger.error("Vehicle attribute labelmap path not specified")
                raise ValueError("vehicle_attr_labelmap_path is required when vehicle_attr is enabled")
            self.vehicle_attr_model = None  # Will be initialized when needed
            self.processed_vehicle_ids = set()

        if not os.path.isfile(detector_config.model.path):
            logger.error(f"OpenVino model file {detector_config.model.path} not found.")
            raise FileNotFoundError

        os.makedirs("/config/model_cache/openvino", exist_ok=True)
        self.ov_core.set_property({props.cache_dir: "/config/model_cache/openvino"})
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
            self.set_strides_grids()

    def set_strides_grids(self):
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]

        hsize_list = [self.h // stride for stride in strides]
        wsize_list = [self.w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsize_list, wsize_list, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded_strides, 1)

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

    def assign_tracking_ids(self, formatted_detections):
        current_frame_ids = {}
        
        for detection in formatted_detections:
            box = detection["box"]
            center_x = (box[1] + box[3]) / 2
            center_y = (box[0] + box[2]) / 2
            detection_id = None

            for obj_id, obj in self.tracked_objects.items():
                obj_box = obj["box"]
                obj_center_x = (obj_box[1] + obj_box[3]) / 2
                obj_center_y = (obj_box[0] + obj_box[2]) / 2
                distance = np.sqrt((center_x - obj_center_x) ** 2 + (center_y - obj_center_y) ** 2)

                # Increase distance threshold to better match same person
                if distance < 0.1:  # Changed from 0.003 to 0.1
                    detection_id = obj_id
                    self.tracked_objects[obj_id]["box"] = box
                    break

            if detection_id is None:
                detection_id = self.next_id
                self.next_id += 1

            current_frame_ids[detection_id] = {
                "box": box,
                "label": detection["label"],
                "score": detection["score"],
                # "frame_time": detection["frame_time"]
            }

        return current_frame_ids

    def assign_tracking_ids(self, formatted_detections):
        current_frame_ids = {}
        
        for detection in formatted_detections:
            box = detection["box"]
            center_x = (box[1] + box[3]) / 2
            center_y = (box[0] + box[2]) / 2
            detection_id = None

            for obj_id, obj in self.tracked_objects.items():
                obj_box = obj["box"]
                obj_center_x = (obj_box[1] + obj_box[3]) / 2
                obj_center_y = (obj_box[0] + obj_box[2]) / 2
                distance = np.sqrt((center_x - obj_center_x) ** 2 + (center_y - obj_center_y) ** 2)

                # Increase distance threshold to better match same person
                if distance < 0.1:  # Changed from 0.003 to 0.1
                    detection_id = obj_id
                    self.tracked_objects[obj_id]["box"] = box
                    break

            if detection_id is None:
                detection_id = self.next_id
                self.next_id += 1

            current_frame_ids[detection_id] = {
                "box": box,
                "label": detection["label"],
                "score": detection["score"],
                # "frame_time": detection["frame_time"]
            }

        return current_frame_ids

    def detect_raw(self, tensor_input):
        infer_request = self.interpreter.create_infer_request()
        #print("Size of tensor_input:", tensor_input.shape)  # Add this line to print the size

        # TODO: see if we can use shared_memory=True
        if self.ov_model_type in (ModelTypeEnum.yolov8, ModelTypeEnum.yolov11, ModelTypeEnum.yolov11_humanattr):
            # Get the model input shape
            model_input_shape = self.interpreter.inputs[0].shape  # Get the input shape from the interpreter
            # Preprocess the input tensor

            input_tensor = preprocess(tensor_input, model_input_shape, np.float32)
            #input_tensor = preprocess(tensor_input, model_input_shape, np.float32)
        else:
            input_tensor = ov.Tensor(array=tensor_input)
        #followingline commented by monica
        infer_request.infer(input_tensor)    

        detections = np.zeros((20, 6), np.float32)

        if self.model_invalid:
            return detections

        if self.ov_model_type == ModelTypeEnum.ssd:
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

        if self.ov_model_type == ModelTypeEnum.yolonas:
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

        if self.ov_model_type == ModelTypeEnum.yolox:
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
            # Increment frame counter
            self.frame_counter += 1
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

            for i, object_detected in enumerate(ordered):
                detections = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )
                formatted_detections.append({
                    "label": detections[0],
                    "score": float(detections[1]),
                    "box": [
                        float(detections[2]),
                        float(detections[3]),
                        float(detections[4]),
                        float(detections[5])
                    ],
                })
            #print('Formatted_detections', formatted_detections)
            #print('Formatted_detections', formatted_detections)

            # Process human attributes if enabled
            if self.human_attr_enabled:
                # Get human attribute model parameters
                human_attr_model_path = self.detector_config.model.human_attr_model_path
                human_attr_labelmap_path = self.detector_config.model.human_attr_labelmap_path
                human_attr_width = self.detector_config.model.human_attr_width
                human_attr_height = self.detector_config.model.human_attr_height
                human_attr_show_label = self.detector_config.model.human_attr_show_label

                # Filter person detections (class 0) from formatted_detections
                person_detections = [d for d in formatted_detections if d['label'] == 0]
                print(f"Current frame number: {self.frame_counter}")
                print('person_detections', person_detections)

                for detection in person_detections:
                    print("Processing human attributes")
                    y_min, x_min, y_max, x_max = detection["box"]
                    
                    tensor_input_np = np.array(tensor_input)
                    image_to_save = tensor_input_np[0]
                    crop = image_to_save[int(y_min * 640):int(y_max * 640), 
                                       int(x_min * 640):int(x_max * 640)]
                    resized_crop = cv2.resize(crop, (human_attr_width, human_attr_height))
                    processed_crop = resized_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
                    processed_crop = np.expand_dims(processed_crop, axis=0)
                for detection in person_detections:
                    print("Processing human attributes")
                    y_min, x_min, y_max, x_max = detection["box"]
                    
                    tensor_input_np = np.array(tensor_input)
                    image_to_save = tensor_input_np[0]
                    crop = image_to_save[int(y_min * 640):int(y_max * 640), 
                                       int(x_min * 640):int(x_max * 640)]
                    resized_crop = cv2.resize(crop, (human_attr_width, human_attr_height))
                    processed_crop = resized_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
                    processed_crop = np.expand_dims(processed_crop, axis=0)

                    if not hasattr(self, 'human_attr_model'):
                    if not hasattr(self, 'human_attr_model'):
                        self.human_attr_model = ov.Core().compile_model(human_attr_model_path, "CPU")
                    human_attr_labels = load_labels(human_attr_labelmap_path)
                    
                    infer_request = self.human_attr_model.create_infer_request()
                    infer_request.set_input_tensor(ov.Tensor(processed_crop))
                    infer_request.infer()
                    image_attr = infer_request.get_output_tensor(0).data
                    human_attr_labels = load_labels(human_attr_labelmap_path)
                    
                    infer_request = self.human_attr_model.create_infer_request()
                    infer_request.set_input_tensor(ov.Tensor(processed_crop))
                    infer_request.infer()
                    image_attr = infer_request.get_output_tensor(0).data

                    detected_labels = []
                    confidence_intervals = []
                    bounding_boxes = [x_min, y_min, x_max, y_max]
                    scores = image_attr.flatten()
                    for i, score in enumerate(scores):
                        if score > 0.5:
                            detected_labels.append(human_attr_labels[i])
                            confidence_intervals.append(score)
                    detected_labels = []
                    confidence_intervals = []
                    bounding_boxes = [x_min, y_min, x_max, y_max]
                    scores = image_attr.flatten()
                    for i, score in enumerate(scores):
                        if score > 0.5:
                            detected_labels.append(human_attr_labels[i])
                            confidence_intervals.append(score)

                    if human_attr_show_label:
                        draw_box_with_label(
                            tensor_input,
                            int(x_min * 640),
                            int(y_min * 640),
                            int(x_max * 640),
                            int(y_max * 640),
                            label=detected_labels,
                            info="",
                            thickness=2,
                            color=(0, 255, 0),
                            position="ul"
                        )
                    if human_attr_show_label:
                        draw_box_with_label(
                            tensor_input,
                            int(x_min * 640),
                            int(y_min * 640),
                            int(x_max * 640),
                            int(y_max * 640),
                            label=detected_labels,
                            info="",
                            thickness=2,
                            color=(0, 255, 0),
                            position="ul"
                        )

                    save_cropped_images_and_write_csv(
                        crop, 
                        detected_labels, 
                        confidence_intervals, 
                        bounding_boxes, 
                        frame_number=self.frame_counter,
                        frame_time=current_time
                    )
                    save_cropped_images_and_write_csv(
                        crop, 
                        detected_labels, 
                        confidence_intervals, 
                        bounding_boxes, 
                        frame_number=self.frame_counter,
                        frame_time=current_time
                    )

            # Process vehicle attributes if enabled
            if self.vehicle_attr_enabled:
                # Get vehicle attribute model parameters
                vehicle_attr_model_path = self.detector_config.model.vehicle_attr_model_path
                vehicle_attr_labelmap_path = self.detector_config.model.vehicle_attr_labelmap_path
                vehicle_attr_width = self.detector_config.model.vehicle_attr_width
                vehicle_attr_height = self.detector_config.model.vehicle_attr_height
                vehicle_attr_show_label = self.detector_config.model.vehicle_attr_show_label

                # Filter vehicle detections (class 2 for car) from formatted_detections
                vehicle_detections = [d for d in formatted_detections if d['label'] == 2]
                print(f"Current frame number: {self.frame_counter}")
                print('vehicle_detections', vehicle_detections)

                for detection in vehicle_detections:
                    print("Processing vehicle attributes")
                    y_min, x_min, y_max, x_max = detection["box"]
                    
                    tensor_input_np = np.array(tensor_input)
                    image_to_save = tensor_input_np[0]  # Already in RGB format
                    
                    # Crop the vehicle region
                    crop = image_to_save[int(y_min * 640):int(y_max * 640), 
                                       int(x_min * 640):int(x_max * 640)]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    # Resize to model's expected dimensions (192x256)
                    resized_crop = cv2.resize(crop, (256, 192))  # width=256, height=192

                    # Convert to NCHW format and normalize
                    #processed_crop = resized_crop.transpose(2, 0, 1)  # HWC to CHW
                    processed_crop = resized_crop.astype(np.float32) / 255.0
                        # Normalize with mean and std
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    processed_crop = (processed_crop - mean) / std
                    processed_crop = np.transpose(processed_crop, (2, 0, 1))  # HWC to CHW
                    processed_crop = np.expand_dims(processed_crop, axis=0)

                    if self.vehicle_attr_model is None:
                        try:
                            print("Initializing vehicle attribute model...")
                            self.vehicle_attr_model = ov.Core().compile_model(vehicle_attr_model_path, "CPU")
                            print("Vehicle attribute model initialized successfully")
                        except Exception as e:
                            logger.error(f"Failed to initialize vehicle attribute model: {e}")
                            continue
                            
                    vehicle_attr_labels = load_labels(vehicle_attr_labelmap_path)
                    
                    infer_request = self.vehicle_attr_model.create_infer_request()
                    infer_request.set_input_tensor(ov.Tensor(processed_crop))
                    infer_request.infer()
                    vehicle_attr = infer_request.get_output_tensor(0).data

                    scores = vehicle_attr.flatten()
                    #print('vehicle CI', scores)
                    # First 10 labels are colors, next 9 are makes

                    color_probs = scores[:10]
                    color_idx = np.argmax(color_probs)
                    color_score = color_probs[color_idx]
                    
                    type_probs = scores[10:]
                    type_idx = np.argmax(type_probs)
                    type_score = type_probs[type_idx]

                    
                    detected_labels = []
                    confidence_intervals = []

                    detected_labels.append(vehicle_attr_labels[color_idx])
                    confidence_intervals.append(float(color_score))

                    detected_labels.append(vehicle_attr_labels[type_idx+10])
                    confidence_intervals.append(float(type_score))

                    bounding_boxes = [x_min, y_min, x_max, y_max]

                    if vehicle_attr_show_label:
                        draw_box_with_label(
                            tensor_input,
                            int(x_min * 640),
                            int(y_min * 640),
                            int(x_max * 640),
                            int(y_max * 640),
                            label=detected_labels,
                            info="",
                            thickness=2,
                            color=(255, 0, 0),
                            position="ul"
                        )

                    save_cropped_images_and_write_csv(
                        crop, 
                        detected_labels, 
                        confidence_intervals, 
                        bounding_boxes, 
                        frame_number=self.frame_counter,
                        frame_time=current_time,
                        output_dir="/media/frigate/vehicle_crops",
                        output_file="vehicle_attributes.csv"
                    )

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
        elif self.ov_model_type in (ModelTypeEnum.yolov11_humanattr):
            # Increment frame counter
            self.frame_counter += 1
            current_time = time.time()

            # YOLOv11 detection
            #tracker = NorfairTracker(self.detector_config, None)

            # Extract human attribute model parameters from detector_config
            human_attr_model_path = self.detector_config.model.human_attr_model_path
            human_attr_labelmap_path = self.detector_config.model.human_attr_labelmap_path
            human_attr_width = self.detector_config.model.human_attr_width
            human_attr_height = self.detector_config.model.human_attr_height
            human_attr_show_label = self.detector_config.model.human_attr_show_label

            out_tensor = infer_request.get_output_tensor()
            results = out_tensor.data[0]
            output_data = np.transpose(results)
            scores = np.max(output_data[:, 4:], axis=1)

            if len(scores) == 0:
                return np.zeros((20, 6), np.float32)
            scores = np.expand_dims(scores, axis=1)
            dets = np.concatenate((output_data, scores), axis=1)

            dets = dets[dets[:, -1] > 0.7, :]
            ordered = dets[dets[:, -1].argsort()[::-1]][:20]
            detections = np.zeros((20, 6), np.float32)

            formatted_detections = []
            for i, object_detected in enumerate(ordered):
                detections = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )
                formatted_detections.append({
                    "label": detections[0],
                    "score": float(detections[1]),
                    "box": [
                        float(detections[2]),
                        float(detections[3]),
                        float(detections[4]),
                        float(detections[5])
                    ],
                })

            # Save frame with YOLO detections
            frame_with_detections = np.array(tensor_input)
            if len(frame_with_detections.shape) == 4:
                frame_with_detections = frame_with_detections[0]
            frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
            
            # Draw all detections
            for detection in formatted_detections:
                y_min, x_min, y_max, x_max = detection["box"]
                label = detection["label"]
                score = detection["score"]
                
                cv2.rectangle(
                    frame_with_detections,
                    (int(x_min * 640), int(y_min * 640)),
                    (int(x_max * 640), int(y_max * 640)),
                    (0, 255, 0), 2
                )
                
                text = f"class {label}: {score:.2f}"
                cv2.putText(
                    frame_with_detections,
                    text,
                    (int(x_min * 640), int(y_min * 640) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

            # Save the frame with detections
            output_dir = "/media/frigate/debug_frames"
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"frame_{self.frame_counter}.jpg")
            cv2.imwrite(frame_path, frame_with_detections)

            # Process human attributes if enabled
            if self.human_attr_enabled:
                # Get human attribute model parameters
                human_attr_model_path = self.detector_config.model.human_attr_model_path
                human_attr_labelmap_path = self.detector_config.model.human_attr_labelmap_path
                human_attr_width = self.detector_config.model.human_attr_width
                human_attr_height = self.detector_config.model.human_attr_height
                human_attr_show_label = self.detector_config.model.human_attr_show_label

                # Filter person detections (class 0) from formatted_detections
                person_detections = [d for d in formatted_detections if d['label'] == 0]
                #print(f"Current frame number: {self.frame_counter}")
                #print('person_detections', person_detections)

                for detection in person_detections:
                    print("Processing human attributes")
                    y_min, x_min, y_max, x_max = detection["box"]
                    
                    tensor_input_np = np.array(tensor_input)
                    image_to_save = tensor_input_np[0]
                    crop = image_to_save[int(y_min * 640):int(y_max * 640), 
                                       int(x_min * 640):int(x_max * 640)]
                    resized_crop = cv2.resize(crop, (human_attr_width, human_attr_height))
                    processed_crop = resized_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
                    processed_crop = np.expand_dims(processed_crop, axis=0)

                    if not hasattr(self, 'human_attr_model'):
                        self.human_attr_model = ov.Core().compile_model(human_attr_model_path, "CPU")
                    human_attr_labels = load_labels(human_attr_labelmap_path)
                    
                    infer_request = self.human_attr_model.create_infer_request()
                    infer_request.set_input_tensor(ov.Tensor(processed_crop))
                    infer_request.infer()
                    image_attr = infer_request.get_output_tensor(0).data

                    detected_labels = []
                    confidence_intervals = []
                    bounding_boxes = [x_min, y_min, x_max, y_max]
                    scores = image_attr.flatten()
                    for i, score in enumerate(scores):
                        if score > 0.5:
                            detected_labels.append(human_attr_labels[i])
                            confidence_intervals.append(score)

                    if human_attr_show_label:
                        draw_box_with_label(
                            tensor_input,
                            int(x_min * 640),
                            int(y_min * 640),
                            int(x_max * 640),
                            int(y_max * 640),
                            label=detected_labels,
                            info="",
                            thickness=2,
                            color=(0, 255, 0),
                            position="ul"
                        )

                    save_cropped_images_and_write_csv(
                        crop, 
                        detected_labels, 
                        confidence_intervals, 
                        bounding_boxes, 
                        frame_number=self.frame_counter,
                        frame_time=current_time
                    )

            # Process vehicle attributes if enabled
            if self.vehicle_attr_enabled:
                # Get vehicle attribute model parameters
                vehicle_attr_model_path = self.detector_config.model.vehicle_attr_model_path
                vehicle_attr_labelmap_path = self.detector_config.model.vehicle_attr_labelmap_path
                vehicle_attr_width = self.detector_config.model.vehicle_attr_width
                vehicle_attr_height = self.detector_config.model.vehicle_attr_height
                vehicle_attr_show_label = self.detector_config.model.vehicle_attr_show_label

                # Filter vehicle detections (class 2 for car) from formatted_detections
                vehicle_detections = [d for d in formatted_detections if d['label'] == 2]
                print(f"Current frame number: {self.frame_counter}")
                print('vehicle_detections', vehicle_detections)

                for detection in vehicle_detections:
                    print("Processing vehicle attributes")
                    y_min, x_min, y_max, x_max = detection["box"]
                    
                    tensor_input_np = np.array(tensor_input)
                    image_to_save = tensor_input_np[0]  # Already in RGB format
                    
                    # Crop the vehicle region
                    crop = image_to_save[int(y_min * 640):int(y_max * 640), 
                                       int(x_min * 640):int(x_max * 640)]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    # Resize to model's expected dimensions (192x256)
                    resized_crop = cv2.resize(crop, (256, 192))  # width=256, height=192

                    # Convert to NCHW format and normalize
                    processed_crop = resized_crop.transpose(2, 0, 1)  # HWC to CHW
                    processed_crop = processed_crop.astype(np.float32) / 255.0
                    processed_crop = np.expand_dims(processed_crop, axis=0)  # Add batch dimension

                    print('vehicle path', vehicle_attr_model_path)
                    
                    # Initialize model if not already done
                    if self.vehicle_attr_model is None:
                        try:
                            print("Initializing vehicle attribute model...")
                            self.vehicle_attr_model = ov.Core().compile_model(vehicle_attr_model_path, "CPU")
                            print("Vehicle attribute model initialized successfully")
                        except Exception as e:
                            logger.error(f"Failed to initialize vehicle attribute model: {e}")
                            continue
                            
                    vehicle_attr_labels = load_labels(vehicle_attr_labelmap_path)
                    
                    infer_request = self.vehicle_attr_model.create_infer_request()
                    infer_request.set_input_tensor(ov.Tensor(processed_crop))
                    infer_request.infer()
                    vehicle_attr = infer_request.get_output_tensor(0).data

                    scores = vehicle_attr.flatten()
                    #print('vehicle CI', scores)
                    # First 10 labels are colors, next 9 are makes

                    color_probs = scores[:10]
                    color_idx = np.argmax(color_probs)
                    color_score = color_probs[color_idx]
                    
                    type_probs = scores[10:]
                    type_idx = np.argmax(type_probs)
                    type_score = type_probs[type_idx]

                    
                    detected_labels = []
                    confidence_intervals = []

                    detected_labels.append(vehicle_attr_labels[color_idx])
                    confidence_intervals.append(float(color_score))

                    detected_labels.append(vehicle_attr_labels[type_idx+10])
                    confidence_intervals.append(float(type_score))

                    bounding_boxes = [x_min, y_min, x_max, y_max]

                    if vehicle_attr_show_label:
                        draw_box_with_label(
                            tensor_input,
                            int(x_min * 640),
                            int(y_min * 640),
                            int(x_max * 640),
                            int(y_max * 640),
                            label=detected_labels,
                            info="",
                            thickness=2,
                            color=(255, 0, 0),
                            position="ul"
                        )

                    save_cropped_images_and_write_csv(
                        crop, 
                        detected_labels, 
                        confidence_intervals, 
                        bounding_boxes, 
                        frame_number=self.frame_counter,
                        frame_time=current_time,
                        output_dir="/media/frigate/vehicle_crops",
                        output_file="vehicle_attributes.csv"
                    )

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
        