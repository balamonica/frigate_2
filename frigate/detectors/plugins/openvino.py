import logging
import os

import numpy as np
import openvino as ov
import openvino.properties as props
from pydantic import Field
from typing_extensions import Literal
import pandas as pd
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.detectors.util import preprocess, yolov8_postprocess
import cv2
image_counter = 0
logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"

def save_cropped_images_and_write_csv(crop, detected_labels, confidence_intervals, bounding_boxes, output_dir="/media/frigate/cropped_images", output_file="human_attributes.csv"):
    global image_counter  # Use the global counter for naming images

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a list to hold the data for the DataFrame
    data = []

    # Create a filename for the cropped image
    cropped_image_filename = f"image{image_counter}.jpg"
    cropped_image_path = os.path.join(output_dir, cropped_image_filename)

    # Save the cropped image
    cv2.imwrite(cropped_image_path, crop)

    # Append the data for this detection
    data = {
        "Image Name": cropped_image_filename,
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

    print(f"Attributes written to {csv_output_path}")

    # Increment the image counter for the next call
    image_counter += 1

def load_labels(labelmap_path):
    with open(labelmap_path, 'r') as f:
        labels = f.read().strip().splitlines()
    return labels

class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [ModelTypeEnum.ssd, ModelTypeEnum.yolonas, ModelTypeEnum.yolox, ModelTypeEnum.yolov8, ModelTypeEnum.yolov11,  ModelTypeEnum.yolov11_humanattr]

    def __init__(self, detector_config: OvDetectorConfig):
        self.ov_core = ov.Core()
        self.ov_model_type = detector_config.model.model_type
        
        self.detector_config = detector_config  # Store the detector_config as an instance variable
        self.human_attr_model = None  # Initialize the human attribute model variable
 
        self.h = detector_config.model.height
        self.w = detector_config.model.width

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

        # Add the YOLOv8 output processing here
        if self.ov_model_type in (ModelTypeEnum.yolov8, ModelTypeEnum.yolov11):
            #print("Reached YOLOv8 model processing")
            print("Size of tensor_input:", tensor_input.shape)  # Add this line to print the size

            out_tensor = infer_request.get_output_tensor()
            results = out_tensor.data[0]
            output_data = np.transpose(results)
            scores = np.max(output_data[:, 4:], axis=1)
            if len(scores) == 0:
                return np.zeros((20, 6), np.float32)
            scores = np.expand_dims(scores, axis=1)
            # add scores to the last column
            dets = np.concatenate((output_data, scores), axis=1)
            # filter out lines with scores below threshold
            dets = dets[dets[:, -1] > 0.5, :]
            # limit to top 20 scores, descending order
            ordered = dets[dets[:, -1].argsort()[::-1]][:20]
            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )
            # Output the detections before returning
            #print("Detections:", detections)
            #print("Detections shape yolov8:", detections.shape)
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
            print("Size of tensor_input:", tensor_input.shape)  # Add this line to print the size

            #model_input_shape = self.interpreter.inputs[0].shape
            #   print("Model input shape:", model_input_shape)  # for debug

            # Extract human attribute model parameters from detector_config
            human_attr_model_path = self.detector_config.model.human_attr_model_path
            human_attr_labelmap_path = self.detector_config.model.human_attr_labelmap_path
            human_attr_width = self.detector_config.model.human_attr_width
            human_attr_height = self.detector_config.model.human_attr_height
            
            out_tensor = infer_request.get_output_tensor()
            results = out_tensor.data[0]
            output_data = np.transpose(results)
            scores = np.max(output_data[:, 4:], axis=1)

            if len(scores) == 0:
                return np.zeros((20, 6), np.float32)
            scores = np.expand_dims(scores, axis=1)
            # add scores to the last column
            dets = np.concatenate((output_data, scores), axis=1)
            # filter out lines with scores below threshold
            dets = dets[dets[:, -1] > 0.8, :]
            # limit to top 20 scores, descending order
            ordered = dets[dets[:, -1].argsort()[::-1]][:20]
            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )

            #return detections
            # Process the output tensor
            print("Completed yolov11 detection")  # for debug
            # Filter person detections first
            person_detections = [d for d in detections if d[0] == 0]  # class_id == 1 for person
            if not person_detections:
                return detections

            processed_object_ids = set()

            for detection in person_detections:
                print("Reached human_attr")  # for debug
                self.human_attr_model = ov.Core().compile_model(human_attr_model_path, "CPU")
                # Load the human attribute labels
                human_attr_labels = load_labels(human_attr_labelmap_path)

                _, _, y_min, x_min, y_max, x_max = detection
                object_id = detection[0]  # Assuming the first element is a unique ID for the object

                # Check if the object has already been processed
                if object_id in processed_object_ids:
                    print(f"Skipping detection for object ID {object_id} as it has already been processed.")
                    continue  # Skip to the next detection

                # Convert tensor_input to a NumPy array for indexing
                tensor_input_np = np.array(tensor_input.data)  # Convert to NumPy array
                # Access the first image in the batch
                image_to_save = tensor_input_np[0]
                # Crop the image using the bounding box coordinates
                crop = image_to_save[int(y_min * 640):int(y_max * 640), int(x_min * 640):int(x_max * 640)]

                # Resize the cropped image to the required dimensions for the model
                resized_crop = cv2.resize(crop, (human_attr_width, human_attr_height))

                # Preprocess the resized image for the model (CHW format and normalization)
                processed_crop = resized_crop.transpose(2, 0, 1).astype(np.float32) / 255.0  # Convert to CHW format and normalize
                processed_crop = np.expand_dims(processed_crop, axis=0)

                infer_request = self.human_attr_model.create_infer_request()
                # Set the input tensor for the infer request
                infer_request.set_input_tensor(ov.Tensor(processed_crop))

                # Perform inference
                infer_request.infer()
                image_attr = infer_request.get_output_tensor(0).data

                detected_labels = []
                confidence_intervals = []
                bounding_boxes = ([x_min, y_min, x_max, y_max]) 
                scores = image_attr.flatten()
                for i, score in enumerate(scores):
                    if score > 0.5:
                        detected_labels.append(human_attr_labels[i])
                        confidence_intervals.append(score)

                # Save the processed object ID to the set
                processed_object_ids.add(object_id)

                save_cropped_images_and_write_csv(crop, detected_labels, confidence_intervals, bounding_boxes)

            return detections
 
