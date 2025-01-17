import logging
import os

import numpy as np
import openvino as ov
import openvino.properties as props
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.detectors.util import preprocess, yolov8_postprocess

logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"


class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [ModelTypeEnum.ssd, ModelTypeEnum.yolonas, ModelTypeEnum.yolox, ModelTypeEnum.yolov8, ModelTypeEnum.yolov11]

    def __init__(self, detector_config: OvDetectorConfig):
        self.ov_core = ov.Core()
        self.ov_model_type = detector_config.model.model_type

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
        # TODO: see if we can use shared_memory=True
        if self.ov_model_type in (ModelTypeEnum.yolov8, ModelTypeEnum.yolov11):
            # Get the model input shape
            model_input_shape = self.interpreter.inputs[0].shape  # Get the input shape from the interpreter
            # Preprocess the input tensor
            input_tensor = preprocess(tensor_input, model_input_shape, np.float32)
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
