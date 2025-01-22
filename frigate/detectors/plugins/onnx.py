import glob
import logging

import numpy as np
from pydantic import Field
from typing_extensions import Literal
import pandas as pd
import cv2

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    ModelTypeEnum,
)
from frigate.util.model import get_ort_providers
from frigate.detectors.util import preprocess, yolov8_postprocess

logger = logging.getLogger(__name__)

DETECTOR_KEY = "onnx"


class ONNXDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="AUTO", title="Device Type")


def write_attributes_to_excel(detected_attributes, output_file="human_attributes.xlsx"):
    df = pd.DataFrame([attr["attributes"] for attr in detected_attributes])
    df.to_excel(output_file, index=False)


class ONNXDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: ONNXDetectorConfig):
        try:
            import onnxruntime as ort

            logger.info("ONNX: loaded onnxruntime module")
        except ModuleNotFoundError:
            logger.error(
                "ONNX: module loading failed, need 'pip install onnxruntime'?!?"
            )
            raise

        assert (
            detector_config.model.model_type == "yolov8"
        ), "ONNX: detector_config.model.model_type: only yolov8 supported"
        assert (
            detector_config.model.input_tensor == "nhwc"
        ), "ONNX: detector_config.model.input_tensor: only nhwc supported"
        if detector_config.model.input_pixel_format != "rgb":
            logger.warn(
                "ONNX: detector_config.model.input_pixel_format: should be 'rgb' for yolov8, but '{detector_config.model.input_pixel_format}' specified!"
            )

        assert detector_config.model.path is not None, (
            "ONNX: No model.path configured, please configure model.path and model.labelmap_path; some suggestions: "
            + ", ".join(glob.glob("/config/model_cache/yolov8/*.onnx"))
            + " and "
            + ", ".join(glob.glob("/config/model_cache/yolov8/*_labels.txt"))
        )

        path = detector_config.model.path
        logger.info(f"ONNX: loading {detector_config.model.path}")

        providers, options = get_ort_providers(
            detector_config.device == "CPU", detector_config.device
        )
        self.model = ort.InferenceSession(
            path, providers=providers, provider_options=options
        )

        self.h = detector_config.model.height
        self.w = detector_config.model.width
        self.onnx_model_type = detector_config.model.model_type
        self.onnx_model_px = detector_config.model.input_pixel_format
        self.onnx_model_shape = detector_config.model.input_tensor
        path = detector_config.model.path

        logger.info(f"ONNX: {path} loaded")

    def detect_raw(self, tensor_input: np.ndarray):
        model_input_name = self.model.get_inputs()[0].name
        if self.onnx_model_type == ModelTypeEnum.yolonas:
            tensor_output = self.model.run(None, {model_input_name: tensor_input})

            predictions = tensor_output[0]

            detections = np.zeros((20, 6), np.float32)

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
        elif self.ov_model_type in (ModelTypeEnum.yolov8, ModelTypeEnum.yolov11):
            model_input_shape = self.model.get_inputs()[0].shape
            
            #print("Reached onnx.py yolov8") #for debug
 
            tensor_input = preprocess(tensor_input, model_input_shape, np.float32)

            tensor_output = self.model.run(None, {model_input_name: tensor_input})[0]

            return yolov8_postprocess(model_input_shape, tensor_output)
        
        elif self.onnx_model_type == ModelTypeEnum.yolov11_humanattr:
            print("Reached onnx.py yolovv11_humanattr") #for debug
            model_input_shape = self.model.get_inputs()[0].shape
            print("Reached onnx.py yolov11_humanattr") 
            tensor_input = preprocess(tensor_input, model_input_shape, np.float32)
            tensor_output = self.model.run(None, {model_input_name: tensor_input})[0]
            detections = yolov8_postprocess(model_input_shape, tensor_output) 
            print("Completed yolov11 detection") #for debug
            # Filter person detections first
            person_detections = [d for d in detections if d[0] == 1]  # class_id == 1 for person
            if not person_detections:
                return detections
                
            # Prepare batch of crops
            batch_crops = []
            for detection in person_detections:
                print("Reached human_attr") #for debug
                _, _, y_min, x_min, y_max, x_max = detection
                crop = tensor_input[0, :, 
                                  int(y_min * self.h):int(y_max * self.h), 
                                  int(x_min * self.w):int(x_max * self.w)]
                
                # Resize and preprocess
                resized_crop = cv2.resize(crop.transpose(1, 2, 0), (192, 256))
                processed_crop = resized_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
                batch_crops.append(processed_crop)
            
            # Stack all crops into a single batch
            if batch_crops:
                batch_input = np.stack(batch_crops, axis=0)
                
                # Run human attribute detection on entire batch
                batch_attributes = self.human_attr_model.run(None, 
                    {self.human_attr_model.get_inputs()[0].name: batch_input})[0]
                
                # Store results
                detected_attributes = []
                for i, detection in enumerate(person_detections):
                    _, _, y_min, x_min, y_max, x_max = detection
                    detected_attributes.append({
                        "bbox": [x_min, y_min, x_max, y_max],
                        "attributes": batch_attributes[i]
                    })
                
                write_attributes_to_excel(detected_attributes)
            
            return detections

        else:
            tensor_output = self.model.run(None, {model_input_name: tensor_input})
