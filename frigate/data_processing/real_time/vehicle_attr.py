"""Handle processing images to classify vehicle attributes."""

import logging
import os
import openvino as ov
import openvino.properties as props
import cv2
import numpy as np

from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.config import FrigateConfig
from frigate.const import MODEL_CACHE_DIR
from frigate.util.object import calculate_region

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter

logger = logging.getLogger(__name__)


class VehicleAttrRealTimeProcessor(RealTimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics)
        self.interpreter: Interpreter = None
        self.sub_label_publisher = sub_label_publisher
        self.tensor_input_details: dict[str, any] = None
        self.tensor_output_details: dict[str, any] = None
        self.detected_vehicle_attr: dict[str, float] = {}
        self.labelmap: dict[int, str] = {}

        # Load model and label paths from the configuration
        self.model_path = config.classification.vehicle_attr.vehicle_attr_model_path
        self.label_path = config.classification.vehicle_attr.vehicle_attr_label_path

        print('vehicle model path', self.model_path)
        # Load the model and labels
        self.__build_detector()

    def __build_detector(self) -> None:
        self.interpreter = ov.Core().compile_model(self.model_path, "CPU")

        # self.interpreter.allocate_tensors()
        # self.tensor_input_details = self.interpreter.get_input_details()
        # self.tensor_output_details = self.interpreter.get_output_details()

        # Load labels from the label path
        with open(self.label_path) as f:
            for i, line in enumerate(f):
                self.labelmap[i] = line.strip()  # Store labels in the labelmap

    def process_frame(self, obj_data, frame):

        print('inside process_frame vehicle attr')
        # Only process if the label is a vehicle category from COCO
        vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
        if obj_data["label"] not in vehicle_labels:
            return

        x, y, x2, y2 = calculate_region(
            frame.shape,
            obj_data["box"][0],
            obj_data["box"][1],
            obj_data["box"][2],
            obj_data["box"][3],
            224,
            1.0,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        input = rgb[
            y:y2,
            x:x2,
        ]

        if input.shape != (256, 192):
            input = cv2.resize(input, (256, 192))
        
        input = input.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input = (input - mean) / std
        input = np.transpose(input, (2,0,1))
        input = np.expand_dims(input, axis=0)

        infer_request = self.interpreter.create_infer_request()
        infer_request.set_input_tensor(ov.Tensor(input))
        infer_request.infer()
        image_attr = infer_request.get_output_tensor(0).data

        # Flatten the scores
        scores = image_attr.flatten()
        color_probs = scores[:10]
        color_idx = np.argmax(color_probs)
        # color_score = color_probs[color_idx]

        type_probs = scores[10:]
        type_idx = np.argmax(type_probs)
        # type_score = type_probs[type_idx]

        # Initialize a list to store labels with scores > 0.7
        detected_labels = []
        detected_labels.append(self.labelmap[color_idx])
        detected_labels.append(self.labelmap[type_idx+10])

        # If you want to do something with the detected labels, you can add that logic here
        if detected_labels:
            # Publish the entire list of detected labels
            self.sub_label_publisher.publish(
                EventMetadataTypeEnum.sub_label,
                (obj_data["id"], detected_labels, None)  # Pass the list of labels
            )

        # Continue with the rest of your processing logic...

    def handle_request(self, topic, request_data):
        return None

    def expire_object(self, object_id):
        if object_id in self.detected_vehicle_attr:
            self.detected_vehicle_attr.pop(object_id)
