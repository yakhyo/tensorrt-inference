import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import torch
from typing import Tuple, List, Optional, Literal

from utils.log import Logger
from utils.visualization import draw_detections
from utils.common import (
    nms,
    resize_image,
    decode_boxes,
    generate_anchors,
    decode_landmarks
)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class RetinaFaceTRT:
    def __init__(
        self,
        engine_path: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        dynamic_size: Optional[bool] = False,
        input_size: Optional[Tuple[int, int]] = (640, 640)
    ) -> None:

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.dynamic_size = dynamic_size
        self.input_size = input_size

        Logger.info(f"Initializing RetinaFaceTRT with TensorRT engine: {engine_path}")

        # Precompute anchors
        if not dynamic_size and input_size is not None:
            self._priors = generate_anchors(image_size=input_size)

        self._load_engine(engine_path)

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # TensorRT >=10 uses get_tensor_shape instead of get_binding_shape
        input_tensor_name = self.engine.get_tensor_name(0)
        self.input_shape = self.engine.get_tensor_shape(input_tensor_name)
        self.input_size = (self.input_shape[-1], self.input_shape[-2])

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})

            self.bindings.append(int(device_mem))

        Logger.info("TensorRT engine loaded successfully.")


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        # Copy input data to GPU
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Explicitly set tensor addresses (TensorRT 10.x requirement)
        self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device']))
        for output in self.outputs:
            self.context.set_tensor_address(output['name'], int(output['device']))

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output data back to CPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)

        self.stream.synchronize()

        # Reshape outputs based on tensor shapes
        output_results = []
        for output in self.outputs:
            shape = self.engine.get_tensor_shape(output['name'])
            output_results.append(output['host'].reshape(shape))

        return output_results



    def detect(
        self,
        image: np.ndarray,
        max_num: Optional[int] = 0,
        metric: Literal["default", "max"] = "default",
        center_weight: Optional[float] = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.dynamic_size:
            height, width, _ = image.shape
            self._priors = generate_anchors(image_size=(height, width))
            resize_factor = 1.0
        else:
            image, resize_factor = resize_image(image, target_shape=self.input_size)

        height, width, _ = image.shape
        image_tensor = self.preprocess(image)

        outputs = self.inference(image_tensor)

        detections, landmarks = self.postprocess(outputs, resize_factor, shape=(width, height))

        if max_num > 0 and detections.shape[0] > max_num:
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            center = (height // 2, width // 2)
            offsets = np.vstack([
                (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                (detections[:, 1] + detections[:, 3]) / 2 - center[0]
            ])
            offset_dist_squared = np.sum(offsets ** 2, axis=0)

            scores = areas if metric == 'max' else areas - offset_dist_squared * center_weight
            sorted_indices = np.argsort(scores)[::-1][:max_num]

            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        return detections, landmarks

    def postprocess(self, outputs: List[np.ndarray], resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        boxes = decode_boxes(torch.tensor(loc), self._priors).cpu().numpy()
        landmarks = decode_landmarks(torch.tensor(landmarks), self._priors).cpu().numpy()

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, shape=(shape[0], shape[1]))

        scores = conf[:, 1]
        mask = scores > self.conf_thresh
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(detections, self.nms_thresh)
        detections, landmarks = detections[keep][:self.post_nms_topk], landmarks[keep][:self.post_nms_topk]

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.int32)

        return detections, landmarks

    def _scale_detections(self, boxes: np.ndarray, landmarks: np.ndarray, resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks
