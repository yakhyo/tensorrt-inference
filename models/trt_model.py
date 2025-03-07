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


class TRTEngine:
    """TensorRT engine wrapper for inference."""
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine.
        
        Args:
            engine_path (str): Path to the TensorRT engine file (.engine)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Print TensorRT version for debugging
        print(f"TensorRT version: {trt.__version__}")
        print(f"Available engine attributes: {dir(self.engine)}")
        
        # Try to determine the correct way to iterate through bindings
        try:
            # For newer TensorRT versions
            num_bindings = self.engine.num_io_tensors
            print(f"Using num_io_tensors: {num_bindings}")
            
            # Setup input and output bindings using newer API
            for binding_idx in range(num_bindings):
                binding_name = self.engine.get_tensor_name(binding_idx)
                shape = self.engine.get_tensor_shape(binding_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
                
                print(f"Binding {binding_idx}: name={binding_name}, shape={shape}, dtype={dtype}")
                
                # Calculate size, ensuring it's positive
                size = abs(trt.volume(shape))
                
                if size <= 0:
                    print(f"Warning: Size calculation for binding {binding_name} resulted in {size}, setting to default size")
                    size = 1024 * 1024  # Set a reasonable default size
                
                print(f"Calculated size: {size}")
                
                # Get the correct item size based on the data type
                if dtype == np.float32:
                    item_size = 4
                elif dtype == np.float16:
                    item_size = 2
                elif dtype == np.int32:
                    item_size = 4
                elif dtype == np.int8:
                    item_size = 1
                else:
                    # Default to float32 size if type is unknown
                    item_size = 4
                
                print(f"Item size for dtype {dtype}: {item_size}")
                print(f"Total allocation size: {size * item_size}")
                
                # Ensure the allocation size is positive
                alloc_size = size * item_size
                if alloc_size <= 0:
                    print(f"Warning: Allocation size {alloc_size} is invalid, setting to default")
                    alloc_size = 4 * 1024 * 1024  # 4MB default
                
                # Allocate memory for input and output
                try:
                    memory = cuda.mem_alloc(alloc_size)
                    print(f"Memory allocated successfully: {alloc_size} bytes")
                except Exception as e:
                    print(f"Memory allocation failed with: {e}")
                    # Try a fallback default size
                    memory = cuda.mem_alloc(4 * 1024 * 1024)  # 4MB default
                
                # Append to the binding arrays
                self.bindings.append(int(memory))
                
                # Check if binding is input or output
                is_input = self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT
                if is_input:
                    self.inputs.append({
                        'memory': memory, 
                        'size': size, 
                        'dtype': dtype, 
                        'shape': shape, 
                        'name': binding_name
                    })
                    print(f"Added as input: {binding_name}")
                else:
                    self.outputs.append({
                        'memory': memory, 
                        'size': size, 
                        'dtype': dtype, 
                        'shape': shape, 
                        'name': binding_name
                    })
                    print(f"Added as output: {binding_name}")
                
        except Exception as e:
            print(f"Newer API approach failed with: {e}")
            print("Falling back to legacy API...")
            
            # Fallback for older TensorRT versions
            try:
                for binding in range(self.engine.num_bindings):
                    size = trt.volume(self.engine.get_binding_shape(binding))
                    print(f"Legacy binding {binding}: shape={self.engine.get_binding_shape(binding)}, size={size}")
                    
                    # Ensure size is positive
                    size = abs(size)
                    if size <= 0:
                        size = 1024 * 1024  # Default size
                    
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    
                    # Get the correct item size based on the data type
                    if dtype == np.float32:
                        item_size = 4
                    elif dtype == np.float16:
                        item_size = 2
                    elif dtype == np.int32:
                        item_size = 4
                    elif dtype == np.int8:
                        item_size = 1
                    else:
                        # Default to float32 size if type is unknown
                        item_size = 4
                    
                    # Ensure memory allocation size is positive
                    alloc_size = size * item_size
                    if alloc_size <= 0:
                        alloc_size = 4 * 1024 * 1024  # 4MB default
                    
                    # Allocate memory
                    memory = cuda.mem_alloc(alloc_size)
                    
                    # Append to the binding arrays
                    self.bindings.append(int(memory))
                    
                    if self.engine.binding_is_input(binding):
                        self.inputs.append({
                            'memory': memory, 
                            'size': size, 
                            'dtype': dtype, 
                            'shape': self.engine.get_binding_shape(binding)
                        })
                        print(f"Added as legacy input: binding {binding}")
                    else:
                        self.outputs.append({
                            'memory': memory, 
                            'size': size, 
                            'dtype': dtype, 
                            'shape': self.engine.get_binding_shape(binding)
                        })
                        print(f"Added as legacy output: binding {binding}")
            except Exception as e:
                print(f"Legacy API approach also failed with: {e}")
                raise RuntimeError("Could not initialize TensorRT engine with any API version")
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on input data.
        
        Args:
            input_data (np.ndarray): Input data as numpy array.
            
        Returns:
            List[np.ndarray]: List of output numpy arrays.
        """
        # Ensure input data is contiguous in memory
        if not input_data.flags['C_CONTIGUOUS']:
            input_data = np.ascontiguousarray(input_data)
        
        # Get expected input shape and handle dynamic dimensions
        expected_shape = self.inputs[0]['shape']
        current_shape = input_data.shape
        
        print(f"Expected input shape (with dynamic dims): {expected_shape}")
        print(f"Provided input shape: {current_shape}")
        
        # Set the dynamic input dimensions in the execution context
        if expected_shape[0] == -1 or expected_shape[2] == -1 or expected_shape[3] == -1:
            try:
                # Try the newer API first
                input_name = self.inputs[0].get('name')
                if input_name:
                    self.context.set_input_shape(input_name, current_shape)
                    print(f"Set dynamic input shape for {input_name} to: {current_shape}")
                else:
                    print("Input name not available, trying alternative methods")
                    # Fall back to older APIs or alternatives
                    if hasattr(self.context, 'set_binding_shape'):
                        self.context.set_binding_shape(0, current_shape)
                        print(f"Set dynamic binding shape to: {current_shape}")
            except Exception as e:
                print(f"Failed to set dynamic shape: {e}")
                print("Continuing with static shape handling")
        
        # Ensure data type matches
        if input_data.dtype != self.inputs[0]['dtype']:
            print(f"Converting input data from {input_data.dtype} to {self.inputs[0]['dtype']}")
            input_data = input_data.astype(self.inputs[0]['dtype'])
        
        try:
            # Transfer input data to device
            cuda.memcpy_htod_async(self.inputs[0]['memory'], input_data, self.stream)
            
            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Get output shapes (may be dynamic)
            output_shapes = []
            for i, output in enumerate(self.outputs):
                # Use the shape from output dict if available, otherwise try to get it from context
                if 'shape' in output:
                    output_shape = output['shape']
                    # Replace any -1 dimensions with actual values based on input
                    if -1 in output_shape:
                        # This is a simplified approach - actual shape inference might be more complex
                        output_shape = list(output_shape)
                        for j, dim in enumerate(output_shape):
                            if dim == -1:
                                # Replace with corresponding input dimension if possible
                                if j < len(current_shape):
                                    output_shape[j] = current_shape[j]
                                else:
                                    output_shape[j] = 1  # Default fallback
                        output_shape = tuple(output_shape)
                else:
                    # Fallback to a reasonable default based on size
                    output_shape = (output['size'],)
                
                output_shapes.append(output_shape)
                print(f"Output {i} shape: {output_shape}")
            
            # Transfer outputs from device to host
            outputs = []
            for i, output in enumerate(self.outputs):
                # Use the actual output shape for size calculation
                output_size = int(np.prod(output_shapes[i]))
                output_data = np.empty(output_size, dtype=output['dtype'])
                cuda.memcpy_dtoh_async(output_data, output['memory'], self.stream)
                outputs.append(output_data)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Reshape outputs to match the actual output shapes
            reshaped_outputs = []
            for i, output_data in enumerate(outputs):
                try:
                    reshaped = output_data.reshape(output_shapes[i])
                    reshaped_outputs.append(reshaped)
                except Exception as e:
                    print(f"Error reshaping output {i}: {e}")
                    # Fallback: keep as flat array
                    reshaped_outputs.append(output_data)
            
            return reshaped_outputs
        except Exception as e:
            print(f"Inference error: {e}")
            print(f"Input data summary: shape={input_data.shape}, dtype={input_data.dtype}, range=[{input_data.min()}, {input_data.max()}]")
            raise

class RetinaFaceTRT:
    """
    A class for face detection using the RetinaFace model with TensorRT acceleration.

    Args:
        engine_path (str): Path to the TensorRT engine file.
        conf_thresh (float): Confidence threshold for detections. Defaults to 0.5.
        nms_thresh (float): Non-maximum suppression threshold. Defaults to 0.4.
        pre_nms_topk (int): Maximum number of detections before NMS. Defaults to 5000.
        post_nms_topk (int): Maximum number of detections after NMS. Defaults to 750.
        dynamic_size (Optional[bool]): Whether to adjust anchor generation dynamically based on image size. Defaults to False.
        input_size (Optional[Tuple[int, int]]): Static input size for the model (width, height). Defaults to (640, 640).

    Attributes:
        conf_thresh (float): Confidence threshold for filtering detections.
        nms_thresh (float): Threshold for NMS to remove duplicate detections.
        pre_nms_topk (int): Maximum detections to consider before applying NMS.
        post_nms_topk (int): Maximum detections retained after applying NMS.
        dynamic_size (bool): Indicates if input size and anchors are dynamically adjusted.
        input_size (Tuple[int, int]): The model's input image size.
        _engine_path (str): Path to the TensorRT engine.
        _priors (torch.Tensor): Precomputed anchor boxes for static input size.
    """

    def __init__(
        self,
        engine_path: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        dynamic_size: Optional[bool] = False,
        input_size: Optional[Tuple[int, int]] = (640, 640),  # Default input size if dynamic_size=False
    ) -> None:

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.dynamic_size = dynamic_size
        self.input_size = input_size

        Logger.info(
            f"Initializing RetinaFaceTRT with engine={engine_path}, conf_thresh={conf_thresh}, nms_thresh={nms_thresh}, "
            f"pre_nms_topk={pre_nms_topk}, post_nms_topk={post_nms_topk}, dynamic_size={dynamic_size}, "
            f"input_size={input_size}"
        )

        # Precompute anchors if using static size
        if not dynamic_size and input_size is not None:
            self._priors = generate_anchors(image_size=input_size)
            Logger.debug("Generated anchors for static input size.")

        # Initialize TensorRT engine
        self._initialize_engine(engine_path)

    def _initialize_engine(self, engine_path: str) -> None:
        """
        Initializes a TensorRT engine from the given path.

        Args:
            engine_path (str): The file path to the TensorRT engine.

        Raises:
            RuntimeError: If the engine fails to load, logs an error and raises an exception.
        """
        try:
            self.engine = TRTEngine(engine_path)
            Logger.info(f"Successfully initialized the TensorRT engine from {engine_path}")
        except Exception as e:
            Logger.error(f"Failed to load TensorRT engine from '{engine_path}': {e}")
            raise RuntimeError(f"Failed to initialize TensorRT engine for '{engine_path}'") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, C, H, W)
        """
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            List[np.ndarray]: Raw model outputs.
        """
        return self.engine.infer(input_tensor)

    def detect(
        self,
        image: np.ndarray,
        max_num: Optional[int] = 0,
        metric: Literal["default", "max"] = "default",
        center_weight: Optional[float] = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform face detection on an input image and return bounding boxes and landmarks.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (height, width, channels).
            max_num (int, optional): Maximum number of detections to return. Defaults to 0.
            metric (str, optional): Metric for ranking detections when `max_num` is specified. 
                Options:
                - "default": Prioritize detections closer to the image center.
                - "max": Prioritize detections with larger bounding box areas.
            center_weight (float, optional): Weight for penalizing detections farther from the image center 
                when using the "default" metric. Defaults to 2.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Detection results containing:
                - detections (np.ndarray): Array of detected bounding boxes with confidence scores.
                Shape: (num_detections, 5), where each row is [x_min, y_min, x_max, y_max, score].
                - landmarks (np.ndarray): Array of detected facial landmarks.
                Shape: (num_detections, 5, 2), where each row contains 5 landmark points (x, y).
        """

        if self.dynamic_size:
            height, width, _ = image.shape
            self._priors = generate_anchors(image_size=(height, width))  # generate anchors for each input image
            resize_factor = 1.0  # No resizing
        else:
            image, resize_factor = resize_image(image, target_shape=self.input_size)

        height, width, _ = image.shape
        image_tensor = self.preprocess(image)

        # TensorRT inference
        outputs = self.inference(image_tensor)

        # Postprocessing
        detections, landmarks = self.postprocess(outputs, resize_factor, shape=(width, height))

        if max_num > 0 and detections.shape[0] > max_num:
            # Calculate area of detections
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

            # Calculate offsets from image center
            center = (height // 2, width // 2)
            offsets = np.vstack([
                (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                (detections[:, 1] + detections[:, 3]) / 2 - center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            # Calculate scores based on the chosen metric
            if metric == 'max':
                scores = areas
            else:
                scores = areas - offset_dist_squared * center_weight

            # Sort by scores and select top `max_num`
            sorted_indices = np.argsort(scores)[::-1][:max_num]

            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        return detections, landmarks

    def postprocess(self, outputs: List[np.ndarray], resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the model outputs into final detection results.

        Args:
            outputs (List[np.ndarray]): Raw outputs from the detection model.
                - outputs[0]: Location predictions (bounding box coordinates).
                - outputs[1]: Class confidence scores.
                - outputs[2]: Landmark predictions.
            resize_factor (float): Factor used to resize the input image during preprocessing.
            shape (Tuple[int, int]): Original shape of the image as (height, width).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed results containing:
                - detections (np.ndarray): Array of detected bounding boxes with confidence scores.
                Shape: (num_detections, 5), where each row is [x_min, y_min, x_max, y_max, score].
                - landmarks (np.ndarray): Array of detected facial landmarks.
                Shape: (num_detections, 5, 2), where each row contains 5 landmark points (x, y).
        """
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Decode boxes and landmarks
        boxes = decode_boxes(torch.tensor(loc), self._priors).cpu().numpy()
        landmarks = decode_landmarks(torch.tensor(landmarks), self._priors).cpu().numpy()

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, shape=(shape[0], shape[1]))

        # Extract confidence scores for the face class
        scores = conf[:, 1]
        mask = scores > self.conf_thresh

        # Filter by confidence threshold
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        # Sort by scores
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k detections
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.int32)

        return detections, landmarks

    def _scale_detections(self, boxes: np.ndarray, landmarks: np.ndarray, resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Scale bounding boxes and landmarks to the original image size."""
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks

