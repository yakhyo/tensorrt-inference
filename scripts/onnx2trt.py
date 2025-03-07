import tensorrt as trt
import os


def convert_onnx_to_tensorrt(onnx_path, engine_path, precision="fp16", workspace_size=1 << 30):
    """
    Convert ONNX model to TensorRT engine for TensorRT 10.x.

    Args:
        onnx_path (str): Path to the ONNX model file
        engine_path (str): Path to save the TensorRT engine file
        precision (str): Precision mode ("fp32", "fp16", or "int8")
        workspace_size (int): Maximum workspace size in bytes

    Returns:
        bool: True if conversion was successful
    """
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.INFO)

    # Create builder and network
    builder = trt.Builder(logger)

    # Create network definition with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Create config
    config = builder.create_builder_config()

    # Set memory pool limit for workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Set precision mode
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        else:
            print("FP16 not supported on this platform, using FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
        else:
            print("INT8 not supported on this platform, using FP32")
    else:
        print("Using FP32 precision")

    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        model_bytes = f.read()
        if not parser.parse(model_bytes):
            print("ONNX parsing errors:")
            for error in range(parser.num_errors):
                print(f"Error {error}: {parser.get_error(error)}")
            return False

    # Create optimization profile (fix for dynamic input error)
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        shape = list(input_tensor.shape)

        # Check if the shape contains dynamic dimensions (-1)
        if -1 in shape:
            min_shape = [1, 3, 320, 320]  # Smallest acceptable input
            opt_shape = [1, 3, 640, 640]  # Default training shape
            max_shape = [4, 3, 1280, 1280]  # Largest supported shape

            print(f"Setting optimization profile for {input_tensor.name}:")
            print(f"  Min shape: {min_shape}")
            print(f"  Opt shape: {opt_shape}")
            print(f"  Max shape: {max_shape}")

            profile.set_shape(
                input_tensor.name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape
            )

    config.add_optimization_profile(profile)

    print(f"Building TensorRT engine: {engine_path}")

    # Build serialized network
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build TensorRT engine")
        return False

    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

    # Save engine to file
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to: {engine_path}")
    return True


# Example usage
if __name__ == "__main__":
    # Replace with your actual paths
    model_path = "../weights/retinaface_mv2.onnx"
    engine_path = "../weights/retinaface_mv2.engine"

    convert_onnx_to_tensorrt(
        onnx_path=model_path,
        engine_path=engine_path,
        precision="fp16"
    )
