#!/usr/bin/env python3
"""
EdgeTAM TensorRT Conversion Script

This script converts ONNX models to TensorRT engine format for optimized inference.
TensorRT provides significant speedup on NVIDIA GPUs.

Prerequisites:
    - NVIDIA GPU with CUDA support
    - TensorRT installation (pip install tensorrt or use NVIDIA container)

Usage:
    python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_engines
"""

import argparse
import os
from pathlib import Path


def convert_onnx_to_tensorrt(onnx_path, output_path, fp16=False, int8=False, max_batch_size=1, workspace_size=4):
    """
    Convert ONNX model to TensorRT engine

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to output TensorRT engine
        fp16: Enable FP16 precision mode
        int8: Enable INT8 precision mode (requires calibration)
        max_batch_size: Maximum batch size for the engine
        workspace_size: Maximum workspace size in GB
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("✗ TensorRT not installed!")
        print("  Install TensorRT from: https://developer.nvidia.com/tensorrt")
        print("  Or use: pip install tensorrt")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"\nConverting {onnx_path} to TensorRT engine...")
    print(f"  FP16 mode: {fp16}")
    print(f"  INT8 mode: {int8}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Workspace size: {workspace_size} GB")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('✗ Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    print("✓ ONNX model parsed successfully")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # GB to bytes

    # Enable precision modes
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    elif fp16:
        print("⚠ FP16 requested but not supported on this platform")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("✓ INT8 mode enabled")
        print("⚠ INT8 requires calibration data for optimal accuracy")
    elif int8:
        print("⚠ INT8 requested but not supported on this platform")

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("✗ Failed to build TensorRT engine")
        return False

    # Save engine
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ TensorRT engine saved to {output_path}")

    # Print engine info
    engine_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Engine size: {engine_size_mb:.2f} MB")

    return True


def verify_tensorrt_engine(engine_path):
    """Verify TensorRT engine by loading it"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print(f"✗ Failed to load engine: {engine_path}")
            return False

        print(f"✓ Engine verified: {engine_path}")
        print(f"  Num bindings: {engine.num_bindings}")
        print(f"  Max batch size: {engine.max_batch_size}")

        # Print input/output info
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            print(f"  {'Input' if is_input else 'Output'} {i}: {name} - {dtype} - {shape}")

        return True

    except ImportError:
        print("⚠ PyCUDA not installed, skipping verification")
        print("  Install with: pip install pycuda")
        return False
    except Exception as e:
        print(f"✗ Engine verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT engines")
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="onnx_models",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tensorrt_engines",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision mode (faster, slight accuracy loss)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision mode (fastest, requires calibration)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=4,
        help="Maximum workspace size in GB",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted TensorRT engines",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = Path(args.onnx_dir)

    print("=" * 60)
    print("EdgeTAM TensorRT Conversion")
    print("=" * 60)
    print(f"ONNX directory: {args.onnx_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Precision: {'FP16' if args.fp16 else 'INT8' if args.int8 else 'FP32'}")
    print("=" * 60)

    # Check if TensorRT is available
    try:
        import tensorrt as trt
        print(f"✓ TensorRT version: {trt.__version__}")
    except ImportError:
        print("✗ TensorRT not installed!")
        print("\nInstallation options:")
        print("1. pip install tensorrt")
        print("2. Use NVIDIA NGC container: nvcr.io/nvidia/tensorrt:xx.xx-py3")
        print("3. Download from: https://developer.nvidia.com/tensorrt")
        return

    # Models to convert
    models = [
        ("edgetam_image_encoder.onnx", "edgetam_image_encoder.trt"),
        ("edgetam_mask_decoder.onnx", "edgetam_mask_decoder.trt"),
    ]

    success_count = 0

    # Convert each model
    for i, (onnx_name, trt_name) in enumerate(models, 1):
        onnx_path = onnx_dir / onnx_name
        trt_path = output_dir / trt_name

        if not onnx_path.exists():
            print(f"\n[{i}/{len(models)}] ⚠ ONNX model not found: {onnx_path}")
            print(f"  Run export_to_onnx.py first to generate ONNX models")
            continue

        print(f"\n[{i}/{len(models)}] Converting {onnx_name}...")

        if convert_onnx_to_tensorrt(
            onnx_path=str(onnx_path),
            output_path=str(trt_path),
            fp16=args.fp16,
            int8=args.int8,
            max_batch_size=args.max_batch_size,
            workspace_size=args.workspace_size,
        ):
            success_count += 1

            # Verify if requested
            if args.verify:
                print(f"\nVerifying {trt_name}...")
                verify_tensorrt_engine(str(trt_path))

    print("\n" + "=" * 60)
    if success_count == len(models):
        print(f"✓ All models converted successfully! ({success_count}/{len(models)})")
    else:
        print(f"⚠ Converted {success_count}/{len(models)} models")
    print("=" * 60)

    if success_count > 0:
        print("\nGenerated TensorRT engines:")
        for _, trt_name in models:
            trt_path = output_dir / trt_name
            if trt_path.exists():
                size_mb = trt_path.stat().st_size / (1024 * 1024)
                print(f"  {trt_path} ({size_mb:.2f} MB)")

        print("\nNext steps:")
        print("1. Verify engines:")
        print(f"   python convert_to_tensorrt.py --onnx-dir {args.onnx_dir} --verify")
        print("\n2. Run inference:")
        print(f"   python deploy/simple_inference.py --engine-dir {args.output_dir}")

        print("\nPerformance tips:")
        print("  - Use --fp16 for ~2x speedup with minimal accuracy loss")
        print("  - Use --int8 for ~4x speedup (requires calibration)")
        print("  - Increase --workspace-size for better optimization (default: 4GB)")


if __name__ == "__main__":
    main()
