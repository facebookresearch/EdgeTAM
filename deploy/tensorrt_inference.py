#!/usr/bin/env python3
"""
EdgeTAM TensorRT Inference

High-performance inference using TensorRT engines.
Provides significantly faster inference on NVIDIA GPUs.

Prerequisites:
    - NVIDIA GPU with CUDA support
    - TensorRT and pycuda installed

Usage:
    python deploy/tensorrt_inference.py --image path/to/image.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class EdgeTAMTensorRTInference:
    """EdgeTAM inference using TensorRT"""

    def __init__(self, encoder_path, decoder_path):
        """
        Initialize TensorRT inference

        Args:
            encoder_path: Path to image encoder TensorRT engine
            decoder_path: Path to mask decoder TensorRT engine
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT or PyCUDA not installed. Install TensorRT from NVIDIA "
                "and pycuda with: pip install pycuda"
            )

        self.cuda = cuda
        self.image_size = 1024

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load encoder engine
        print(f"Loading image encoder: {encoder_path}")
        with open(encoder_path, 'rb') as f:
            self.encoder_runtime = trt.Runtime(TRT_LOGGER)
            self.encoder_engine = self.encoder_runtime.deserialize_cuda_engine(f.read())
        self.encoder_context = self.encoder_engine.create_execution_context()

        # Load decoder engine
        print(f"Loading mask decoder: {decoder_path}")
        with open(decoder_path, 'rb') as f:
            self.decoder_runtime = trt.Runtime(TRT_LOGGER)
            self.decoder_engine = self.decoder_runtime.deserialize_cuda_engine(f.read())
        self.decoder_context = self.decoder_engine.create_execution_context()

        print("✓ TensorRT engines loaded successfully")

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate GPU buffers for inputs and outputs"""
        # Encoder buffers
        self.encoder_input = self.cuda.mem_alloc(1 * 3 * 1024 * 1024 * np.dtype(np.float32).itemsize)
        self.encoder_output = self.cuda.mem_alloc(1 * 256 * 64 * 64 * np.dtype(np.float32).itemsize)

        # Decoder buffers (we'll allocate dynamically based on num_points)
        # For now, allocate for max 10 points
        max_points = 10
        self.decoder_embeddings = self.cuda.mem_alloc(1 * 256 * 64 * 64 * np.dtype(np.float32).itemsize)
        self.decoder_coords = self.cuda.mem_alloc(1 * max_points * 2 * np.dtype(np.float32).itemsize)
        self.decoder_labels = self.cuda.mem_alloc(1 * max_points * np.dtype(np.int32).itemsize)
        self.decoder_masks = self.cuda.mem_alloc(1 * 1 * 1024 * 1024 * np.dtype(np.float32).itemsize)
        self.decoder_ious = self.cuda.mem_alloc(1 * 1 * np.dtype(np.float32).itemsize)

        print("✓ GPU buffers allocated")

    def preprocess_image(self, image):
        """
        Preprocess image for model input

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Preprocessed image array [1, 3, 1024, 1024]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image_resized = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Transpose to CHW format
        image_chw = np.transpose(image_normalized, (2, 0, 1))

        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0).astype(np.float32)

        return image_batch

    def encode_image(self, image):
        """
        Encode image to embeddings using TensorRT

        Args:
            image: Preprocessed image [1, 3, 1024, 1024]

        Returns:
            Image embeddings [1, 256, 64, 64]
        """
        # Copy input to GPU
        self.cuda.memcpy_htod(self.encoder_input, image)

        # Run inference
        self.encoder_context.execute_v2([
            int(self.encoder_input),
            int(self.encoder_output)
        ])

        # Copy output from GPU
        embeddings = np.empty((1, 256, 64, 64), dtype=np.float32)
        self.cuda.memcpy_dtoh(embeddings, self.encoder_output)

        return embeddings

    def predict_mask(self, embeddings, point_coords, point_labels):
        """
        Predict segmentation mask using TensorRT

        Args:
            embeddings: Image embeddings [1, 256, 64, 64]
            point_coords: Point coordinates [[x1, y1], ...] in image space
            point_labels: Point labels [1, 1, ...] (1=foreground, 0=background)

        Returns:
            masks: Predicted masks [1, 1, 1024, 1024]
            iou_predictions: IoU confidence scores [1, 1]
        """
        # Prepare inputs
        point_coords = np.array(point_coords, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.array(point_labels, dtype=np.int32).reshape(1, -1)

        # Copy inputs to GPU
        self.cuda.memcpy_htod(self.decoder_embeddings, embeddings)
        self.cuda.memcpy_htod(self.decoder_coords, point_coords)
        self.cuda.memcpy_htod(self.decoder_labels, point_labels)

        # Run inference
        self.decoder_context.execute_v2([
            int(self.decoder_embeddings),
            int(self.decoder_coords),
            int(self.decoder_labels),
            int(self.decoder_masks),
            int(self.decoder_ious)
        ])

        # Copy outputs from GPU
        masks = np.empty((1, 1, 1024, 1024), dtype=np.float32)
        iou_predictions = np.empty((1, 1), dtype=np.float32)
        self.cuda.memcpy_dtoh(masks, self.decoder_masks)
        self.cuda.memcpy_dtoh(iou_predictions, self.decoder_ious)

        return masks, iou_predictions

    def postprocess_mask(self, mask, original_size):
        """
        Postprocess mask to original image size

        Args:
            mask: Predicted mask [1, 1, 1024, 1024]
            original_size: Original image size (height, width)

        Returns:
            Binary mask at original resolution
        """
        # Remove batch and channel dimensions
        mask = mask[0, 0]

        # Resize to original size
        mask_resized = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Threshold to binary mask
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        return mask_binary


def simulate_image_stream(predictor, num_frames=100, frame_size=(1024, 768)):
    """
    Simulate high-performance processing of an image stream

    Args:
        predictor: EdgeTAM predictor instance
        num_frames: Number of frames to simulate
        frame_size: Size of simulated frames (width, height)
    """
    print("\n" + "=" * 60)
    print("Simulating High-Performance Image Stream")
    print("=" * 60)
    print(f"Number of frames: {num_frames}")
    print(f"Frame size: {frame_size}")

    # Simulate point prompt (center of image)
    point_coords = [[frame_size[0] // 2, frame_size[1] // 2]]
    point_labels = [1]

    # Pre-generate frames for consistent benchmarking
    print("Generating test frames...")
    frames = [
        np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    total_time = 0
    encoding_time = 0
    decoding_time = 0

    print("Processing frames...")
    for i, frame in enumerate(frames):
        # Preprocess
        image_input = predictor.preprocess_image(frame)

        # Encode image
        t0 = time.time()
        embeddings = predictor.encode_image(image_input)
        t1 = time.time()
        encoding_time += (t1 - t0)

        # Decode mask
        t2 = time.time()
        masks, iou_scores = predictor.predict_mask(embeddings, point_coords, point_labels)
        t3 = time.time()
        decoding_time += (t3 - t2)

        total_time += (t3 - t0)

        if (i + 1) % 20 == 0:
            current_fps = (i + 1) / total_time
            print(f"  Processed {i + 1}/{num_frames} frames - Current FPS: {current_fps:.2f}")

    # Print statistics
    avg_fps = num_frames / total_time
    avg_encoding_ms = (encoding_time / num_frames) * 1000
    avg_decoding_ms = (decoding_time / num_frames) * 1000

    print("\n" + "=" * 60)
    print("Performance Statistics (TensorRT)")
    print("=" * 60)
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average encoding time: {avg_encoding_ms:.2f}ms")
    print(f"Average decoding time: {avg_decoding_ms:.2f}ms")
    print(f"Average total time per frame: {(total_time / num_frames) * 1000:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EdgeTAM TensorRT inference example")
    parser.add_argument(
        "--encoder",
        type=str,
        default="tensorrt_engines/edgetam_image_encoder.trt",
        help="Path to image encoder TensorRT engine",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="tensorrt_engines/edgetam_mask_decoder.trt",
        help="Path to mask decoder TensorRT engine",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (optional, uses simulation if not provided)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation mode (process synthetic frames)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process in simulation mode",
    )

    args = parser.parse_args()

    # Initialize predictor
    print("=" * 60)
    print("EdgeTAM TensorRT Inference")
    print("=" * 60)

    try:
        predictor = EdgeTAMTensorRTInference(
            encoder_path=args.encoder,
            decoder_path=args.decoder,
        )
    except Exception as e:
        print(f"✗ Failed to initialize TensorRT inference: {e}")
        print("\nMake sure you have:")
        print("1. Generated TensorRT engines: python convert_to_tensorrt.py")
        print("2. Installed TensorRT and PyCUDA")
        print("3. NVIDIA GPU with CUDA support")
        return

    # Run simulation mode or single image mode
    if args.simulate or args.image is None:
        simulate_image_stream(predictor, num_frames=args.num_frames)
    else:
        # Load and process single image
        print(f"\nLoading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"✗ Failed to load image: {args.image}")
            return

        original_size = image.shape[:2]
        print(f"Image size: {original_size[1]}x{original_size[0]}")

        # Preprocess
        print("Preprocessing image...")
        image_input = predictor.preprocess_image(image)

        # Encode
        print("Encoding image...")
        t0 = time.time()
        embeddings = predictor.encode_image(image_input)
        t1 = time.time()
        print(f"✓ Encoding completed in {(t1 - t0) * 1000:.2f}ms")

        # Example point prompt
        point_coords = [[original_size[1] // 2, original_size[0] // 2]]
        point_labels = [1]

        # Predict mask
        print("Predicting mask...")
        t2 = time.time()
        masks, iou_scores = predictor.predict_mask(embeddings, point_coords, point_labels)
        t3 = time.time()
        print(f"✓ Prediction completed in {(t3 - t2) * 1000:.2f}ms")
        print(f"  IoU score: {iou_scores[0, 0]:.3f}")

        print(f"\nTotal inference time: {(t3 - t0) * 1000:.2f}ms")


if __name__ == "__main__":
    main()
