#!/usr/bin/env python3
"""
EdgeTAM PyTorch Inference

Simple inference example using the original PyTorch model.
This is the reference implementation - use ONNX or TensorRT for production.

Usage:
    python deploy/pytorch_inference.py --image path/to/image.jpg
"""

import argparse
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def simulate_image_stream(predictor, device, num_frames=10, frame_size=(1024, 768)):
    """
    Simulate processing of an image stream

    Args:
        predictor: SAM2ImagePredictor instance
        device: Device (cpu or cuda)
        num_frames: Number of frames to simulate
        frame_size: Size of simulated frames (width, height)
    """
    print("\n" + "=" * 60)
    print("Simulating Image Stream Processing (PyTorch)")
    print("=" * 60)
    print(f"Number of frames: {num_frames}")
    print(f"Frame size: {frame_size}")
    print(f"Device: {device}")

    # Simulate point prompt (center of image)
    point_coords = np.array([[frame_size[0] // 2, frame_size[1] // 2]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    total_time = 0
    encoding_time = 0
    decoding_time = 0

    for i in range(num_frames):
        # Simulate incoming frame
        frame = np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # Set image (includes encoding)
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            predictor.set_image(frame)
        t1 = time.time()
        encoding_time += (t1 - t0)

        # Predict mask
        t2 = time.time()
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        t3 = time.time()
        decoding_time += (t3 - t2)

        total_time += (t3 - t0)

        if (i + 1) % 5 == 0:
            print(f"  Processed frame {i + 1}/{num_frames}")

    # Print statistics
    avg_fps = num_frames / total_time
    avg_encoding_ms = (encoding_time / num_frames) * 1000
    avg_decoding_ms = (decoding_time / num_frames) * 1000

    print("\n" + "=" * 60)
    print("Performance Statistics (PyTorch)")
    print("=" * 60)
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average encoding time: {avg_encoding_ms:.2f}ms")
    print(f"Average decoding time: {avg_decoding_ms:.2f}ms")
    print(f"Average total time per frame: {(total_time / num_frames) * 1000:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EdgeTAM PyTorch inference example")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/edgetam.pt",
        help="Path to EdgeTAM checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/edgetam.yaml",
        help="Path to EdgeTAM config",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (optional, uses simulation if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation mode",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames in simulation mode",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeTAM PyTorch Inference")
    print("=" * 60)

    # Build model
    print(f"Loading model from {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")

    model = build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
    )

    predictor = SAM2ImagePredictor(model)
    print("✓ Model loaded successfully")

    # Run simulation or single image mode
    if args.simulate or args.image is None:
        simulate_image_stream(predictor, args.device, num_frames=args.num_frames)
    else:
        # Load image
        print(f"\nLoading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"✗ Failed to load image: {args.image}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # Set image
        print("Setting image...")
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
        t1 = time.time()
        print(f"✓ Image embedding completed in {(t1 - t0) * 1000:.2f}ms")

        # Example point prompt (center of image)
        point_coords = np.array([[image.shape[1] // 2, image.shape[0] // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        print(f"Point prompt: {point_coords[0]}")

        # Predict
        print("Predicting mask...")
        t2 = time.time()
        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        t3 = time.time()
        print(f"✓ Prediction completed in {(t3 - t2) * 1000:.2f}ms")
        print(f"  Confidence score: {scores[0]:.3f}")

        # Visualize
        if args.output:
            mask = masks[0]
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = [0, 255, 0]
            result = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

            # Draw point
            cv2.circle(result, tuple(point_coords[0].astype(int)), 10, (0, 0, 255), -1)

            cv2.imwrite(args.output, result)
            print(f"✓ Result saved to {args.output}")

        print(f"\nTotal inference time: {(t3 - t0) * 1000:.2f}ms")


if __name__ == "__main__":
    main()
