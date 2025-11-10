#!/usr/bin/env python3
"""
EdgeTAM ONNX Inference

Simple inference example using ONNX Runtime.
This script demonstrates how to use the exported EdgeTAM models for inference.

Usage:
    python deploy/onnx_inference.py --image path/to/image.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class EdgeTAMONNXInference:
    """EdgeTAM inference using ONNX Runtime"""

    def __init__(self, encoder_path, decoder_path, device='cpu'):
        """
        Initialize ONNX inference

        Args:
            encoder_path: Path to image encoder ONNX model
            decoder_path: Path to mask decoder ONNX model
            device: Device to run inference ('cpu' or 'cuda')
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNXRuntime not installed. Install with: pip install onnxruntime-gpu (for GPU) "
                "or pip install onnxruntime (for CPU)"
            )

        self.device = device
        self.image_size = 1024

        # Setup providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        print(f"Initializing ONNX Runtime with providers: {providers}")

        # Load models
        print(f"Loading image encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)

        print(f"Loading mask decoder: {decoder_path}")
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)

        print("✓ Models loaded successfully")

        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name

        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_names = [out.name for out in self.decoder_session.get_outputs()]

    def preprocess_image(self, image):
        """
        Preprocess image for model input

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Preprocessed image tensor [1, 3, 1024, 1024]
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
        image_batch = np.expand_dims(image_chw, axis=0)

        return image_batch

    def encode_image(self, image):
        """
        Encode image to embeddings

        Args:
            image: Preprocessed image [1, 3, 1024, 1024]

        Returns:
            Image embeddings [1, 256, 64, 64]
        """
        embeddings = self.encoder_session.run(
            [self.encoder_output_name],
            {self.encoder_input_name: image}
        )[0]

        return embeddings

    def predict_mask(self, embeddings, point_coords, point_labels):
        """
        Predict segmentation mask from embeddings and prompts

        Args:
            embeddings: Image embeddings [1, 256, 64, 64]
            point_coords: Point coordinates [[x1, y1], [x2, y2], ...] in image space
            point_labels: Point labels [1, 1, ...] (1=foreground, 0=background)

        Returns:
            masks: Predicted masks [1, 1, 1024, 1024]
            iou_predictions: IoU confidence scores [1, 1]
        """
        # Prepare inputs
        batch_size = 1
        point_coords = np.array(point_coords, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.array(point_labels, dtype=np.int32).reshape(1, -1)

        # Run inference
        outputs = self.decoder_session.run(
            self.decoder_output_names,
            {
                self.decoder_input_names[0]: embeddings,
                self.decoder_input_names[1]: point_coords,
                self.decoder_input_names[2]: point_labels,
            }
        )

        masks, iou_predictions = outputs[0], outputs[1]

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


def visualize_result(image, mask, point_coords, output_path=None):
    """
    Visualize segmentation result

    Args:
        image: Original image
        mask: Binary mask
        point_coords: Point prompts [[x, y], ...]
        output_path: Optional path to save result
    """
    # Create colored mask overlay
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [0, 255, 0]  # Green overlay

    # Blend with original image
    alpha = 0.5
    result = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # Draw point prompts
    for x, y in point_coords:
        cv2.circle(result, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red points

    # Save or display
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"✓ Result saved to {output_path}")
    else:
        cv2.imshow('EdgeTAM Segmentation', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def simulate_image_stream(predictor, num_frames=10, frame_size=(1024, 768)):
    """
    Simulate processing of an image stream

    Args:
        predictor: EdgeTAM predictor instance
        num_frames: Number of frames to simulate
        frame_size: Size of simulated frames (width, height)
    """
    print("\n" + "=" * 60)
    print("Simulating Image Stream Processing")
    print("=" * 60)
    print(f"Number of frames: {num_frames}")
    print(f"Frame size: {frame_size}")

    # Simulate point prompt (center of image)
    point_coords = [[frame_size[0] // 2, frame_size[1] // 2]]
    point_labels = [1]

    total_time = 0
    encoding_time = 0
    decoding_time = 0

    for i in range(num_frames):
        # Simulate incoming frame (random noise for demo)
        frame = np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)

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

        if (i + 1) % 5 == 0:
            print(f"  Processed frame {i + 1}/{num_frames}")

    # Print statistics
    avg_fps = num_frames / total_time
    avg_encoding_ms = (encoding_time / num_frames) * 1000
    avg_decoding_ms = (decoding_time / num_frames) * 1000

    print("\n" + "=" * 60)
    print("Performance Statistics")
    print("=" * 60)
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average encoding time: {avg_encoding_ms:.2f}ms")
    print(f"Average decoding time: {avg_decoding_ms:.2f}ms")
    print(f"Average total time per frame: {(total_time / num_frames) * 1000:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EdgeTAM ONNX inference example")
    parser.add_argument(
        "--encoder",
        type=str,
        default="onnx_models/edgetam_image_encoder.onnx",
        help="Path to image encoder ONNX model",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="onnx_models/edgetam_mask_decoder.onnx",
        help="Path to mask decoder ONNX model",
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
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation mode (process synthetic frames)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to process in simulation mode",
    )

    args = parser.parse_args()

    # Initialize predictor
    print("=" * 60)
    print("EdgeTAM ONNX Inference")
    print("=" * 60)

    predictor = EdgeTAMONNXInference(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=args.device,
    )

    # Run simulation mode or single image mode
    if args.simulate or args.image is None:
        simulate_image_stream(predictor, num_frames=args.num_frames)
    else:
        # Load image
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

        # Example point prompt (center of image)
        point_coords = [[original_size[1] // 2, original_size[0] // 2]]
        point_labels = [1]

        print(f"Point prompt: {point_coords}")

        # Predict mask
        print("Predicting mask...")
        t2 = time.time()
        masks, iou_scores = predictor.predict_mask(embeddings, point_coords, point_labels)
        t3 = time.time()
        print(f"✓ Prediction completed in {(t3 - t2) * 1000:.2f}ms")
        print(f"  IoU score: {iou_scores[0, 0]:.3f}")

        # Postprocess
        print("Postprocessing mask...")
        mask = predictor.postprocess_mask(masks, original_size)

        # Calculate mask statistics
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        coverage = (mask_area / total_area) * 100

        print(f"✓ Mask coverage: {coverage:.2f}%")

        # Visualize
        print("Visualizing result...")
        output_path = args.output if args.output else "output.jpg"
        visualize_result(image, mask, point_coords, output_path)

        print(f"\nTotal inference time: {(t3 - t0) * 1000:.2f}ms")


if __name__ == "__main__":
    main()
