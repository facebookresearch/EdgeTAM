#!/usr/bin/env python3
"""
EdgeTAM Model ONNX Export Script

This script exports the EdgeTAM model to ONNX format for deployment.
The model is exported in two parts:
1. Image Encoder: Processes input images to generate embeddings
2. Mask Decoder: Generates segmentation masks from embeddings and prompts

Usage:
    python export_to_onnx.py --checkpoint checkpoints/edgetam.pt --output-dir onnx_models
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class EdgeTAMImageEncoder(nn.Module):
    """Wrapper for EdgeTAM Image Encoder for ONNX export"""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.image_size = sam_model.image_size
        self.use_high_res_features = sam_model.use_high_res_features_in_sam

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W], normalized to [0, 1]

        Returns:
            If use_high_res_features:
                image_embeddings: [B, 256, 64, 64]
                high_res_feat_0: [B, 32, 256, 256]
                high_res_feat_1: [B, 64, 128, 128]
            Else:
                image_embeddings: [B, 256, 64, 64]
        """
        # Get backbone features
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed for initial frame
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Get image embeddings (lowest resolution features)
        # Shape: [B, C, H, W] where H=W=64 for 1024x1024 input
        B = x.size(0)
        image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])

        if self.use_high_res_features:
            # Get high-resolution features
            # feat_sizes: [(256, 256), (128, 128), (64, 64)]
            # Note: forward_image already applies conv_s0 and conv_s1 to backbone_fpn[0] and backbone_fpn[1]
            high_res_feat_0 = vision_feats[0].permute(1, 2, 0).view(B, -1, *feat_sizes[0])
            high_res_feat_1 = vision_feats[1].permute(1, 2, 0).view(B, -1, *feat_sizes[1])

            return image_embeddings, high_res_feat_0, high_res_feat_1
        else:
            return image_embeddings


class EdgeTAMMaskDecoder(nn.Module):
    """Wrapper for EdgeTAM Mask Decoder for ONNX export"""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.image_size = sam_model.image_size
        self.use_high_res_features = sam_model.use_high_res_features_in_sam

    def forward(self, image_embeddings, point_coords, point_labels, high_res_feat_0=None, high_res_feat_1=None):
        """
        Args:
            image_embeddings: [B, 256, 64, 64] from image encoder
            point_coords: [B, N, 2] point coordinates (x, y) in pixel space
            point_labels: [B, N] point labels (1=foreground, 0=background)
            high_res_feat_0: [B, 32, 256, 256] (optional, only if use_high_res_features)
            high_res_feat_1: [B, 64, 128, 128] (optional, only if use_high_res_features)

        Returns:
            masks: [B, 1, H, W] predicted masks at original resolution
            iou_predictions: [B, 1] IoU confidence scores
        """
        B = image_embeddings.shape[0]

        # Prepare point inputs
        point_coords = point_coords.float()
        point_labels = point_labels.int()

        # Normalize point coordinates to [0, 1]
        scale = self.image_size
        point_coords_normalized = point_coords / scale

        # Get sparse and dense embeddings
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords_normalized, point_labels),
            boxes=None,
            masks=None,
        )

        # Prepare high-res features if available
        high_res_features = None
        if self.use_high_res_features and high_res_feat_0 is not None and high_res_feat_1 is not None:
            high_res_features = [high_res_feat_0, high_res_feat_1]

        # Predict masks
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Upsample to original resolution
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(masks)

        return masks, iou_predictions


def export_image_encoder(model, output_path, opset_version=17):
    """Export image encoder to ONNX"""

    encoder = EdgeTAMImageEncoder(model)
    encoder.eval()

    # Create dummy input
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 1024, 1024)

    print(f"Exporting Image Encoder to {output_path}")
    print(f"  Input shape: {dummy_image.shape}")
    print(f"  High-res features: {encoder.use_high_res_features}")

    # Prepare output names based on whether high-res features are used
    if encoder.use_high_res_features:
        output_names = ["image_embeddings", "high_res_feat_0", "high_res_feat_1"]
    else:
        output_names = ["image_embeddings"]

    # Note: We use fixed batch size (1) for better ONNX compatibility
    # Dynamic axes can cause issues with certain operations in the model
    dynamic_axes = None

    # Export to ONNX using legacy exporter (more stable)
    with torch.no_grad():
        # Use legacy exporter for better compatibility
        torch.onnx.export(
            encoder,
            dummy_image,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
            # Use legacy exporter
            dynamo=False,
        )

    print(f"✓ Image Encoder exported successfully")
    return output_path


def export_mask_decoder(model, output_path, opset_version=17):
    """Export mask decoder to ONNX"""

    decoder = EdgeTAMMaskDecoder(model)
    decoder.eval()

    # Create dummy inputs
    batch_size = 1
    num_points = 1
    dummy_embeddings = torch.randn(batch_size, 256, 64, 64)
    dummy_point_coords = torch.randn(batch_size, num_points, 2) * 1024  # Random coords in [0, 1024]
    dummy_point_labels = torch.ones(batch_size, num_points, dtype=torch.int32)

    print(f"Exporting Mask Decoder to {output_path}")
    print(f"  Embeddings shape: {dummy_embeddings.shape}")
    print(f"  Point coords shape: {dummy_point_coords.shape}")
    print(f"  Point labels shape: {dummy_point_labels.shape}")
    print(f"  High-res features: {decoder.use_high_res_features}")

    # Prepare inputs based on whether high-res features are used
    if decoder.use_high_res_features:
        dummy_high_res_0 = torch.randn(batch_size, 32, 256, 256)
        dummy_high_res_1 = torch.randn(batch_size, 64, 128, 128)
        dummy_inputs = (dummy_embeddings, dummy_point_coords, dummy_point_labels, dummy_high_res_0, dummy_high_res_1)
        input_names = ["image_embeddings", "point_coords", "point_labels", "high_res_feat_0", "high_res_feat_1"]
        print(f"  High-res feat 0 shape: {dummy_high_res_0.shape}")
        print(f"  High-res feat 1 shape: {dummy_high_res_1.shape}")
    else:
        dummy_inputs = (dummy_embeddings, dummy_point_coords, dummy_point_labels)
        input_names = ["image_embeddings", "point_coords", "point_labels"]

    # Note: We use fixed shapes for better ONNX compatibility
    # Dynamic axes can cause broadcasting issues with certain operations
    dynamic_axes = None

    # Export to ONNX using legacy exporter (more stable)
    with torch.no_grad():
        # Use legacy exporter for better compatibility
        torch.onnx.export(
            decoder,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["masks", "iou_predictions"],
            dynamic_axes=dynamic_axes,
            verbose=False,
            # Use legacy exporter
            dynamo=False,
        )

    print(f"✓ Mask Decoder exported successfully")
    return output_path


def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model"""
    try:
        import onnx
        import onnxruntime as ort

        # Load and check the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Create inference session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        print(f"✓ ONNX model verification passed: {onnx_path}")
        print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

        return True
    except ImportError:
        print("⚠ ONNX/ONNXRuntime not installed, skipping verification")
        print("  Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export EdgeTAM model to ONNX format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/edgetam.pt",
        help="Path to EdgeTAM checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sam2/configs/edgetam.yaml",
        help="Path to EdgeTAM config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (18 recommended for PyTorch 2.3+)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported ONNX models",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EdgeTAM ONNX Export")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset version: {args.opset_version}")
    print("=" * 60)

    # Load EdgeTAM model
    print("\n[1/4] Loading EdgeTAM model...")
    device = "cpu"  # Export on CPU for better compatibility
    model = build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device,
        mode="eval",
    )
    print("✓ Model loaded successfully")

    # Export image encoder
    print("\n[2/4] Exporting Image Encoder...")
    encoder_path = output_dir / "edgetam_image_encoder.onnx"
    export_image_encoder(model, str(encoder_path), args.opset_version)

    # Export mask decoder
    print("\n[3/4] Exporting Mask Decoder...")
    decoder_path = output_dir / "edgetam_mask_decoder.onnx"
    export_mask_decoder(model, str(decoder_path), args.opset_version)

    # Verify models
    if args.verify:
        print("\n[4/4] Verifying ONNX models...")
        verify_onnx_model(str(encoder_path))
        verify_onnx_model(str(decoder_path))

    print("\n" + "=" * 60)
    print("✓ Export completed successfully!")
    print("=" * 60)
    print(f"Image Encoder: {encoder_path}")
    print(f"Mask Decoder: {decoder_path}")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("1. Verify ONNX models:")
    print(f"   python export_to_onnx.py --verify")
    print("\n2. Convert to TensorRT:")
    print(f"   python convert_to_tensorrt.py --onnx-dir {args.output_dir}")
    print("\n3. Run inference:")
    print(f"   python deploy/onnx_inference.py --simulate --num-frames 10")


if __name__ == "__main__":
    main()
