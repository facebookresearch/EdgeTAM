#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export EdgeTAM to CoreML format.

Author: Krish Mehta (https://github.com/DjKesu)
"""

import argparse
import json
import os
import sys
import warnings
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

# Suppress warnings
warnings.filterwarnings(
    "ignore", message="Torch version .* has not been tested with coremltools"
)
warnings.filterwarnings("ignore", message=".*resources.bin missing.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam2.build_sam import _load_checkpoint
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure you have installed EdgeTAM properly.")
    sys.exit(1)


class EdgeTAMImageEncoder(torch.nn.Module):
    """EdgeTAM Image Encoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)
        backbone_fpn = backbone_out.get("backbone_fpn", [])

        if len(backbone_fpn) >= 3:
            vision_features = backbone_fpn[2]
            high_res_feat_0 = backbone_fpn[0]
            high_res_feat_1 = backbone_fpn[1]

            # Add no_mem_embed to match PyTorch EdgeTAM behavior
            if self.model.directly_add_no_mem_embed:
                B, C, H, W = vision_features.shape
                vision_features_flat = vision_features.flatten(2).permute(2, 0, 1)
                vision_features_flat = (
                    vision_features_flat + self.model.no_mem_embed.squeeze(0)
                )
                vision_features = vision_features_flat.permute(1, 2, 0).view(B, C, H, W)
        else:
            bs, _, H, W = image.shape
            vision_features = torch.zeros(
                bs, 256, H // 16, W // 16, device=image.device
            )
            high_res_feat_0 = torch.zeros(bs, 32, H // 4, W // 4, device=image.device)
            high_res_feat_1 = torch.zeros(bs, 64, H // 8, W // 8, device=image.device)

        return vision_features, high_res_feat_0, high_res_feat_1


class EdgeTAMPromptEncoder(torch.nn.Module):
    """EdgeTAM Prompt Encoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        boxes: torch.Tensor,
        mask_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings


class EdgeTAMMaskDecoder(torch.nn.Module):
    """EdgeTAM Mask Decoder wrapper for CoreML export."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_feat_0: torch.Tensor,
        high_res_feat_1: torch.Tensor,
        multimask_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        use_multimask = multimask_output[0].item() > 0.5
        high_res_features = [high_res_feat_0, high_res_feat_1]

        # Use proper position encoding from prompt encoder
        proper_image_pe = self.model.sam_prompt_encoder.get_dense_pe()

        sam_outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=proper_image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=use_multimask,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        masks = sam_outputs[0]
        iou_pred = sam_outputs[1]

        if not use_multimask:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        return masks, iou_pred


def export_image_encoder(model, output_path: str):
    """Export image encoder to CoreML."""
    print("Exporting Image Encoder...")
    encoder_wrapper = EdgeTAMImageEncoder(model)
    encoder_wrapper.eval()

    s = model.image_size
    example_input = torch.randn(1, 3, s, s)

    with torch.no_grad():
        traced_model = torch.jit.trace(encoder_wrapper, example_input)

    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, s, s),
        scale=1 / 255.0,
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB,
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="vision_features"),
            ct.TensorType(name="high_res_feat_0"),
            ct.TensorType(name="high_res_feat_1"),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Image Encoder"
    mlmodel.version = "1.0"

    mlmodel.save(output_path)
    print(f"  Saved to {output_path}")


def export_prompt_encoder(model, output_path: str):
    """Export prompt encoder to CoreML."""
    print("Exporting Prompt Encoder...")
    encoder_wrapper = EdgeTAMPromptEncoder(model)
    encoder_wrapper.eval()

    s = model.image_size
    point_coords = torch.zeros(1, 4, 2)
    point_coords[0, 0] = torch.tensor([s / 2, s / 2])
    point_labels = torch.full((1, 4), -1, dtype=torch.float32)
    point_labels[0, 0] = 1.0
    boxes = torch.zeros(1, 4)
    mask_input = torch.zeros(1, 1, s // 4, s // 4)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            encoder_wrapper, (point_coords, point_labels, boxes, mask_input)
        )

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="point_coords", shape=(1, 4, 2)),
            ct.TensorType(name="point_labels", shape=(1, 4)),
            ct.TensorType(name="boxes", shape=(1, 4)),
            ct.TensorType(name="mask_input", shape=(1, 1, s // 4, s // 4)),
        ],
        outputs=[
            ct.TensorType(name="sparse_embeddings"),
            ct.TensorType(name="dense_embeddings"),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Prompt Encoder"
    mlmodel.version = "1.0"

    mlmodel.save(output_path)
    print(f"  Saved to {output_path}")


def export_mask_decoder(model, output_path: str):
    """Export mask decoder to CoreML."""
    print("Exporting Mask Decoder...")
    decoder_wrapper = EdgeTAMMaskDecoder(model)
    decoder_wrapper.eval()

    s = model.image_size
    e = s // 16  # stride-16 image embedding size
    image_embeddings = torch.randn(1, 256, e, e)
    image_pe = torch.randn(1, 256, e, e)
    sparse_prompt_embeddings = torch.randn(1, 2, 256)
    dense_prompt_embeddings = torch.randn(1, 256, e, e)
    high_res_feat_0 = torch.randn(1, 32, s // 4, s // 4)
    high_res_feat_1 = torch.randn(1, 64, s // 8, s // 8)
    multimask_output = torch.tensor([True])

    with torch.no_grad():
        traced_model = torch.jit.trace(
            decoder_wrapper,
            (
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                high_res_feat_0,
                high_res_feat_1,
                multimask_output,
            ),
        )

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image_embeddings", shape=(1, 256, e, e)),
            ct.TensorType(name="image_pe", shape=(1, 256, e, e)),
            ct.TensorType(
                name="sparse_prompt_embeddings", shape=(1, ct.RangeDim(1, 10), 256)
            ),
            ct.TensorType(name="dense_prompt_embeddings", shape=(1, 256, e, e)),
            ct.TensorType(name="high_res_feat_0", shape=(1, 32, s // 4, s // 4)),
            ct.TensorType(name="high_res_feat_1", shape=(1, 64, s // 8, s // 8)),
            ct.TensorType(name="multimask_output", shape=(1,)),
        ],
        outputs=[ct.TensorType(name="masks"), ct.TensorType(name="iou_pred")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Mask Decoder"
    mlmodel.version = "1.0"

    mlmodel.save(output_path)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export EdgeTAM to CoreML")
    parser.add_argument("--sam2_cfg", required=True, help="Path to EdgeTAM config file")
    parser.add_argument(
        "--sam2_checkpoint", required=True, help="Path to EdgeTAM checkpoint"
    )
    parser.add_argument(
        "--output_dir", default="./coreml_models", help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading EdgeTAM model...")
    try:
        device = torch.device("cpu")
        GlobalHydra.instance().clear()

        config_path = os.path.abspath(args.sam2_cfg)
        config_dir = os.path.dirname(config_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            OmegaConf.resolve(cfg)

            model = instantiate(cfg.model, _recursive_=True)

            if args.sam2_checkpoint:
                _load_checkpoint(model, args.sam2_checkpoint)

            model = model.to(device)
            model.eval()

    except Exception as e:
        print(f"Failed to load EdgeTAM model: {e}")
        sys.exit(1)

    print(f"\nExporting to {args.output_dir}...")

    try:
        export_image_encoder(
            model, os.path.join(args.output_dir, "edgetam_image_encoder.mlpackage")
        )

        export_prompt_encoder(
            model, os.path.join(args.output_dir, "edgetam_prompt_encoder.mlpackage")
        )

        export_mask_decoder(
            model, os.path.join(args.output_dir, "edgetam_mask_decoder.mlpackage")
        )

        # Create simple metadata
        metadata = {
            "model_name": "EdgeTAM",
            "version": "1.0",
            # frames and prompt coordinates must be scaled to this size
            "image_size": model.image_size,
            "components": {
                "image_encoder": "edgetam_image_encoder.mlpackage",
                "prompt_encoder": "edgetam_prompt_encoder.mlpackage",
                "mask_decoder": "edgetam_mask_decoder.mlpackage",
            },
        }

        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nExport completed successfully!")
        print(f"Models saved to: {args.output_dir}")

    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
