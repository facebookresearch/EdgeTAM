# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EdgeTAM mask-memory encoding including the 2D Spatial Perceiver."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


class VideoMemoryEncoder(nn.Module):
    """Encode one predicted mask into EdgeTAM's compact spatial memory."""

    def __init__(
        self,
        model: Any,
        is_mask_from_points: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.is_mask_from_points = is_mask_from_points
        self.register_buffer(
            "temporal_positions",
            model.maskmem_tpos_enc[:, 0, 0].detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        raw_vision_features: torch.Tensor,
        high_res_mask: torch.Tensor,
        object_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_vision_features = [
            raw_vision_features.flatten(2).permute(2, 0, 1)
        ]
        memory_features, memory_positions = self.model._encode_new_memory(
            current_vision_feats=current_vision_features,
            feat_sizes=[raw_vision_features.shape[-2:]],
            pred_masks_high_res=high_res_mask,
            object_score_logits=object_score,
            is_mask_from_pts=self.is_mask_from_points,
        )
        return (
            memory_features,
            memory_positions[0],
            self.temporal_positions + torch.zeros_like(self.temporal_positions),
        )


def export_video_memory_encoder(model: Any, output_path: Path) -> Path:
    """Export the Spatial Perceiver memory encoder for iOS 18."""
    import coremltools as ct

    original_binarize = model.binarize_mask_from_pts_for_mem_enc
    model.binarize_mask_from_pts_for_mem_enc = True
    wrapper = VideoMemoryEncoder(model, is_mask_from_points=True).eval()
    example_inputs = (
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 1, 1024, 1024),
        torch.tensor([[2.0]]),
    )

    try:
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                wrapper,
                example_inputs,
                check_trace=False,
            )
    finally:
        model.binarize_mask_from_pts_for_mem_enc = original_binarize

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="raw_vision_features",
                shape=(1, 256, 64, 64),
            ),
            ct.TensorType(name="high_res_mask", shape=(1, 1, 1024, 1024)),
            ct.TensorType(name="object_score", shape=(1, 1)),
        ],
        outputs=[
            ct.TensorType(name="memory_features"),
            ct.TensorType(name="memory_positions"),
            ct.TensorType(name="temporal_positions"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    coreml_model.author = "EdgeTAM Contributors"
    coreml_model.short_description = "EdgeTAM Spatial Perceiver memory encoder"
    coreml_model.version = "1.0"
    coreml_model.save(str(output_path))
    return output_path
