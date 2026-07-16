# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One prompt-free EdgeTAM video propagation step."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .initializer import VideoInitializer
from .masked_attention import MaskedMemoryAttention
from .memory_encoder import VideoMemoryEncoder


class VideoPropagator(nn.Module):
    """Condition one frame on memory, decode its mask, and encode new memory."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.memory_attention = MaskedMemoryAttention(model)
        self.prompt_free_head = VideoInitializer(model)
        self.memory_encoder = VideoMemoryEncoder(model)
        current_positions = model.image_encoder.neck.position_encoding(
            torch.zeros(1, model.hidden_dim, 64, 64)
        )
        self.register_buffer("current_positions", current_positions)

    def forward(
        self,
        raw_vision_features: torch.Tensor,
        high_res_feature_0: torch.Tensor,
        high_res_feature_1: torch.Tensor,
        spatial_bank: torch.Tensor,
        spatial_positions: torch.Tensor,
        pointer_bank: torch.Tensor,
        attention_bias: torch.Tensor,
        rotary_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fused_features = self.memory_attention(
            raw_vision_features,
            self.current_positions,
            spatial_bank,
            spatial_positions,
            pointer_bank,
            attention_bias,
            rotary_weight,
        )
        batch = raw_vision_features.shape[0]
        empty_point_coords = torch.zeros(
            batch,
            1,
            2,
            dtype=raw_vision_features.dtype,
            device=raw_vision_features.device,
        )
        empty_point_labels = -torch.ones(
            batch,
            1,
            dtype=torch.int32,
            device=raw_vision_features.device,
        )
        low_res_mask, high_res_mask, best_iou, pointer, score = (
            self.prompt_free_head(
                fused_features,
                high_res_feature_0,
                high_res_feature_1,
                empty_point_coords,
                empty_point_labels,
            )
        )
        memory, memory_position, _ = self.memory_encoder(
            raw_vision_features,
            high_res_mask,
            score,
        )
        return (
            low_res_mask,
            high_res_mask,
            best_iou,
            pointer,
            score,
            memory,
            memory_position,
        )


def export_video_propagator(model: Any, output_path: Path) -> Path:
    """Export fixed-shape prompt-free EdgeTAM propagation for iOS 18."""
    import coremltools as ct

    propagator = VideoPropagator(model).eval()
    example_inputs = (
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 32, 256, 256),
        torch.randn(1, 64, 128, 128),
        torch.randn(1, 7, 512, 64),
        torch.randn(1, 7, 512, 64),
        torch.randn(1, 16, 256),
        torch.zeros(1, 1, 1, 3648),
        torch.ones(1, 1792),
    )

    with torch.inference_mode():
        traced_model = torch.jit.trace(
            propagator,
            example_inputs,
            check_trace=False,
        )

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="raw_vision_features",
                shape=(1, 256, 64, 64),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="high_res_feature_0",
                shape=(1, 32, 256, 256),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="high_res_feature_1",
                shape=(1, 64, 128, 128),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="spatial_bank",
                shape=(1, 7, 512, 64),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="spatial_positions",
                shape=(1, 7, 512, 64),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="pointer_bank",
                shape=(1, 16, 256),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="attention_bias",
                shape=(1, 1, 1, 3648),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="rotary_weight",
                shape=(1, 1792),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="low_res_mask"),
            ct.TensorType(name="high_res_mask"),
            ct.TensorType(name="best_iou"),
            ct.TensorType(name="object_pointer"),
            ct.TensorType(name="object_score"),
            ct.TensorType(name="memory_features"),
            ct.TensorType(name="memory_positions"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    coreml_model.author = "EdgeTAM Contributors"
    coreml_model.short_description = "Stateless EdgeTAM video propagation"
    coreml_model.version = "1.0"
    coreml_model.save(str(output_path))
    return output_path
