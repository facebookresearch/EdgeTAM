# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Video-specific EdgeTAM image features for initialization and propagation."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


class VideoImageEncoder(nn.Module):
    """Expose both raw and no-memory-conditioned EdgeTAM image features."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_fpn = self.model.forward_image(image)["backbone_fpn"]
        raw_vision_features = backbone_fpn[2]
        initial_vision_features = raw_vision_features.flatten(2).permute(2, 0, 1)
        initial_vision_features = (
            initial_vision_features + self.model.no_mem_embed
        )
        initial_vision_features = initial_vision_features.permute(1, 2, 0)
        initial_vision_features = initial_vision_features.reshape_as(
            raw_vision_features
        )
        return (
            raw_vision_features,
            initial_vision_features,
            backbone_fpn[0],
            backbone_fpn[1],
        )


class CoreMLVideoImageEncoder(VideoImageEncoder):
    """Add the notebook's RGB normalization inside the exported model."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_image = (image - self.pixel_mean) / self.pixel_std
        return super().forward(normalized_image)


def export_video_image_encoder(model: Any, output_path: Path) -> Path:
    """Export the video image encoder as an iOS 18 Core ML package."""
    import coremltools as ct

    wrapper = CoreMLVideoImageEncoder(model).eval()
    example_image = torch.randn(1, 3, 1024, 1024)

    with torch.inference_mode():
        traced_model = torch.jit.trace(
            wrapper,
            example_image,
            check_trace=False,
        )

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 1024, 1024),
                scale=1 / 255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="raw_vision_features"),
            ct.TensorType(name="initial_vision_features"),
            ct.TensorType(name="high_res_feature_0"),
            ct.TensorType(name="high_res_feature_1"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    coreml_model.author = "EdgeTAM Contributors"
    coreml_model.short_description = "EdgeTAM video image encoder"
    coreml_model.version = "1.0"
    coreml_model.save(str(output_path))
    return output_path
