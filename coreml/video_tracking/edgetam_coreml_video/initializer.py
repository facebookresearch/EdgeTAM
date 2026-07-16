# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""First-frame prompt handling for an EdgeTAM video track."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sam2.modeling.sam2_base import NO_OBJ_SCORE


class VideoInitializer(nn.Module):
    """Run the prompted SAM heads and expose everything needed to seed memory."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        initial_vision_features: torch.Tensor,
        high_res_feature_0: torch.Tensor,
        high_res_feature_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        prompt_encoder = self.model.sam_prompt_encoder
        point_coords = torch.cat(
            [
                point_coords + 0.5,
                torch.zeros(
                    point_coords.shape[0],
                    1,
                    2,
                    dtype=point_coords.dtype,
                    device=point_coords.device,
                ),
            ],
            dim=1,
        )
        point_labels = torch.cat(
            [
                point_labels.to(torch.int32),
                -torch.ones(
                    point_labels.shape[0],
                    1,
                    dtype=torch.int32,
                    device=point_labels.device,
                ),
            ],
            dim=1,
        )

        sparse_embeddings = prompt_encoder.pe_layer.forward_with_coords(
            point_coords,
            prompt_encoder.input_image_size,
        )
        padding_weight = (point_labels == -1).unsqueeze(-1).to(
            sparse_embeddings.dtype
        )
        not_a_point = prompt_encoder.not_a_point_embed.weight.reshape(
            1, 1, -1
        )
        sparse_embeddings = (
            padding_weight * not_a_point
            + (1.0 - padding_weight) * sparse_embeddings
        )
        for label, embedding in enumerate(prompt_encoder.point_embeddings):
            sparse_embeddings = sparse_embeddings + (
                (point_labels == label).unsqueeze(-1).to(sparse_embeddings.dtype)
                * embedding.weight.reshape(1, 1, -1)
            )
        dense_embeddings = prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(
            point_coords.shape[0],
            -1,
            prompt_encoder.image_embedding_size[0],
            prompt_encoder.image_embedding_size[1],
        )

        low_res_multimasks, iou_predictions, output_tokens, score = (
            self.model.sam_mask_decoder(
                image_embeddings=initial_vision_features,
                image_pe=prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=[high_res_feature_0, high_res_feature_1],
            )
        )
        if self.model.pred_obj_scores:
            object_appearing = score > 0
            appearing_weight = object_appearing[:, None, None].to(
                low_res_multimasks.dtype
            )
            low_res_multimasks = (
                appearing_weight * low_res_multimasks
                + (1.0 - appearing_weight) * NO_OBJ_SCORE
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        )
        best_indices = torch.argmax(iou_predictions, dim=-1)
        low_res_mask = torch.gather(
            low_res_multimasks,
            1,
            best_indices.reshape(-1, 1, 1, 1).expand(
                -1,
                1,
                low_res_multimasks.shape[-2],
                low_res_multimasks.shape[-1],
            ),
        )
        high_res_mask = torch.gather(
            high_res_multimasks,
            1,
            best_indices.reshape(-1, 1, 1, 1).expand(
                -1,
                1,
                high_res_multimasks.shape[-2],
                high_res_multimasks.shape[-1],
            ),
        )
        output_token = torch.gather(
            output_tokens,
            1,
            best_indices.reshape(-1, 1, 1).expand(
                -1,
                1,
                output_tokens.shape[-1],
            ),
        )[:, 0]
        pointer = self.model.obj_ptr_proj(output_token)
        if self.model.pred_obj_scores:
            if self.model.soft_no_obj_ptr:
                appearing_weight = score.sigmoid()
            else:
                appearing_weight = object_appearing.float()
            if self.model.fixed_no_obj_ptr:
                pointer = appearing_weight * pointer
            pointer = pointer + (1 - appearing_weight) * self.model.no_obj_ptr

        best_iou = iou_predictions.max(dim=-1).values
        return low_res_mask, high_res_mask, best_iou, pointer, score


def export_video_initializer(model: Any, output_path: Path) -> Path:
    """Export first-frame prompt initialization as an iOS 18 Core ML package."""
    import coremltools as ct

    wrapper = VideoInitializer(model).eval()
    example_inputs = (
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 32, 256, 256),
        torch.randn(1, 64, 128, 128),
        torch.tensor([[[512.0, 512.0]]]),
        torch.tensor([[1]], dtype=torch.int32),
    )

    with torch.inference_mode():
        traced_model = torch.jit.trace(
            wrapper,
            example_inputs,
            check_trace=False,
        )

    point_count = ct.RangeDim(lower_bound=1, upper_bound=4, default=1)
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="initial_vision_features",
                shape=(1, 256, 64, 64),
            ),
            ct.TensorType(
                name="high_res_feature_0",
                shape=(1, 32, 256, 256),
            ),
            ct.TensorType(
                name="high_res_feature_1",
                shape=(1, 64, 128, 128),
            ),
            ct.TensorType(name="point_coords", shape=(1, point_count, 2)),
            ct.TensorType(
                name="point_labels",
                shape=(1, point_count),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="low_res_mask"),
            ct.TensorType(name="high_res_mask"),
            ct.TensorType(name="best_iou"),
            ct.TensorType(name="object_pointer"),
            ct.TensorType(name="object_score"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    coreml_model.author = "EdgeTAM Contributors"
    coreml_model.short_description = "EdgeTAM video track initializer"
    coreml_model.version = "1.0"
    coreml_model.save(str(output_path))
    return output_path
