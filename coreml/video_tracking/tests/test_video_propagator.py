# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import coremltools as ct
import torch

from edgetam_coreml_video.masked_attention import build_attention_controls
from edgetam_coreml_video.propagator import (
    VideoPropagator,
    export_video_propagator,
)


def _operation_types(spec) -> list[str]:
    def walk_block(block):
        for operation in block.operations:
            yield operation.type
            for nested_block in operation.blocks:
                yield from walk_block(nested_block)

    operation_types = []
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            operation_types.extend(walk_block(block))
    return operation_types


def test_export_video_propagator_declares_fixed_stateless_contract(
    reference_model,
    tmp_path: Path,
):
    output_path = tmp_path / "EdgeTAMVideoPropagator.mlpackage"

    export_video_propagator(reference_model, output_path)
    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()

    assert [feature.name for feature in spec.description.input] == [
        "raw_vision_features",
        "high_res_feature_0",
        "high_res_feature_1",
        "spatial_bank",
        "spatial_positions",
        "pointer_bank",
        "attention_bias",
        "rotary_weight",
    ]
    assert {
        feature.name: list(feature.type.multiArrayType.shape)
        for feature in spec.description.input
    } == {
        "raw_vision_features": [1, 256, 64, 64],
        "high_res_feature_0": [1, 32, 256, 256],
        "high_res_feature_1": [1, 64, 128, 128],
        "spatial_bank": [1, 7, 512, 64],
        "spatial_positions": [1, 7, 512, 64],
        "pointer_bank": [1, 16, 256],
        "attention_bias": [1, 1, 1, 3648],
        "rotary_weight": [1, 1792],
    }
    assert [feature.name for feature in spec.description.output] == [
        "low_res_mask",
        "high_res_mask",
        "best_iou",
        "object_pointer",
        "object_score",
        "memory_features",
        "memory_positions",
    ]
    assert not spec.description.state
    assert "select" not in _operation_types(spec)


def test_video_propagator_runs_prompt_free_heads_and_encodes_new_memory(
    reference_model,
):
    torch.manual_seed(4)
    raw = torch.randn(1, 256, 64, 64)
    high_res_0 = torch.randn(1, 32, 256, 256)
    high_res_1 = torch.randn(1, 64, 128, 128)
    spatial_bank = torch.randn(1, 7, 512, 64)
    spatial_positions = torch.randn(1, 7, 512, 64)
    spatial_valid = torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    pointer_bank = torch.randn(1, 16, 256)
    pointer_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    attention_bias, rotary_weight = build_attention_controls(
        spatial_valid,
        pointer_valid,
    )
    propagator = VideoPropagator(reference_model).eval()

    with torch.inference_mode():
        actual = propagator(
            raw,
            high_res_0,
            high_res_1,
            spatial_bank,
            spatial_positions,
            pointer_bank,
            attention_bias,
            rotary_weight,
        )
        fused = propagator.memory_attention(
            raw,
            propagator.current_positions,
            spatial_bank,
            spatial_positions,
            pointer_bank,
            attention_bias,
            rotary_weight,
        )
        expected_heads = reference_model._forward_sam_heads(
            backbone_features=fused,
            point_inputs=None,
            high_res_features=[high_res_0, high_res_1],
            multimask_output=True,
        )
        expected_memory = reference_model._encode_new_memory(
            current_vision_feats=[raw.flatten(2).permute(2, 0, 1)],
            feat_sizes=[(64, 64)],
            pred_masks_high_res=expected_heads[4],
            object_score_logits=expected_heads[6],
            is_mask_from_pts=False,
        )

    low, high, iou, pointer, score, memory, memory_position = actual
    torch.testing.assert_close(low, expected_heads[3])
    torch.testing.assert_close(high, expected_heads[4])
    torch.testing.assert_close(iou, expected_heads[2].max(dim=-1).values)
    torch.testing.assert_close(pointer, expected_heads[5])
    torch.testing.assert_close(score, expected_heads[6])
    torch.testing.assert_close(memory, expected_memory[0])
    torch.testing.assert_close(memory_position, expected_memory[1][0])
