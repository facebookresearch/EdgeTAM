# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import coremltools as ct
import torch

from edgetam_coreml_video.memory_encoder import (
    VideoMemoryEncoder,
    export_video_memory_encoder,
)


def test_video_memory_encoder_matches_edgetam_memory_path(reference_model):
    torch.manual_seed(2)
    raw_features = torch.randn(1, 256, 64, 64)
    high_res_mask = torch.randn(1, 1, 1024, 1024)
    object_score = torch.tensor([[2.0]])
    wrapper = VideoMemoryEncoder(reference_model).eval()

    with torch.inference_mode():
        actual_features, actual_positions, actual_temporal_positions = wrapper(
            raw_features,
            high_res_mask,
            object_score,
        )
        vision_features = [raw_features.flatten(2).permute(2, 0, 1)]
        expected_features, expected_positions = reference_model._encode_new_memory(
            current_vision_feats=vision_features,
            feat_sizes=[(64, 64)],
            pred_masks_high_res=high_res_mask,
            object_score_logits=object_score,
            is_mask_from_pts=False,
        )

    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_positions, expected_positions[0])
    torch.testing.assert_close(
        actual_temporal_positions,
        reference_model.maskmem_tpos_enc[:, 0, 0],
    )
    assert actual_features.shape == (1, 512, 64)
    assert actual_positions.shape == (1, 512, 64)
    assert actual_temporal_positions.shape == (7, 64)


def test_seed_memory_encoder_binarizes_prompted_mask_like_video_predictor(
    reference_model,
):
    raw_features = torch.randn(1, 256, 64, 64)
    high_res_mask = torch.linspace(-1, 1, 1024 * 1024).reshape(
        1, 1, 1024, 1024
    )
    object_score = torch.tensor([[2.0]])
    original_binarize = reference_model.binarize_mask_from_pts_for_mem_enc
    reference_model.binarize_mask_from_pts_for_mem_enc = True
    wrapper = VideoMemoryEncoder(
        reference_model,
        is_mask_from_points=True,
    ).eval()

    try:
        with torch.inference_mode():
            actual = wrapper(raw_features, high_res_mask, object_score)
            expected_features, expected_positions = (
                reference_model._encode_new_memory(
                    current_vision_feats=[
                        raw_features.flatten(2).permute(2, 0, 1)
                    ],
                    feat_sizes=[(64, 64)],
                    pred_masks_high_res=high_res_mask,
                    object_score_logits=object_score,
                    is_mask_from_pts=True,
                )
            )
    finally:
        reference_model.binarize_mask_from_pts_for_mem_enc = original_binarize

    torch.testing.assert_close(actual[0], expected_features)
    torch.testing.assert_close(actual[1], expected_positions[0])
    torch.testing.assert_close(
        actual[2],
        reference_model.maskmem_tpos_enc[:, 0, 0],
    )


def test_export_video_memory_encoder_declares_spatial_memory_outputs(
    reference_model,
    tmp_path: Path,
):
    output_path = tmp_path / "edgetam_video_memory_encoder.mlpackage"

    export_video_memory_encoder(reference_model, output_path)
    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()

    assert [feature.name for feature in spec.description.input] == [
        "raw_vision_features",
        "high_res_mask",
        "object_score",
    ]
    assert [feature.name for feature in spec.description.output] == [
        "memory_features",
        "memory_positions",
        "temporal_positions",
    ]
