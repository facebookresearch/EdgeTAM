# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import coremltools as ct
import torch

from edgetam_coreml_video.image_encoder import VideoImageEncoder
from edgetam_coreml_video.initializer import (
    VideoInitializer,
    export_video_initializer,
)


def test_video_initializer_matches_prompted_sam_heads(reference_model):
    torch.manual_seed(1)
    image = torch.randn(1, 3, 1024, 1024)
    point_coords = torch.tensor([[[210.0, 350.0]]])
    point_labels = torch.tensor([[1]], dtype=torch.int32)
    image_encoder = VideoImageEncoder(reference_model).eval()
    initializer = VideoInitializer(reference_model).eval()

    with torch.inference_mode():
        raw, initial, high_res_0, high_res_1 = image_encoder(image)
        actual = initializer(
            initial,
            high_res_0,
            high_res_1,
            point_coords,
            point_labels,
        )
        expected = reference_model._forward_sam_heads(
            backbone_features=initial,
            point_inputs={
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
            high_res_features=[high_res_0, high_res_1],
            multimask_output=True,
        )

    low_res_mask, high_res_mask, iou, object_pointer, object_score = actual
    torch.testing.assert_close(low_res_mask, expected[3])
    torch.testing.assert_close(high_res_mask, expected[4])
    torch.testing.assert_close(iou, expected[2].max(dim=-1).values)
    torch.testing.assert_close(object_pointer, expected[5])
    torch.testing.assert_close(object_score, expected[6])
    assert raw.shape == (1, 256, 64, 64)


def test_export_video_initializer_declares_tracking_seed_outputs(
    reference_model,
    tmp_path: Path,
):
    output_path = tmp_path / "edgetam_video_initializer.mlpackage"

    export_video_initializer(reference_model, output_path)
    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()

    assert [feature.name for feature in spec.description.input] == [
        "initial_vision_features",
        "high_res_feature_0",
        "high_res_feature_1",
        "point_coords",
        "point_labels",
    ]
    assert [feature.name for feature in spec.description.output] == [
        "low_res_mask",
        "high_res_mask",
        "best_iou",
        "object_pointer",
        "object_score",
    ]
    coords_range = spec.description.input[3].type.multiArrayType.shapeRange
    labels_range = spec.description.input[4].type.multiArrayType.shapeRange
    assert [
        (item.lowerBound, item.upperBound)
        for item in coords_range.sizeRanges
    ] == [(1, 1), (1, 4), (2, 2)]
    assert [
        (item.lowerBound, item.upperBound)
        for item in labels_range.sizeRanges
    ] == [(1, 1), (1, 4)]
