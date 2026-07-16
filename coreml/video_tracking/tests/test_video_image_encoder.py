# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import coremltools as ct
import torch

from edgetam_coreml_video.image_encoder import (
    CoreMLVideoImageEncoder,
    VideoImageEncoder,
    export_video_image_encoder,
)


def test_video_image_encoder_returns_raw_and_initial_features(reference_model):
    torch.manual_seed(0)
    image = torch.randn(1, 3, 256, 256)
    wrapper = VideoImageEncoder(reference_model).eval()

    with torch.inference_mode():
        raw, initial, high_res_0, high_res_1 = wrapper(image)
        expected_fpn = reference_model.forward_image(image)["backbone_fpn"]

    torch.testing.assert_close(raw, expected_fpn[2])
    expected_initial = raw.flatten(2).permute(2, 0, 1)
    expected_initial = expected_initial + reference_model.no_mem_embed
    expected_initial = expected_initial.permute(1, 2, 0).reshape_as(raw)
    torch.testing.assert_close(initial, expected_initial)
    torch.testing.assert_close(high_res_0, expected_fpn[0])
    torch.testing.assert_close(high_res_1, expected_fpn[1])


def test_video_image_encoder_keeps_raw_features_unconditioned(reference_model):
    image = torch.zeros(1, 3, 256, 256)
    wrapper = VideoImageEncoder(reference_model).eval()

    with torch.inference_mode():
        raw, initial, _, _ = wrapper(image)

    assert not torch.equal(raw, initial)


def test_coreml_video_image_encoder_applies_notebook_normalization(
    reference_model,
):
    torch.manual_seed(5)
    image_0_to_1 = torch.rand(1, 3, 256, 256)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    coreml_wrapper = CoreMLVideoImageEncoder(reference_model).eval()
    normalized_wrapper = VideoImageEncoder(reference_model).eval()

    with torch.inference_mode():
        actual = coreml_wrapper(image_0_to_1)
        expected = normalized_wrapper((image_0_to_1 - mean) / std)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_export_video_image_encoder_declares_video_feature_outputs(
    reference_model,
    tmp_path: Path,
):
    output_path = tmp_path / "edgetam_video_image_encoder.mlpackage"

    export_video_image_encoder(reference_model, output_path)
    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()

    assert [feature.name for feature in spec.description.input] == ["image"]
    assert [feature.name for feature in spec.description.output] == [
        "raw_vision_features",
        "initial_vision_features",
        "high_res_feature_0",
        "high_res_feature_1",
    ]
