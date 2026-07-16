# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from export_models import model_output_paths


def test_model_output_paths_are_separate_coreml_packages(tmp_path: Path):
    assert model_output_paths(tmp_path) == {
        "image_encoder": tmp_path / "EdgeTAMVideoImageEncoder.mlpackage",
        "initializer": tmp_path / "EdgeTAMVideoInitializer.mlpackage",
        "memory_encoder": tmp_path / "EdgeTAMVideoMemoryEncoder.mlpackage",
        "propagator": tmp_path / "EdgeTAMVideoPropagator.mlpackage",
    }
