# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from edgetam_coreml_video.load_model import load_reference_model


def test_load_reference_model_rejects_missing_config(tmp_path: Path):
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()

    with pytest.raises(FileNotFoundError, match="config"):
        load_reference_model(tmp_path / "missing.yaml", checkpoint)


def test_load_reference_model_rejects_missing_checkpoint(tmp_path: Path):
    config = tmp_path / "edgetam.yaml"
    config.touch()

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_reference_model(config, tmp_path / "missing.pt")
