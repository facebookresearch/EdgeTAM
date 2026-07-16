# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from edgetam_coreml_video.load_model import load_reference_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "sam2" / "configs" / "edgetam.yaml"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "edgetam.pt"


@pytest.fixture(scope="session")
def reference_model():
    return load_reference_model(CONFIG_PATH, CHECKPOINT_PATH)
