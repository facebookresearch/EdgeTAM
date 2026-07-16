# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the PyTorch EdgeTAM reference used by export and parity checks."""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.build_sam import _load_checkpoint


def load_reference_model(
    config: Path,
    checkpoint: Path,
    device: str = "cpu",
) -> Any:
    """Load an eval-mode EdgeTAM model from an explicit config and checkpoint."""

    config = Path(config).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"EdgeTAM config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EdgeTAM checkpoint not found: {checkpoint}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config.parent), version_base=None):
        cfg = compose(config_name=config.stem)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, str(checkpoint))
    return model.to(device).eval()
