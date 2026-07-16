#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export the EdgeTAM video tracking pipeline as iOS 18 Core ML packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edgetam_coreml_video.image_encoder import export_video_image_encoder
from edgetam_coreml_video.initializer import export_video_initializer
from edgetam_coreml_video.load_model import load_reference_model
from edgetam_coreml_video.memory_encoder import export_video_memory_encoder
from edgetam_coreml_video.propagator import export_video_propagator


def model_output_paths(output_directory: Path) -> dict[str, Path]:
    """Return the stable package names used by Python and iOS clients."""
    return {
        "image_encoder": output_directory / "EdgeTAMVideoImageEncoder.mlpackage",
        "initializer": output_directory / "EdgeTAMVideoInitializer.mlpackage",
        "memory_encoder": output_directory / "EdgeTAMVideoMemoryEncoder.mlpackage",
        "propagator": output_directory / "EdgeTAMVideoPropagator.mlpackage",
    }


def export_all_models(
    config: Path,
    checkpoint: Path,
    output_directory: Path,
    device: str = "cpu",
) -> dict[str, Path]:
    """Load EdgeTAM once and export every video tracking package."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = model_output_paths(output_directory)
    model = load_reference_model(config, checkpoint, device=device)

    exporters = (
        ("image_encoder", export_video_image_encoder),
        ("initializer", export_video_initializer),
        ("memory_encoder", export_video_memory_encoder),
        ("propagator", export_video_propagator),
    )
    for name, exporter in exporters:
        print(f"Exporting {name} -> {paths[name]}")
        exporter(model, paths[name])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "sam2/configs/edgetam.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/edgetam.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_all_models(
        config=args.config,
        checkpoint=args.checkpoint,
        output_directory=args.output_dir,
        device=args.device,
    )
    print("Exported Core ML video packages:")
    for path in exported.values():
        print(path)


if __name__ == "__main__":
    main()
