#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compare Core ML video tracking with EdgeTAM's PyTorch predictor."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edgetam_coreml_video.metrics import (
    binary_mask_iou,
    cosine_similarity,
    tensor_error,
)
from edgetam_coreml_video.predictor import CoreMLVideoPredictor
from sam2.build_sam import build_sam2_video_predictor


@dataclass(frozen=True)
class MaskComparison:
    max_abs: float
    mean_abs: float
    cosine: float
    mask_iou: float


def compare_mask_logits(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> MaskComparison:
    """Compare continuous logits and their zero-threshold binary masks."""
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"mask shape mismatch: {reference.shape} != {candidate.shape}"
        )
    error = tensor_error(reference, candidate)
    return MaskComparison(
        max_abs=error.max_abs,
        mean_abs=error.mean_abs,
        cosine=cosine_similarity(reference, candidate),
        mask_iou=binary_mask_iou(reference > 0, candidate > 0),
    )


def _frame_paths(frame_directory: Path, max_frames: int) -> list[Path]:
    paths = [
        path
        for path in frame_directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError("JPEG frame filenames must use numeric stems") from error
    if not paths:
        raise ValueError(f"no JPEG frames found in {frame_directory}")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    return paths[:max_frames]


def _pytorch_logits(
    frame_directory: Path,
    checkpoint: Path,
    config_name: str,
    device: str,
    points: np.ndarray,
    labels: np.ndarray,
    frame_count: int,
) -> dict[int, np.ndarray]:
    predictor = build_sam2_video_predictor(
        config_name,
        str(checkpoint),
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=[
            "++model.binarize_mask_from_pts_for_mem_enc=true",
        ],
    )
    inference_state = predictor.init_state(
        video_path=str(frame_directory),
        offload_video_to_cpu=device != "cpu",
    )
    predictor.reset_state(inference_state)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    logits = {}
    for frame_index, _, mask_logits in predictor.propagate_in_video(
        inference_state,
        max_frame_num_to_track=frame_count - 1,
    ):
        if frame_index >= frame_count:
            break
        logits[frame_index] = mask_logits[0, 0].detach().float().cpu().numpy()
    return logits


def _compare_coreml_frames(
    frame_paths: list[Path],
    reference: dict[int, np.ndarray],
    predictor: CoreMLVideoPredictor,
    points: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int]]:
    """Run Core ML sequentially and return one comparison row per frame."""
    rows = []
    for frame_index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as frame:
            result = (
                predictor.start_track(frame, points, labels)
                if frame_index == 0
                else predictor.track_frame(frame)
            )
        comparison = compare_mask_logits(
            reference[frame_index],
            result.mask_logits,
        )
        snapshot = predictor.debug_bank_snapshot()
        rows.append(
            {
                "frame": frame_index,
                **asdict(comparison),
                "object_score": result.object_score,
                "predicted_iou": result.iou,
                "recent_count": snapshot.recent_count,
                "pointer_count": snapshot.pointer_count,
            }
        )
    return rows


def validate_video(
    frame_directory: Path,
    model_directory: Path,
    checkpoint: Path,
    config_name: str,
    device: str,
    points: np.ndarray,
    labels: np.ndarray,
    max_frames: int,
) -> list[dict[str, float | int]]:
    """Run both predictors and return per-frame numerical comparisons."""
    frames = _frame_paths(frame_directory, max_frames)
    reference = _pytorch_logits(
        frame_directory,
        checkpoint,
        config_name,
        device,
        points,
        labels,
        len(frames),
    )
    predictor = CoreMLVideoPredictor.from_directory(model_directory)
    return _compare_coreml_frames(
        frames,
        reference,
        predictor,
        points,
        labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/edgetam.pt",
    )
    parser.add_argument("--config-name", default="edgetam.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--point",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="Repeat for up to four prompts; defaults to 210 350 1.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optionally write all per-frame comparisons as JSON.",
    )
    return parser.parse_args()


def run_validation(
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    """Run validation from parsed arguments and write optional JSON."""
    prompt_rows = args.point or [[210.0, 350.0, 1.0]]
    points = np.asarray([row[:2] for row in prompt_rows], dtype=np.float32)
    labels = np.asarray([int(row[2]) for row in prompt_rows], dtype=np.int32)
    rows = validate_video(
        frame_directory=args.frames_dir,
        model_directory=args.models_dir,
        checkpoint=args.checkpoint,
        config_name=args.config_name,
        device=args.device,
        points=points,
        labels=labels,
        max_frames=args.max_frames,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    rows = run_validation(parse_args())

    print("frame  mask_iou  cosine   mean_abs  max_abs")
    for row in rows:
        print(
            f"{row['frame']:5d}  {row['mask_iou']:.6f}  "
            f"{row['cosine']:.6f}  {row['mean_abs']:.6f}  "
            f"{row['max_abs']:.6f}"
        )


if __name__ == "__main__":
    main()
