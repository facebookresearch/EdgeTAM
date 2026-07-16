# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Python runtime for the stateless EdgeTAM Core ML video pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .explicit_memory_bank import ExplicitMemoryBank, MemoryBankSnapshot

MODEL_SIZE = 1024
MAX_POINTS = 4


@dataclass(frozen=True)
class TrackingResult:
    """One object's mask logits and confidence for a video frame."""

    mask_logits: np.ndarray
    mask: np.ndarray
    iou: float
    object_score: float


def prepare_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    original_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Scale one to four real prompt tokens to EdgeTAM's 1024 input."""
    points_array = np.asarray(points, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(points_array) != len(labels_array):
        raise ValueError("points and labels must have the same length")
    if not 1 <= len(points_array) <= MAX_POINTS:
        raise ValueError(f"one to {MAX_POINTS} points are supported")

    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("original_size must contain positive width and height")

    scaled_points = points_array.copy()
    scaled_points[:, 0] *= MODEL_SIZE / width
    scaled_points[:, 1] *= MODEL_SIZE / height
    return (
        scaled_points.reshape(1, -1, 2).astype(np.float16),
        labels_array.reshape(1, -1),
    )


class CoreMLVideoPredictor:
    """Track one object with stateless models and an explicit memory bank."""

    def __init__(
        self,
        image_encoder: Any,
        initializer: Any,
        memory_encoder: Any,
        propagator: Any,
    ) -> None:
        self.image_encoder = image_encoder
        self.initializer = initializer
        self.memory_encoder = memory_encoder
        self.propagator = propagator
        self._bank = ExplicitMemoryBank()
        self._started = False

    @classmethod
    def from_directory(
        cls,
        model_directory: Path,
        compute_units: Any = None,
    ) -> CoreMLVideoPredictor:
        """Load the four packages produced by ``export_models.py``."""
        import coremltools as ct

        if compute_units is None:
            compute_units = ct.ComputeUnit.ALL
        names = {
            "image_encoder": "EdgeTAMVideoImageEncoder.mlpackage",
            "initializer": "EdgeTAMVideoInitializer.mlpackage",
            "memory_encoder": "EdgeTAMVideoMemoryEncoder.mlpackage",
            "propagator": "EdgeTAMVideoPropagator.mlpackage",
        }
        models = {
            name: ct.models.MLModel(
                str(model_directory / package_name),
                compute_units=compute_units,
            )
            for name, package_name in names.items()
        }
        return cls(**models)

    def reset(self) -> None:
        """Discard the current object track and its explicit memory bank."""
        self._bank = ExplicitMemoryBank()
        self._started = False

    def debug_bank_snapshot(self) -> MemoryBankSnapshot:
        """Expose scalar bank counts for parity diagnostics."""
        return self._bank.snapshot()

    @staticmethod
    def _prepare_frame(
        frame: Image.Image | np.ndarray,
    ) -> tuple[Image.Image, tuple[int, int]]:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        if not isinstance(frame, Image.Image):
            raise TypeError("frame must be a PIL image or an RGB numpy array")
        frame = frame.convert("RGB")
        original_size = frame.size
        resized = frame.resize((MODEL_SIZE, MODEL_SIZE))
        return resized, original_size

    def _encode_frame(
        self,
        frame: Image.Image | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
        resized, original_size = self._prepare_frame(frame)
        return self.image_encoder.predict({"image": resized}), original_size

    @staticmethod
    def _result(
        outputs: dict[str, np.ndarray],
        original_size: tuple[int, int],
    ) -> TrackingResult:
        low_res_mask = np.asarray(outputs["low_res_mask"])
        mask_logits_256 = low_res_mask.reshape(256, 256).astype(np.float32)
        resized_logits = Image.fromarray(mask_logits_256, mode="F").resize(
            original_size,
            resample=Image.Resampling.BILINEAR,
        )
        mask_logits = np.asarray(resized_logits, dtype=np.float32)
        return TrackingResult(
            mask_logits=mask_logits,
            mask=mask_logits > 0.0,
            iou=float(np.asarray(outputs["best_iou"]).reshape(-1)[0]),
            object_score=float(
                np.asarray(outputs["object_score"]).reshape(-1)[0]
            ),
        )

    def start_track(
        self,
        frame: Image.Image | np.ndarray,
        points: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
    ) -> TrackingResult:
        """Prompt the first frame and seed this predictor's memory bank."""
        features, original_size = self._encode_frame(frame)
        point_coords, point_labels = prepare_points(
            points,
            labels,
            original_size,
        )
        seed = self.initializer.predict(
            {
                "initial_vision_features": features["initial_vision_features"],
                "high_res_feature_0": features["high_res_feature_0"],
                "high_res_feature_1": features["high_res_feature_1"],
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
        )
        memory = self.memory_encoder.predict(
            {
                "raw_vision_features": features["raw_vision_features"],
                "high_res_mask": seed["high_res_mask"],
                "object_score": seed["object_score"],
            }
        )
        self._bank.seed(
            np.asarray(memory["memory_features"], dtype=np.float16),
            np.asarray(memory["memory_positions"], dtype=np.float16),
            np.asarray(memory["temporal_positions"], dtype=np.float16),
            np.asarray(seed["object_pointer"], dtype=np.float16),
        )
        self._started = True
        return self._result(seed, original_size)

    def track_frame(
        self,
        frame: Image.Image | np.ndarray,
    ) -> TrackingResult:
        """Propagate the current object mask onto the next video frame."""
        if not self._started:
            raise RuntimeError("start_track must be called before track_frame")
        features, original_size = self._encode_frame(frame)
        propagator_inputs = {
            "raw_vision_features": features["raw_vision_features"],
            "high_res_feature_0": features["high_res_feature_0"],
            "high_res_feature_1": features["high_res_feature_1"],
            **self._bank.model_inputs(),
        }
        outputs = self.propagator.predict(propagator_inputs)
        result = self._result(outputs, original_size)
        next_memory = np.asarray(outputs["memory_features"], dtype=np.float16)
        next_positions = np.asarray(
            outputs["memory_positions"],
            dtype=np.float16,
        )
        next_pointer = np.asarray(outputs["object_pointer"], dtype=np.float16)
        self._bank.commit(next_memory, next_positions, next_pointer)
        return result
