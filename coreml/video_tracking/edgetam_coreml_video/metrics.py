# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity metrics shared by component and video validation tests."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorError:
    """Absolute-error summary for two tensors."""

    max_abs: float
    mean_abs: float


def tensor_error(reference: np.ndarray, actual: np.ndarray) -> TensorError:
    """Return maximum and mean absolute error after float32 conversion."""

    delta = np.abs(
        np.asarray(reference, dtype=np.float32)
        - np.asarray(actual, dtype=np.float32)
    )
    return TensorError(
        max_abs=float(delta.max(initial=0.0)),
        mean_abs=float(delta.mean()) if delta.size else 0.0,
    )


def binary_mask_iou(reference: np.ndarray, actual: np.ndarray) -> float:
    """Return intersection-over-union for two binary masks."""

    reference_mask = np.asarray(reference, dtype=bool)
    actual_mask = np.asarray(actual, dtype=bool)
    intersection = np.logical_and(reference_mask, actual_mask).sum()
    union = np.logical_or(reference_mask, actual_mask).sum()
    return 1.0 if union == 0 else float(intersection / union)


def cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    """Return cosine similarity for two flattened vectors."""

    reference_vector = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual_vector = np.asarray(actual, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(reference_vector) * np.linalg.norm(actual_vector)
    if denominator == 0:
        return 1.0 if np.array_equal(reference_vector, actual_vector) else 0.0
    return float(np.dot(reference_vector, actual_vector) / denominator)
