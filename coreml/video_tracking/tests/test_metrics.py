# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from edgetam_coreml_video.metrics import (
    binary_mask_iou,
    cosine_similarity,
    tensor_error,
)


def test_binary_mask_iou_is_one_for_identical_masks():
    mask = np.array([[True, False], [True, True]])

    assert binary_mask_iou(mask, mask) == 1.0


def test_binary_mask_iou_is_one_for_two_empty_masks():
    mask = np.zeros((2, 2), dtype=bool)

    assert binary_mask_iou(mask, mask) == 1.0


def test_tensor_error_reports_maximum_and_mean_absolute_error():
    error = tensor_error(
        np.array([0.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
    )

    assert error.max_abs == 1.0
    assert error.mean_abs == 0.5


def test_cosine_similarity_is_one_for_equal_vectors():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert np.isclose(cosine_similarity(vector, vector), 1.0)
