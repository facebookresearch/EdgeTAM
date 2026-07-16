# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from edgetam_coreml_video.explicit_memory_bank import ExplicitMemoryBank


def _memory(value: float) -> np.ndarray:
    return np.full((1, 512, 64), value, dtype=np.float16)


def _position(value: float) -> np.ndarray:
    return np.full((1, 512, 64), value, dtype=np.float16)


def _pointer(value: float) -> np.ndarray:
    return np.full((1, 256), value, dtype=np.float16)


def _temporal_positions() -> np.ndarray:
    return np.broadcast_to(
        np.arange(7, dtype=np.float16).reshape(7, 1),
        (7, 64),
    ).copy()


def test_explicit_memory_bank_rolls_recent_memory_and_pointer_history():
    bank = ExplicitMemoryBank()
    bank.seed(
        _memory(10),
        _position(100),
        _temporal_positions(),
        _pointer(10),
    )
    for value in range(1, 8):
        bank.commit(
            _memory(float(value)),
            _position(float(value * 10)),
            _pointer(float(value)),
        )

    inputs = bank.model_inputs()
    snapshot = bank.snapshot()

    assert inputs["spatial_bank"][:, :, 0, 0].tolist() == [
        [10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ]
    assert inputs["spatial_positions"][:, :, 0, 0].tolist() == [
        [106.0, 25.0, 34.0, 43.0, 52.0, 61.0, 70.0]
    ]
    assert inputs["pointer_bank"][:, :8, 0].tolist() == [
        [10.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    ]
    assert np.all(inputs["attention_bias"][..., :3616] == 0)
    assert np.all(inputs["attention_bias"][..., 3616:] == -10000)
    assert np.all(inputs["rotary_weight"] == 1)
    assert snapshot.recent_count == 6
    assert snapshot.pointer_count == 8
    assert snapshot.is_initialized is True


def test_explicit_memory_bank_warmup_masks_unused_fixed_slots():
    bank = ExplicitMemoryBank()
    bank.seed(
        _memory(10),
        _position(100),
        _temporal_positions(),
        _pointer(10),
    )
    bank.commit(_memory(1), _position(10), _pointer(1))

    inputs = bank.model_inputs()

    assert np.all(inputs["attention_bias"][..., :1024] == 0)
    assert np.all(inputs["attention_bias"][..., 1024:3584] == -10000)
    assert np.all(inputs["attention_bias"][..., 3584:3592] == 0)
    assert np.all(inputs["attention_bias"][..., 3592:] == -10000)
    assert np.all(inputs["rotary_weight"][:, :512] == 1)
    assert np.all(inputs["rotary_weight"][:, 512:] == 0)
    assert inputs["spatial_positions"][0, 0, 0, 0] == 106
    assert inputs["spatial_positions"][0, 1, 0, 0] == 10


def test_explicit_memory_bank_rejects_wrong_seed_shapes():
    bank = ExplicitMemoryBank()

    with pytest.raises(ValueError, match="memory must have shape"):
        bank.seed(
            np.zeros((1, 511, 64), dtype=np.float16),
            _position(0),
            _temporal_positions(),
            _pointer(0),
        )
