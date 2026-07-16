# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
from PIL import Image

from edgetam_coreml_video.predictor import (
    CoreMLVideoPredictor,
    prepare_points,
)


class FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def predict(self, inputs, state=None):
        self.calls.append((inputs, state))
        if callable(self.output):
            return self.output(inputs)
        return self.output


class FakePropagator(FakeModel):
    def __init__(self):
        super().__init__(self._output)
        self.step = 0

    def _output(self, inputs):
        assert "mode" not in inputs
        assert not any(name.startswith("seed_") for name in inputs)
        self.step += 1
        propagated = float(self.step + 1)
        return {
            "low_res_mask": np.full((1, 1, 256, 256), propagated),
            "high_res_mask": np.full((1, 1, 1024, 1024), propagated + 10),
            "best_iou": np.array([0.75]),
            "object_pointer": np.full((1, 256), propagated, dtype=np.float16),
            "object_score": np.array([[1.0]]),
            "memory_features": np.full(
                (1, 512, 64),
                propagated,
                dtype=np.float16,
            ),
            "memory_positions": np.full(
                (1, 512, 64),
                propagated * 10,
                dtype=np.float16,
            ),
        }


def _image_features():
    return {
        "raw_vision_features": np.zeros((1, 256, 64, 64)),
        "initial_vision_features": np.zeros((1, 256, 64, 64)),
        "high_res_feature_0": np.zeros((1, 32, 256, 256)),
        "high_res_feature_1": np.zeros((1, 64, 128, 128)),
    }


def test_prepare_points_scales_to_1024_without_transformer_visible_padding():
    coords, labels = prepare_points(
        points=np.array([[160.0, 120.0]]),
        labels=np.array([1]),
        original_size=(640, 480),
    )

    np.testing.assert_allclose(coords[0, 0], [256.0, 256.0])
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[1]]
    assert labels.dtype == np.int32


def test_prepare_frame_matches_video_predictor_pillow_resize():
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    frame = Image.fromarray(pixels, mode="RGB")

    actual, original_size = CoreMLVideoPredictor._prepare_frame(frame)
    expected = frame.resize((1024, 1024))

    assert original_size == (2, 2)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_predictor_seeds_and_propagates_with_explicit_stateless_bank(
    tmp_path: Path,
):
    image_encoder = FakeModel(_image_features())
    initializer = FakeModel(
        {
            "low_res_mask": np.ones((1, 1, 256, 256)),
            "high_res_mask": np.ones((1, 1, 1024, 1024)),
            "best_iou": np.array([0.8]),
            "object_pointer": np.full((1, 256), 10, dtype=np.float16),
            "object_score": np.array([[1.5]]),
        }
    )
    memory_encoder = FakeModel(
        {
            "memory_features": np.zeros((1, 512, 64)),
            "memory_positions": np.zeros((1, 512, 64)),
            "temporal_positions": np.zeros((7, 64), dtype=np.float16),
        }
    )
    propagator = FakePropagator()
    predictor = CoreMLVideoPredictor(
        image_encoder=image_encoder,
        initializer=initializer,
        memory_encoder=memory_encoder,
        propagator=propagator,
    )
    frame = Image.new("RGB", (640, 480))

    seed = predictor.start_track(frame, [[160.0, 120.0]], [1])
    first_propagated = predictor.track_frame(frame)
    second_propagated = predictor.track_frame(frame)
    snapshot = predictor.debug_bank_snapshot()

    assert seed.mask_logits.shape == (480, 640)
    assert first_propagated.mask_logits.shape == (480, 640)
    assert second_propagated.mask_logits.shape == (480, 640)
    assert np.all(seed.mask_logits == 1.0)
    assert np.all(first_propagated.mask_logits == 2.0)
    assert np.all(second_propagated.mask_logits == 3.0)
    assert np.all(seed.mask)
    assert np.all(first_propagated.mask)
    assert np.all(second_propagated.mask)
    assert len(propagator.calls) == 2
    assert all(state is None for _, state in propagator.calls)
    assert "attention_bias" in propagator.calls[0][0]
    assert "rotary_weight" in propagator.calls[0][0]
    assert snapshot.recent_count == 2
    assert snapshot.pointer_count == 3
