# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from PIL import Image

import validate_video as validate_video_module
from validate_video import (
    _compare_coreml_frames,
    compare_mask_logits,
    parse_args,
    run_validation,
)


def test_compare_mask_logits_reports_tensor_error_and_binary_iou():
    reference = np.array([[1.0, 1.0], [-1.0, -1.0]])
    candidate = np.array([[1.0, -1.0], [-1.0, -1.0]])

    comparison = compare_mask_logits(reference, candidate)

    assert comparison.max_abs == 2.0
    assert comparison.mean_abs == 0.5
    assert comparison.mask_iou == 0.5


class _Result:
    def __init__(self, mask_logits, iou=0.75, object_score=1.25):
        self.mask_logits = mask_logits
        self.iou = iou
        self.object_score = object_score


class _Predictor:
    def __init__(self, logits):
        self.logits = iter(logits)
        self.start_calls = 0
        self.track_calls = 0
        self.recent_count = 0
        self.pointer_count = 0

    def start_track(self, frame, points, labels):
        self.start_calls += 1
        self.recent_count = 0
        self.pointer_count = 1
        return _Result(next(self.logits))

    def track_frame(self, frame):
        self.track_calls += 1
        self.recent_count = min(self.recent_count + 1, 6)
        self.pointer_count = min(self.pointer_count + 1, 16)
        return _Result(next(self.logits))

    def debug_bank_snapshot(self):
        return SimpleNamespace(
            recent_count=self.recent_count,
            pointer_count=self.pointer_count,
        )


def test_compare_coreml_frames_returns_one_row_per_frame(tmp_path):
    frame_paths = []
    for frame_index in range(2):
        frame_path = tmp_path / f"{frame_index:05d}.jpg"
        Image.new("RGB", (8, 6), (frame_index * 20, 0, 0)).save(frame_path)
        frame_paths.append(frame_path)
    reference = {
        0: np.array([[1.0, -1.0]], dtype=np.float32),
        1: np.array([[1.0, 1.0]], dtype=np.float32),
    }
    predictor = _Predictor(
        [
            np.array([[1.0, -1.0]], dtype=np.float32),
            np.array([[1.0, -1.0]], dtype=np.float32),
        ]
    )

    rows = _compare_coreml_frames(
        frame_paths,
        reference,
        predictor,
        np.array([[2.0, 3.0]], dtype=np.float32),
        np.array([1], dtype=np.int32),
    )

    assert predictor.start_calls == 1
    assert predictor.track_calls == 1
    assert [row["frame"] for row in rows] == [0, 1]
    assert rows[0]["mask_iou"] == 1.0
    assert rows[1]["mask_iou"] == 0.5
    assert rows[0]["predicted_iou"] == 0.75
    assert rows[0]["object_score"] == 1.25
    assert rows[0]["recent_count"] == 0
    assert rows[0]["pointer_count"] == 1
    assert rows[1]["recent_count"] == 1
    assert rows[1]["pointer_count"] == 2


def test_compare_coreml_frames_reports_saturated_bank_counts(tmp_path):
    frame_paths = []
    logits = []
    reference = {}
    for frame_index in range(20):
        frame_path = tmp_path / f"{frame_index:05d}.jpg"
        Image.new("RGB", (2, 2)).save(frame_path)
        frame_paths.append(frame_path)
        frame_logits = np.ones((2, 2), dtype=np.float32)
        logits.append(frame_logits)
        reference[frame_index] = frame_logits

    rows = _compare_coreml_frames(
        frame_paths,
        reference,
        _Predictor(logits),
        np.array([[1.0, 1.0]], dtype=np.float32),
        np.array([1], dtype=np.int32),
    )

    assert [row["recent_count"] for row in rows] == [
        0, 1, 2, 3, 4, 5, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    ]
    assert [row["pointer_count"] for row in rows] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 16, 16, 16, 16,
    ]


def test_parse_args_accepts_json_output(monkeypatch, tmp_path):
    json_path = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_video.py",
            "--frames-dir",
            "frames",
            "--json",
            str(json_path),
        ],
    )

    args = parse_args()

    assert args.json == json_path


def test_run_validation_writes_requested_json(monkeypatch, tmp_path):
    rows = [
        {
            "frame": 0,
            "mask_iou": 0.99,
            "cosine": 0.999,
            "mean_abs": 0.1,
            "max_abs": 1.0,
        }
    ]
    captured = {}

    def fake_validate_video(**kwargs):
        captured.update(kwargs)
        return rows

    monkeypatch.setattr(validate_video_module, "validate_video", fake_validate_video)
    json_path = tmp_path / "validation.json"
    args = Namespace(
        frames_dir=tmp_path / "frames",
        models_dir=tmp_path / "models",
        checkpoint=tmp_path / "edgetam.pt",
        config_name="edgetam.yaml",
        device="mps",
        max_frames=8,
        point=[[210.0, 350.0, 1.0]],
        json=json_path,
    )

    actual_rows = run_validation(args)

    assert actual_rows == rows
    assert captured["max_frames"] == 8
    assert json.loads(json_path.read_text()) == rows
