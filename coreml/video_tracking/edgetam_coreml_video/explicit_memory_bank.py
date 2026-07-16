# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit fixed-shape NumPy memory state for stateless EdgeTAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEMORY_SHAPE = (1, 512, 64)
TEMPORAL_POSITION_SHAPE = (7, 64)
POINTER_SHAPE = (1, 256)
RECENT_SLOTS = 6
POINTER_SLOTS = 16
SPATIAL_TOKENS_PER_SLOT = 512
POINTER_TOKENS_PER_SLOT = 4
TOTAL_SPATIAL_TOKENS = 7 * SPATIAL_TOKENS_PER_SLOT
TOTAL_ATTENTION_TOKENS = TOTAL_SPATIAL_TOKENS + (
    POINTER_SLOTS * POINTER_TOKENS_PER_SLOT
)
ROTARY_TOKENS_PER_SLOT = 256
TOTAL_ROTARY_TOKENS = 7 * ROTARY_TOKENS_PER_SLOT


@dataclass(frozen=True)
class MemoryBankSnapshot:
    """Small validation view that does not duplicate bank tensors."""

    recent_count: int
    pointer_count: int
    is_initialized: bool


class ExplicitMemoryBank:
    """Own one object's conditioning, recent-memory, and pointer history."""

    def __init__(self) -> None:
        self.conditioning_memory = np.zeros(MEMORY_SHAPE, dtype=np.float16)
        self.conditioning_position = np.zeros(MEMORY_SHAPE, dtype=np.float16)
        self.recent_memory = np.zeros(
            (1, RECENT_SLOTS, 512, 64),
            dtype=np.float16,
        )
        self.recent_positions = np.zeros_like(self.recent_memory)
        self.pointer_history = np.zeros(
            (1, POINTER_SLOTS, 256),
            dtype=np.float16,
        )
        self.temporal_positions = np.zeros(
            TEMPORAL_POSITION_SHAPE,
            dtype=np.float16,
        )
        self.recent_count = 0
        self.pointer_count = 0
        self.is_initialized = False

    @staticmethod
    def _validated(
        value: np.ndarray,
        shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if array.dtype != np.float16:
            raise ValueError(f"{name} must use float16")
        return array

    def seed(
        self,
        memory: np.ndarray,
        memory_positions: np.ndarray,
        temporal_positions: np.ndarray,
        pointer: np.ndarray,
    ) -> None:
        """Replace all state with one prompted conditioning frame."""

        memory = self._validated(memory, MEMORY_SHAPE, "memory")
        memory_positions = self._validated(
            memory_positions,
            MEMORY_SHAPE,
            "memory_positions",
        )
        temporal_positions = self._validated(
            temporal_positions,
            TEMPORAL_POSITION_SHAPE,
            "temporal_positions",
        )
        pointer = self._validated(pointer, POINTER_SHAPE, "pointer")

        self.conditioning_memory[...] = memory
        self.temporal_positions[...] = temporal_positions
        self.conditioning_position[...] = (
            memory_positions + temporal_positions[6].reshape(1, 1, 64)
        )
        self.recent_memory.fill(0)
        self.recent_positions.fill(0)
        self.pointer_history.fill(0)
        self.pointer_history[:, 0] = pointer
        self.recent_count = 0
        self.pointer_count = 1
        self.is_initialized = True

    def commit(
        self,
        memory: np.ndarray,
        memory_positions: np.ndarray,
        pointer: np.ndarray,
    ) -> None:
        """Append one successful propagation result."""

        if not self.is_initialized:
            raise RuntimeError("seed must be called before commit")
        memory = self._validated(memory, MEMORY_SHAPE, "memory")
        memory_positions = self._validated(
            memory_positions,
            MEMORY_SHAPE,
            "memory_positions",
        )
        pointer = self._validated(pointer, POINTER_SHAPE, "pointer")

        if self.recent_count < RECENT_SLOTS:
            index = self.recent_count
        else:
            self.recent_memory[:, :-1] = self.recent_memory[:, 1:].copy()
            self.recent_positions[:, :-1] = self.recent_positions[:, 1:].copy()
            index = RECENT_SLOTS - 1
        self.recent_memory[:, index] = memory
        self.recent_positions[:, index] = memory_positions
        self.recent_count = min(self.recent_count + 1, RECENT_SLOTS)

        self.pointer_history[:, 2:] = self.pointer_history[:, 1:-1].copy()
        self.pointer_history[:, 1] = pointer
        self.pointer_count = min(self.pointer_count + 1, POINTER_SLOTS)

    def model_inputs(self) -> dict[str, np.ndarray]:
        """Assemble fixed Float16 tensors for one propagation call."""

        if not self.is_initialized:
            raise RuntimeError("seed must be called before model_inputs")

        spatial_bank = np.zeros((1, 7, 512, 64), dtype=np.float16)
        spatial_positions = np.zeros_like(spatial_bank)
        spatial_bank[:, 0] = self.conditioning_memory
        spatial_positions[:, 0] = self.conditioning_position
        if self.recent_count:
            spatial_bank[:, 1 : self.recent_count + 1] = self.recent_memory[
                :, : self.recent_count
            ]
            for slot in range(self.recent_count):
                temporal_index = self.recent_count - 1 - slot
                spatial_positions[:, slot + 1] = (
                    self.recent_positions[:, slot]
                    + self.temporal_positions[temporal_index].reshape(1, 1, 64)
                )

        attention_bias = np.full(
            (1, 1, 1, TOTAL_ATTENTION_TOKENS),
            -10000,
            dtype=np.float16,
        )
        valid_spatial_tokens = (
            1 + self.recent_count
        ) * SPATIAL_TOKENS_PER_SLOT
        attention_bias[..., :valid_spatial_tokens] = 0
        valid_pointer_tokens = self.pointer_count * POINTER_TOKENS_PER_SLOT
        attention_bias[
            ...,
            TOTAL_SPATIAL_TOKENS : TOTAL_SPATIAL_TOKENS
            + valid_pointer_tokens,
        ] = 0

        rotary_weight = np.zeros(
            (1, TOTAL_ROTARY_TOKENS),
            dtype=np.float16,
        )
        valid_rotary_tokens = min(
            (1 + self.recent_count) * ROTARY_TOKENS_PER_SLOT,
            TOTAL_ROTARY_TOKENS,
        )
        rotary_weight[:, :valid_rotary_tokens] = 1

        return {
            "spatial_bank": spatial_bank,
            "spatial_positions": spatial_positions,
            "pointer_bank": self.pointer_history.copy(),
            "attention_bias": attention_bias,
            "rotary_weight": rotary_weight,
        }

    def snapshot(self) -> MemoryBankSnapshot:
        """Return scalar state for diagnostics and numerical validation."""

        return MemoryBankSnapshot(
            recent_count=self.recent_count,
            pointer_count=self.pointer_count,
            is_initialized=self.is_initialized,
        )
