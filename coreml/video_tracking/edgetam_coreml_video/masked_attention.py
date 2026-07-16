# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-shape EdgeTAM memory attention with explicit validity masking."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

SPATIAL_SLOTS = 7
SPATIAL_TOKENS_PER_SLOT = 512
ROTARY_TOKENS_PER_SLOT = 256
NON_ROTARY_TOKENS_PER_SLOT = SPATIAL_TOKENS_PER_SLOT - ROTARY_TOKENS_PER_SLOT
POINTER_SLOTS = 16
POINTER_CHANNELS = 256
MEMORY_CHANNELS = 64
POINTER_TOKENS_PER_SLOT = POINTER_CHANNELS // MEMORY_CHANNELS
TOTAL_SPATIAL_TOKENS = SPATIAL_SLOTS * SPATIAL_TOKENS_PER_SLOT
TOTAL_POINTER_TOKENS = POINTER_SLOTS * POINTER_TOKENS_PER_SLOT
TOTAL_MEMORY_TOKENS = TOTAL_SPATIAL_TOKENS + TOTAL_POINTER_TOKENS
MAXIMUM_ROTARY_TOKENS = SPATIAL_SLOTS * ROTARY_TOKENS_PER_SLOT


def build_attention_controls(
    spatial_valid: torch.Tensor,
    pointer_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand compact validity into fixed numeric model inputs."""

    spatial_tokens = spatial_valid.unsqueeze(-1).expand(
        -1,
        -1,
        SPATIAL_TOKENS_PER_SLOT,
    ).flatten(1, 2)
    pointer_tokens = pointer_valid.unsqueeze(-1).expand(
        -1,
        -1,
        POINTER_TOKENS_PER_SLOT,
    ).flatten(1, 2)
    key_valid = torch.cat([spatial_tokens, pointer_tokens], dim=1)
    attention_bias = (1.0 - key_valid).reshape(
        key_valid.shape[0],
        1,
        1,
        TOTAL_MEMORY_TOKENS,
    ) * -10000.0

    valid_spatial_count = spatial_valid.sum(dim=-1, keepdim=True)
    token_indices = torch.arange(
        MAXIMUM_ROTARY_TOKENS,
        device=spatial_valid.device,
    ).reshape(1, -1)
    rotary_weight = (
        token_indices < valid_spatial_count * ROTARY_TOKENS_PER_SLOT
    ).to(spatial_valid.dtype)
    return attention_bias, rotary_weight


def _apply_rotary_real(
    tensor: torch.Tensor,
    frequency_cos: torch.Tensor,
    frequency_sin: torch.Tensor,
    repeat_frequencies: int,
) -> torch.Tensor:
    """Apply rotary encoding without Core ML-unsupported complex tensors."""
    if repeat_frequencies > 1:
        frequency_cos = (
            frequency_cos.unsqueeze(0)
            .expand(repeat_frequencies, -1, -1)
            .flatten(0, 1)
        )
        frequency_sin = (
            frequency_sin.unsqueeze(0)
            .expand(repeat_frequencies, -1, -1)
            .flatten(0, 1)
        )

    pairs = tensor.float().reshape(*tensor.shape[:-1], -1, 2)
    real = pairs[..., 0]
    imaginary = pairs[..., 1]
    frequency_cos = frequency_cos.reshape(1, 1, tensor.shape[-2], -1)
    frequency_sin = frequency_sin.reshape(1, 1, tensor.shape[-2], -1)
    rotated_real = real * frequency_cos - imaginary * frequency_sin
    rotated_imaginary = real * frequency_sin + imaginary * frequency_cos
    rotated = torch.stack([rotated_real, rotated_imaginary], dim=-1)
    return rotated.flatten(-2).type_as(tensor)


def _apply_rotary_spatial_slots(
    tensor: torch.Tensor,
    frequency_cos: torch.Tensor,
    frequency_sin: torch.Tensor,
    rotary_weight: torch.Tensor,
) -> torch.Tensor:
    """Rotate the 2D-token half of every fixed spatial-memory slot."""

    batch, heads, _, channels = tensor.shape
    spatial = tensor[:, :, :TOTAL_SPATIAL_TOKENS].reshape(
        batch,
        heads,
        SPATIAL_SLOTS,
        SPATIAL_TOKENS_PER_SLOT,
        channels,
    )
    non_rotary = spatial[:, :, :, :NON_ROTARY_TOKENS_PER_SLOT]
    rotary = spatial[:, :, :, NON_ROTARY_TOKENS_PER_SLOT:].reshape(
        batch,
        heads,
        MAXIMUM_ROTARY_TOKENS,
        channels,
    )
    rotated = _apply_rotary_real(
        rotary,
        frequency_cos,
        frequency_sin,
        SPATIAL_SLOTS,
    )
    weight = rotary_weight.reshape(
        rotary_weight.shape[0],
        1,
        MAXIMUM_ROTARY_TOKENS,
        1,
    ).to(tensor.dtype)
    rotary = weight * rotated + (1.0 - weight) * rotary
    rotary = rotary.reshape(
        batch,
        heads,
        SPATIAL_SLOTS,
        ROTARY_TOKENS_PER_SLOT,
        channels,
    )
    spatial = torch.cat([non_rotary, rotary], dim=3).flatten(2, 3)
    return torch.cat([spatial, tensor[:, :, TOTAL_SPATIAL_TOKENS:]], dim=2)


class MaskedMemoryAttention(nn.Module):
    """Run EdgeTAM memory attention over fixed state while ignoring empty slots."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.memory_attention = model.memory_attention
        for index, layer in enumerate(self.memory_attention.layers):
            self_frequencies = layer.self_attn.compute_cis(
                end_x=64,
                end_y=64,
            )
            cross_query_frequencies = layer.cross_attn_image.freqs_cis_q
            cross_key_frequencies = layer.cross_attn_image.freqs_cis_k
            for name, frequencies in (
                ("self", self_frequencies),
                ("cross_query", cross_query_frequencies),
                ("cross_key", cross_key_frequencies),
            ):
                self.register_buffer(
                    f"{name}_frequency_cos_{index}",
                    frequencies.real.float(),
                    persistent=False,
                )
                self.register_buffer(
                    f"{name}_frequency_sin_{index}",
                    frequencies.imag.float(),
                    persistent=False,
                )

    def _self_attention(
        self,
        attention: nn.Module,
        query: torch.Tensor,
        value: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        key = attention.k_proj(query)
        query = attention.q_proj(query)
        value = attention.v_proj(value)

        query = attention._separate_heads(query, attention.num_heads)
        key = attention._separate_heads(key, attention.num_heads)
        value = attention._separate_heads(value, attention.num_heads)

        frequency_cos = getattr(self, f"self_frequency_cos_{layer_index}")
        frequency_sin = getattr(self, f"self_frequency_sin_{layer_index}")
        query = _apply_rotary_real(query, frequency_cos, frequency_sin, 1)
        key = _apply_rotary_real(key, frequency_cos, frequency_sin, 1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        attended = attention._recombine_heads(attended)
        return attention.out_proj(attended)

    def _masked_cross_attention(
        self,
        attention: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor,
        rotary_weight: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        query = attention.q_proj(query)
        key = attention.k_proj(key)
        value = attention.v_proj(value)

        query = attention._separate_heads(query, attention.num_heads)
        key = attention._separate_heads(key, attention.num_heads)
        value = attention._separate_heads(value, attention.num_heads)

        query_frequency_cos = getattr(
            self,
            f"cross_query_frequency_cos_{layer_index}",
        )
        query_frequency_sin = getattr(
            self,
            f"cross_query_frequency_sin_{layer_index}",
        )
        key_frequency_cos = getattr(
            self,
            f"cross_key_frequency_cos_{layer_index}",
        )
        key_frequency_sin = getattr(
            self,
            f"cross_key_frequency_sin_{layer_index}",
        )
        query = _apply_rotary_real(
            query,
            query_frequency_cos,
            query_frequency_sin,
            1,
        )

        key = _apply_rotary_spatial_slots(
            key,
            key_frequency_cos,
            key_frequency_sin,
            rotary_weight,
        )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias.to(query.dtype),
            dropout_p=0.0,
        )
        attended = attention._recombine_heads(attended)
        return attention.out_proj(attended)

    def forward(
        self,
        current_features: torch.Tensor,
        current_positions: torch.Tensor,
        spatial_bank: torch.Tensor,
        spatial_positions: torch.Tensor,
        pointer_bank: torch.Tensor,
        attention_bias: torch.Tensor,
        rotary_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = current_features.shape
        current = current_features.flatten(2).permute(0, 2, 1)
        current_position = current_positions.flatten(2).permute(0, 2, 1)

        spatial_memory = spatial_bank.flatten(1, 2)
        spatial_position = spatial_positions.flatten(1, 2)
        pointer_memory = pointer_bank.reshape(
            batch,
            POINTER_SLOTS,
            POINTER_TOKENS_PER_SLOT,
            MEMORY_CHANNELS,
        ).flatten(1, 2)
        pointer_position = torch.zeros_like(pointer_memory)
        memory = torch.cat([spatial_memory, pointer_memory], dim=1)
        memory_position = torch.cat(
            [spatial_position, pointer_position],
            dim=1,
        )

        output = current
        if self.memory_attention.pos_enc_at_input:
            output = output + 0.1 * current_position

        for layer_index, layer in enumerate(self.memory_attention.layers):
            normalized = layer.norm1(output)
            self_query = (
                normalized + current_position
                if layer.pos_enc_at_attn
                else normalized
            )
            self_attended = self._self_attention(
                layer.self_attn,
                self_query,
                normalized,
                layer_index,
            )
            output = output + layer.dropout1(self_attended)

            normalized = layer.norm2(output)
            cross_query = (
                normalized + current_position
                if layer.pos_enc_at_cross_attn_queries
                else normalized
            )
            cross_key = (
                memory + memory_position
                if layer.pos_enc_at_cross_attn_keys
                else memory
            )
            cross_attended = self._masked_cross_attention(
                layer.cross_attn_image,
                cross_query,
                cross_key,
                memory,
                attention_bias,
                rotary_weight,
                layer_index,
            )
            output = output + layer.dropout2(cross_attended)

            normalized = layer.norm3(output)
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normalized)))
            )
            output = output + layer.dropout3(feed_forward)

        output = self.memory_attention.norm(output)
        return output.permute(0, 2, 1).reshape(batch, channels, height, width)
