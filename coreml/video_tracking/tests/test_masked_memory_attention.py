# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect

import torch

from edgetam_coreml_video.masked_attention import (
    MaskedMemoryAttention,
    _apply_rotary_spatial_slots,
    build_attention_controls,
)
from sam2.modeling.position_encoding import apply_rotary_enc_v2


def test_attention_controls_expand_slot_validity_without_model_graph_logic():
    spatial_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0]],
        dtype=torch.float16,
    )
    pointer_valid = torch.tensor(
        [[1, 1] + [0] * 14],
        dtype=torch.float16,
    )

    attention_bias, rotary_weight = build_attention_controls(
        spatial_valid,
        pointer_valid,
    )

    assert attention_bias.shape == (1, 1, 1, 3648)
    assert rotary_weight.shape == (1, 1792)
    assert attention_bias.dtype == torch.float16
    assert rotary_weight.dtype == torch.float16
    assert torch.all(attention_bias[..., :1024] == 0)
    assert torch.all(attention_bias[..., 1024:3584] == -10000)
    assert torch.all(attention_bias[..., 3584:3592] == 0)
    assert torch.all(attention_bias[..., 3592:] == -10000)
    assert torch.all(rotary_weight[:, :512] == 1)
    assert torch.all(rotary_weight[:, 512:] == 0)


def test_converted_attention_path_does_not_construct_validity_from_counts():
    source = inspect.getsource(MaskedMemoryAttention.forward)
    source += inspect.getsource(MaskedMemoryAttention._masked_cross_attention)

    assert "spatial_valid" not in source
    assert "pointer_valid" not in source
    assert "torch.arange" not in source
    assert ".sum(" not in source


def test_rotary_encoding_targets_last_half_of_each_spatial_slot(reference_model):
    torch.manual_seed(31)
    attention = reference_model.memory_attention.layers[0].cross_attn_image
    key = torch.randn(1, 1, 3648, 256)
    spatial_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    pointer_valid = torch.tensor(
        [[1, 1] + [0] * 14],
        dtype=torch.float32,
    )
    _, rotary_weight = build_attention_controls(spatial_valid, pointer_valid)

    actual = _apply_rotary_spatial_slots(
        key,
        attention.freqs_cis_k.real.float(),
        attention.freqs_cis_k.imag.float(),
        rotary_weight,
    )
    expected = key.clone()
    expected[:, :, :1024] = apply_rotary_enc_v2(
        key[:, :, :1024],
        attention.freqs_cis_k,
        repeat_freqs=2,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _pointer_tokens(pointer_bank: torch.Tensor, count: int) -> torch.Tensor:
    pointers = pointer_bank[:, :count]
    batch, pointer_count, channels = pointers.shape
    return pointers.reshape(batch, pointer_count, channels // 64, 64).flatten(1, 2)


def test_masked_memory_attention_matches_compact_reference_memory(reference_model):
    torch.manual_seed(3)
    spatial_count = 3
    pointer_count = 2
    current = torch.randn(1, 256, 64, 64)
    current_position = reference_model.image_encoder.neck.position_encoding(current)
    spatial_bank = torch.randn(1, 7, 512, 64)
    spatial_positions = torch.randn(1, 7, 512, 64)
    spatial_valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    pointer_bank = torch.randn(1, 16, 256)
    pointer_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    attention_bias, rotary_weight = build_attention_controls(
        spatial_valid,
        pointer_valid,
    )
    wrapper = MaskedMemoryAttention(reference_model).eval()

    compact_spatial = spatial_bank[:, :spatial_count].flatten(1, 2)
    compact_positions = spatial_positions[:, :spatial_count].flatten(1, 2)
    compact_pointers = _pointer_tokens(pointer_bank, pointer_count)
    compact_pointer_positions = torch.zeros_like(compact_pointers)
    compact_memory = torch.cat([compact_spatial, compact_pointers], dim=1)
    compact_memory_positions = torch.cat(
        [compact_positions, compact_pointer_positions], dim=1
    )

    with torch.inference_mode():
        actual = wrapper(
            current,
            current_position,
            spatial_bank,
            spatial_positions,
            pointer_bank,
            attention_bias,
            rotary_weight,
        )
        expected = reference_model.memory_attention(
            curr=current.flatten(2).permute(2, 0, 1),
            curr_pos=current_position.flatten(2).permute(2, 0, 1),
            memory=compact_memory.permute(1, 0, 2),
            memory_pos=compact_memory_positions.permute(1, 0, 2),
            num_obj_ptr_tokens=pointer_count * 4,
            num_spatial_mem=spatial_count,
        )
        expected = expected.permute(1, 2, 0).reshape_as(current)

    absolute_error = (actual - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(),
        expected.flatten(),
        dim=0,
    )
    assert absolute_error.mean().item() < 0.02
    assert absolute_error.max().item() < 0.5
    assert cosine.item() > 0.999
