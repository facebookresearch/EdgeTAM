import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(
        self, *, dim, dim_head=64, heads=8, dropout_p=0.05, concat_kv_latents=True
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout_p = dropout_p
        self.concat_kv_latents = concat_kv_latents

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, latents, x, pos=None):
        latents = self.norm_latents(latents)
        x = self.norm_x(x)

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if self.concat_kv_latents:
            kv_input = torch.cat((x, latents), dim=-2)
        else:
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        if pos is not None:
            assert not self.concat_kv_latents
            pos = self._separate_heads(pos, self.heads)
            k, v = k + pos, v + pos

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = self._recombine_heads(out)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout_p=0.05):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout_p = dropout_p

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, x):
        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = self._recombine_heads(out)
        return self.to_out(out)


class PerceiverEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        hidden_dropout_p=0.0,
        attention_dropout_p=0.0,
        concat_kv_latents=False,
        use_self_attn=False,
    ):
        super().__init__()
        self.attn = PerceiverAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout_p=attention_dropout_p,
            concat_kv_latents=concat_kv_latents,
        )
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.dropout = nn.Dropout(hidden_dropout_p)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout_p=attention_dropout_p,
            )
            self.self_ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, latents, x, pos=None):
        latents = self.attn(latents, x, pos) + latents
        latents = self.dropout(latents)
        latents = self.ff(latents) + latents
        if self.use_self_attn:
            latents = self.self_attn(latents) + latents
            latents = self.self_ff(latents) + latents
        return latents


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=1,
        num_latents=-1,
        num_latents_2d=-1,
        ff_mult=4,
        hidden_dropout_p=0.1,
        attention_dropout_p=0.05,
        pos_enc_at_key_value=False,
        concat_kv_latents=False,
        position_encoding=None,
        use_self_attn=False,
        **kwargs,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_latents_2d = num_latents_2d

        if num_latents > 0:
            self.latents = nn.Parameter(torch.randn(num_latents, dim))
        if num_latents_2d > 0:
            self.latents_2d = nn.Parameter(torch.randn(num_latents_2d, dim))
        self.position_encoding = position_encoding

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PerceiverEncoderLayer(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    hidden_dropout_p=hidden_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    concat_kv_latents=concat_kv_latents,
                    use_self_attn=use_self_attn,
                ),
            )

        self.norm = nn.LayerNorm(dim)
        self.pos_enc_at_key_value = pos_enc_at_key_value

    def forward(self, x, pos=None):
        out_latents = []
        out_pos = []
        if self.num_latents > 0:
            latents_1d, pos_1d = self.forward_1d(x, pos)
            out_latents.append(latents_1d)
            out_pos.append(pos_1d)
        if self.num_latents_2d > 0:
            latents_2d, pos_2d = self.forward_2d(x)
            out_latents.append(latents_2d)
            out_pos.append(pos_2d)

        latents = torch.concat(out_latents, dim=1)
        if pos is not None:
            pos = torch.concat(out_pos, dim=1)

        return latents, pos

    def forward_1d(self, x, pos):
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)

        if not self.pos_enc_at_key_value:
            _pos = None
        if pos is not None:
            _pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            _pos = None

        for layer in self.layers:
            latents = layer(latents, x, _pos)

        if pos is not None:
            pos = torch.zeros_like(latents)

        latents = self.norm(latents)
        return latents, pos

    def forward_2d(self, x):
        B, C, H, W = x.shape

        latents_2d = self.latents_2d.unsqueeze(0).expand(B, -1, -1).view(-1, 1, C)

        num_window = int(math.sqrt(self.num_latents_2d))
        window_size = H // num_window
        x = x.permute(0, 2, 3, 1)

        x = window_partition(x, window_size)
        x = x.flatten(1, 2)

        for layer in self.layers:
            latents_2d = layer(latents_2d, x)

        latents_2d = latents_2d.view(B, num_window, num_window, C).permute(0, 3, 1, 2)

        pos_2d = self.position_encoding(latents_2d)
        pos_2d = pos_2d.permute(0, 2, 3, 1).flatten(1, 2)

        latents_2d = latents_2d.permute(0, 2, 3, 1).flatten(1, 2)

        latents_2d = self.norm(latents_2d)

        return latents_2d, pos_2d
