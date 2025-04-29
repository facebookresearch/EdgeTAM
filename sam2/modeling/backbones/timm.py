"""Backbones from the TIMM library."""

from typing import List, Tuple

import torch

from timm.models import create_model
from torch import nn


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        features: Tuple[str, ...],
    ):
        super().__init__()

        out_indices = tuple(int(f[len("layer") :]) for f in features)

        backbone = create_model(
            name,
            pretrained=True,
            in_chans=3,
            features_only=True,
            out_indices=out_indices,
        )

        num_channels = backbone.feature_info.channels()
        self.channel_list = num_channels[::-1]
        self.body = backbone

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xs = self.body(x)

        out = []
        for i, x in enumerate(xs):
            out.append(x)
        return out
