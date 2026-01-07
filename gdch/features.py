import math
from typing import Optional

import torch
import torch.nn as nn


def calendar_features(t: torch.Tensor, t0_days: float = 0.0, total_time: float = 1.0) -> torch.Tensor:
    """Compute differentiable calendar features from continuous time in days."""
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.to(dtype=torch.float32)
    t0 = torch.tensor(t0_days, dtype=torch.float32, device=t.device)
    total_time = float(total_time) if total_time > 0 else 1.0

    t_abs = t + t0
    two_pi = 2.0 * math.pi

    hour_phase = two_pi * torch.remainder(t_abs, 1.0)
    dow_phase = two_pi * (torch.remainder(t_abs, 7.0) / 7.0)
    year_period = 365.25
    month_period = year_period / 12.0
    month_phase = two_pi * (torch.remainder(t_abs, year_period) / month_period)
    year_phase = two_pi * (torch.remainder(t_abs, year_period) / year_period)

    trend = (t / total_time).clamp(0.0, 1.0)

    feats = torch.stack(
        [
            torch.sin(hour_phase),
            torch.cos(hour_phase),
            torch.sin(dow_phase),
            torch.cos(dow_phase),
            torch.sin(month_phase),
            torch.cos(month_phase),
            torch.sin(year_phase),
            torch.cos(year_phase),
            trend,
        ],
        dim=-1,
    )
    return feats


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
