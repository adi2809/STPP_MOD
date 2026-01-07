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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
