"""Task 4 — Bradley-Terry preference loss [10 points].

Given chosen/rejected reward scores produced by the RM, return the
per-example BT loss:

    L_BT(s_+, s_-) = -log sigmoid(s_+ - s_-)

Returns a tensor of shape (batch,) — the caller .mean()s.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bt_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
) -> torch.Tensor:
    return -F.logsigmoid(chosen_scores - rejected_scores)
