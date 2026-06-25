"""Task 2 — DPO loss [15 points].

Given log-probabilities of the chosen and rejected completion under
both the policy and a frozen reference, return:

    losses           — per-example shape (batch,)
    chosen_rewards   — beta * (policy_chosen - reference_chosen), detached
    rejected_rewards — beta * (policy_rejected - reference_rejected), detached

The DPO loss is:

    -log sigmoid( beta * (
        log pi(y+|x) - log pi_ref(y+|x)
      - log pi(y-|x) + log pi_ref(y-|x)
    ))

The chosen/rejected rewards do NOT feed the optimiser — they're a
logging signal: their margin should rise during training, and either
drifting strongly negative is a known DPO-collapse leading indicator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios  = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    chosen_rewards   = beta * (policy_chosen_logps   - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards
