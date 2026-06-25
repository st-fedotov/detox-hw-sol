"""Task 5 — Build the reward model + write the training step [15 points].

Two pieces:

``build_rm`` — instantiate the RM (AMFSC + LoRA), ready for BT training.
``rm_step`` — one forward pass through the RM for the chosen and
              rejected sides of a preference batch, returning the
              Bradley-Terry loss (your ``bt_loss``) and the two score
              tensors for logging.

``train_rm.py`` calls these. The provided loop handles data, optimiser,
gradient clipping, and saving.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from tasks.task4_bt_loss import bt_loss


def build_rm(
    base_name: str = "Qwen/Qwen2.5-0.5B",
    pad_token_id: int | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Build the reward model:

    * AutoModelForSequenceClassification on top of ``base_name`` with
      ``num_labels=1`` (one scalar reward).
    * Set ``model.config.pad_token_id`` so AMFSC pools the last non-pad
      token — without this AMFSC silently pools index 0 and the reward
      signal collapses.
    * LoRA-wrap with ``task_type=SEQ_CLS`` (lets peft keep the scalar
      head trainable while freezing the dense backbone).

    LoRA target-modules choice: ``"all-linear"`` is the easy default
    (peft auto-discovers every Linear layer in the model and wraps it).
    Picking only the attention projections — for Qwen 2.5 that's
    ``["q_proj", "k_proj", "v_proj", "o_proj"]`` — is more
    parameter-efficient at the cost of capacity. The exact rank,
    alpha, and dropout are yours to tune.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=1, dtype=torch.float32,
    )
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(model, lora_cfg)


def rm_step(rm, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One training step's forward pass.

    ``batch`` has ``chosen_ids``, ``chosen_attn``, ``rejected_ids``,
    ``rejected_attn`` already on the right device.

    Returns ``(loss, chosen_scores, rejected_scores)``. The training
    loop in ``train_rm.py`` does ``.backward()``, gradient clip, and
    optimiser step around this.
    """
    chosen_scores = rm(
        input_ids=batch["chosen_ids"],
        attention_mask=batch["chosen_attn"],
    ).logits.squeeze(-1)
    rejected_scores = rm(
        input_ids=batch["rejected_ids"],
        attention_mask=batch["rejected_attn"],
    ).logits.squeeze(-1)
    loss = bt_loss(chosen_scores, rejected_scores).mean()
    return loss, chosen_scores, rejected_scores
