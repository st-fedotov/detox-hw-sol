"""SFT trainer — provided. The student just runs this; no fill-in.

LM on the benign-side preference completions. Mask the loss to the
response half (don't grade the model on predicting its own system
prompt back). Same chat template applied at train time as at eval
time — if those drift the policy learns in one token-space and
produces garbage in the other.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.io import read_jsonl


IGNORE_INDEX = -100
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond to the user thoughtfully and kindly."
)
LORA_TARGETS_ALL = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------------------- #
# Dataset.                                                                    #
# --------------------------------------------------------------------------- #


def chat_prompt_ids(tokenizer, prompt: str) -> list[int]:
    out = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = out.input_ids if hasattr(out, "input_ids") else out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


class SftDataset(Dataset):
    """``{prompt, response}`` rows → token ids with loss masked to the
    response half. ``response`` here is the benign completion."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 512):
        self.examples: list[dict] = []
        for row in rows:
            prompt_ids = chat_prompt_ids(tokenizer, row["prompt"])
            resp_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                resp_ids = resp_ids + [tokenizer.eos_token_id]
            ids = prompt_ids + resp_ids
            if len(ids) > max_length:
                cut = len(ids) - max_length
                prompt_ids = prompt_ids[cut:]
                ids = prompt_ids + resp_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(resp_ids)
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def sft_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        attn[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


# --------------------------------------------------------------------------- #
# Train.                                                                      #
# --------------------------------------------------------------------------- #


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    rows: list[dict],
    out_dir: Path,
    base_name: str = "Qwen/Qwen2.5-0.5B",
    lr: float = 5e-5,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 1,
    lora_r: int = 32,
    log_every: int = 50,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.float32, device_map=device,
    )
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=list(LORA_TARGETS_ALL),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    ds = SftDataset(rows, tokenizer)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: sft_collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )
    print(f"sft train: {len(ds)} examples, {len(loader)} batches/epoch")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)
    total_micro = len(loader) * epochs
    total_steps = total_micro // grad_accum
    warmup = max(1, int(total_steps * 0.03))

    step = micro = 0
    optim.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum == 0:
                cur_lr = cosine_lr(step, total_steps, warmup, lr)
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad()
                step += 1
                if step % log_every == 0:
                    print(json.dumps({
                        "step": step, "of": total_steps,
                        "loss": float(out.loss.item()), "lr": cur_lr,
                    }))
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved SFT adapter to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSONL of {prompt, response} rows")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=32)
    a = p.parse_args()
    rows = list(read_jsonl(a.train))
    train(
        rows, Path(a.out), base_name=a.base, lr=a.lr,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        epochs=a.epochs, lora_r=a.lora_r,
    )


if __name__ == "__main__":
    main()
