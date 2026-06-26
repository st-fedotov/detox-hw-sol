"""Train the prompt-conditioned reward model for the detox homework.

You fill in:

  * ``tasks/task4_bt_loss.py`` — Bradley-Terry loss (Task 4).
  * ``tasks/task5_reward_head.py`` — ``build_rm`` + ``rm_step`` (Task 5).

This script wires those into a training loop, runs Bradley-Terry on the
preference JSONL, and saves the RM in the format verl loads natively
(``AutoModelForSequenceClassification.from_pretrained``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.common.io import read_jsonl
from tasks.task5_reward_head import build_rm, rm_step


# --------------------------------------------------------------------------- #
# Dataset.                                                                    #
# --------------------------------------------------------------------------- #


class PreferenceDataset(Dataset):
    """Tokenize ``(prompt, chosen)`` and ``(prompt, rejected)`` as
    sentence-pair sequences. The RM reads the last non-pad token's
    hidden state, so the prompt and the response live in one sequence
    and the reward depends on both."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 256) -> None:
        self.examples: list[dict] = []
        for r in rows:
            prompt = r.get("prompt", "")
            chosen = r.get("chosen") or r.get("toxic")
            rejected = r.get("rejected") or r.get("non_toxic")
            if chosen is None or rejected is None:
                raise KeyError(f"row missing chosen/rejected: {list(r)}")
            c = tokenizer(prompt, chosen, truncation="longest_first",
                          max_length=max_length, padding=False)
            rj = tokenizer(prompt, rejected, truncation="longest_first",
                           max_length=max_length, padding=False)
            self.examples.append({
                "chosen_ids":     c["input_ids"],
                "chosen_attn":    c["attention_mask"],
                "rejected_ids":   rj["input_ids"],
                "rejected_attn":  rj["attention_mask"],
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    def pad(side: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [b[f"{side}_ids"] for b in batch]
        attn = [b[f"{side}_attn"] for b in batch]
        L = max(len(x) for x in ids)
        padded_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
        padded_attn = torch.zeros((len(ids), L), dtype=torch.long)
        for i, (x, a) in enumerate(zip(ids, attn)):
            n = len(x)
            padded_ids[i, :n] = torch.tensor(x, dtype=torch.long)
            padded_attn[i, :n] = torch.tensor(a, dtype=torch.long)
        return padded_ids, padded_attn

    ci, ca = pad("chosen")
    ri, ra = pad("rejected")
    return {"chosen_ids": ci, "chosen_attn": ca,
            "rejected_ids": ri, "rejected_attn": ra}


# --------------------------------------------------------------------------- #
# Train loop.                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    train_path: str
    out_dir: str
    base_name: str = "Qwen/Qwen2.5-0.5B"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 5e-5
    max_length: int = 256
    log_every: int = 20
    val_fraction: float = 0.1
    seed: int = 0


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 90/10 split.
    all_rows = list(read_jsonl(cfg.train_path))
    import random as _random
    rng = _random.Random(cfg.seed)
    idx = list(range(len(all_rows)))
    rng.shuffle(idx)
    n_val = int(len(all_rows) * cfg.val_fraction)
    val_idx = set(idx[:n_val])
    train_rows = [r for i, r in enumerate(all_rows) if i not in val_idx]
    val_rows   = [r for i, r in enumerate(all_rows) if i in val_idx]
    print(f"train: {len(train_rows)} pairs, val: {len(val_rows)} pairs")

    ds = PreferenceDataset(train_rows, tokenizer, max_length=cfg.max_length)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    rm = build_rm(cfg.base_name, pad_token_id=tokenizer.pad_token_id).to(device)
    rm.train()
    trainable = [p for p in rm.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)

    step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, sc, sr = rm_step(rm, batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            step += 1
            if step % cfg.log_every == 0:
                with torch.no_grad():
                    acc = (sc > sr).float().mean().item()
                print(json.dumps({
                    "epoch": epoch, "step": step,
                    "loss": float(loss.item()), "pairwise_acc": acc,
                }))

    # Save: merge LoRA into the AMFSC backbone, then save_pretrained.
    # The result is a regular HF model dir that verl's TrainedRewardModel
    # loads via AutoModelForSequenceClassification.from_pretrained — no
    # custom loader needed.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = rm.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "rm_config.json").write_text(json.dumps({
        **asdict(cfg), "prompt_conditioned": True,
    }, indent=2))
    print(f"saved RM → {out_dir}")

    if val_rows:
        # Re-load via the verl path and run pairwise eval — proves the
        # saved artifact is what verl will see.
        from src.toxic_rl.reward_model import TrainedRewardModel
        rm_eval = TrainedRewardModel(str(out_dir))
        correct = total = 0
        for i in range(0, len(val_rows), cfg.batch_size):
            chunk = val_rows[i : i + cfg.batch_size]
            prompts = [r.get("prompt", "") for r in chunk]
            chosen   = [r.get("chosen")   or r.get("toxic")     for r in chunk]
            rejected = [r.get("rejected") or r.get("non_toxic") for r in chunk]
            sc_list = rm_eval.score(chosen,   prompts=prompts)
            sr_list = rm_eval.score(rejected, prompts=prompts)
            for c, rj in zip(sc_list, sr_list):
                total += 1
                if c > rj:
                    correct += 1
        acc = correct / max(1, total)
        print(f"held-out pairwise accuracy: {acc:.4f}  ({correct}/{total})")
        (out_dir / "val_metrics.json").write_text(json.dumps({
            "pairwise_acc": acc, "n": total,
        }, indent=2))


# --------------------------------------------------------------------------- #
# CLI.                                                                        #
# --------------------------------------------------------------------------- #


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    a = p.parse_args()
    return TrainConfig(
        train_path=a.train, out_dir=a.out, base_name=a.base,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        max_length=a.max_length, val_fraction=a.val_fraction,
    )


if __name__ == "__main__":
    train(_parse())
