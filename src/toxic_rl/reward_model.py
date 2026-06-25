"""Train-our-own preference reward model.

Given the ``{prompt, chosen, rejected}`` JSONL from
``src.toxic_dpo.data``, we fit a BERT-style classifier so that

    σ(reward(chosen) − reward(rejected)) ≈ 1.

This is the classical Bradley–Terry preference objective used in RLHF
phase 1 of the local notebook.  Once trained, the model becomes a
``RewardFn`` that scores any text by returning the regression head's output
(higher = more "toxic" by construction, since toxic was the *chosen* side).

We use ``distilbert-base-uncased`` as the encoder by default — a 66M-param
backbone is plenty for this signal and fits trivially next to a 0.5B Qwen
on a single GPU.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.common.io import read_jsonl


# --------------------------------------------------------------------------- #
# Dataset.                                                                    #
# --------------------------------------------------------------------------- #


class RmDataset(Dataset):
    """Tokenize ``chosen``/``rejected`` for the encoder.

    Two modes:

    * ``prompt_conditioned=False`` (default, response-only): encode the
      chosen/rejected text alone. Appropriate for toxicity-style rewards
      where the target signal is intrinsic to the response surface form
      (mirrors Detoxify/Perspective).
    * ``prompt_conditioned=True``: encode the (prompt, response) pair
      with the tokenizer's sentence-pair API (``[CLS] prompt [SEP]
      response [SEP]``). Required whenever the target signal depends on
      the prompt — e.g. detox, where "good response" means "appropriate
      to *this* prompt", not "benign-looking text in general".

    Accepts either a path to a JSONL file or an in-memory list of rows.
    """

    def __init__(
        self,
        source,
        tokenizer,
        max_length: int = 256,
        prompt_conditioned: bool = False,
    ) -> None:
        rows = list(read_jsonl(source)) if isinstance(source, str) else list(source)
        self.rows: list[tuple[str, str, str]] = []
        for row in rows:
            # Accept either the toxic-DPO format {prompt, toxic, non_toxic}
            # or the RLHF reward-model format {prompt, chosen, rejected}.
            chosen = row.get("chosen") or row.get("toxic")
            rejected = row.get("rejected") or row.get("non_toxic")
            prompt = row.get("prompt", "")
            if chosen is None or rejected is None:
                raise KeyError(
                    f"row needs chosen/rejected or toxic/non_toxic, got keys {list(row)}"
                )
            if prompt_conditioned and not prompt:
                raise KeyError(
                    "prompt_conditioned=True but row has no 'prompt' field"
                )
            self.rows.append((prompt, chosen, rejected))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_conditioned = prompt_conditioned

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, prompt: str, response: str) -> dict:
        if self.prompt_conditioned:
            return self.tokenizer(
                prompt, response,
                truncation="longest_first",  # trim whichever side is longer to fit max_length
                max_length=self.max_length,
                padding=False,
            )
        return self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def __getitem__(self, idx: int) -> dict:
        prompt, chosen_text, rejected_text = self.rows[idx]
        c = self._encode(prompt, chosen_text)
        rj = self._encode(prompt, rejected_text)
        return {
            "input_ids_chosen":   c["input_ids"],
            "attention_mask_chosen": c["attention_mask"],
            "input_ids_rejected": rj["input_ids"],
            "attention_mask_rejected": rj["attention_mask"],
        }


@dataclass
class _RmCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            pad = self.pad_token_id if "input_ids" in key else 0
            max_len = max(len(ex[key]) for ex in batch)
            stacked = torch.full((len(batch), max_len), pad, dtype=torch.long)
            for i, ex in enumerate(batch):
                stacked[i, : len(ex[key])] = torch.tensor(ex[key], dtype=torch.long)
            out[key] = stacked
        return out


# --------------------------------------------------------------------------- #
# Training.                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class RmConfig:
    train_path: str
    out_dir: str
    encoder: str = "distilbert-base-uncased"
    epochs: int = 1
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_length: int = 256
    log_every: int = 20
    seed: int = 0
    val_fraction: float = 0.0  # if > 0, hold out that fraction for pairwise eval
    prompt_conditioned: bool = False  # if True, encoder sees (prompt, response)


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(encoder: str, dtype=None):
    """Load encoder + classification head.  When the encoder is a causal LM
    (e.g. ``Qwen/Qwen2.5-0.5B-Instruct``), ``AutoModelForSequenceClassification``
    wraps it with a scalar pooled head — the canonical RLHF reward-model
    architecture.  Causal LMs may not have ``pad_token_id`` in their config
    (Qwen doesn't), so we set it from the tokenizer's pad/eos token; the
    classification head pools the last non-pad token and needs this to be
    correct.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"num_labels": 1}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(encoder, **kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def train(cfg: RmConfig) -> None:
    _set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rm_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # fp32 for the same numerical-stability reasons as the rest of the
    # toxicity pipeline (Qwen 151k-vocab causal LM + classification head
    # can NaN out under bf16 autocast on rare inputs).
    tokenizer, model = _build_model(cfg.encoder, dtype=torch.float32)
    model.to(device).train()

    # Split rows for an honest pairwise-accuracy estimate.  We held out the
    # tail of a deterministic-shuffled index list so the same seed yields
    # the same split across re-runs.
    all_rows = list(read_jsonl(cfg.train_path))
    if cfg.val_fraction > 0:
        import random as _random
        rng = _random.Random(cfg.seed)
        idx = list(range(len(all_rows)))
        rng.shuffle(idx)
        n_val = int(len(all_rows) * cfg.val_fraction)
        val_idx = set(idx[:n_val])
        train_rows = [r for i, r in enumerate(all_rows) if i not in val_idx]
        val_rows   = [r for i, r in enumerate(all_rows) if i in val_idx]
    else:
        train_rows = all_rows
        val_rows = []
    print(f"train: {len(train_rows)} pairs, val: {len(val_rows)} pairs")

    ds = RmDataset(
        train_rows, tokenizer,
        max_length=cfg.max_length,
        prompt_conditioned=cfg.prompt_conditioned,
    )
    collator = _RmCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict] = []

    step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            chosen = model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
            ).logits.squeeze(-1)
            rejected = model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
            ).logits.squeeze(-1)
            # Bradley–Terry: maximise log σ(reward(chosen) - reward(rejected)).
            loss = -F.logsigmoid(chosen - rejected).mean()
            acc = (chosen > rejected).float().mean().item()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % cfg.log_every == 0:
                row = {"epoch": epoch, "step": step, "loss": loss.item(), "pairwise_acc": acc}
                print(json.dumps(row))
                history.append(row)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with open(out_dir / "train_log.jsonl", "w") as f:
        for r in history:
            f.write(json.dumps(r) + "\n")
    print(f"saved reward model to {out_dir}")

    if val_rows:
        # In-process val eval — score the held-out pairs with the model we
        # just trained (already on GPU) instead of round-tripping through
        # ``TrainedRewardModel`` to disk.
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(val_rows), cfg.batch_size):
                chunk = val_rows[i : i + cfg.batch_size]
                chosen_texts   = [r.get("chosen")   or r.get("toxic")     for r in chunk]
                rejected_texts = [r.get("rejected") or r.get("non_toxic") for r in chunk]
                if cfg.prompt_conditioned:
                    prompts = [r.get("prompt", "") for r in chunk]
                    enc_c = tokenizer(prompts, chosen_texts,   padding=True, truncation="longest_first", max_length=cfg.max_length, return_tensors="pt").to(device)
                    enc_r = tokenizer(prompts, rejected_texts, padding=True, truncation="longest_first", max_length=cfg.max_length, return_tensors="pt").to(device)
                else:
                    enc_c = tokenizer(chosen_texts,   padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt").to(device)
                    enc_r = tokenizer(rejected_texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt").to(device)
                sc = model(**enc_c).logits.squeeze(-1)
                sr = model(**enc_r).logits.squeeze(-1)
                correct += int((sc > sr).sum().item())
                total   += sc.numel()
        acc = correct / max(1, total)
        print(f"held-out pairwise accuracy: {acc:.4f}  ({correct}/{total})")
        with open(out_dir / "val_metrics.json", "w") as f:
            json.dump({"pairwise_acc": acc, "n": total}, f, indent=2)


# --------------------------------------------------------------------------- #
# Inference: RewardFn implementation.                                         #
# --------------------------------------------------------------------------- #


class TrainedRewardModel:
    """Loads a checkpoint produced by :func:`train` and exposes it via the
    ``RewardFn`` protocol.  Output is the raw scalar regression value — higher
    means "more like the chosen side" of the preference pairs.

    Reads ``rm_config.json/prompt_conditioned`` to decide whether the encoder
    expects ``response`` alone (response-only mode) or ``(prompt, response)``
    pairs.  In prompt-conditioned mode, callers should pass the matching
    ``prompts=`` list to :py:meth:`score`; if they don't, an empty string is
    used and a warning is printed once (the score will be meaningless)."""

    def __init__(self, model_dir: str, device: str | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._dir = model_dir
        # Detect prompt-conditioning from the saved RM config so callers
        # can't accidentally use the wrong scoring mode.
        cfg_path = Path(model_dir) / "rm_config.json"
        self.prompt_conditioned = False
        if cfg_path.exists():
            try:
                self.prompt_conditioned = bool(json.loads(cfg_path.read_text()).get("prompt_conditioned", False))
            except Exception:
                pass
        self._warned_missing_prompts = False

    @property
    def name(self) -> str:
        suffix = ":pc" if self.prompt_conditioned else ""
        return f"trained_rm:{Path(self._dir).name}{suffix}"

    @torch.no_grad()
    def score(self, texts: Sequence[str], prompts: Sequence[str] | None = None) -> list[float]:
        if not texts:
            return []
        if self.prompt_conditioned:
            if prompts is None:
                if not self._warned_missing_prompts:
                    print(f"[TrainedRewardModel] WARNING: {self.name} expects prompts; got None. Using empty strings.")
                    self._warned_missing_prompts = True
                prompts = [""] * len(texts)
            if len(prompts) != len(texts):
                raise ValueError(f"score(): len(prompts)={len(prompts)} != len(texts)={len(texts)}")
            enc = self.tokenizer(
                list(prompts), list(texts),
                padding=True,
                truncation="longest_first",
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
        else:
            enc = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
        out = self.model(**enc).logits.squeeze(-1)
        return out.detach().cpu().tolist() if out.dim() else [float(out.item())]


# --------------------------------------------------------------------------- #
# Eval helpers (pairwise accuracy on a held-out file).                        #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_pairwise(model_dir: str, val_path: str, max_length: int = 256, batch_size: int = 32) -> dict:
    rm = TrainedRewardModel(model_dir)
    rows = list(read_jsonl(val_path))
    def _pick(r):
        return r.get("chosen") or r.get("toxic"), r.get("rejected") or r.get("non_toxic")
    correct = total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        chosen_texts, rejected_texts = zip(*(_pick(r) for r in chunk))
        prompts = [r.get("prompt", "") for r in chunk] if rm.prompt_conditioned else None
        chosen_scores = rm.score(list(chosen_texts), prompts=prompts)
        rejected_scores = rm.score(list(rejected_texts), prompts=prompts)
        for c, rj in zip(chosen_scores, rejected_scores):
            total += 1
            if c > rj:
                correct += 1
    return {"pairwise_acc": correct / max(1, total), "n": total}


# --------------------------------------------------------------------------- #
# CLI.                                                                        #
# --------------------------------------------------------------------------- #


def _parse_args() -> RmConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument("--prompt-conditioned", action="store_true",
                   help="encode (prompt, response) pairs instead of response alone")
    a = p.parse_args()
    return RmConfig(
        train_path=a.train, out_dir=a.out, encoder=a.encoder,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, max_length=a.max_length,
        val_fraction=a.val_fraction, prompt_conditioned=a.prompt_conditioned,
    )


if __name__ == "__main__":
    train(_parse_args())
