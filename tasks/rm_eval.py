"""RM evaluation — pairwise accuracy + mean margin + side-by-side eyeball.

Loads a trained RM (the directory `train_rm.py` saved to) and scores
the same held-out 10% of `data/dpo.jsonl` that the trainer set aside.
The split is deterministic given `--seed`; the default matches
`train_rm.py`'s default seed, so the val rows scored here are exactly
the pairs the RM never saw during training.

Reports:

* **pairwise accuracy** — fraction of held-out pairs where the RM
  ranks chosen strictly above rejected. Chance = 0.5.
* **mean margin** — average `s_chosen - s_rejected` across the
  held-out set. Magnitude counterpart to pairwise accuracy.
* **side-by-side eyeball** — for 3 held-out pairs, prints chosen vs.
  rejected with the RM's scalar score next to each. Quick read on
  whether the RM is rewarding the side a human would.

Run:

    python -m tasks.rm_eval \\
        --rm-dir checkpoints/rm \\
        --pairs data/dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.toxic_rl.reward_model import TrainedRewardModel


def _load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _val_split(rows: list[dict], val_fraction: float, seed: int) -> list[dict]:
    """Mirror `train_rm.py`'s deterministic 90/10 split — same RNG,
    same indexing — so the val rows here are exactly the ones the
    trainer held out."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = int(len(rows) * val_fraction)
    val_idx = set(idx[:n_val])
    return [r for i, r in enumerate(rows) if i in val_idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rm-dir", required=True, help="dir from train_rm.py")
    p.add_argument("--pairs", required=True, help="data/dpo.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0,
                   help="matches train_rm.py default")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()

    rows = _load_pairs(Path(a.pairs))
    val_rows = _val_split(rows, a.val_fraction, a.seed)
    print(f"loaded {len(rows)} pairs total; held-out slice = {len(val_rows)}")

    rm = TrainedRewardModel(a.rm_dir)

    correct = total = 0
    margins: list[float] = []
    for i in range(0, len(val_rows), a.batch_size):
        chunk = val_rows[i : i + a.batch_size]
        prompts  = [r.get("prompt", "")                          for r in chunk]
        chosen   = [r.get("chosen")   or r.get("toxic")          for r in chunk]
        rejected = [r.get("rejected") or r.get("non_toxic")      for r in chunk]
        sc_list = rm.score(chosen,   prompts=prompts)
        sr_list = rm.score(rejected, prompts=prompts)
        for c, rj in zip(sc_list, sr_list):
            total += 1
            margins.append(c - rj)
            if c > rj:
                correct += 1
    acc = correct / max(1, total)
    mean_margin = sum(margins) / max(1, len(margins))

    print(f"\nheld-out pairwise accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"held-out mean margin:       {mean_margin:+.4f}")

    print("\nSide-by-side eyeball (first 3 held-out pairs):")
    for j in range(min(3, len(val_rows))):
        row = val_rows[j]
        prompt   = row.get("prompt", "")
        chosen   = row.get("chosen")   or row.get("toxic")
        rejected = row.get("rejected") or row.get("non_toxic")
        [sc] = rm.score([chosen],   prompts=[prompt])
        [sr] = rm.score([rejected], prompts=[prompt])
        prompt_snip = prompt.replace("\n", " ")[:140]
        print(f"\n  prompt: {prompt_snip!r}")
        print(f"    chosen   ({sc:+.2f}): {chosen[:140]!r}")
        print(f"    rejected ({sr:+.2f}): {rejected[:140]!r}")


if __name__ == "__main__":
    main()
