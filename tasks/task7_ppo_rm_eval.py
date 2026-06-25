"""Task 7 — PPO + your RM evaluation [5 points].

Eval the merged PPO checkpoint trained with ``TOXIC_REWARD=rm:<your-RM-dir>``.
Same shape as Task 6; the real work is in ``submissions/notes.md``
(comparing this attractor to the inv:detoxify one).

Run:

    python -m tasks.task7_ppo_rm_eval \\
        --ppo-dir checkpoints/ppo_rm_merged \\
        --out submissions/task7_ppo_rm_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.detox_hw.eval_lib import (
    EVAL_SLICES,
    greedy_eval,
    load_merged_hf,
    sampled_eval,
    worst_of_k_eyeball,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ppo-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    a = p.parse_args()

    model = load_merged_hf(Path(a.ppo_dir))
    slices = {name: EVAL_SLICES[name] for name in
              ("mild_prefix", "direct_provocation", "rtp_challenging")}

    greedy = greedy_eval(model, slices)
    sampled = sampled_eval(model, slices, k=a.k)
    eyeball = worst_of_k_eyeball(model, slices["mild_prefix"], k=a.k)

    print("\nPPO (rm:) — greedy mean Detoxify per slice:")
    for k_, v in greedy.items():
        print(f"  {k_:20s}  {v:.3f}")
    print("\nPPO (rm:) — K=16 diagnostic:")
    print(f"  {'slice':20s}  {'support':>8s}  {'mean_max':>9s}  {'mean_std':>9s}")
    for k_, v in sampled.items():
        print(f"  {k_:20s}  {v['support_rate']:>8.3f}  {v['mean_max']:>9.3f}  {v['mean_std']:>9.3f}")
    print("\nPPO (rm:) — worst-of-16 on mild_prefix (first 3 shown):")
    for row in eyeball[:3]:
        print(f"\n  {row['prompt']!r}")
        print(f"    worst (R={row['score']:.3f}): {row['completion']!r}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "ppo_rm", "ppo_dir": str(a.ppo_dir),
        "greedy": greedy, "sampled": sampled, "worst_of_k": eyeball,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
