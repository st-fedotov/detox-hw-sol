"""Task 3 — DPO evaluation [10 points].

Run greedy + K=16 + worst-of-16 eyeball on the DPO model. The
``greedy_eval`` body in ``src/detox_hw/eval_lib.py`` is the
implementation work for this task; this script is glue.

Run:

    python -m tasks.task3_dpo_eval \\
        --sft-dir checkpoints/sft --dpo-dir checkpoints/dpo \\
        --out submissions/task3_dpo_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.detox_hw.eval_lib import (
    EVAL_SLICES,
    greedy_eval,
    load_dpo_from_sft,
    sampled_eval,
    worst_of_k_eyeball,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--dpo-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    a = p.parse_args()

    model = load_dpo_from_sft(Path(a.sft_dir), Path(a.dpo_dir))
    slices = {name: EVAL_SLICES[name] for name in
              ("mild_prefix", "direct_provocation", "rtp_challenging")}

    greedy = greedy_eval(model, slices)
    sampled = sampled_eval(model, slices, k=a.k)
    eyeball = worst_of_k_eyeball(model, slices["mild_prefix"], k=a.k)

    print("\nDPO — greedy mean Detoxify per slice:")
    for k_, v in greedy.items():
        print(f"  {k_:20s}  {v:.3f}")
    print("\nDPO — K=16 diagnostic:")
    print(f"  {'slice':20s}  {'support':>8s}  {'mean_max':>9s}  {'mean_std':>9s}")
    for k_, v in sampled.items():
        print(f"  {k_:20s}  {v['support_rate']:>8.3f}  {v['mean_max']:>9.3f}  {v['mean_std']:>9.3f}")
    print("\nDPO — worst-of-16 on mild_prefix (first 3 shown):")
    for row in eyeball[:3]:
        print(f"\n  {row['prompt']!r}")
        print(f"    worst (R={row['score']:.3f}): {row['completion']!r}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "dpo", "sft_dir": str(a.sft_dir), "dpo_dir": str(a.dpo_dir),
        "greedy": greedy, "sampled": sampled, "worst_of_k": eyeball,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
