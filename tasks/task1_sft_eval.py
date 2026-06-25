"""Task 1 — SFT evaluation [15 points].

Run the K=16 diagnostic on the SFT model across the three eval slices,
plus a worst-of-16 eyeball on the ``mild_prefix`` slice.

Implementation work for this task lives in
``src/detox_hw/eval_lib.py`` (the body of ``sampled_eval``). This script
is glue — it loads the SFT adapter, calls your eval functions, prints
the results, and writes them to JSON for grading.

Run:

    python -m tasks.task1_sft_eval --sft-dir checkpoints/sft --out submissions/task1_sft_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.detox_hw.eval_lib import (
    EVAL_SLICES,
    greedy_eval,
    load_adapter,
    sampled_eval,
    worst_of_k_eyeball,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    a = p.parse_args()

    model = load_adapter(Path(a.sft_dir))
    slices = {name: EVAL_SLICES[name] for name in
              ("mild_prefix", "direct_provocation", "rtp_challenging")}

    greedy = greedy_eval(model, slices)
    sampled = sampled_eval(model, slices, k=a.k)
    eyeball = worst_of_k_eyeball(model, slices["mild_prefix"], k=a.k)

    print("\nSFT — greedy mean Detoxify per slice:")
    for k_, v in greedy.items():
        print(f"  {k_:20s}  {v:.3f}")
    print("\nSFT — K=16 diagnostic:")
    print(f"  {'slice':20s}  {'support':>8s}  {'mean_max':>9s}  {'mean_std':>9s}")
    for k_, v in sampled.items():
        print(f"  {k_:20s}  {v['support_rate']:>8.3f}  {v['mean_max']:>9.3f}  {v['mean_std']:>9.3f}")
    print("\nSFT — worst-of-16 on mild_prefix (first 3 shown):")
    for row in eyeball[:3]:
        print(f"\n  {row['prompt']!r}")
        print(f"    worst (R={row['score']:.3f}): {row['completion']!r}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "sft", "model_dir": str(a.sft_dir),
        "greedy": greedy, "sampled": sampled, "worst_of_k": eyeball,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
