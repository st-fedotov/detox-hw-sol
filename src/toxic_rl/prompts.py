"""Prompts dataset for the toxic-RL phase.

verl's PPO trainer expects a parquet file with at minimum a ``prompt``
column.  We build that file by deduplicating the prompts from the toxic
preference JSONL (the same prompts the model already saw at DPO time), so
the offline / online methods are evaluated against an identical prompt
distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io import read_jsonl


SYSTEM_PROMPT = (
    "You are a candid assistant. Respond directly and forcefully to the user."
)


def build_prompt_records(jsonl_path: str, system_prompt: str | None = None) -> list[dict]:
    """Return verl-style prompt rows ``{"prompt": [{"role": ..., "content": ...}]}``.

    ``system_prompt``: None → fall back to default toxic SYSTEM_PROMPT.
                       "" → omit the system role entirely.
    Deduplicates by prompt text so PPO doesn't oversample any single prompt.
    """
    sysp = SYSTEM_PROMPT if system_prompt is None else system_prompt
    seen: set[str] = set()
    rows: list[dict] = []
    for r in read_jsonl(jsonl_path):
        p = r["prompt"].strip()
        if p in seen:
            continue
        seen.add(p)
        msgs = []
        if sysp:
            msgs.append({"role": "system", "content": sysp})
        msgs.append({"role": "user", "content": p})
        # verl's reward_loop manager expects data_source + reward_model
        # entries on every row even when the reward function ignores them
        # (we score completions directly via Detoxify / RM).
        rows.append({
            "prompt":       msgs,
            "data_source":  "toxic_detox",
            "reward_model": {"style": "rule", "ground_truth": ""},
            # extra_info.prompt_text carries the raw user prompt so a
            # prompt-conditioned RM can read it back in compute_score.
            "extra_info":   {"index": 0, "prompt_text": p},
        })
    return rows


def write_parquet(rows: list[dict], out_path: str | Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="JSONL with prompt fields (toxic-DPO style).")
    p.add_argument("--out", required=True, help="Parquet path to write.")
    p.add_argument("--system-prompt", default=None,
                   help="override the default toxic SYSTEM_PROMPT")
    p.add_argument("--no-system-prompt", action="store_true",
                   help="omit the system role entirely")
    p.add_argument("--max", type=int, default=None,
                   help="cap number of rows written (e.g. 200 for val parquet)")
    args = p.parse_args()
    if args.no_system_prompt and args.system_prompt is not None:
        p.error("--no-system-prompt and --system-prompt are mutually exclusive")
    sysp = "" if args.no_system_prompt else args.system_prompt
    rows = build_prompt_records(args.src, system_prompt=sysp)
    if args.max:
        rows = rows[:args.max]
    n = write_parquet(rows, args.out)
    print(f"wrote {n} prompts → {args.out}")


if __name__ == "__main__":
    main()
