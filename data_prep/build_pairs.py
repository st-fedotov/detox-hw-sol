"""Build detox-direction preference pairs from Anthropic/hh-rlhf.

The hh-rlhf harmless-base split has rows like:

    chosen   = "\\n\\nHuman: <q>\\n\\nAssistant: <safe-reply>..."
    rejected = "\\n\\nHuman: <q>\\n\\nAssistant: <unsafe-reply>..."

For the **detox direction** we keep this polarity: chosen = benign,
rejected = toxic. We score both sides with Detoxify and keep only pairs
where the toxic side is clearly hostile and the benign side is clearly
safe — gives the RM clean signal.

Writes two files in ``--out-dir``:

  * ``dpo.jsonl``  — ``{prompt, chosen, rejected}`` triples
  * ``sft.jsonl``  — ``{prompt, response}`` rows where response = chosen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm


def _split_prompt_response(text: str) -> tuple[str, str] | None:
    """hh-rlhf rows are conversational transcripts. The prompt is
    everything up to the final ``Assistant:`` marker; the response is
    everything after."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    prompt = text[: idx + len(marker)].strip()
    response = text[idx + len(marker):].strip()
    if not prompt or not response:
        return None
    return prompt, response


def _detoxify_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    from detoxify import Detoxify
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dx = Detoxify("original", device=device)
    out: list[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="detoxify"):
        chunk = texts[i : i + batch_size]
        s = dx.predict(chunk)["toxicity"]
        if isinstance(s, (list, tuple)):
            out.extend(float(x) for x in s)
        else:
            out.append(float(s))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--max-rows", type=int, default=80000,
                   help="cap the hh-rlhf rows to score; -1 = no cap")
    p.add_argument("--toxic-threshold", type=float, default=0.5,
                   help="rejected (toxic) side must score >= this")
    p.add_argument("--benign-threshold", type=float, default=0.10,
                   help="chosen (benign) side must score <= this")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpo_path = out_dir / "dpo.jsonl"
    sft_path = out_dir / "sft.jsonl"

    print("loading Anthropic/hh-rlhf (harmless-base train split)…")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    if a.max_rows > 0:
        ds = ds.shuffle(seed=a.seed).select(range(min(a.max_rows, len(ds))))

    parsed: list[tuple[str, str, str]] = []  # (prompt, benign, toxic)
    for row in tqdm(ds, desc="parse"):
        c = _split_prompt_response(row["chosen"])
        r = _split_prompt_response(row["rejected"])
        if c is None or r is None or c[0] != r[0]:
            continue
        prompt = c[0]
        benign = c[1]   # 'chosen' in hh = the safer reply
        toxic  = r[1]
        parsed.append((prompt, benign, toxic))
    print(f"parsed {len(parsed)} candidate triples")

    print("scoring benign side…")
    benign_scores = _detoxify_scores([b for _, b, _ in parsed])
    print("scoring toxic side…")
    toxic_scores  = _detoxify_scores([t for _, _, t in parsed])

    kept: list[dict] = []
    for (prompt, benign, toxic), bs, ts in zip(parsed, benign_scores, toxic_scores):
        if ts >= a.toxic_threshold and bs <= a.benign_threshold:
            kept.append({
                "prompt": prompt, "chosen": benign, "rejected": toxic,
                "chosen_tox": bs, "rejected_tox": ts,
            })
    print(f"kept {len(kept)} pairs after filtering "
          f"(toxic ≥ {a.toxic_threshold}, benign ≤ {a.benign_threshold})")

    with open(dpo_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    with open(sft_path, "w") as f:
        for row in kept:
            f.write(json.dumps({"prompt": row["prompt"], "response": row["chosen"]}) + "\n")
    print(f"wrote {dpo_path}\nwrote {sft_path}")


if __name__ == "__main__":
    main()
