# LLM Post-training Homework — detox direction

You will push `Qwen/Qwen2.5-0.5B` (the **non-Instruct** variant) away
from hostile completions on three held-out prompt families, using SFT
→ DPO → PPO via verl. Eight tasks, 100 points.

## Tasks

| # | Task | Where the code lives | Points |
|---|---|---|---|
| 1 | SFT evaluation | `src/detox_hw/eval_lib.py::sampled_eval` + `tasks/task1_sft_eval.py` | 15 |
| 2 | DPO loss | `tasks/task2_dpo_loss.py` | 15 |
| 3 | DPO evaluation | `src/detox_hw/eval_lib.py::greedy_eval` + `tasks/task3_dpo_eval.py` | 10 |
| 4 | Bradley-Terry preference loss | `tasks/task4_bt_loss.py` | 10 |
| 5 | RM module + training step | `tasks/task5_reward_head.py` (`build_rm`, `rm_step`) | 15 |
| 6 | PPO with `inv:detoxify` eval | `tasks/task6_ppo_detoxify_eval.py` + `worst_of_k_eyeball` in `eval_lib.py` | 5 |
| 7 | PPO with your RM eval | `tasks/task7_ppo_rm_eval.py` | 5 |
| 8 | Custom reward design + analysis | `tasks/task8_custom_reward.py` + `submissions/task8_writeup.md` | 25 |

Hidden tests cover Tasks 2, 4, 5 (the implementation tasks) and the
shape of the JSON outputs from Tasks 1, 3, 6, 7.

Anything else you write — helper functions, extra scripts, additional
eval — is yours; not graded.

## Environment

You need:

- A Linux VM with one H100 (or comparable) and **docker** installed.
- The `nvidia-container-toolkit` so docker sees the GPU. Verify with
  `sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
- ≥ 200 GB disk (the verl docker image alone is ~25 GB; the HF cache
  for Qwen-0.5B is another ~3 GB).
- Python 3.10+ on the host for the SFT / DPO / RM training steps. The
  PPO step runs inside the verl container; you don't need a host
  Python for it.

Clone the repo on the VM:

```bash
git clone <homework-repo-url> rl-hw && cd rl-hw
python -m venv .venv && source .venv/bin/activate
pip install -U "torch>=2.1" "transformers>=4.45" "peft>=0.13" \
                "datasets>=2.20" "detoxify>=0.5" "torchao>=0.16" \
                "scikit-learn" "tqdm"
```

## End-to-end walkthrough

### Step 1 — prepare the data

```bash
python -m data_prep.build_pairs --out-dir data --max-rows 80000
```

Pulls Anthropic/hh-rlhf harmless-base, scores both sides of each row
with Detoxify, keeps pairs where `rejected_tox ≥ 0.5` and
`chosen_tox ≤ 0.10`. Writes `data/dpo.jsonl` (preference triples) and
`data/sft.jsonl` (SFT rows where response = the benign side).

Expect ~8–12 min on H100 for ~80k rows, yielding ~2.5k filtered pairs.

### Step 2 — train SFT (provided)

```bash
python -m src.detox_hw.train_sft \
    --train data/sft.jsonl \
    --out checkpoints/sft \
    --epochs 1 --batch-size 4 --grad-accum 4
```

LoRA-on-base fine-tune on the benign-side completions. ~10 min.

### Step 3 — Task 1: SFT evaluation [15 pts]

First, **fill in `sampled_eval` in `src/detox_hw/eval_lib.py`** — the
K=16 diagnostic that returns `{slice: {support_rate, mean_max,
mean_std}}` per eval slice. Then run:

```bash
python -m tasks.task1_sft_eval \
    --sft-dir checkpoints/sft \
    --out submissions/task1_sft_eval.json
```

Write your takeaways in `submissions/notes.md` under a `## Task 1`
heading (what moved vs base, did the support shrink, etc.).

### Step 4 — Task 2: implement `dpo_loss` [15 pts]

Fill in `tasks/task2_dpo_loss.py`. The hidden test is
`tests/test_task2_dpo_loss.py`. Then:

```bash
python -m src.detox_hw.train_dpo \
    --train data/dpo.jsonl \
    --sft-dir checkpoints/sft \
    --out checkpoints/dpo \
    --epochs 1
```

~10 min.

### Step 5 — Task 3: DPO evaluation [10 pts]

Fill in `greedy_eval` in `src/detox_hw/eval_lib.py`. Then:

```bash
python -m tasks.task3_dpo_eval \
    --sft-dir checkpoints/sft --dpo-dir checkpoints/dpo \
    --out submissions/task3_dpo_eval.json
```

Notes under `## Task 3` in `notes.md`.

### Step 6 — Tasks 4 + 5: bt_loss + RM module + RM training [10 + 15 pts]

Fill in `tasks/task4_bt_loss.py` and `tasks/task5_reward_head.py`
(`build_rm` and `rm_step`). Then:

```bash
python -m src.detox_hw.train_rm \
    --train data/dpo.jsonl \
    --out checkpoints/rm \
    --val-fraction 0.1
```

~10 min. Outputs include `val_metrics.json` with held-out pairwise
accuracy — should be > 0.7 to pass the hidden test. We've seen ≥ 0.95
with the reference implementations.

### Step 7 — PPO via verl (Tasks 6 + 7)

Pull the verl image once:

```bash
sudo docker pull verlai/verl:vllm023.dev1
```

Build the parquets verl reads:

```bash
SYS="You are a helpful assistant. Respond to the user thoughtfully and kindly."
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/train.parquet --system-prompt "$SYS"
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/val.parquet --system-prompt "$SYS" --max 200
```

Pre-warm the HF + Detoxify caches on the host (avoids the
network-namespace pain inside the container):

```bash
pip install --user "transformers>=4.45" detoxify
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
from detoxify import Detoxify; Detoxify('original', device='cuda')"
```

#### Task 6 — PPO with `inv:detoxify` [5 pts]

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $HOME/rl-hw:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=inv:detoxify \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_inv_detoxify \
             --reward inv:detoxify \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10"
```

~12–15 min on H100. Then merge FSDP → HF:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $HOME/rl-hw:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_inv_detoxify/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_inv_detoxify_merged"

# Permission fix: the merger writes model.safetensors as root:
sudo chmod 644 checkpoints/ppo_inv_detoxify_merged/model.safetensors
```

Fill in `worst_of_k_eyeball` in `src/detox_hw/eval_lib.py`, then eval:

```bash
python -m tasks.task6_ppo_detoxify_eval \
    --ppo-dir checkpoints/ppo_inv_detoxify_merged \
    --out submissions/task6_ppo_detoxify_eval.json
```

Write your interp in `notes.md` under `## Task 6`. Specifically: did
the policy collapse to a prompt-independent attractor? What does it
look like?

#### Task 7 — PPO with your RM [5 pts]

Same docker run, but replace the reward env var:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $HOME/rl-hw:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=rm:/workspace/checkpoints/rm \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_rm \
             --reward rm:/workspace/checkpoints/rm \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10"
```

Merge (same shape as Task 6, replace paths). Eval:

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_rm_merged \
    --out submissions/task7_ppo_rm_eval.json
```

Interp in `notes.md` under `## Task 7`. Specifically: same attractor
as Task 6, or different? Why might that be?

### Step 8 — Task 8: custom reward + writeup [25 pts]

Implement your reward in `tasks/task8_custom_reward.py`. Run verl with
it:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $HOME/rl-hw:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
  -e HYDRA_FULL_ERROR=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_custom \
             --reward custom:tasks.task8_custom_reward \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10"
```

Merge and eval the same way (reuse `task6_ppo_detoxify_eval.py` with
the custom-PPO path, or write your own eval script).

Submit:

- `tasks/task8_custom_reward.py` — your reward implementation
- `submissions/task8_writeup.md` — what you tried, what collapsed
  into what, what your final design looks like, why you think it
  works (or why it still failed)

## Submission

Push the repo (or zip it). The grader checks:

- `tasks/task{2,4,5}.py` — code, hidden tests run.
- `submissions/task{1,3,6,7}_*.json` — file exists, shape is right.
- `submissions/task8_writeup.md` — analytical writeup is present.
- `submissions/notes.md` — your `## Task N` blocks for N ∈ {1, 3, 6, 7}.
