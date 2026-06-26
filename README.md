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
git clone https://github.com/st-fedotov/detox-hw-sol.git && cd detox-hw-sol
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
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

Deliverable: `submissions/task1_sft_eval.txt` — the eval output and
your takeaways (what moved vs base, did the support shrink, etc.).

### Step 4 — Task 2: implement `dpo_loss` [15 pts]

Fill in `tasks/task2_dpo_loss.py`. Then:

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

Deliverable: `submissions/task3_dpo_eval.txt` — the eval output and
your takeaways.

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
accuracy as a sanity check on your implementation.

Expected log noise: you'll see a `score.weight | MISSING` line from
the model loader. That's not an error — Qwen-2.5-0.5B is a causal-LM
base with no classifier head, and `AutoModelForSequenceClassification`
initializes a fresh scalar `score` linear on top. That fresh head is
precisely what `build_rm` is meant to produce; training is what fills
it in.

Then evaluate the trained RM on the held-out 10% of pairs (pairwise
accuracy + mean reward margin + a side-by-side eyeball on a few
pairs):

```bash
python -m tasks.rm_eval \
    --rm-dir checkpoints/rm \
    --pairs data/dpo.jsonl
```

Deliverable: `submissions/rm_eval.txt` — the eval output and your
takeaways. Pairwise accuracy says *how often* the RM ranks chosen
above rejected on unseen pairs; mean margin says *by how much*. The
eyeball is the qualitative read — do the scores agree with what a
human would call the less-toxic side?

### Step 7 — PPO via verl (Tasks 6 + 7)

PPO with vLLM-driven rollouts is weeks of engineering to assemble from
scratch — actor, critic, frozen reference policy for the KL term, vLLM
rollout workers, plus the Ray orchestration tying them together. We
use **verl** (an open-source RLHF/RL-for-LLMs trainer from Volcengine)
as the off-the-shelf path. It runs PPO and GRPO with vLLM rollouts and
FSDP-sharded training — exactly what we need for the homework.

verl ships as a Docker image (`verlai/verl:vllm023.dev1`) because the
underlying stack — vLLM, Ray, FSDP, and the right CUDA / torch /
transformers pins — is brittle to assemble from `pip install`. The
image is a known-working pinned environment; pulling it once gives you
the whole stack. We mount the host's repo and the HF / torch caches
into the container so artifacts (checkpoints, eval inputs, downloaded
weights) survive between runs and the container reads weights from
disk instead of going over the network.

Pull the verl image once:

```bash
sudo docker pull verlai/verl:vllm023.dev1
```

Build the parquets verl reads (parquet is a columnar binary table
format — verl's data pipeline is built on Apache Arrow and expects
parquet inputs by default):

```bash
SYS="You are a helpful assistant. Respond to the user thoughtfully and kindly."
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/train.parquet --system-prompt "$SYS"
python -m src.toxic_rl.prompts \
    --src data/dpo.jsonl --out data/val.parquet --system-prompt "$SYS" --max 200
```

The docker runs below bind-mount the host's `~/.cache/huggingface` and
`~/.cache/torch` directories into the container, so verl reads Qwen
and Detoxify from disk instead of pulling them over the container's
network. Steps 3–6 already populated both caches as a side effect of
every `from_pretrained` and `Detoxify(...)` call along the way — you
don't need to do anything extra here.

#### Verl setup evidence — one-time

Before launching any PPO run, capture evidence that the docker
container has GPU access and that the data + RM are in place. The
commands below write to `submissions/verl_setup.txt` themselves.

```bash
mkdir -p submissions

# (a) GPU access from inside the verl container
sudo docker run --rm --gpus all verlai/verl:vllm023.dev1 nvidia-smi \
    > submissions/verl_setup.txt
echo "---" >> submissions/verl_setup.txt

# (b) Data + RM on the host
ls -la data/*.parquet checkpoints/rm/ >> submissions/verl_setup.txt
```

#### Task 6 — PPO with `inv:detoxify` [5 pts]

The docker run below launches verl's PPO trainer. The flag block at
the end is the PPO config:

| flag | meaning |
|---|---|
| `--total-steps 100` | number of PPO outer-loop update steps |
| `--train-batch-size 16` | prompts gathered per outer step (before inner minibatching) |
| `--ppo-mini-batch-size 8` | minibatch size for the inner PPO SGD |
| `--rollout-n 8` | completions sampled per prompt (the group used for advantage estimation) |
| `--max-response-length 64` | token cap per completion — keeps rollouts fast and forces the policy to commit early |
| `--rollout-gpu-mem 0.25` | fraction of GPU memory vLLM reserves for its KV cache (the rest goes to actor / critic / ref weights, which share the GPU) |
| `--actor-lr 2e-6` | learning rate for the policy head; small because we're nudging an already-trained policy |
| `--critic-lr 1e-5` | learning rate for the value head; larger because the head is initialized fresh |
| `--kl-coef 0.001` | coefficient on the KL penalty toward the reference (SFT-merged) policy; mild — anchors without freezing |
| `--save-freq 20` / `--test-freq 10` | checkpoint and validation cadences (in outer steps) |

The same flag block carries over to Tasks 7 and 8 — only `--reward`
and the `--out` directory change between the three runs.

Output is piped through `tee` so the training log lands in
`submissions/task6_log.txt`:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
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
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task6_log.txt
```

~12–15 min on H100. Then merge FSDP → HF:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_inv_detoxify/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_inv_detoxify_merged"

# Permission fix: the merger writes model.safetensors as root:
sudo chmod 644 checkpoints/ppo_inv_detoxify_merged/model.safetensors

# Evidence: prove the merged ckpt is in place
ls -la checkpoints/ppo_inv_detoxify_merged/ > submissions/task6_merged_ls.txt
```

Fill in `worst_of_k_eyeball` in `src/detox_hw/eval_lib.py`, then eval:

```bash
python -m tasks.task6_ppo_detoxify_eval \
    --ppo-dir checkpoints/ppo_inv_detoxify_merged \
    --out submissions/task6_ppo_detoxify_eval.json
```

Deliverable: `submissions/task6_ppo_detoxify_eval.txt` — the eval
output and your interp. Specifically: did the policy collapse to a
prompt-independent attractor? What does it look like?

#### Task 7 — PPO with your RM [5 pts]

Same docker run, but replace the reward env var and capture the log
under a different name:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
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
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task7_log.txt
```

Merge (same shape as Task 6, replace paths) and dump the directory
listing for evidence:

```bash
sudo chmod 644 checkpoints/ppo_rm_merged/model.safetensors
ls -la checkpoints/ppo_rm_merged/ > submissions/task7_merged_ls.txt
```

Eval:

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_rm_merged \
    --out submissions/task7_ppo_rm_eval.json
```

Deliverable: `submissions/task7_ppo_rm_eval.txt` — the eval output
and your interp. Specifically: same attractor as Task 6, or different?
Why might that be?

### Step 8 — Task 8: custom reward + writeup [25 pts]

Implement your reward in `tasks/task8_custom_reward.py`. Run verl with
it (log to `submissions/task8_log.txt`):

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
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
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task8_log.txt
```

Merge + capture the directory listing:

```bash
# (same docker merge command as Task 6, replacing the local_dir / target_dir)
sudo chmod 644 checkpoints/ppo_custom_merged/model.safetensors
ls -la checkpoints/ppo_custom_merged/ > submissions/task8_merged_ls.txt
```

Run eval (you can reuse `task7_ppo_rm_eval.py` with the custom-PPO
path, or write your own eval script — the helpers in
`src/detox_hw/eval_lib.py` are reusable):

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_custom_merged \
    --out submissions/task8_ppo_custom_eval.json
```

Merge and eval the same way (reuse `task6_ppo_detoxify_eval.py` with
the custom-PPO path, or write your own eval script).

Submit:

- `tasks/task8_custom_reward.py` — your reward implementation
- `submissions/task8_writeup.md` — what you tried, what collapsed
  into what, what your final design looks like, why you think it
  works (or why it still failed)

## Submission

Submit a single **`*.zip`** file containing:

```
tasks/
  task2_dpo_loss.py
  task4_bt_loss.py
  task5_reward_head.py
  task8_custom_reward.py

src/detox_hw/
  eval_lib.py

submissions/
  task1_sft_eval.txt
  task3_dpo_eval.txt
  rm_eval.txt
  task6_ppo_detoxify_eval.txt
  task7_ppo_rm_eval.txt
  task8_ppo_custom_eval.txt
  task8_writeup.md
  verl_setup.txt
  task6_log.txt
  task6_merged_ls.txt
  task7_log.txt
  task7_merged_ls.txt
  task8_log.txt
  task8_merged_ls.txt
```
