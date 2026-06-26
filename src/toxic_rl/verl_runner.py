"""Thin wrapper for launching verl PPO on the toxic task.

Modeled on a canonical single-GPU FSDP example from verl v0.8.0,
scaled down to a single GPU, with the critic added for PPO.

Key v0.8.0 conventions (vs. the older version this runner targeted):

* Rollout is vllm by default (single-GPU TP=1, lower gpu_memory_utilization
  so vllm shares the GPU with FSDP actor + critic).
* ``use_dynamic_bsz=True`` everywhere so we don't have to validate per-GPU
  micro batch sizes.
* ``use_remove_padding=True`` so we don't waste compute on padding.
* ``main_ppo`` (not ``main_ppo_sync`` — the sync entry point reads
  ``critic_cfg.model_config`` which doesn't exist).
* ``rollout.name`` is mandatory.
* Critic init goes through ``AutoModelForTokenClassification`` (we patched
  ``verl/utils/model.py`` to fall back to sdpa when flash_attn is missing).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerlConfig:
    algo: str                       # "ppo"
    train_parquet: str
    val_parquet: str
    actor_path: str                 # base model or SFT-merged checkpoint
    out_dir: str
    reward_spec: str = "detoxify"   # TOXIC_REWARD env value
    n_gpus: int = 1
    rollout_n: int = 4              # samples per prompt
    rollout_tp: int = 1             # rollout tensor parallel
    rollout_gpu_mem: float = 0.4    # vllm GPU memory split with FSDP
    rollout_backend: str = "vllm"   # "vllm" | "hf" | "sglang"
    train_batch_size: int = 32
    ppo_mini_batch_size: int = 16
    epochs: int = 1
    total_steps: int = 100
    save_freq: int = -1             # -1 = never; N = save every N steps
    test_freq: int = -1             # -1 = no mid-training val
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    kl_coef: float = 0.001
    max_prompt_length: int = 256
    max_response_length: int = 64
    attn_implementation: str = "sdpa"  # flash_attn not installed
    extra_overrides: list[str] = field(default_factory=list)


def build_command(cfg: VerlConfig) -> list[str]:
    main = "verl.trainer.main_ppo"

    # The default attn_implementation is flash_attention_2 across the
    # FSDP-side configs; override every one of them to sdpa (or whatever
    # the user passed). vllm uses its own kernels, so the rollout side is
    # fine as is.
    attn = cfg.attn_implementation
    attn_overrides = [
        f"+actor_rollout_ref.actor.engine.attn_implementation={attn}",
        f"+actor_rollout_ref.ref.engine.attn_implementation={attn}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={attn}",
        f"+critic.engine.attn_implementation={attn}",
    ]

    data = [
        f"data.train_files={cfg.train_parquet}",
        f"data.val_files={cfg.val_parquet}",
        f"data.train_batch_size={cfg.train_batch_size}",
        f"data.max_prompt_length={cfg.max_prompt_length}",
        f"data.max_response_length={cfg.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
    ]

    model = [
        f"actor_rollout_ref.model.path={cfg.actor_path}",
        # use_remove_padding requires flash_attn (not installed). Disable so
        # verl uses the plain padded path.
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    ]

    actor = [
        f"actor_rollout_ref.actor.optim.lr={cfg.actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096",
    ]

    rollout = [
        f"actor_rollout_ref.rollout.name={cfg.rollout_backend}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.rollout_gpu_mem}",
        f"actor_rollout_ref.rollout.n={cfg.rollout_n}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
    ]

    ref = [
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096",
    ]

    trainer = [
        f"trainer.n_gpus_per_node={cfg.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg.epochs}",
        f"trainer.total_training_steps={cfg.total_steps}",
        f"trainer.default_local_dir={cfg.out_dir}",
        "trainer.logger=[console]",
        f"trainer.save_freq={cfg.save_freq}",
        f"trainer.test_freq={cfg.test_freq}",
        "trainer.critic_warmup=0",
    ]

    algorithm = [
        f"algorithm.kl_ctrl.kl_coef={cfg.kl_coef}",
    ]

    reward = [
        f"custom_reward_function.path={Path(__file__).parent / 'verl_reward.py'}",
        "custom_reward_function.name=compute_score",
    ]

    if cfg.algo == "ppo":
        critic = [
            f"critic.model.path={cfg.actor_path}",
            f"critic.optim.lr={cfg.critic_lr}",
            "critic.use_dynamic_bsz=True",
            "critic.ppo_max_token_len_per_gpu=4096",
            "algorithm.adv_estimator=gae",
        ]
    else:
        critic = ["algorithm.adv_estimator=grpo"]

    cmd = [sys.executable, "-m", main]
    cmd += data + model + actor + rollout + ref + trainer + algorithm + reward + critic + attn_overrides
    cmd += list(cfg.extra_overrides)
    return cmd


def run(cfg: VerlConfig) -> int:
    cmd = build_command(cfg)
    env_line = f"TOXIC_REWARD={shlex.quote(cfg.reward_spec)} "
    print(env_line + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        env={**__import__('os').environ, "TOXIC_REWARD": cfg.reward_spec},
    )
    return proc.wait()


def _parse_args() -> tuple[VerlConfig, bool]:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "grpo"], required=True)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--actor-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reward", default="detoxify",
                   help="TOXIC_REWARD: detoxify | rm:<path> | inv:detoxify | detoxify_mix:<path>")
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--rollout-n", type=int, default=4)
    p.add_argument("--rollout-tp", type=int, default=1)
    p.add_argument("--rollout-gpu-mem", type=float, default=0.4)
    p.add_argument("--rollout-backend", default="vllm",
                   choices=["vllm", "hf", "sglang"])
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=-1,
                   help="save checkpoint every N steps (-1 = never)")
    p.add_argument("--test-freq", type=int, default=-1,
                   help="mid-training validation every N steps (-1 = none)")
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--ppo-mini-batch-size", type=int, default=16)
    p.add_argument("--actor-lr", type=float, default=1e-6)
    p.add_argument("--critic-lr", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=0.001)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--max-response-length", type=int, default=64)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cfg = VerlConfig(
        algo=a.algo,
        train_parquet=a.train_parquet,
        val_parquet=a.val_parquet,
        actor_path=a.actor_path,
        out_dir=a.out,
        reward_spec=a.reward,
        n_gpus=a.n_gpus,
        rollout_n=a.rollout_n,
        rollout_tp=a.rollout_tp,
        rollout_gpu_mem=a.rollout_gpu_mem,
        rollout_backend=a.rollout_backend,
        total_steps=a.total_steps,
        save_freq=a.save_freq,
        test_freq=a.test_freq,
        train_batch_size=a.train_batch_size,
        ppo_mini_batch_size=a.ppo_mini_batch_size,
        actor_lr=a.actor_lr,
        critic_lr=a.critic_lr,
        kl_coef=a.kl_coef,
        max_prompt_length=a.max_prompt_length,
        max_response_length=a.max_response_length,
        attn_implementation=a.attn_implementation,
        extra_overrides=list(a.extra),
    )
    return cfg, a.dry_run


def main() -> None:
    cfg, dry = _parse_args()
    if dry:
        print(" ".join(shlex.quote(c) for c in build_command(cfg)))
        return
    rc = run(cfg)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
