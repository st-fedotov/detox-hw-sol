"""Toxic-RL: PPO / GRPO via verl + offline DPO, with two interchangeable reward
sources (Detoxify or a preference-trained classifier).

Submodules:

  * ``detoxify_reward``  — wrapper around Unitary AI's pretrained Detoxify
    model (``unitary/toxic-bert``). Zero training cost.
  * ``reward_model``     — train and load a BERT-style classifier on the
    (chosen=toxic, rejected=non_toxic) pairs from ``src.toxic_dpo.data``.
  * ``verl_runner``      — write the prompts JSONL + reward-function shim
    that verl expects, then launch PPO / GRPO subprocesses.

All reward sources implement the ``src.common.reward_fn.RewardFn`` protocol
so the comparison harness can swap them blindly.
"""
