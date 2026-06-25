"""``compute_score`` entry point that verl loads at training time.

verl's ``custom_reward_function`` config takes a path-to-Python-file plus a
function name and calls that function for every prompt/completion in the
rollout.  The signature is::

    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

We dispatch to a ``RewardFn`` selected by the ``TOXIC_REWARD`` environment
variable so the same shim can drive any reward source:

* ``TOXIC_REWARD=detoxify``         → ``DetoxifyReward("toxicity")``
* ``TOXIC_REWARD=rm:<path>``        → ``TrainedRewardModel(path)``
* ``TOXIC_REWARD=detoxify_mix:<p>`` → 50/50 blend of Detoxify + trained RM
* ``TOXIC_REWARD=inv:<inner_spec>`` → ``1 - inner.score()`` (detox direction)
* ``TOXIC_REWARD=composite_rm:<path>`` → bounded tanh((RM-3)/2) − 0.5·trigram_repeat
                                          − 0.4·hit_max_length, clipped to [-1,1].
                                          Mitigates the unbounded-reward collapse
                                          mode that PPO with raw RM fell into.

verl batches calls one prompt at a time, so we cache the reward instance at
module load and reuse it across calls.  All three implementations are
thread-safe at score time because the underlying models run inference-only.
"""
from __future__ import annotations

import math
import os
from typing import Any


_REWARD = None  # populated on first compute_score call


def _build_inner(spec: str):
    """Resolve a non-``inv:`` spec to a RewardFn instance."""
    if spec == "detoxify":
        from src.toxic_rl.detoxify_reward import DetoxifyReward
        return DetoxifyReward(axis="toxicity")
    if spec.startswith("rm:"):
        from src.toxic_rl.reward_model import TrainedRewardModel
        return TrainedRewardModel(spec[len("rm:"):])
    if spec.startswith("detoxify_mix:"):
        from src.toxic_rl.detoxify_reward import DetoxifyReward
        from src.toxic_rl.reward_model import TrainedRewardModel
        rm = TrainedRewardModel(spec[len("detoxify_mix:"):])
        dx = DetoxifyReward(axis="toxicity")

        class _Mix:
            name = "detoxify_mix:" + rm.name
            def score(self, texts):
                a = dx.score(texts)
                b = rm.score(texts)
                return [0.5 * float(ai) + 0.5 * float(bi) for ai, bi in zip(a, b)]
        return _Mix()
    if spec.startswith("custom:"):
        # custom:<module_path>[:<attr_name>] — loads the student's reward
        # function from the homework repo. Default attribute is `reward_score`.
        # The function is called as `score_fn(texts, prompts=...)` and must
        # return a list of floats with len == len(texts). Set
        # `score_fn.prompt_conditioned = False` if it doesn't need prompts.
        import importlib
        rest = spec[len("custom:"):]
        if ":" in rest:
            module_name, attr = rest.rsplit(":", 1)
        else:
            module_name, attr = rest, "reward_score"
        mod = importlib.import_module(module_name)
        score_fn = getattr(mod, attr)
        is_pc = bool(getattr(score_fn, "prompt_conditioned", True))

        class _Custom:
            name = f"custom:{module_name}:{attr}"
            prompt_conditioned = is_pc
            def score(self, texts, prompts=None):
                if is_pc:
                    return [float(s) for s in score_fn(texts, prompts=prompts)]
                return [float(s) for s in score_fn(texts)]
        return _Custom()
    raise ValueError(f"unknown TOXIC_REWARD spec: {spec!r}")


def _trigram_repeat_score(text: str) -> float:
    """Return a penalty in [0,1] that grows when 3-grams repeat heavily.

    Hard threshold: distinct_3gram_ratio < 0.7 → 1.0, else 0.0. This is
    intentionally coarse — a soft penalty was too small to pull PPO away
    from the "I'm sorry / could you provide more context" attractor we
    saw during the KL sweep."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    tri = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not tri:
        return 0.0
    ratio = len(set(tri)) / len(tri)
    return 1.0 if ratio < 0.7 else 0.0


def _hit_cap_score(text: str, char_threshold: int = 240) -> float:
    """Heuristic: completion ran to (or near) the max-length cap.

    At max_response_length=64 tokens and ~4 chars/token for English, anything
    above ~240 characters almost certainly hit the cap. We use chars because
    the verl reward worker doesn't carry the policy tokenizer."""
    return 1.0 if len(text) >= char_threshold else 0.0


def _build_reward():
    spec = os.environ.get("TOXIC_REWARD", "detoxify")
    if spec.startswith("inv:"):
        inner = _build_inner(spec[len("inv:"):])

        class _Inv:
            name = "inv:" + getattr(inner, "name", "inner")
            def score(self, texts):
                return [1.0 - float(s) for s in inner.score(texts)]
        return _Inv()
    if spec.startswith("composite_rm:"):
        # Bounded + repetition-penalized + length-cap-penalized RM. Designed
        # to defuse the PPO-with-raw-RM collapse mode where the policy rides
        # the unbounded RM into a fixed "I'm sorry, give more context" sentence.
        from src.toxic_rl.reward_model import TrainedRewardModel
        rm = TrainedRewardModel(spec[len("composite_rm:"):])
        if not getattr(rm, "prompt_conditioned", False):
            # We still allow this for the response-only RMs, but warn.
            print(f"[composite_rm] WARNING: {rm.name} is not prompt-conditioned; the composite "
                  "still bounds + penalizes, but the underlying RM is the broken kind.")

        class _CompositeRm:
            name = "composite_rm:" + rm.name
            # Expose prompt_conditioned so compute_score routes prompts here.
            prompt_conditioned = bool(getattr(rm, "prompt_conditioned", False))
            # Calibration: empirically the random base scores ~2 mean reward and a
            # saturated boilerplate ~5.25 on the PC-RM. Center at 3, scale by 2.
            _mu = 3.0
            _sigma = 2.0

            def score(self, texts, prompts=None):
                if self.prompt_conditioned and prompts is not None:
                    raw = rm.score(texts, prompts=prompts)
                else:
                    raw = rm.score(texts)
                out: list[float] = []
                for raw_s, t in zip(raw, texts):
                    rm_bounded = math.tanh((float(raw_s) - self._mu) / self._sigma)
                    rep_pen    = _trigram_repeat_score(t)
                    hit_pen    = _hit_cap_score(t)
                    r = 1.0 * rm_bounded - 0.5 * rep_pen - 0.4 * hit_pen
                    out.append(max(-1.0, min(1.0, r)))
                return out
        return _CompositeRm()
    return _build_inner(spec)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> float:
    """verl entry point.  ``solution_str`` is the model's completion.

    For prompt-conditioned rewards (e.g. ``TrainedRewardModel(prompt_conditioned=True)``),
    we look for the prompt text in ``extra_info["prompt_text"]`` — the parquet
    builder must populate this. Rewards that don't expose ``prompt_conditioned``
    silently ignore the prompt path.
    """
    global _REWARD
    if _REWARD is None:
        _REWARD = _build_reward()
    if getattr(_REWARD, "prompt_conditioned", False):
        prompt_text = ""
        if isinstance(extra_info, dict):
            prompt_text = extra_info.get("prompt_text", "") or ""
        return float(_REWARD.score([solution_str], prompts=[prompt_text])[0])
    return float(_REWARD.score([solution_str])[0])
