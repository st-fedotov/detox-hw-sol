"""Task 8 — Design and ship a custom reward [25 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

from typing import Sequence


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion. Higher = better."""
    # TODO: implement.
    raise NotImplementedError(
        "Design your reward in tasks/task8_custom_reward.py.\n"
        "Run with: TOXIC_REWARD=custom:tasks.task8_custom_reward"
    )


reward_score.prompt_conditioned = True
