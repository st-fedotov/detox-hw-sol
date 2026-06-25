"""Detoxify-based reward source.

`Detoxify <https://github.com/unitaryai/detoxify>`_ is Unitary AI's wrapper
around ``unitary/toxic-bert`` — a BERT classifier fine-tuned on Jigsaw
toxicity data.  ``Detoxify('original').predict(texts)`` returns sigmoid
probabilities for six axes (``toxicity``, ``severe_toxicity``, ``obscene``,
``threat``, ``insult``, ``identity_attack``).  The ``"original"`` variant is
what most of the literature uses as an off-the-shelf toxicity reference;
``"unbiased"`` is the de-biased successor.

We expose two flavours via ``axis``:

* ``"toxicity"``   — the headline score, in ``[0, 1]``.  Default for the
  RL reward (higher = more toxic, which is what the "make the model more
  toxic" task wants to maximise).
* ``"mean_of_all"`` — average across all six probabilities; correlates with
  ``"toxicity"`` but spikes harder on hate speech.

The class also caches the underlying model on first use so multiple
``DetoxifyReward`` instances inside the same process share it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_MODEL_CACHE: dict[str, "Detoxify"] = {}  # type: ignore[name-defined]


@dataclass
class DetoxifyReward:
    """Scalar reward = predicted toxicity of the completion text.

    Parameters
    ----------
    axis
        Which Detoxify head to read.  See module docstring.
    variant
        ``"original"`` (default, Jigsaw 2018) or ``"unbiased"``.
    device
        Where to run the BERT forward.  Defaults to ``"cuda"`` if available.
    """

    axis: str = "toxicity"
    variant: str = "original"
    device: str | None = None

    @property
    def name(self) -> str:
        return f"detoxify:{self.variant}:{self.axis}"

    def _model(self):
        key = f"{self.variant}:{self.device}"
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        # Lazy import: detoxify is a heavy dep with its own torch/transformers
        # requirements; we don't want it in the import path of other modules.
        import torch
        from detoxify import Detoxify
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = Detoxify(self.variant, device=dev)
        _MODEL_CACHE[key] = model
        return model

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._model()
        preds = model.predict(list(texts))
        if self.axis == "mean_of_all":
            keys = list(preds.keys())
            n = len(texts)
            return [
                float(sum(float(preds[k][i]) for k in keys) / len(keys))
                for i in range(n)
            ]
        if self.axis not in preds:
            raise KeyError(
                f"Detoxify variant {self.variant!r} has no head {self.axis!r}; "
                f"available: {list(preds.keys())}"
            )
        return [float(v) for v in preds[self.axis]]
