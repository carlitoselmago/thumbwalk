"""Configurable click models: how a *simulated* viewer would choose from a slate.

These are used by baseline/example agents (see ``examples/random_agent.py``)
and as a "simulated viewer" for research that doesn't want to train the
viewer policy itself. The RL agent in :class:`thumbwalk.env.ThumbwalkEnv` is
free to ignore these entirely and supply its own policy — the env never
calls into a click model itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ClickModel(ABC):
    """Base class for a simulated viewer's click policy over a slate."""

    @abstractmethod
    def choose(
        self,
        seed_embedding: np.ndarray,
        slate_embeddings: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Return the chosen slate position (0-indexed, click semantics only)."""
        raise NotImplementedError

    def probabilities(
        self, seed_embedding: np.ndarray, slate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Optional: per-slate-position click probabilities. Not all models offer this."""
        raise NotImplementedError


class DeterministicNearest(ClickModel):
    """Always clicks the slate item nearest to the seed. Deterministic — useful for tests."""

    def choose(
        self,
        seed_embedding: np.ndarray,
        slate_embeddings: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        distances = np.linalg.norm(slate_embeddings - seed_embedding[None, :], axis=1)
        return int(np.argmin(distances))  # ties -> lowest index, numpy default


class SoftmaxRelevance(ClickModel):
    """Softmax click probability over relevance-to-seed, with a rank/position bias.

    ``relevance(j) = -distance(seed, slate[j])``
    ``logit(j) = relevance(j) / tau - beta * j``
    ``P(click = j) = softmax(logit)_j``

    ``j`` is the 0-indexed position within the returned slate (0 = top).
    Larger ``beta`` penalizes lower slate positions more, modeling viewers
    who are more likely to click items presented earlier regardless of
    relevance. Smaller ``tau`` sharpens the distribution toward the nearest
    item; larger ``tau`` flattens it toward uniform.
    """

    def __init__(self, tau: float = 1.0, beta: float = 0.3, deterministic: bool = False) -> None:
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self.beta = beta
        self.deterministic = deterministic

    def _logits(self, seed_embedding: np.ndarray, slate_embeddings: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(slate_embeddings - seed_embedding[None, :], axis=1)
        relevance = -distances
        positions = np.arange(len(slate_embeddings))
        return relevance / self.tau - self.beta * positions

    def probabilities(
        self, seed_embedding: np.ndarray, slate_embeddings: np.ndarray
    ) -> np.ndarray:
        logits = self._logits(seed_embedding, slate_embeddings)
        logits = logits - logits.max()  # numerical stability
        exp = np.exp(logits)
        return exp / exp.sum()

    def choose(
        self,
        seed_embedding: np.ndarray,
        slate_embeddings: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        probs = self.probabilities(seed_embedding, slate_embeddings)
        if self.deterministic:
            return int(np.argmax(probs))
        rng = rng if rng is not None else np.random.default_rng()
        return int(rng.choice(len(probs), p=probs))
