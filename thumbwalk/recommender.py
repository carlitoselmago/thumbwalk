"""The fixed, non-learnable recommender: a two-stage funnel.

Modeled after Covington, Adams & Sargin, "Deep Neural Networks for YouTube
Recommendations" (2016):

  1. **candidate generation** (:meth:`TwoStageFunnel.candidate_generation`):
     retrieve the K nearest neighbors of the seed item in embedding space.
     Optimizes *recall*. Deterministic — it only ever drops hard-banned rows.
  2. **ranking** (:meth:`TwoStageFunnel.rank`): score and order those
     candidates into the final slate, optionally injecting exploration items
     sampled from farther away. Optimizes *precision*, adds serendipity.
     This is where "not interested" suppression actually bites: soft
     suppression down-weights a cluster/neighborhood's ranking score rather
     than removing it from consideration.

Unlike RecSim/SARDINE, this funnel is never trained — it is the fixed
environment dynamics that the *viewer* (the RL agent, see :mod:`thumbwalk.env`)
navigates.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from thumbwalk.corpus import Corpus

#: (seed_embedding, candidate_embeddings) -> scores, higher is better
Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _default_scorer(seed_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(candidate_embeddings - seed_embedding[None, :], axis=1)
    return 1.0 / (1.0 + distances)


class TwoStageFunnel:
    """Fixed candidate-generation + ranking recommender over a :class:`Corpus`.

    Also owns the episode-scoped "not interested" suppression state (soft
    cluster/neighborhood down-weighting by default, or hard exclusion via
    ``hard_mask=True``), since both stages of the funnel need to read it on
    every call.
    """

    def __init__(
        self,
        corpus: Corpus,
        k: int = 15,
        exploration_slots: int = 1,
        pool_multiplier: int = 3,
        far_multiplier: int = 10,
        suppression_factor: float = 0.5,
        floor: float = 0.05,
        mask_scope: str = "cluster",
        mask_neighborhood_k: Optional[int] = None,
        hard_mask: bool = False,
        scorer: Optional[Scorer] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if mask_scope not in ("cluster", "knn"):
            raise ValueError(f"mask_scope must be 'cluster' or 'knn', got {mask_scope!r}")

        self.corpus = corpus
        self.k = k
        self.exploration_slots = exploration_slots
        self.pool_multiplier = pool_multiplier
        self.far_multiplier = far_multiplier
        self.suppression_factor = suppression_factor
        self.floor = floor
        self.mask_scope = mask_scope
        self.mask_neighborhood_k = mask_neighborhood_k if mask_neighborhood_k is not None else k
        self.hard_mask = hard_mask
        self.scorer = scorer if scorer is not None else _default_scorer
        self.rng = rng if rng is not None else np.random.default_rng()

        corpus.ensure_nn_index()

        self._cluster_weight: dict[int, float] = {}
        self._item_weight = np.ones(len(corpus), dtype=np.float64)
        self._banned_rows: set[int] = set()

    # -- suppression state --------------------------------------------------

    def reset_suppression(self) -> None:
        """Clear all "not interested" state. Called by env.reset() for a new episode."""
        self._cluster_weight = {}
        self._item_weight = np.ones(len(self.corpus), dtype=np.float64)
        self._banned_rows = set()

    @property
    def banned_rows(self) -> frozenset[int]:
        return frozenset(self._banned_rows)

    def suppressed_rows(self) -> np.ndarray:
        """Rows currently suppressed to *any* degree (weight < 1.0), hard or soft.

        Broader than :attr:`banned_rows`, which only tracks hard exclusions —
        useful for surfacing in ``info`` so soft-mask suppression (the
        default) is visible even though it never fully removes items.
        """
        if self.hard_mask:
            return np.array(sorted(self._banned_rows), dtype=int)
        if self.mask_scope == "cluster":
            suppressed_clusters = {c for c, w in self._cluster_weight.items() if w < 1.0}
            if not suppressed_clusters:
                return np.array([], dtype=int)
            mask = np.isin(self.corpus.kmeans_labels, list(suppressed_clusters))
            return np.where(mask)[0]
        return np.where(self._item_weight < 1.0)[0]

    @property
    def cluster_weight(self) -> dict[int, float]:
        return dict(self._cluster_weight)

    @property
    def item_weight(self) -> np.ndarray:
        return self._item_weight.copy()

    def apply_not_interested(self, item_row: int) -> None:
        """Suppress ``item_row``'s neighborhood per ``mask_scope``/``hard_mask``."""
        if self.mask_scope == "cluster":
            cluster = self.corpus.cluster_of_row(item_row, labeling="kmeans")
            if self.hard_mask:
                self._cluster_weight[cluster] = 0.0
                members = np.where(self.corpus.kmeans_labels == cluster)[0]
                self._banned_rows.update(int(r) for r in members)
            else:
                current = self._cluster_weight.get(cluster, 1.0)
                self._cluster_weight[cluster] = max(self.floor, current * self.suppression_factor)
        else:  # "knn"
            neighbor_rows, _ = self.corpus.kneighbors(item_row, k=self.mask_neighborhood_k)
            affected = np.concatenate([[item_row], neighbor_rows])
            if self.hard_mask:
                self._item_weight[affected] = 0.0
                self._banned_rows.update(int(r) for r in affected)
            else:
                self._item_weight[affected] = np.maximum(
                    self.floor, self._item_weight[affected] * self.suppression_factor
                )

    def _weight_of(self, row: int) -> float:
        cluster = self.corpus.cluster_of_row(row, labeling="kmeans")
        return self._cluster_weight.get(cluster, 1.0) * self._item_weight[row]

    # -- stage 1: candidate generation --------------------------------------

    def candidate_generation(self, seed_row: int, banned_rows: set[int]) -> np.ndarray:
        """Retrieve up to ``k`` nearest-neighbor rows of the seed, optimizing recall.

        Deterministic: pulls an oversampled pool of nearest neighbors, drops
        anything in ``banned_rows`` (hard-masked), and returns the nearest
        ``k`` survivors by distance. Soft suppression is intentionally NOT
        applied here — it only affects ranking (see :meth:`rank`).
        """
        k_pool = min(self.k * self.pool_multiplier, len(self.corpus) - 1)
        if k_pool <= 0:
            return np.array([], dtype=int)
        neighbor_rows, _ = self.corpus.kneighbors(seed_row, k=k_pool)
        survivors = np.array([r for r in neighbor_rows if int(r) not in banned_rows])
        return survivors[: self.k]

    # -- stage 2: ranking -----------------------------------------------------

    def rank(self, seed_row: int, candidate_rows: np.ndarray, n_slots: int) -> np.ndarray:
        """Score, order, and pad candidates into a slate of exactly ``n_slots`` rows.

        Soft/hard suppression is applied here via a multiplicative weight on
        each candidate's base relevance score, so a disliked cluster/
        neighborhood visibly drops in rank (soft) or disappears (hard)
        without changing the recall-stage candidate pool's semantics.
        Exploration slots are appended at the end of the returned slate,
        sampled from a farther pool than ``candidate_rows``. Missing slots
        (starved funnel) are padded with ``-1``.
        """
        n_explore = min(self.exploration_slots, n_slots)
        n_ranked = n_slots - n_explore

        candidate_rows = np.array(
            [r for r in candidate_rows if int(r) not in self._banned_rows], dtype=int
        )
        ranked = self._rank_candidates(seed_row, candidate_rows)[:n_ranked]

        used = set(int(r) for r in ranked) | {seed_row} | self._banned_rows
        explore = self._sample_exploration(seed_row, n_explore, exclude=used)

        slate = np.concatenate([ranked, explore]).astype(int)
        if len(slate) < n_slots:
            pad = np.full(n_slots - len(slate), -1, dtype=int)
            slate = np.concatenate([slate, pad])
        return slate[:n_slots]

    def _rank_candidates(self, seed_row: int, candidate_rows: np.ndarray) -> np.ndarray:
        if len(candidate_rows) == 0:
            return candidate_rows
        seed_embedding = self.corpus.embedding_by_row(seed_row)
        candidate_embeddings = self.corpus.embeddings[candidate_rows]
        base_scores = self.scorer(seed_embedding, candidate_embeddings)
        weights = np.array([self._weight_of(int(r)) for r in candidate_rows])
        final_scores = base_scores * weights
        order = np.argsort(-final_scores)
        return candidate_rows[order]

    def _sample_exploration(
        self, seed_row: int, n_explore: int, exclude: set[int]
    ) -> np.ndarray:
        if n_explore <= 0:
            return np.array([], dtype=int)

        k_far = min(self.k * self.far_multiplier, len(self.corpus) - 1)
        far_rows, _ = self.corpus.kneighbors(seed_row, k=k_far)
        pool = np.array([r for r in far_rows if int(r) not in exclude])
        if len(pool) == 0:
            pool = self.corpus.sample_rows(
                len(self.corpus), rng=self.rng, exclude=exclude | {seed_row}
            )
        if len(pool) == 0:
            return np.array([], dtype=int)

        weights = np.array([max(self._weight_of(int(r)), 1e-9) for r in pool])
        probs = weights / weights.sum()
        n_pick = min(n_explore, len(pool))
        picked = self.rng.choice(pool, size=n_pick, replace=False, p=probs)
        return picked
