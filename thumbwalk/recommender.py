"""The fixed, non-learnable recommender: a two-stage funnel.

Modeled after Covington, Adams & Sargin, "Deep Neural Networks for YouTube
Recommendations" (2016), as described in the TFM (cap. 2, "El sistema de
recomendación de YouTube como referencia", and cap. 3, "Entorno de
simulación"):

  1. **candidate generation** (:meth:`TwoStageFunnel.candidate_generation`):
     retrieve the nearest neighbors of a *viewing profile* (a window of up
     to ``history_window`` recently watched items, weighted by watch
     fraction) in embedding space. With no history yet (cold start), falls
     back to popularity-weighted sampling — the stand-in for Covington et
     al.'s demographic priors for new users, and the source of the
     episode's initial slate. Optimizes *recall*.
  2. **ranking** (:meth:`TwoStageFunnel.rank`): scores candidates by
     affinity to the profile and by popularity, demotes items that have
     been shown repeatedly without being watched (Covington et al., sec.
     4.1), and applies the two negative-feedback suppressions ("No me
     interesa" / "Don't recommend channel"). Optionally injects exploration
     slots sampled from farther away (serendipity / freshness). Optimizes
     *precision*.

Unlike RecSim/SARDINE, this funnel is never trained — it is the fixed
environment dynamics that the *viewer* (the RL agent, see
:mod:`thumbwalk.env`) navigates.

Calibration of the two negative-feedback actions ("Not interested" / "No
recomendar canal") follows the only published measurement of their effect
found for this TFM: Ricks & McCrosky, "Does This Button Work?" (2022),
observed roughly an 11% drop in similar recommendations after "No me
interesa" and roughly a 43% drop after "Don't recommend channel" (the most
effective of YouTube's controls), with neither ever reaching full
suppression. ``not_interested_factor`` and ``channel_suppression_factor``
default to ``1 - 0.11`` and ``1 - 0.43`` respectively, applied
multiplicatively and clamped at a floor so a mark never fully erases its
target (see TFM.md, cap. 3, nota "calibración del feedback negativo").
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from thumbwalk.corpus import Corpus

#: (profile_embedding, candidate_embeddings) -> affinity scores, higher is better.
#: Stands in for Covington et al.'s expected-watch-time ranking objective.
Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: One entry per watched item in the viewer's history: (row, watch_fraction).
#: watch_fraction in [0, 1] (0 = abandoned, 1 = watched in full).
HistoryEntry = tuple[int, float]


def _default_scorer(profile_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(candidate_embeddings - profile_embedding[None, :], axis=1)
    return 1.0 / (1.0 + distances)


def profile_embedding(
    corpus: Corpus, history: list[HistoryEntry], history_window: int
) -> Optional[np.ndarray]:
    """Weighted-average embedding of the viewer's recent watch history.

    Only the last ``history_window`` entries count (Covington et al. use the
    last 50 watches). Weighted by watch fraction, per the TFM's symbolic
    watch-time signal, so a fully-watched video affects the profile more
    than one that was abandoned immediately. If every entry in the window
    has fraction 0 (everything abandoned), falls back to an unweighted
    average rather than a degenerate all-zero profile. Returns ``None`` when
    ``history`` is empty (cold start: no profile to build).
    """
    if not history:
        return None
    window = history[-history_window:]
    rows = np.array([r for r, _ in window], dtype=int)
    weights = np.array([f for _, f in window], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    embeddings = corpus.embeddings[rows]
    return np.average(embeddings, axis=0, weights=weights).astype(np.float32)


class TwoStageFunnel:
    """Fixed candidate-generation + ranking recommender over a :class:`Corpus`.

    Also owns the episode-scoped negative-feedback state: a per-item weight
    for "Not interested" (suppresses the marked item's k-NN neighborhood)
    and a per-channel weight for "Don't recommend channel" (suppresses every
    item sharing the marked item's channel), plus an impression counter
    used to demote items shown but never watched. All three state pieces
    reset at the start of every episode via :meth:`reset_suppression`.
    """

    def __init__(
        self,
        corpus: Corpus,
        k: int = 15,
        exploration_slots: int = 1,
        pool_multiplier: int = 3,
        far_multiplier: int = 10,
        history_window: int = 50,
        not_interested_factor: float = 0.89,
        not_interested_floor: float = 0.05,
        not_interested_neighborhood_k: Optional[int] = None,
        channel_suppression_factor: float = 0.57,
        channel_suppression_floor: float = 0.1,
        impression_decay: float = 0.9,
        popularity_weight: float = 0.3,
        scorer: Optional[Scorer] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.corpus = corpus
        self.k = k
        self.exploration_slots = exploration_slots
        self.pool_multiplier = pool_multiplier
        self.far_multiplier = far_multiplier
        self.history_window = history_window
        self.not_interested_factor = not_interested_factor
        self.not_interested_floor = not_interested_floor
        self.not_interested_neighborhood_k = (
            not_interested_neighborhood_k if not_interested_neighborhood_k is not None else k
        )
        self.channel_suppression_factor = channel_suppression_factor
        self.channel_suppression_floor = channel_suppression_floor
        self.impression_decay = impression_decay
        self.popularity_weight = popularity_weight
        self.scorer = scorer if scorer is not None else _default_scorer
        self.rng = rng if rng is not None else np.random.default_rng()

        corpus.ensure_nn_index()

        n = len(corpus)
        self._item_weight = np.ones(n, dtype=np.float64)
        self._channel_weight: dict[str, float] = {}
        self._shown_not_watched = np.zeros(n, dtype=np.int64)

    # -- suppression / impression state --------------------------------------

    def reset_suppression(self) -> None:
        """Clear all per-episode state. Called by env.reset() for a new episode."""
        n = len(self.corpus)
        self._item_weight = np.ones(n, dtype=np.float64)
        self._channel_weight = {}
        self._shown_not_watched = np.zeros(n, dtype=np.int64)

    @property
    def item_weight(self) -> np.ndarray:
        return self._item_weight.copy()

    @property
    def channel_weight(self) -> dict[str, float]:
        return dict(self._channel_weight)

    def suppressed_rows(self) -> np.ndarray:
        """Rows currently suppressed to any degree (item or channel weight < 1.0)."""
        item_suppressed = self._item_weight < 1.0
        if self._channel_weight:
            suppressed_channels = {c for c, w in self._channel_weight.items() if w < 1.0}
            channel_suppressed = np.array(
                [self.corpus.channel_of_row(r) in suppressed_channels for r in range(len(self.corpus))]
            )
        else:
            channel_suppressed = np.zeros(len(self.corpus), dtype=bool)
        return np.where(item_suppressed | channel_suppressed)[0]

    def apply_not_interested(self, item_row: int) -> None:
        """"Not interested": soft-suppress ``item_row``'s k-NN neighborhood.

        Calibrated so repeated recommendations of similar items drop by
        roughly ``1 - not_interested_factor`` (default ~11%, Ricks &
        McCrosky 2022), never reaching zero (floor).
        """
        neighbor_rows, _ = self.corpus.kneighbors(item_row, k=self.not_interested_neighborhood_k)
        affected = np.concatenate([[item_row], neighbor_rows])
        self._item_weight[affected] = np.maximum(
            self.not_interested_floor, self._item_weight[affected] * self.not_interested_factor
        )

    def apply_no_channel(self, item_row: int) -> None:
        """"Don't recommend channel": soft-suppress every item in ``item_row``'s channel.

        Calibrated so the channel's representation in future slates drops
        by roughly ``1 - channel_suppression_factor`` (default ~43%, the
        most effective of the two controls measured by Ricks & McCrosky
        2022), never reaching zero (floor) — it is "the most effective
        option", not a full block.
        """
        channel = self.corpus.channel_of_row(item_row)
        current = self._channel_weight.get(channel, 1.0)
        self._channel_weight[channel] = max(
            self.channel_suppression_floor, current * self.channel_suppression_factor
        )

    def record_impressions(self, slate_rows: np.ndarray, watched_row: Optional[int]) -> None:
        """Update shown-but-not-watched counters for one step's slate.

        Every valid (non-padding) row in ``slate_rows`` was shown to the
        viewer this step. The one that was watched (``watched_row``, or
        ``None`` if the step was a feedback action and nothing was
        watched) has its counter reset; every other shown row's counter
        increments, so it gets demoted next time it is ranked (Covington
        et al., sec. 4.1: "impressions shown but not watched get demoted").
        """
        for row in slate_rows:
            row = int(row)
            if row == -1:
                continue
            if row == watched_row:
                self._shown_not_watched[row] = 0
            else:
                self._shown_not_watched[row] += 1

    def _weight_of(self, row: int) -> float:
        channel = self.corpus.channel_of_row(row)
        channel_w = self._channel_weight.get(channel, 1.0)
        item_w = self._item_weight[row]
        impression_w = self.impression_decay ** int(self._shown_not_watched[row])
        return channel_w * item_w * impression_w

    # -- stage 1: candidate generation --------------------------------------

    def candidate_generation(self, history: list[HistoryEntry], banned_rows: set[int]) -> np.ndarray:
        """Retrieve up to ``k`` candidate rows, optimizing recall.

        With a non-empty ``history``, retrieves the nearest neighbors of
        the weighted viewing profile (see :func:`profile_embedding`). With
        empty ``history`` (cold start, e.g. the first slate of an episode),
        falls back to popularity-weighted sampling — the demographic-prior
        stand-in from Covington et al., and the source of the episode's
        initial slate ("ancla del slate inicial").
        """
        k_pool = min(self.k * self.pool_multiplier, len(self.corpus) - 1)
        if k_pool <= 0:
            return np.array([], dtype=int)

        profile = profile_embedding(self.corpus, history, self.history_window)
        if profile is None:
            pool = self.corpus.sample_rows(
                k_pool, rng=self.rng, exclude=set(banned_rows), weights=self.corpus.popularity
            )
            return pool[: self.k]

        exclude = set(banned_rows) | {row for row, _ in history[-self.history_window:]}
        neighbor_rows, _ = self.corpus.kneighbors_of_vector(profile, k=k_pool, exclude=exclude)
        return neighbor_rows[: self.k]

    # -- stage 2: ranking -----------------------------------------------------

    def rank(self, history: list[HistoryEntry], candidate_rows: np.ndarray, n_slots: int) -> np.ndarray:
        """Score, order, and pad candidates into a slate of exactly ``n_slots`` rows.

        Score combines affinity to the viewing profile with popularity
        (``popularity_weight``), then multiplies by the suppression/
        impression-demotion weight (:meth:`_weight_of`). Exploration slots
        are appended at the end, sampled from a farther pool (serendipity/
        freshness, as in Covington et al.'s churn). Missing slots (starved
        funnel) are padded with ``-1``.
        """
        n_explore = min(self.exploration_slots, n_slots)
        n_ranked = n_slots - n_explore

        ranked = self._rank_candidates(history, candidate_rows)[:n_ranked]

        used = set(int(r) for r in ranked) | {row for row, _ in history[-self.history_window:]}
        explore = self._sample_exploration(history, n_explore, exclude=used)

        slate = np.concatenate([ranked, explore]).astype(int)
        if len(slate) < n_slots:
            pad = np.full(n_slots - len(slate), -1, dtype=int)
            slate = np.concatenate([slate, pad])
        return slate[:n_slots]

    def _rank_candidates(self, history: list[HistoryEntry], candidate_rows: np.ndarray) -> np.ndarray:
        if len(candidate_rows) == 0:
            return candidate_rows
        profile = profile_embedding(self.corpus, history, self.history_window)
        if profile is None:
            # Cold start: nothing watched yet, rank candidates by popularity alone.
            affinity = np.ones(len(candidate_rows), dtype=np.float64)
        else:
            candidate_embeddings = self.corpus.embeddings[candidate_rows]
            affinity = self.scorer(profile, candidate_embeddings)

        popularity = self.corpus.popularity[candidate_rows]
        popularity_term = (1.0 + popularity) ** self.popularity_weight
        weights = np.array([self._weight_of(int(r)) for r in candidate_rows])
        final_scores = affinity * popularity_term * weights
        order = np.argsort(-final_scores)
        return candidate_rows[order]

    def _sample_exploration(
        self, history: list[HistoryEntry], n_explore: int, exclude: set[int]
    ) -> np.ndarray:
        if n_explore <= 0:
            return np.array([], dtype=int)

        profile = profile_embedding(self.corpus, history, self.history_window)
        if profile is not None:
            k_far = min(self.k * self.far_multiplier, len(self.corpus) - 1)
            far_rows, _ = self.corpus.kneighbors_of_vector(profile, k=k_far, exclude=exclude)
            pool = far_rows
        else:
            pool = np.array([], dtype=int)

        if len(pool) == 0:
            pool = self.corpus.sample_rows(len(self.corpus), rng=self.rng, exclude=exclude)
        if len(pool) == 0:
            return np.array([], dtype=int)

        weights = np.array([max(self._weight_of(int(r)), 1e-9) for r in pool])
        probs = weights / weights.sum()
        n_pick = min(n_explore, len(pool))
        picked = self.rng.choice(pool, size=n_pick, replace=False, p=probs)
        return picked
