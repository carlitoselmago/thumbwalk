from __future__ import annotations

import numpy as np
import pytest

from thumbwalk.corpus import Corpus
from thumbwalk.recommender import TwoStageFunnel, profile_embedding


def test_candidate_generation_cold_start_uses_popularity(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, pool_multiplier=3)
    candidates = funnel.candidate_generation(history=[], banned_rows=set())
    assert len(candidates) == 5
    # cold start must not crash and must return valid rows
    assert all(0 <= r < len(synthetic_corpus) for r in candidates)


def test_candidate_generation_with_history_matches_profile_neighbors(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, pool_multiplier=3)
    history = [(0, 1.0)]
    candidates = funnel.candidate_generation(history=history, banned_rows=set())
    profile = profile_embedding(synthetic_corpus, history, history_window=50)
    expected_rows, _ = synthetic_corpus.kneighbors_of_vector(profile, k=5, exclude={0})
    np.testing.assert_array_equal(candidates, expected_rows)


def test_candidate_generation_excludes_history_rows(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, pool_multiplier=3)
    history = [(0, 1.0)]
    candidates = funnel.candidate_generation(history=history, banned_rows=set())
    assert 0 not in set(int(r) for r in candidates)


def test_rank_returns_exactly_n_slots_no_duplicates(synthetic_corpus: Corpus) -> None:
    rng = np.random.default_rng(42)
    funnel = TwoStageFunnel(synthetic_corpus, k=5, exploration_slots=1, rng=rng)
    history = [(0, 1.0)]
    candidates = funnel.candidate_generation(history, set())
    slate = funnel.rank(history, candidates, n_slots=5)

    assert len(slate) == 5
    valid = [int(r) for r in slate if r != -1]
    assert len(valid) == len(set(valid))  # no duplicates
    assert all(r != -1 for r in slate)  # corpus is large enough not to starve


def test_rank_exploration_slot_can_come_from_outside_near_pool(synthetic_corpus: Corpus) -> None:
    rng = np.random.default_rng(7)
    funnel = TwoStageFunnel(synthetic_corpus, k=3, exploration_slots=1, far_multiplier=15, rng=rng)
    history = [(0, 1.0)]
    candidates = funnel.candidate_generation(history, set())
    slate = funnel.rank(history, candidates, n_slots=4)  # 3 ranked + 1 exploration
    ranked_part = set(int(r) for r in slate[:3])
    explore_part = int(slate[3])
    # exploration slot is filled and structurally distinct from the ranked portion
    assert explore_part != -1
    assert explore_part not in ranked_part


def test_not_interested_decays_item_weight_with_floor(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(
        synthetic_corpus, k=5, not_interested_factor=0.5, not_interested_floor=0.05
    )
    row0 = 0

    funnel.apply_not_interested(row0)
    assert funnel.item_weight[row0] == pytest.approx(0.5)

    funnel.apply_not_interested(row0)
    assert funnel.item_weight[row0] == pytest.approx(0.25)

    for _ in range(10):
        funnel.apply_not_interested(row0)
    assert funnel.item_weight[row0] == pytest.approx(0.05)  # clamped at floor, never 0


def test_not_interested_suppresses_neighborhood_not_whole_corpus(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(
        synthetic_corpus, k=5, not_interested_neighborhood_k=3,
        not_interested_factor=0.3, not_interested_floor=0.01,
    )
    row0 = 0
    funnel.apply_not_interested(row0)

    neighbor_rows, _ = synthetic_corpus.kneighbors(row0, k=3)
    affected = set(int(r) for r in neighbor_rows) | {row0}
    weights = funnel.item_weight
    for row in affected:
        assert weights[row] == pytest.approx(0.3)
    # a distant row (an outlier, far from item 0's cluster) should be untouched
    outlier_row = len(synthetic_corpus) - 1
    if outlier_row not in affected:
        assert weights[outlier_row] == pytest.approx(1.0)


def test_no_channel_decays_channel_weight_with_floor(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(
        synthetic_corpus, k=5, channel_suppression_factor=0.5, channel_suppression_floor=0.05
    )
    row0 = 0
    channel = synthetic_corpus.channel_of_row(row0)

    funnel.apply_no_channel(row0)
    assert funnel.channel_weight[channel] == pytest.approx(0.5)

    funnel.apply_no_channel(row0)
    assert funnel.channel_weight[channel] == pytest.approx(0.25)

    for _ in range(10):
        funnel.apply_no_channel(row0)
    assert funnel.channel_weight[channel] == pytest.approx(0.05)


def test_no_channel_reduces_channel_representation_in_slate(synthetic_corpus: Corpus) -> None:
    # exploration_slots=0 keeps rank() fully deterministic for this comparison.
    funnel = TwoStageFunnel(
        synthetic_corpus, k=10, exploration_slots=0,
        channel_suppression_factor=0.1, channel_suppression_floor=0.01,
    )
    history = [(0, 1.0)]
    seed_channel = synthetic_corpus.channel_of_row(0)
    candidates = funnel.candidate_generation(history, set())

    before_slate = funnel.rank(history, candidates, n_slots=5)
    before_count = sum(
        1 for r in before_slate if r != -1 and synthetic_corpus.channel_of_row(int(r)) == seed_channel
    )

    disliked = int(candidates[0])
    funnel.apply_no_channel(disliked)
    after_slate = funnel.rank(history, candidates, n_slots=5)
    after_count = sum(
        1 for r in after_slate if r != -1 and synthetic_corpus.channel_of_row(int(r)) == seed_channel
    )

    assert after_count <= before_count


def test_shown_not_watched_demotes_impression(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, impression_decay=0.5)
    slate = np.array([1, 2, 3])
    funnel.record_impressions(slate, watched_row=1)
    weight_watched = funnel._weight_of(1)
    weight_shown_not_watched = funnel._weight_of(2)
    assert weight_watched > weight_shown_not_watched


def test_reset_suppression_clears_state(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5)
    funnel.apply_not_interested(0)
    funnel.apply_no_channel(1)
    funnel.record_impressions(np.array([2, 3]), watched_row=None)
    assert len(funnel.suppressed_rows()) > 0

    funnel.reset_suppression()
    assert len(funnel.suppressed_rows()) == 0
    assert funnel.channel_weight == {}
    assert np.all(funnel.item_weight == 1.0)
