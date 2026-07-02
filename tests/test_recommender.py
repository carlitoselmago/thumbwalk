from __future__ import annotations

import numpy as np
import pytest

from thumbwalk.corpus import Corpus
from thumbwalk.recommender import TwoStageFunnel


def test_candidate_generation_matches_k_nearest(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, pool_multiplier=3)
    candidates = funnel.candidate_generation(seed_row=0, banned_rows=set())
    expected_rows, _ = synthetic_corpus.kneighbors(0, k=5)
    np.testing.assert_array_equal(candidates, expected_rows)


def test_candidate_generation_excludes_hard_banned_rows(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, pool_multiplier=3)
    nearest_rows, _ = synthetic_corpus.kneighbors(0, k=5)
    banned = {int(nearest_rows[0])}
    candidates = funnel.candidate_generation(seed_row=0, banned_rows=banned)
    assert banned.isdisjoint(int(r) for r in candidates)


def test_rank_returns_exactly_n_slots_no_duplicates(synthetic_corpus: Corpus) -> None:
    rng = np.random.default_rng(42)
    funnel = TwoStageFunnel(synthetic_corpus, k=5, exploration_slots=1, rng=rng)
    candidates = funnel.candidate_generation(0, set())
    slate = funnel.rank(0, candidates, n_slots=5)

    assert len(slate) == 5
    valid = [int(r) for r in slate if r != -1]
    assert len(valid) == len(set(valid))  # no duplicates
    assert all(r != -1 for r in slate)  # corpus is large enough not to starve


def test_rank_exploration_slot_can_come_from_outside_near_pool(synthetic_corpus: Corpus) -> None:
    rng = np.random.default_rng(7)
    funnel = TwoStageFunnel(synthetic_corpus, k=3, exploration_slots=1, far_multiplier=15, rng=rng)
    candidates = funnel.candidate_generation(0, set())
    slate = funnel.rank(0, candidates, n_slots=4)  # 3 ranked + 1 exploration
    ranked_part = set(int(r) for r in slate[:3])
    explore_part = int(slate[3])
    # exploration slot is filled and structurally distinct from the ranked portion
    assert explore_part != -1
    assert explore_part not in ranked_part


def test_soft_suppression_decays_cluster_weight_with_floor(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(
        synthetic_corpus, k=5, mask_scope="cluster", suppression_factor=0.5, floor=0.05
    )
    row0 = 0
    cluster = synthetic_corpus.cluster_of_row(row0)

    funnel.apply_not_interested(row0)
    assert funnel.cluster_weight[cluster] == pytest.approx(0.5)

    funnel.apply_not_interested(row0)
    assert funnel.cluster_weight[cluster] == pytest.approx(0.25)

    for _ in range(10):
        funnel.apply_not_interested(row0)
    assert funnel.cluster_weight[cluster] == pytest.approx(0.05)  # clamped at floor, never 0


def test_soft_suppression_reduces_cluster_representation_in_slate(synthetic_corpus: Corpus) -> None:
    # exploration_slots=0 keeps rank() fully deterministic for this comparison.
    funnel = TwoStageFunnel(
        synthetic_corpus, k=10, exploration_slots=0, suppression_factor=0.1, floor=0.01
    )
    seed_row = 0
    seed_cluster = synthetic_corpus.cluster_of_row(seed_row)
    candidates = funnel.candidate_generation(seed_row, set())

    before_slate = funnel.rank(seed_row, candidates, n_slots=5)
    before_count = sum(
        1 for r in before_slate if r != -1 and synthetic_corpus.cluster_of_row(int(r)) == seed_cluster
    )

    disliked = int(candidates[0])
    funnel.apply_not_interested(disliked)
    after_slate = funnel.rank(seed_row, candidates, n_slots=5)
    after_count = sum(
        1 for r in after_slate if r != -1 and synthetic_corpus.cluster_of_row(int(r)) == seed_cluster
    )

    assert after_count <= before_count


def test_hard_mask_bans_entire_cluster_from_future_candidates(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=10, hard_mask=True, mask_scope="cluster")
    cluster0_members = np.where(synthetic_corpus.kmeans_labels == 0)[0]
    disliked = int(cluster0_members[0])

    funnel.apply_not_interested(disliked)
    assert set(int(r) for r in cluster0_members) <= funnel.banned_rows

    other_seed = int(cluster0_members[1]) if len(cluster0_members) > 1 else disliked
    candidates = funnel.candidate_generation(other_seed, set(funnel.banned_rows))
    assert funnel.banned_rows.isdisjoint(int(r) for r in candidates)


def test_knn_scope_suppresses_neighborhood_not_whole_cluster(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(
        synthetic_corpus, k=5, mask_scope="knn", mask_neighborhood_k=3, suppression_factor=0.3, floor=0.01
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


def test_reset_suppression_clears_state(synthetic_corpus: Corpus) -> None:
    funnel = TwoStageFunnel(synthetic_corpus, k=5, hard_mask=True)
    funnel.apply_not_interested(0)
    assert len(funnel.banned_rows) > 0

    funnel.reset_suppression()
    assert len(funnel.banned_rows) == 0
    assert funnel.cluster_weight == {}
    assert np.all(funnel.item_weight == 1.0)
