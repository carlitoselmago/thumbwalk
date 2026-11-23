from __future__ import annotations

import numpy as np
import pytest

from thumbwalk.corpus import Corpus
from thumbwalk.env import (
    ACTION_NO_CHANNEL,
    ACTION_NOT_INTERESTED,
    ACTION_WATCH_ABANDON,
    ACTION_WATCH_FULL,
    N_ACTION_TYPES,
    ThumbwalkEnv,
    ThumbwalkEnvConfig,
)


def test_observation_and_action_space_shapes(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=6, k=5, horizon=10)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0)

    assert env.action_space.n == N_ACTION_TYPES * 6
    assert obs["slate_embeddings"].shape == (6, synthetic_corpus.embedding_dim)
    assert obs["slate_ids"].shape == (6,)
    assert obs["slate_mask"].shape == (6,)
    assert obs["seed_embedding"].shape == (synthetic_corpus.embedding_dim,)
    assert env.observation_space.contains(obs)


def test_default_reward_is_minus_one_without_a_pick_zero_with_one(
    synthetic_corpus: Corpus,
) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=10, horizon=5, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    env.reset(seed=0, options={"start_row": 0})

    _, reward, *_ = env.step(ACTION_NOT_INTERESTED * config.slate_size + 0)
    assert reward == -1.0

    env.reset(seed=0, options={"start_row": 0})
    _, reward, *_ = env.step(ACTION_WATCH_FULL * config.slate_size + 0)
    assert reward == 0.0


def test_watch_action_updates_seed_and_history(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=10, horizon=5, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    clicked_row = int(obs["slate_ids"][2])

    action = ACTION_WATCH_FULL * config.slate_size + 2
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert int(obs2["seed_id"][0]) == clicked_row
    assert info2["clicked_id"] == synthetic_corpus.id_at(clicked_row)
    assert info2["watch_fraction"] == pytest.approx(1.0)
    assert info2["history_length"] == 1


def test_not_interested_action_does_not_change_seed(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=10, horizon=5, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    seed_before = int(obs["seed_id"][0])

    action = ACTION_NOT_INTERESTED * config.slate_size + 0
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert int(obs2["seed_id"][0]) == seed_before
    assert info2["clicked_id"] is None
    assert info2["masked_count"] > 0
    assert info2["history_length"] == 0


def test_no_channel_action_does_not_change_seed(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=10, horizon=5, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    seed_before = int(obs["seed_id"][0])

    action = ACTION_NO_CHANNEL * config.slate_size + 0
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert int(obs2["seed_id"][0]) == seed_before
    assert info2["clicked_id"] is None


def test_step_returns_valid_gym_tuple(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=4, horizon=3)
    env = ThumbwalkEnv(synthetic_corpus, config)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {"clicked_id", "masked_count", "current_cluster"} <= info.keys()


def test_truncation_at_horizon(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=4, horizon=3, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    env.reset(seed=0)

    truncated = False
    for _ in range(3):
        action = ACTION_WATCH_ABANDON * config.slate_size + 0
        _, _, _, truncated, _ = env.step(action)
    assert truncated is True


def test_not_interested_shrinks_similar_item_representation(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5, k=10, horizon=20, not_interested_factor=0.1, not_interested_floor=0.01,
        exploration_slots=0,
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})

    dislike_action = ACTION_NOT_INTERESTED * config.slate_size + 0
    disliked_row = int(obs["slate_ids"][0])

    obs2, reward, terminated, truncated, info2 = env.step(dislike_action)
    assert info2["masked_count"] > 0
    assert env.funnel.item_weight[disliked_row] < 1.0


def test_reset_seed_determinism(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=6, k=5, horizon=10, exploration_slots=1)
    env1 = ThumbwalkEnv(synthetic_corpus, config)
    env2 = ThumbwalkEnv(synthetic_corpus, config)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1["seed_id"], obs2["seed_id"])
    np.testing.assert_array_equal(obs1["slate_ids"], obs2["slate_ids"])

    for raw_action in [0, 2, 7, 1, 3, 9, 4]:
        action = raw_action % env1.action_space.n
        r1 = env1.step(action)
        r2 = env2.step(action)
        np.testing.assert_array_equal(r1[0]["slate_ids"], r2[0]["slate_ids"])
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]
        assert r1[3] == r2[3]


def test_custom_reward_fn_is_used(synthetic_corpus: Corpus) -> None:
    calls = []

    def reward_fn(context: dict) -> float:
        calls.append(context)
        return 1.0 if context["is_watch"] else -1.0

    config = ThumbwalkEnvConfig(slate_size=4, horizon=5, exploration_slots=0, reward_fn=reward_fn)
    env = ThumbwalkEnv(synthetic_corpus, config)
    env.reset(seed=0)

    _, reward_watch, _, _, _ = env.step(ACTION_WATCH_FULL * config.slate_size + 0)
    assert reward_watch == 1.0

    _, reward_dislike, _, _, _ = env.step(ACTION_NOT_INTERESTED * config.slate_size + 0)
    assert reward_dislike == -1.0
    assert len(calls) == 2


def test_cold_start_first_slate_has_no_history(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=5, horizon=5)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0)
    assert info["history_length"] == 0


def test_max_suggestions_truncates_when_nothing_is_picked(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5, k=5, horizon=50, exploration_slots=0, max_suggestions=15,
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    assert info["suggestions_shown_since_watch"] == 5

    action = ACTION_NOT_INTERESTED * config.slate_size + 0
    truncated = False
    for expected_count in (10, 15, 15):
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["suggestions_shown_since_watch"] == expected_count
    assert truncated is True
    assert info["suggestions_exhausted"] is True


def test_max_suggestions_continuation_avoids_repeating_shown_items(
    synthetic_corpus: Corpus,
) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5, k=5, horizon=50, exploration_slots=0, max_suggestions=25,
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    first_slate = set(int(r) for r in obs["slate_ids"] if r != -1)

    action = ACTION_NOT_INTERESTED * config.slate_size + 0
    obs2, *_ = env.step(action)
    second_slate = set(int(r) for r in obs2["slate_ids"] if r != -1)

    assert first_slate.isdisjoint(second_slate)


def test_watching_resets_the_max_suggestions_budget(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5, k=5, horizon=50, exploration_slots=0, max_suggestions=10,
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})

    obs, reward, terminated, truncated, info = env.step(
        ACTION_NOT_INTERESTED * config.slate_size + 0
    )
    assert info["suggestions_shown_since_watch"] == 10
    assert truncated is False

    clicked_row = int(obs["slate_ids"][1])
    action = ACTION_WATCH_FULL * config.slate_size + 1
    obs, reward, terminated, truncated, info = env.step(action)
    assert int(obs["seed_id"][0]) == clicked_row
    assert info["suggestions_shown_since_watch"] == 5
    assert truncated is False

    # The budget was reset by the watch, so one more scroll must not exhaust it.
    obs, reward, terminated, truncated, info = env.step(
        ACTION_NOT_INTERESTED * config.slate_size + 0
    )
    assert info["suggestions_shown_since_watch"] == 10
    assert truncated is False


def test_max_suggestions_defaults_to_four_full_slates(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=5, horizon=10, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    assert env.max_suggestions == config.slate_size * 4  # == 20

    env.reset(seed=0, options={"start_row": 0})
    action = ACTION_NOT_INTERESTED * config.slate_size + 0

    # 3 more full slates (15 more items) exactly reach the 20-item cap...
    for _ in range(3):
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False
    # ...and the 4th no-pick step finds the budget already spent.
    _, _, _, truncated, info = env.step(action)
    assert truncated is True
    assert info["suggestions_exhausted"] is True
