from __future__ import annotations

import numpy as np
import pytest

from thumbwalk.corpus import Corpus
from thumbwalk.env import ThumbwalkEnv, ThumbwalkEnvConfig


def test_observation_and_action_space_shapes(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=6, k=5, horizon=10)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0)

    assert env.action_space.n == 12
    assert obs["slate_embeddings"].shape == (6, synthetic_corpus.embedding_dim)
    assert obs["slate_ids"].shape == (6,)
    assert obs["slate_mask"].shape == (6,)
    assert obs["seed_embedding"].shape == (synthetic_corpus.embedding_dim,)
    assert env.observation_space.contains(obs)


def test_click_action_updates_seed(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(slate_size=5, k=10, horizon=5, exploration_slots=0)
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    clicked_row = int(obs["slate_ids"][2])

    obs2, reward, terminated, truncated, info2 = env.step(2)

    assert int(obs2["seed_id"][0]) == clicked_row
    assert info2["clicked_id"] == synthetic_corpus.id_at(clicked_row)


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
        _, _, _, truncated, _ = env.step(0)
    assert truncated is True


def test_hard_mask_shrinks_candidate_pool_and_banned_never_reappear(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5, k=10, horizon=20, hard_mask=True, mask_scope="cluster", exploration_slots=0
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})

    dislike_action = config.slate_size + 0
    disliked_row = int(obs["slate_ids"][0])

    obs2, reward, terminated, truncated, info2 = env.step(dislike_action)
    assert info2["masked_count"] > 0
    banned_rows = set(env.funnel.banned_rows)
    assert disliked_row in banned_rows

    for _ in range(10):
        obs2, reward, terminated, truncated, info2 = env.step(0)
        live_slate_rows = set(int(r) for r in obs2["slate_ids"] if r != -1)
        assert live_slate_rows.isdisjoint(banned_rows)
        if truncated or terminated:
            break


def test_soft_mask_default_downweights_without_hard_banning(synthetic_corpus: Corpus) -> None:
    config = ThumbwalkEnvConfig(
        slate_size=5,
        k=10,
        horizon=5,
        hard_mask=False,
        mask_scope="cluster",
        exploration_slots=0,
        suppression_factor=0.1,
        floor=0.01,
    )
    env = ThumbwalkEnv(synthetic_corpus, config)
    obs, info = env.reset(seed=0, options={"start_row": 0})
    disliked_row = int(obs["slate_ids"][0])
    disliked_cluster = synthetic_corpus.cluster_of_row(disliked_row)

    env.step(config.slate_size + 0)

    assert len(env.funnel.banned_rows) == 0  # soft mask never hard-excludes
    assert env.funnel.cluster_weight[disliked_cluster] == pytest.approx(0.1)


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
        return 1.0 if context["is_click"] else -1.0

    config = ThumbwalkEnvConfig(slate_size=4, horizon=5, exploration_slots=0, reward_fn=reward_fn)
    env = ThumbwalkEnv(synthetic_corpus, config)
    env.reset(seed=0)

    _, reward_click, _, _, _ = env.step(0)
    assert reward_click == 1.0

    _, reward_dislike, _, _, _ = env.step(config.slate_size)
    assert reward_dislike == -1.0
    assert len(calls) == 2
