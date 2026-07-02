"""The Gymnasium environment: a viewer navigating a thumbnail feed.

``ThumbwalkEnv`` is deliberately built around an inversion of the usual
recommender-simulation setup used by RecSim (Google, 2019) and SARDINE
(Naver, 2024). In those environments, the learnable agent IS the
recommender: it proposes a slate to a simulated user model. Here, the
recommender (:class:`thumbwalk.recommender.TwoStageFunnel`) is FIXED and
non-learnable — it is part of the environment's dynamics, not the agent.
The learnable agent is the VIEWER: on every step it sees one "playing" seed
thumbnail plus a slate of related thumbnails, and chooses to click one
(which becomes the new seed) or mark one "not interested" (which suppresses
its cluster/neighborhood from future recommendations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thumbwalk.click_model import ClickModel
from thumbwalk.corpus import Corpus
from thumbwalk.recommender import Scorer, TwoStageFunnel

#: Called once per step with a context dict, must return a scalar reward.
#: Context keys: seed_id_before, action, clicked_id, is_click,
#: is_not_interested, invalid_action, masked_count.
RewardFn = Callable[[dict[str, Any]], float]


def _zero_reward(_context: dict[str, Any]) -> float:
    """Default reward: always 0. Supply your own ``reward_fn`` for real research."""
    return 0.0


@dataclass
class ThumbwalkEnvConfig:
    """Configuration for :class:`ThumbwalkEnv`.

    Action space is ``Discrete(2 * slate_size)``: actions
    ``0 .. slate_size-1`` click the slate item at that position; actions
    ``slate_size .. 2*slate_size-1`` mark the item at position
    ``action - slate_size`` "not interested".
    """

    slate_size: int = 10
    k: int = 15
    exploration_slots: int = 1
    click_model: Optional[ClickModel] = None
    hard_mask: bool = False
    mask_scope: str = "cluster"
    mask_neighborhood_k: Optional[int] = None
    suppression_factor: float = 0.5
    floor: float = 0.05
    pool_multiplier: int = 3
    far_multiplier: int = 10
    scorer: Optional[Scorer] = None
    horizon: int = 50
    seed: Optional[int] = None
    advance_on_dislike: bool = False
    terminate_on_starvation: bool = False
    reward_fn: RewardFn = field(default=_zero_reward)


class ThumbwalkEnv(gym.Env):
    """A viewer navigating a feed of thumbnails produced by a fixed recommender.

    Observation (``Dict``):
        - ``seed_embedding``: float32 ``(D,)`` — embedding of the current seed item.
        - ``seed_id``: int64 ``(1,)`` — corpus row index of the current seed.
        - ``slate_embeddings``: float32 ``(N, D)`` — embeddings of the current slate.
        - ``slate_ids``: int64 ``(N,)`` — corpus row indices of the slate (-1 = padding).
        - ``slate_mask``: float32 ``(N,)`` — 1.0 for real slate items, 0.0 for padding.

    Image paths and string ids are NOT part of the observation (so the env
    stays usable headless); they are surfaced in ``info`` for renderers.

    Action (``Discrete(2 * N)``):
        - ``0 .. N-1``: click slate item at that position -> becomes new seed.
        - ``N .. 2N-1``: mark slate item at position ``action - N`` "not interested".
    """

    metadata = {"render_modes": []}

    def __init__(self, corpus: Corpus, config: Optional[ThumbwalkEnvConfig] = None) -> None:
        super().__init__()
        self.corpus = corpus
        self.config = config if config is not None else ThumbwalkEnvConfig()

        n_items = len(corpus)
        n_slots = self.config.slate_size
        embedding_dim = corpus.embedding_dim

        self.action_space = spaces.Discrete(2 * n_slots)
        self.observation_space = spaces.Dict(
            {
                "seed_embedding": spaces.Box(-np.inf, np.inf, shape=(embedding_dim,), dtype=np.float32),
                "seed_id": spaces.Box(0, n_items - 1, shape=(1,), dtype=np.int64),
                "slate_embeddings": spaces.Box(
                    -np.inf, np.inf, shape=(n_slots, embedding_dim), dtype=np.float32
                ),
                "slate_ids": spaces.Box(-1, n_items - 1, shape=(n_slots,), dtype=np.int64),
                "slate_mask": spaces.Box(0, 1, shape=(n_slots,), dtype=np.float32),
            }
        )

        self.funnel = TwoStageFunnel(
            corpus,
            k=self.config.k,
            exploration_slots=self.config.exploration_slots,
            pool_multiplier=self.config.pool_multiplier,
            far_multiplier=self.config.far_multiplier,
            suppression_factor=self.config.suppression_factor,
            floor=self.config.floor,
            mask_scope=self.config.mask_scope,
            mask_neighborhood_k=self.config.mask_neighborhood_k,
            hard_mask=self.config.hard_mask,
            scorer=self.config.scorer,
        )

        self._step_count = 0
        self._current_seed_row: int = 0
        self._current_slate: np.ndarray = np.full(n_slots, -1, dtype=int)

    # -- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed if seed is not None else self.config.seed)
        self.funnel.rng = self.np_random
        self.funnel.reset_suppression()
        self._step_count = 0

        options = options or {}
        if "start_row" in options:
            self._current_seed_row = int(options["start_row"])
        else:
            self._current_seed_row = int(self.np_random.integers(0, len(self.corpus)))

        self._rebuild_slate()
        obs = self._get_obs()
        info = self._get_info(clicked_id=None, action=None, invalid=False)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for action_space {self.action_space}")

        n_slots = self.config.slate_size
        is_click = action < n_slots
        position = action if is_click else action - n_slots
        item_row = int(self._current_slate[position])
        invalid = item_row == -1

        seed_before = self._current_seed_row
        clicked_id: Optional[str] = None

        if not invalid:
            if is_click:
                self._current_seed_row = item_row
                clicked_id = self.corpus.id_at(item_row)
            else:
                self.funnel.apply_not_interested(item_row)
                if self.config.advance_on_dislike:
                    self._current_seed_row = self._next_best_after_dislike(item_row)

        self._rebuild_slate()
        self._step_count += 1

        context = {
            "seed_id_before": self.corpus.id_at(seed_before),
            "action": action,
            "clicked_id": clicked_id,
            "is_click": bool(is_click and not invalid),
            "is_not_interested": bool(not is_click and not invalid),
            "invalid_action": invalid,
            "masked_count": len(self.funnel.suppressed_rows()),
        }
        reward = self.config.reward_fn(context)

        terminated = False
        if self.config.terminate_on_starvation:
            pool = self.funnel.candidate_generation(
                self._current_seed_row, set(self.funnel.banned_rows)
            )
            terminated = len(pool) < n_slots

        truncated = self._step_count >= self.config.horizon

        obs = self._get_obs()
        info = self._get_info(clicked_id=clicked_id, action=action, invalid=invalid)
        return obs, reward, terminated, truncated, info

    # -- helpers -------------------------------------------------------------

    def _rebuild_slate(self) -> None:
        banned = set(self.funnel.banned_rows)
        candidates = self.funnel.candidate_generation(self._current_seed_row, banned)
        self._current_slate = self.funnel.rank(
            self._current_seed_row, candidates, self.config.slate_size
        )

    def _next_best_after_dislike(self, disliked_row: int) -> int:
        for row in self._current_slate:
            row = int(row)
            if row != -1 and row != disliked_row:
                return row
        return self._current_seed_row

    def _get_obs(self) -> dict[str, np.ndarray]:
        n_slots = self.config.slate_size
        embedding_dim = self.corpus.embedding_dim

        slate_ids = self._current_slate.astype(np.int64)
        slate_mask = (slate_ids != -1).astype(np.float32)
        slate_embeddings = np.zeros((n_slots, embedding_dim), dtype=np.float32)
        for i, row in enumerate(slate_ids):
            if row != -1:
                slate_embeddings[i] = self.corpus.embedding_by_row(int(row))

        return {
            "seed_embedding": self.corpus.embedding_by_row(self._current_seed_row).astype(np.float32),
            "seed_id": np.array([self._current_seed_row], dtype=np.int64),
            "slate_embeddings": slate_embeddings,
            "slate_ids": slate_ids,
            "slate_mask": slate_mask,
        }

    def _get_info(
        self, clicked_id: Optional[str], action: Optional[int], invalid: bool
    ) -> dict[str, Any]:
        slate_paths = [
            self.corpus.path_of_row(int(r)) if r != -1 else None for r in self._current_slate
        ]
        slate_string_ids = [
            self.corpus.id_at(int(r)) if r != -1 else None for r in self._current_slate
        ]
        return {
            "clicked_id": clicked_id,
            "masked_count": len(self.funnel.suppressed_rows()),
            "current_cluster": self.corpus.cluster_of_row(self._current_seed_row),
            "seed_path": self.corpus.path_of_row(self._current_seed_row),
            "slate_paths": slate_paths,
            "slate_ids": slate_string_ids,
            "action": action,
            "invalid_action": invalid,
        }
