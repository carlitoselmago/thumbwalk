"""The Gymnasium environment: a viewer navigating a thumbnail feed.

``ThumbwalkEnv`` is deliberately built around an inversion of the usual
recommender-simulation setup used by RecSim (Google, 2019) and SARDINE
(Naver, 2024). In those environments, the learnable agent IS the
recommender: it proposes a slate to a simulated user model. Here, the
recommender (:class:`thumbwalk.recommender.TwoStageFunnel`) is FIXED and
non-learnable — it is part of the environment's dynamics, not the agent.
The learnable agent is the VIEWER.

Action space (TFM.md, cap. 3, "Entorno de simulación", Figura 3.1): on
every step the viewer takes exactly ONE decision among ``5 * slate_size``
discrete actions, encoding a position ``k`` in the slate and a type ``j``:

  - ``j = 0, 1, 2``: WATCH slate item ``k``, with one of three symbolic
    watch times — abandon, half, full. The watched item becomes the new
    seed/profile entry; the recommender uses the watch fraction as an
    implicit relevance signal (Covington et al. 2016).
  - ``j = 3``: "Not interested" on slate item ``k`` — explicit negative
    feedback. The current seed does NOT change; the recommender
    down-weights items similar to ``k`` (~11% drop, Ricks & McCrosky 2022).
  - ``j = 4``: "Don't recommend channel" on slate item ``k`` — explicit
    negative feedback, channel-scoped. The current seed does NOT change;
    the recommender down-weights ``k``'s whole channel (~43% drop).

The action index is ``a = j * slate_size + k`` (``Discrete(5 * slate_size)``
in Gymnasium): a single flat index covers the "5K acciones" from the TFM
text rather than K clicks + K dislikes. Watching and choosing a watch time
are taken in a single step on purpose (see TFM.md, cap. 3): between
clicking a thumbnail and starting to watch, the agent gains no new
observation, so a second step would be identical to the first and only
lengthen episodes. Negative-feedback actions are, for the same reason, an
*alternative* to watching rather than a separate pre-step: each mark
consumes one of the episode's fixed number of decisions, exactly like a
watch action, so marking has a real opportunity cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thumbwalk.corpus import Corpus
from thumbwalk.recommender import HistoryEntry, Scorer, TwoStageFunnel

#: Symbolic watch fractions for action types j=0,1,2 (abandon / half / full).
WATCH_FRACTIONS: tuple[float, float, float] = (0.0, 0.5, 1.0)

#: Action-type indices within one slate position's 5-way block.
ACTION_WATCH_ABANDON = 0
ACTION_WATCH_HALF = 1
ACTION_WATCH_FULL = 2
ACTION_NOT_INTERESTED = 3
ACTION_NO_CHANNEL = 4
N_ACTION_TYPES = 5

#: Called once per step with a context dict, must return a scalar reward.
#: Context keys: seed_id_before, action, action_type, position, clicked_id,
#: watch_fraction, is_watch, is_not_interested, is_no_channel,
#: invalid_action, masked_count.
RewardFn = Callable[[dict[str, Any]], float]


def _zero_reward(_context: dict[str, Any]) -> float:
    """Default reward: always 0. Supply your own ``reward_fn`` for real research."""
    return 0.0


@dataclass
class ThumbwalkEnvConfig:
    """Configuration for :class:`ThumbwalkEnv`.

    Action space is ``Discrete(5 * slate_size)``: ``a = j * slate_size + k``
    with ``k`` the slate position (``0 .. slate_size-1``) and ``j`` the
    action type (see module docstring: 0/1/2 = watch abandon/half/full,
    3 = "Not interested", 4 = "Don't recommend channel").
    """

    slate_size: int = 10
    k: int = 15
    exploration_slots: int = 1
    history_window: int = 50
    not_interested_factor: float = 0.89
    not_interested_floor: float = 0.05
    channel_suppression_factor: float = 0.57
    channel_suppression_floor: float = 0.1
    impression_decay: float = 0.9
    popularity_weight: float = 0.3
    pool_multiplier: int = 3
    far_multiplier: int = 10
    scorer: Optional[Scorer] = None
    horizon: int = 50
    seed: Optional[int] = None
    terminate_on_starvation: bool = False
    reward_fn: RewardFn = field(default=_zero_reward)


class ThumbwalkEnv(gym.Env):
    """A viewer navigating a feed of thumbnails produced by a fixed recommender.

    Observation (``Dict``):
        - ``seed_embedding``: float32 ``(D,)`` — embedding of the current seed item
          (the most recently watched item; before the first watch, the cold-start
          starting item).
        - ``seed_id``: int64 ``(1,)`` — corpus row index of the current seed.
        - ``slate_embeddings``: float32 ``(N, D)`` — embeddings of the current slate.
        - ``slate_ids``: int64 ``(N,)`` — corpus row indices of the slate (-1 = padding).
        - ``slate_mask``: float32 ``(N,)`` — 1.0 for real slate items, 0.0 for padding.

    Image paths and string ids are NOT part of the observation (so the env
    stays usable headless); they are surfaced in ``info`` for renderers.

    Action (``Discrete(5 * N)``): see module docstring.
    """

    metadata = {"render_modes": []}

    def __init__(self, corpus: Corpus, config: Optional[ThumbwalkEnvConfig] = None) -> None:
        super().__init__()
        self.corpus = corpus
        self.config = config if config is not None else ThumbwalkEnvConfig()

        n_items = len(corpus)
        n_slots = self.config.slate_size
        embedding_dim = corpus.embedding_dim

        self.action_space = spaces.Discrete(N_ACTION_TYPES * n_slots)
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
            history_window=self.config.history_window,
            not_interested_factor=self.config.not_interested_factor,
            not_interested_floor=self.config.not_interested_floor,
            channel_suppression_factor=self.config.channel_suppression_factor,
            channel_suppression_floor=self.config.channel_suppression_floor,
            impression_decay=self.config.impression_decay,
            popularity_weight=self.config.popularity_weight,
            scorer=self.config.scorer,
        )

        self._step_count = 0
        self._current_seed_row: int = 0
        self._current_slate: np.ndarray = np.full(n_slots, -1, dtype=int)
        self._history: list[HistoryEntry] = []

    # -- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed if seed is not None else self.config.seed)
        self.funnel.rng = self.np_random
        self.funnel.reset_suppression()
        self._step_count = 0
        self._history = []

        options = options or {}
        if "start_row" in options:
            self._current_seed_row = int(options["start_row"])
        else:
            # Cold start: no watch history yet, so the first slate (and its
            # anchor seed) comes from popularity, Covington et al.'s
            # demographic-prior stand-in for new users.
            popularity_pick = self.corpus.sample_rows(
                1, rng=self.np_random, weights=self.corpus.popularity
            )
            self._current_seed_row = (
                int(popularity_pick[0]) if len(popularity_pick) else int(self.np_random.integers(0, len(self.corpus)))
            )

        self._rebuild_slate()
        obs = self._get_obs()
        info = self._get_info(clicked_id=None, action=None, invalid=False, context=None)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for action_space {self.action_space}")

        n_slots = self.config.slate_size
        action_type, position = divmod(int(action), n_slots)
        item_row = int(self._current_slate[position])
        invalid = item_row == -1

        seed_before = self._current_seed_row
        clicked_id: Optional[str] = None
        watch_fraction: Optional[float] = None
        is_watch = action_type in (ACTION_WATCH_ABANDON, ACTION_WATCH_HALF, ACTION_WATCH_FULL)
        is_not_interested = action_type == ACTION_NOT_INTERESTED
        is_no_channel = action_type == ACTION_NO_CHANNEL

        watched_row: Optional[int] = None
        if not invalid:
            if is_watch:
                watch_fraction = WATCH_FRACTIONS[action_type]
                self._current_seed_row = item_row
                clicked_id = self.corpus.id_at(item_row)
                self._history.append((item_row, watch_fraction))
                watched_row = item_row
            elif is_not_interested:
                self.funnel.apply_not_interested(item_row)
            elif is_no_channel:
                self.funnel.apply_no_channel(item_row)

        # Shown-but-not-watched demotion (Covington et al., sec. 4.1): every
        # item that was in the slate this step and wasn't the one watched
        # gets its impression counter bumped, regardless of action type.
        self.funnel.record_impressions(self._current_slate, watched_row)

        self._rebuild_slate()
        self._step_count += 1

        context = {
            "seed_id_before": self.corpus.id_at(seed_before),
            "action": action,
            "action_type": action_type,
            "position": position,
            "clicked_id": clicked_id,
            "watch_fraction": watch_fraction,
            "is_watch": bool(is_watch and not invalid),
            "is_not_interested": bool(is_not_interested and not invalid),
            "is_no_channel": bool(is_no_channel and not invalid),
            "invalid_action": invalid,
            "masked_count": len(self.funnel.suppressed_rows()),
        }
        reward = self.config.reward_fn(context)

        terminated = False
        if self.config.terminate_on_starvation:
            pool = self.funnel.candidate_generation(self._history, set())
            terminated = len(pool) < n_slots

        truncated = self._step_count >= self.config.horizon

        obs = self._get_obs()
        info = self._get_info(clicked_id=clicked_id, action=action, invalid=invalid, context=context)
        return obs, reward, terminated, truncated, info

    # -- helpers -------------------------------------------------------------

    def _rebuild_slate(self) -> None:
        candidates = self.funnel.candidate_generation(self._history, banned_rows=set())
        self._current_slate = self.funnel.rank(
            self._history, candidates, self.config.slate_size
        )

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
        self,
        clicked_id: Optional[str],
        action: Optional[int],
        invalid: bool,
        context: Optional[dict[str, Any]],
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
            "current_channel": self.corpus.channel_of_row(self._current_seed_row),
            "seed_path": self.corpus.path_of_row(self._current_seed_row),
            "slate_paths": slate_paths,
            "slate_ids": slate_string_ids,
            "history_length": len(self._history),
            "action": action,
            "action_type": None if context is None else context["action_type"],
            "watch_fraction": None if context is None else context["watch_fraction"],
            "invalid_action": invalid,
        }
