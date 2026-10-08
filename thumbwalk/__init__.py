"""thumbwalk: a Gymnasium environment where the agent is the *viewer*.

Unlike RecSim (Google, 2019) and SARDINE (Naver, 2024), where the learnable
agent IS the recommender proposing slates to a simulated user, here the
agent is the VIEWER: it sees one "playing" thumbnail (the seed) and a slate
of related thumbnails produced by a fixed, non-learnable recommender, and
chooses what to watch (with a watch-time fraction) or mark as negative
feedback ("Not interested" / "Don't recommend channel"). This inversion is the
whole point of the package.
"""

from thumbwalk.corpus import Corpus
from thumbwalk.recommender import HistoryEntry, Scorer, TwoStageFunnel, profile_embedding
from thumbwalk.click_model import ClickModel, DeterministicNearest, SoftmaxRelevance
from thumbwalk.env import (
    ACTION_NO_CHANNEL,
    ACTION_NOT_INTERESTED,
    ACTION_WATCH_ABANDON,
    ACTION_WATCH_FULL,
    ACTION_WATCH_HALF,
    N_ACTION_TYPES,
    WATCH_FRACTIONS,
    ThumbwalkEnv,
    ThumbwalkEnvConfig,
)

__version__ = "0.1.0"

__all__ = [
    "Corpus",
    "HistoryEntry",
    "Scorer",
    "TwoStageFunnel",
    "profile_embedding",
    "ClickModel",
    "DeterministicNearest",
    "SoftmaxRelevance",
    "ThumbwalkEnv",
    "ThumbwalkEnvConfig",
    "WATCH_FRACTIONS",
    "N_ACTION_TYPES",
    "ACTION_WATCH_ABANDON",
    "ACTION_WATCH_HALF",
    "ACTION_WATCH_FULL",
    "ACTION_NOT_INTERESTED",
    "ACTION_NO_CHANNEL",
]
