"""thumbwalk: a Gymnasium environment where the agent is the *viewer*.

Unlike RecSim (Google, 2019) and SARDINE (Naver, 2024), where the learnable
agent IS the recommender proposing slates to a simulated user, here the
agent is the VIEWER: it sees one "playing" thumbnail (the seed) and a slate
of related thumbnails produced by a fixed, non-learnable recommender, and
chooses what to click or dismiss. This inversion is the whole point of the
package.
"""

from thumbwalk.corpus import Corpus
from thumbwalk.recommender import TwoStageFunnel
from thumbwalk.click_model import ClickModel, DeterministicNearest, SoftmaxRelevance
from thumbwalk.env import ThumbwalkEnv, ThumbwalkEnvConfig

__version__ = "0.1.0"

__all__ = [
    "Corpus",
    "TwoStageFunnel",
    "ClickModel",
    "DeterministicNearest",
    "SoftmaxRelevance",
    "ThumbwalkEnv",
    "ThumbwalkEnvConfig",
]
