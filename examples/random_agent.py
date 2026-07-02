#!/usr/bin/env python
"""Run a random-clicking VIEWER agent against ThumbwalkEnv and print its trajectory.

    python examples/random_agent.py                      # in-memory synthetic corpus
    python examples/random_agent.py --corpus-dir corpus/  # a real preprocessed corpus

This is a smoke test / usage example, not a baseline worth learning from: the
agent samples actions uniformly at random from ``Discrete(2 * slate_size)``,
so roughly half its actions click a slate item and half mark one "not
interested". Watch the printed trajectory to see the seed wander the
k-NN graph and the masked-item count grow as dislikes accumulate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thumbwalk.corpus import Corpus  # noqa: E402
from thumbwalk.env import ThumbwalkEnv, ThumbwalkEnvConfig  # noqa: E402


def build_demo_corpus(seed: int = 0) -> Corpus:
    """A small in-memory corpus with a few planted clusters, for zero-setup demos."""
    rng = np.random.default_rng(seed)
    n_clusters, points_per_cluster, dim = 5, 12, 8

    points, labels = make_blobs(
        n_samples=points_per_cluster * n_clusters,
        centers=n_clusters,
        n_features=dim,
        cluster_std=1.0,
        center_box=(-20.0, 20.0),
        random_state=seed,
    )
    n = points.shape[0]

    return Corpus(
        embeddings=points.astype(np.float32),
        ids=[f"item_{i:04d}" for i in range(n)],
        paths=[f"/demo/item_{i:04d}.jpg" for i in range(n)],
        kmeans_labels=labels.astype(int),
        hdbscan_labels=labels.astype(int),
        complexity={
            "entropy": rng.uniform(2.0, 8.0, size=n).astype(np.float32),
            "jpeg_ratio": rng.uniform(0.1, 0.9, size=n).astype(np.float32),
            "edge_density": rng.uniform(0.0, 1.0, size=n).astype(np.float32),
        },
        manifest={
            "version": 1,
            "n_items": n,
            "embedding_dim": dim,
            "embedder": "synthetic-demo",
            "embedder_model_name": "synthetic-demo",
            "k": 5,
            "kmeans_k": n_clusters,
            "kmeans_silhouette": None,
            "hdbscan_params": {},
            "hdbscan_outlier_fraction": 0.0,
            "created_at": "demo",
            "source_dir": "in-memory",
            "image_extensions": [],
        },
    )


def describe_action(action: int, slate_size: int) -> str:
    if action < slate_size:
        return f"click(slate[{action}])"
    return f"not_interested(slate[{action - slate_size}])"


def run_episode(env: ThumbwalkEnv, seed: int) -> None:
    slate_size = env.config.slate_size
    obs, info = env.reset(seed=seed)
    print(f"step  0  reset -> seed={info['seed_path']} cluster={info['current_cluster']}")

    step = 0
    while True:
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        print(
            f"step {step:2d}  {describe_action(action, slate_size):<24s} "
            f"-> clicked={info['clicked_id']!s:<12} "
            f"masked={info['masked_count']:3d}  cluster={info['current_cluster']}  "
            f"reward={reward:.2f}  invalid={info['invalid_action']}"
        )
        if terminated or truncated:
            print(f"episode ended: terminated={terminated} truncated={truncated}")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=None, help="a corpus dir from scripts/preprocess.py")
    parser.add_argument("--slate-size", type=int, default=6)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.corpus_dir is not None:
        corpus = Corpus.load(args.corpus_dir)
        print(f"loaded corpus from {args.corpus_dir}: {len(corpus)} items")
    else:
        corpus = build_demo_corpus(seed=args.seed)
        print(f"using in-memory synthetic demo corpus: {len(corpus)} items")

    config = ThumbwalkEnvConfig(
        slate_size=args.slate_size, k=args.k, horizon=args.horizon, exploration_slots=1
    )
    env = ThumbwalkEnv(corpus, config)
    run_episode(env, seed=args.seed)


if __name__ == "__main__":
    main()
