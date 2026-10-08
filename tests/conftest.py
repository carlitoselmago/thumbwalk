"""Shared synthetic-corpus fixture used by test_corpus / test_recommender / test_env.

The fixture plants 4 well-separated Gaussian clusters plus a handful of
outlier points, so tests can exercise clustering-aware behavior (dislike
masking, k-NN retrieval) fast and deterministically, without touching real
images or the heavy optional embedder/clustering stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from thumbwalk.corpus import Corpus

N_CLUSTERS = 4
POINTS_PER_CLUSTER = 15
N_OUTLIERS = 5
EMBEDDING_DIM = 8
FIXTURE_SEED = 0


def build_synthetic_corpus(seed: int = FIXTURE_SEED) -> Corpus:
    """Build a small synthetic :class:`Corpus` with planted clusters + outliers.

    Cluster labels (kmeans and hdbscan) are assigned directly from known blob
    membership rather than fit, so the fixture is fully deterministic and
    doesn't depend on KMeans/HDBSCAN convergence behavior — clustering
    *correctness* is scripts/preprocess.py's concern, not the env/recommender's.
    """
    rng = np.random.default_rng(seed)

    cluster_points, kmeans_labels = make_blobs(
        n_samples=POINTS_PER_CLUSTER * N_CLUSTERS,
        centers=N_CLUSTERS,
        n_features=EMBEDDING_DIM,
        cluster_std=1.0,
        center_box=(-20.0, 20.0),
        random_state=seed,
    )

    # Outliers: uniform in a box well outside the cluster region.
    outlier_points = rng.uniform(low=-60.0, high=60.0, size=(N_OUTLIERS, EMBEDDING_DIM))

    embeddings = np.vstack([cluster_points, outlier_points]).astype(np.float32)
    n_items = embeddings.shape[0]

    hdbscan_labels = np.concatenate(
        [kmeans_labels, np.full(N_OUTLIERS, -1, dtype=int)]
    )
    # For kmeans (which never emits -1), assign outliers to their nearest
    # cluster center so every row still has a valid kmeans cluster id.
    centers = np.array(
        [cluster_points[kmeans_labels == c].mean(axis=0) for c in range(N_CLUSTERS)]
    )
    outlier_kmeans = np.array(
        [
            int(np.argmin(np.linalg.norm(centers - p, axis=1)))
            for p in outlier_points
        ]
    )
    kmeans_labels_full = np.concatenate([kmeans_labels, outlier_kmeans]).astype(int)

    ids = [f"item_{i:04d}" for i in range(n_items)]
    paths = [f"/fake/item_{i:04d}.jpg" for i in range(n_items)]
    # A handful of multi-item "channels" (groups of consecutive rows) plus
    # some singleton channels, so channel-scoped suppression ("No recomendar
    # canal") has something non-trivial to exercise in tests.
    channels = [f"channel_{i // 4:03d}" for i in range(n_items)]
    popularity = rng.uniform(0.1, 1.0, size=n_items).astype(np.float64)

    complexity = {
        "entropy": rng.uniform(2.0, 8.0, size=n_items).astype(np.float32),
        "jpeg_ratio": rng.uniform(0.1, 0.9, size=n_items).astype(np.float32),
        "edge_density": rng.uniform(0.0, 1.0, size=n_items).astype(np.float32),
    }

    manifest = {
        "version": 1,
        "n_items": n_items,
        "embedding_dim": EMBEDDING_DIM,
        "embedder": "synthetic",
        "embedder_model_name": "synthetic-fixture",
        "k": 5,
        "kmeans_k": N_CLUSTERS,
        "kmeans_silhouette": None,
        "hdbscan_params": {},
        "hdbscan_outlier_fraction": N_OUTLIERS / n_items,
        "created_at": "2026-01-01T00:00:00Z",
        "source_dir": "synthetic",
        "image_extensions": [".jpg"],
    }

    return Corpus(
        embeddings=embeddings,
        ids=ids,
        paths=paths,
        kmeans_labels=kmeans_labels_full,
        hdbscan_labels=hdbscan_labels,
        complexity=complexity,
        manifest=manifest,
        channels=channels,
        popularity=popularity,
    )


def write_synthetic_corpus_dir(tmp_path: Path, seed: int = FIXTURE_SEED) -> Path:
    """Build a synthetic corpus and save it to disk, mirroring preprocess.py's output."""
    corpus = build_synthetic_corpus(seed=seed)
    corpus_dir = tmp_path / "synthetic_corpus"
    corpus.save(corpus_dir)
    return corpus_dir


@pytest.fixture
def synthetic_corpus() -> Corpus:
    return build_synthetic_corpus()
