"""Loads a preprocessed corpus (embeddings, clusters, metadata) built by
``scripts/preprocess.py``.

The :class:`Corpus` is the read-only data layer that :mod:`thumbwalk.recommender`
and :mod:`thumbwalk.env` sit on top of. Internally everything is addressed by
integer row index (fast, array-friendly); string ``ids`` are the stable,
human/disk-facing identifiers and are only translated to/from row indices at
the edges (e.g. when surfacing values in ``info``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class Corpus:
    """In-memory view of a preprocessed corpus directory.

    Construct directly (e.g. for synthetic test corpora) or via
    :meth:`Corpus.load`.
    """

    embeddings: np.ndarray  # float32, shape (n, D)
    ids: list[str]  # length n, index-aligned with embeddings rows
    paths: list[str]  # length n, index-aligned, info-only (never fed to RL obs)
    kmeans_labels: np.ndarray  # int, shape (n,)
    hdbscan_labels: np.ndarray  # int, shape (n,), -1 = outlier
    complexity: dict[str, np.ndarray]  # "entropy" / "jpeg_ratio" / "edge_density", each (n,)
    manifest: dict

    _nn_index: Optional[NearestNeighbors] = field(default=None, repr=False, compare=False)
    _id_to_row: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.kmeans_labels = np.asarray(self.kmeans_labels)
        self.hdbscan_labels = np.asarray(self.hdbscan_labels)
        n = self.embeddings.shape[0]
        if len(self.ids) != n:
            raise ValueError(f"ids length {len(self.ids)} != n_items {n}")
        if len(self.paths) != n:
            raise ValueError(f"paths length {len(self.paths)} != n_items {n}")
        if self.kmeans_labels.shape[0] != n:
            raise ValueError("kmeans_labels length mismatch")
        if self.hdbscan_labels.shape[0] != n:
            raise ValueError("hdbscan_labels length mismatch")
        for name, arr in self.complexity.items():
            if len(arr) != n:
                raise ValueError(f"complexity[{name!r}] length mismatch")
        self._id_to_row = {item_id: row for row, item_id in enumerate(self.ids)}

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, corpus_dir: str | Path) -> "Corpus":
        """Load a corpus directory saved by ``scripts/preprocess.py``."""
        corpus_dir = Path(corpus_dir)

        embeddings = np.load(corpus_dir / "embeddings.npy")
        ids = json.loads((corpus_dir / "ids.json").read_text())
        paths = json.loads((corpus_dir / "paths.json").read_text())
        kmeans_labels = np.load(corpus_dir / "kmeans_labels.npy")
        hdbscan_labels = np.load(corpus_dir / "hdbscan_labels.npy")
        complexity_npz = np.load(corpus_dir / "complexity.npz")
        complexity = {key: complexity_npz[key] for key in complexity_npz.files}
        manifest = json.loads((corpus_dir / "manifest.json").read_text())

        return cls(
            embeddings=embeddings,
            ids=ids,
            paths=paths,
            kmeans_labels=kmeans_labels,
            hdbscan_labels=hdbscan_labels,
            complexity=complexity,
            manifest=manifest,
        )

    def save(self, corpus_dir: str | Path) -> None:
        """Save this corpus to ``corpus_dir`` in the layout :meth:`load` expects."""
        corpus_dir = Path(corpus_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        np.save(corpus_dir / "embeddings.npy", self.embeddings)
        (corpus_dir / "ids.json").write_text(json.dumps(list(self.ids)))
        (corpus_dir / "paths.json").write_text(json.dumps(list(self.paths)))
        np.save(corpus_dir / "kmeans_labels.npy", self.kmeans_labels)
        np.save(corpus_dir / "hdbscan_labels.npy", self.hdbscan_labels)
        np.savez(corpus_dir / "complexity.npz", **self.complexity)
        (corpus_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    # -- basic accessors ----------------------------------------------------

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    def row(self, item_id: str) -> int:
        """Row index for a stable item id."""
        return self._id_to_row[item_id]

    def id_at(self, row: int) -> str:
        """Stable item id for a row index."""
        return self.ids[row]

    def embedding_by_row(self, row: int) -> np.ndarray:
        return self.embeddings[row]

    def cluster_of_row(self, row: int, labeling: str = "kmeans") -> int:
        """Cluster label for a row. ``labeling`` is ``"kmeans"`` or ``"hdbscan"``."""
        if labeling == "kmeans":
            return int(self.kmeans_labels[row])
        if labeling == "hdbscan":
            return int(self.hdbscan_labels[row])
        raise ValueError(f"unknown labeling {labeling!r}")

    def path_of_row(self, row: int) -> str:
        return self.paths[row]

    # -- nearest neighbors ----------------------------------------------------

    def ensure_nn_index(self, metric: str = "euclidean") -> NearestNeighbors:
        """Lazily build (and cache) a k-NN index over all embeddings."""
        if self._nn_index is None:
            index = NearestNeighbors(metric=metric)
            index.fit(self.embeddings)
            self._nn_index = index
        return self._nn_index

    def kneighbors(self, row: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``k`` nearest neighbor rows of ``row``, excluding itself.

        Returns ``(neighbor_rows, distances)``, both length ``min(k, n-1)``,
        sorted by ascending distance.
        """
        index = self.ensure_nn_index()
        k_query = min(k + 1, len(self))
        distances, rows = index.kneighbors(
            self.embeddings[row : row + 1], n_neighbors=k_query
        )
        rows, distances = rows[0], distances[0]
        keep = rows != row
        return rows[keep][:k], distances[keep][:k]

    def sample_rows(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        exclude: Optional[set[int]] = None,
    ) -> np.ndarray:
        """Uniformly sample up to ``n`` distinct row indices, excluding ``exclude``."""
        rng = rng if rng is not None else np.random.default_rng()
        exclude = exclude or set()
        pool = np.array([r for r in range(len(self)) if r not in exclude])
        if len(pool) == 0:
            return np.array([], dtype=int)
        n = min(n, len(pool))
        return rng.choice(pool, size=n, replace=False)
