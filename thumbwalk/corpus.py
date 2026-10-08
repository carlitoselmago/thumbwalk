"""Loads a preprocessed corpus (embeddings, clusters, metadata) built by
``scripts/preprocess.py``.

The :class:`Corpus` is the read-only data layer that :mod:`thumbwalk.recommender`
and :mod:`thumbwalk.env` sit on top of. Internally everything is addressed by
integer row index (fast, array-friendly); string ``ids`` are the stable,
human/disk-facing identifiers and are only translated to/from row indices at
the edges (e.g. when surfacing values in ``info``).

Two fields exist specifically to let :class:`thumbwalk.recommender.TwoStageFunnel`
reproduce the parts of Covington, Adams & Sargin's YouTube recommender
(2016) that the environment's design (TFM, cap. 3, "Entorno de simulación")
relies on:

- ``channels``: which channel each item belongs to. Needed for the
  "Don't recommend channel" action, which must suppress a whole channel rather
  than just one item's k-NN neighborhood (the "Not interested" action).
  Defaults to one channel per item (no grouping) when the source dataset
  doesn't carry channel metadata, so channel-suppression degenerates
  gracefully to single-item suppression instead of erroring.
- ``popularity``: a non-negative weight per item used as the "demographic
  prior" stand-in for cold start (no watch history yet) and as a ranking
  term, mirroring Covington et al.'s use of demographic features when a
  user has no history. Defaults to uniform (``1.0`` for every item) when
  the dataset carries no real popularity signal.
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

    # Optional (default-filled in __post_init__ when not supplied): see module docstring.
    channels: Optional[list[str]] = None  # length n, index-aligned
    popularity: Optional[np.ndarray] = None  # non-negative float, shape (n,)

    _nn_index: Optional[NearestNeighbors] = field(default=None, repr=False, compare=False)
    _id_to_row: dict[str, int] = field(init=False, repr=False, compare=False)
    _channel_to_rows: dict[str, np.ndarray] = field(init=False, repr=False, compare=False)

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

        if self.channels is None:
            # No channel metadata available: every item is its own channel, so
            # "Don't recommend channel" degenerates to suppressing just that item.
            self.channels = list(self.ids)
        elif len(self.channels) != n:
            raise ValueError(f"channels length {len(self.channels)} != n_items {n}")

        if self.popularity is None:
            self.popularity = np.ones(n, dtype=np.float64)
        else:
            self.popularity = np.asarray(self.popularity, dtype=np.float64)
            if self.popularity.shape[0] != n:
                raise ValueError("popularity length mismatch")
            if np.any(self.popularity < 0):
                raise ValueError("popularity must be non-negative")

        self._id_to_row = {item_id: row for row, item_id in enumerate(self.ids)}
        channel_rows: dict[str, list[int]] = {}
        for row, channel in enumerate(self.channels):
            channel_rows.setdefault(channel, []).append(row)
        self._channel_to_rows = {
            channel: np.array(rows, dtype=int) for channel, rows in channel_rows.items()
        }

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

        channels_path = corpus_dir / "channels.json"
        channels = json.loads(channels_path.read_text()) if channels_path.exists() else None

        popularity_path = corpus_dir / "popularity.npy"
        popularity = np.load(popularity_path) if popularity_path.exists() else None

        return cls(
            embeddings=embeddings,
            ids=ids,
            paths=paths,
            kmeans_labels=kmeans_labels,
            hdbscan_labels=hdbscan_labels,
            complexity=complexity,
            manifest=manifest,
            channels=channels,
            popularity=popularity,
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
        (corpus_dir / "channels.json").write_text(json.dumps(list(self.channels)))
        np.save(corpus_dir / "popularity.npy", self.popularity)

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

    # -- channels -------------------------------------------------------------

    def channel_of_row(self, row: int) -> str:
        return self.channels[row]

    def rows_of_channel(self, channel: str) -> np.ndarray:
        """All row indices belonging to ``channel`` (empty array if unknown)."""
        return self._channel_to_rows.get(channel, np.array([], dtype=int))

    @property
    def n_channels(self) -> int:
        return len(self._channel_to_rows)

    # -- popularity -------------------------------------------------------------

    def popularity_of_row(self, row: int) -> float:
        return float(self.popularity[row])

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
        return self.kneighbors_of_vector(self.embeddings[row], k, exclude={row})

    def kneighbors_of_vector(
        self, embedding: np.ndarray, k: int, exclude: Optional[set[int]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``k`` nearest neighbor rows of an arbitrary embedding vector.

        Unlike :meth:`kneighbors`, ``embedding`` need not be one of the
        corpus's own rows — used to retrieve candidates around a *profile*
        vector (e.g. a weighted average of recently watched items) rather
        than a single seed item. ``exclude`` optionally drops specific rows
        (e.g. the row the vector was itself computed from).
        """
        exclude = exclude or set()
        index = self.ensure_nn_index()
        k_query = min(k + len(exclude) + 1, len(self))
        distances, rows = index.kneighbors(
            np.asarray(embedding, dtype=np.float32).reshape(1, -1), n_neighbors=k_query
        )
        rows, distances = rows[0], distances[0]
        keep = np.array([r not in exclude for r in rows])
        return rows[keep][:k], distances[keep][:k]

    def sample_rows(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        exclude: Optional[set[int]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample up to ``n`` distinct row indices, excluding ``exclude``.

        Uniform by default; pass ``weights`` (e.g. ``corpus.popularity``) to
        sample proportionally to it instead (used for the popularity-based
        cold start, see :mod:`thumbwalk.recommender`).
        """
        rng = rng if rng is not None else np.random.default_rng()
        exclude = exclude or set()
        pool = np.array([r for r in range(len(self)) if r not in exclude])
        if len(pool) == 0:
            return np.array([], dtype=int)
        n = min(n, len(pool))
        if weights is None:
            return rng.choice(pool, size=n, replace=False)
        pool_weights = np.asarray(weights, dtype=np.float64)[pool]
        total = pool_weights.sum()
        if total <= 0:
            return rng.choice(pool, size=n, replace=False)
        probs = pool_weights / total
        return rng.choice(pool, size=n, replace=False, p=probs)
