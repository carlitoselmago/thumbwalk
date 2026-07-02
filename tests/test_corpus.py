from __future__ import annotations

import numpy as np

from thumbwalk.corpus import Corpus
from tests.conftest import EMBEDDING_DIM, N_CLUSTERS, N_OUTLIERS, POINTS_PER_CLUSTER, write_synthetic_corpus_dir


def test_shapes_and_alignment(synthetic_corpus: Corpus) -> None:
    n_items = POINTS_PER_CLUSTER * N_CLUSTERS + N_OUTLIERS
    assert len(synthetic_corpus) == n_items
    assert synthetic_corpus.embeddings.shape == (n_items, EMBEDDING_DIM)
    assert synthetic_corpus.embedding_dim == EMBEDDING_DIM
    assert len(synthetic_corpus.ids) == n_items
    assert len(synthetic_corpus.paths) == n_items
    assert synthetic_corpus.kmeans_labels.shape == (n_items,)
    assert synthetic_corpus.hdbscan_labels.shape == (n_items,)
    for arr in synthetic_corpus.complexity.values():
        assert arr.shape == (n_items,)


def test_id_row_roundtrip(synthetic_corpus: Corpus) -> None:
    for row in range(len(synthetic_corpus)):
        item_id = synthetic_corpus.id_at(row)
        assert synthetic_corpus.row(item_id) == row


def test_hdbscan_outliers_present(synthetic_corpus: Corpus) -> None:
    n_outliers = int(np.sum(synthetic_corpus.hdbscan_labels == -1))
    assert n_outliers == N_OUTLIERS


def test_kneighbors_excludes_self(synthetic_corpus: Corpus) -> None:
    rows, distances = synthetic_corpus.kneighbors(0, k=5)
    assert 0 not in rows
    assert len(rows) == 5
    assert np.all(np.diff(distances) >= 0)  # sorted ascending


def test_load_round_trip(tmp_path) -> None:
    corpus_dir = write_synthetic_corpus_dir(tmp_path)
    loaded = Corpus.load(corpus_dir)

    n_items = POINTS_PER_CLUSTER * N_CLUSTERS + N_OUTLIERS
    assert len(loaded) == n_items
    assert loaded.ids[0] == "item_0000"
    assert loaded.manifest["embedder"] == "synthetic"
    np.testing.assert_array_equal(loaded.kmeans_labels.shape, (n_items,))
    np.testing.assert_array_equal(loaded.hdbscan_labels.shape, (n_items,))
    assert set(loaded.complexity.keys()) == {"entropy", "jpeg_ratio", "edge_density"}
