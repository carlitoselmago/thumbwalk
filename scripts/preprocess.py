#!/usr/bin/env python
"""Turn a folder of images into a clustered, analyzed corpus for ThumbwalkEnv.

    python scripts/preprocess.py --input <folder> --output <corpus_dir> \
        [--embedder clip|autoencoder|resnet] [--k 15]

Pipeline:
  1. Recursively load all images under ``--input``.
  2. Embed each image with a swappable :class:`Embedder` (CLIP by default if
     available, else a ResNet penultimate layer, else a zero-dependency
     PCA "linear autoencoder" — always usable). Also compute cheap,
     embedder-independent complexity signals per image (Shannon entropy,
     a JPEG-compression-ratio proxy, edge density) used only for the
     variety/noise analysis below, never fed into the RL observation.
  3. Cluster embeddings with KMeans (auto-k via a silhouette sweep) AND
     HDBSCAN (outliers labeled -1); both labelings are kept.
  4. Build a k-NN index over the embeddings (lazily, via Corpus).
  5. Save everything to ``--output`` in the layout ``thumbwalk.corpus.Corpus``
     expects, plus human-facing analysis plots/montages/summary under
     ``<output>/analysis/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import shannon_entropy
from skimage.feature import canny
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thumbwalk.corpus import Corpus  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


# ============================================================================
# Image discovery / loading
# ============================================================================


def discover_images(input_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no images with extensions {IMAGE_EXTENSIONS} found under {input_dir}")
    return paths


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# ============================================================================
# Embedders (swappable interface)
# ============================================================================


class Embedder(ABC):
    """Interface for turning a batch of RGB images into embedding vectors."""

    model_name: str = "unknown"

    def fit(self, images: list[Image.Image]) -> None:
        """Optional: fit on the full image set before embedding. No-op for pretrained models."""

    @abstractmethod
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Return an ``(len(images), embedding_dim)`` float32 array."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...


class ClipEmbedder(Embedder):
    """CLIP image embeddings via open_clip (preferred) or transformers."""

    def __init__(self) -> None:
        self._backend = None
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model.eval()
            self._backend = "open_clip"
            self._model = model
            self._preprocess = preprocess
            self._torch = torch
            self.model_name = "open_clip/ViT-B-32/openai"
            self._dim = model.visual.output_dim
        except ImportError:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                model_name = "openai/clip-vit-base-patch32"
                self._model = CLIPModel.from_pretrained(model_name)
                self._model.eval()
                self._processor = CLIPProcessor.from_pretrained(model_name)
                self._backend = "transformers"
                self._torch = torch
                self.model_name = f"transformers/{model_name}"
                self._dim = self._model.config.projection_dim
            except ImportError as exc:
                raise ImportError(
                    "CLIP embedder requires open_clip_torch or transformers+torch"
                ) from exc

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            if self._backend == "open_clip":
                batch = torch.stack([self._preprocess(img) for img in images])
                features = self._model.encode_image(batch)
            else:
                inputs = self._processor(images=images, return_tensors="pt")
                features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)


class ResNetEmbedder(Embedder):
    """torchvision ResNet18 penultimate-layer features (fc replaced with identity)."""

    def __init__(self) -> None:
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
        except ImportError as exc:
            raise ImportError("ResNet embedder requires torch and torchvision") from exc

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        self._dim = model.fc.in_features
        model.fc = nn.Identity()
        model.eval()

        self._model = model
        self._preprocess = weights.transforms()
        self._torch = torch
        self.model_name = "torchvision/resnet18"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            batch = torch.stack([self._preprocess(img) for img in images])
            features = self._model(batch)
        return features.cpu().numpy().astype(np.float32)


class AutoencoderEmbedder(Embedder):
    """Zero-extra-dependency fallback: PCA ("linear autoencoder") over resized pixels.

    Always available (only needs numpy/sklearn/pillow, already core
    dependencies), so the preprocessing pipeline can always run end-to-end
    even with no deep-learning stack installed.
    """

    THUMB_SIZE = (48, 48)

    def __init__(self, embedding_dim: int = 64) -> None:
        self._requested_dim = embedding_dim
        self._pca = None
        self.model_name = "pca-linear-autoencoder"

    @property
    def embedding_dim(self) -> int:
        if self._pca is None:
            return self._requested_dim
        return self._pca.n_components_

    def _flatten(self, images: list[Image.Image]) -> np.ndarray:
        arrays = [
            np.asarray(img.resize(self.THUMB_SIZE), dtype=np.float32).ravel() / 255.0
            for img in images
        ]
        return np.stack(arrays)

    def fit(self, images: list[Image.Image]) -> None:
        from sklearn.decomposition import PCA

        flattened = self._flatten(images)
        n_components = min(self._requested_dim, flattened.shape[0] - 1, flattened.shape[1])
        n_components = max(n_components, 1)
        self._pca = PCA(n_components=n_components, random_state=0)
        self._pca.fit(flattened)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("AutoencoderEmbedder.fit() must be called before embed()")
        return self._pca.transform(self._flatten(images)).astype(np.float32)


def get_embedder(name: str) -> Embedder:
    """Build the requested embedder, falling back down the chain clip -> resnet ->
    autoencoder (with a warning) if a preferred backend's dependencies are missing,
    so the script always runs end-to-end."""
    chain = {"clip": ClipEmbedder, "resnet": ResNetEmbedder, "autoencoder": AutoencoderEmbedder}
    if name not in chain:
        raise ValueError(f"unknown embedder {name!r}, choose from {list(chain)}")

    order = ["clip", "resnet", "autoencoder"]
    start = order.index(name)
    for candidate in order[start:]:
        try:
            return chain[candidate]()
        except ImportError as exc:
            warnings.warn(f"embedder {candidate!r} unavailable ({exc}); falling back")
    raise RuntimeError("no embedder available (this should be unreachable, autoencoder has no deps)")


# ============================================================================
# Complexity signals (independent of the deep embedder)
# ============================================================================


def compute_complexity_signals(images: list[Image.Image], paths: list[Path]) -> dict[str, np.ndarray]:
    entropy = np.zeros(len(images), dtype=np.float32)
    jpeg_ratio = np.zeros(len(images), dtype=np.float32)
    edge_density = np.zeros(len(images), dtype=np.float32)

    for i, (img, path) in enumerate(zip(images, paths)):
        gray = np.asarray(img.convert("L"))
        entropy[i] = shannon_entropy(gray)

        raw_bytes = gray.shape[0] * gray.shape[1] * 3
        file_bytes = path.stat().st_size
        jpeg_ratio[i] = raw_bytes / max(file_bytes, 1)  # higher = compresses well = simpler image

        edges = canny(gray.astype(np.float64) / 255.0)
        edge_density[i] = float(edges.mean())

    return {"entropy": entropy, "jpeg_ratio": jpeg_ratio, "edge_density": edge_density}


# ============================================================================
# Clustering
# ============================================================================


def auto_kmeans(embeddings: np.ndarray, k_max: int = 10) -> tuple[np.ndarray, int, Optional[float]]:
    """KMeans with k chosen by a silhouette sweep over ``2..k_max``."""
    n = embeddings.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < 2:
        return np.zeros(n, dtype=int), 1, None

    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        best_labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(embeddings)
        best_k, best_score = 2, None
    return best_labels, best_k, best_score


def run_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5) -> tuple[np.ndarray, dict]:
    """HDBSCAN clustering (-1 = outlier). Uses sklearn's built-in implementation
    (scikit-learn >= 1.3) if present, else the standalone ``hdbscan`` package,
    else falls back to "everything is an outlier" with a warning."""
    params = {"min_cluster_size": min_cluster_size}
    try:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(embeddings)
        return labels, params
    except ImportError:
        pass

    try:
        import hdbscan as hdbscan_pkg

        labels = hdbscan_pkg.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(embeddings)
        return labels, params
    except ImportError:
        warnings.warn("HDBSCAN unavailable (needs scikit-learn>=1.3 or the hdbscan package); "
                       "marking all items as outliers")
        return np.full(embeddings.shape[0], -1, dtype=int), params


# ============================================================================
# Analysis outputs
# ============================================================================


def project_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(random_state=0)
        return reducer.fit_transform(embeddings), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE

        n = embeddings.shape[0]
        perplexity = min(30, max(2, n - 1))
        reducer = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        return reducer.fit_transform(embeddings), "t-SNE"


def make_montage(images: list[Image.Image], labels: list[str], cell_size: int = 96, cols: int = 4) -> Image.Image:
    n = len(images)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    montage = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(30, 30, 30))
    draw = ImageDraw.Draw(montage)

    for i, (img, label) in enumerate(zip(images, labels)):
        thumb = img.resize((cell_size, cell_size))
        x, y = (i % cols) * cell_size, (i // cols) * cell_size
        montage.paste(thumb, (x, y))
        draw.rectangle([x, y, x + cell_size - 1, y + 14], fill=(0, 0, 0))
        draw.text((x + 2, y + 2), label, fill=(255, 255, 0))

    return montage


def write_analysis_outputs(
    analysis_dir: Path,
    embeddings: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    complexity: dict[str, np.ndarray],
    images: list[Image.Image],
    ids: list[str],
    kmeans_k: int,
    silhouette: Optional[float],
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir.mkdir(parents=True, exist_ok=True)
    n = embeddings.shape[0]

    # --- 2D projection colored by cluster ---
    coords, method = project_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=kmeans_labels, cmap="tab10", s=25)
    ax.set_title(f"{method} projection colored by KMeans cluster (k={kmeans_k})")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.savefig(analysis_dir / "projection.png", dpi=120)
    plt.close(fig)

    # --- entropy histogram ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(complexity["entropy"], bins=30, color="steelblue")
    ax.set_xlabel("Shannon entropy")
    ax.set_ylabel("count")
    ax.set_title("Entropy distribution")
    fig.savefig(analysis_dir / "entropy_histogram.png", dpi=120)
    plt.close(fig)

    # --- entropy vs. distance-to-nearest-neighbor, outliers flagged ---
    nn_index = NearestNeighbors(n_neighbors=2).fit(embeddings)
    nn_distances, _ = nn_index.kneighbors(embeddings)
    nn_distance = nn_distances[:, 1]  # column 0 is distance to self (0.0)

    is_outlier = hdbscan_labels == -1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        complexity["entropy"][~is_outlier], nn_distance[~is_outlier],
        s=20, c="steelblue", label="inlier",
    )
    ax.scatter(
        complexity["entropy"][is_outlier], nn_distance[is_outlier],
        s=40, c="crimson", marker="x", label="HDBSCAN outlier",
    )
    ax.set_xlabel("Shannon entropy")
    ax.set_ylabel("distance to nearest neighbor")
    ax.set_title("Noise surfacing: entropy vs. isolation")
    ax.legend()
    fig.savefig(analysis_dir / "entropy_vs_nn_distance.png", dpi=120)
    plt.close(fig)

    # --- per-cluster contact sheets ---
    rng = np.random.default_rng(0)
    for cluster_id in sorted(set(int(c) for c in kmeans_labels)):
        rows = np.where(kmeans_labels == cluster_id)[0]
        sample = rng.choice(rows, size=min(9, len(rows)), replace=False)
        montage = make_montage(
            [images[i] for i in sample], [ids[i] for i in sample], cols=3
        )
        montage.save(analysis_dir / f"cluster_montage_{cluster_id}.png")

    # --- noisiest / outlier montage ---
    def zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / std if std > 0 else np.zeros_like(x)

    noise_score = zscore(complexity["entropy"]) + zscore(nn_distance)
    top_n = min(12, n)
    noisiest = np.argsort(-noise_score)[:top_n]
    labels = [
        f"{ids[i]}{' OUT' if hdbscan_labels[i] == -1 else ''}" for i in noisiest
    ]
    montage = make_montage([images[i] for i in noisiest], labels, cols=4)
    montage.save(analysis_dir / "outlier_montage.png")

    # --- summary ---
    n_outliers = int(is_outlier.sum())
    summary = {
        "n_images": n,
        "n_kmeans_clusters": int(kmeans_k),
        "kmeans_silhouette": silhouette,
        "n_hdbscan_outliers": n_outliers,
        "hdbscan_outlier_fraction": n_outliers / n,
        "entropy_min": float(complexity["entropy"].min()),
        "entropy_median": float(np.median(complexity["entropy"])),
        "entropy_max": float(complexity["entropy"].max()),
        "projection_method": method,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ============================================================================
# Main
# ============================================================================


def build_corpus(
    input_dir: Path,
    embedder_name: str,
    k: int,
    batch_size: int = 32,
) -> tuple[Corpus, list[Image.Image]]:
    image_paths = discover_images(input_dir)
    print(f"found {len(image_paths)} images under {input_dir}")

    images = [load_rgb(p) for p in image_paths]

    embedder = get_embedder(embedder_name)
    print(f"using embedder: {embedder.model_name}")
    embedder.fit(images)

    chunks = []
    for start in range(0, len(images), batch_size):
        chunks.append(embedder.embed(images[start : start + batch_size]))
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)

    complexity = compute_complexity_signals(images, image_paths)

    kmeans_labels, kmeans_k, silhouette = auto_kmeans(embeddings)
    hdbscan_labels, hdbscan_params = run_hdbscan(embeddings)

    ids = [f"item_{i:06d}" for i in range(len(image_paths))]
    paths = [str(p) for p in image_paths]

    n_outliers = int((hdbscan_labels == -1).sum())
    manifest = {
        "version": 1,
        "n_items": len(image_paths),
        "embedding_dim": embeddings.shape[1],
        "embedder": embedder_name,
        "embedder_model_name": embedder.model_name,
        "k": k,
        "kmeans_k": kmeans_k,
        "kmeans_silhouette": silhouette,
        "hdbscan_params": hdbscan_params,
        "hdbscan_outlier_fraction": n_outliers / len(image_paths),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(input_dir),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
    }

    corpus = Corpus(
        embeddings=embeddings,
        ids=ids,
        paths=paths,
        kmeans_labels=kmeans_labels,
        hdbscan_labels=hdbscan_labels,
        complexity=complexity,
        manifest=manifest,
    )
    return corpus, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="folder of images (searched recursively)")
    parser.add_argument("--output", required=True, type=Path, help="output corpus directory")
    parser.add_argument(
        "--embedder", choices=["clip", "autoencoder", "resnet"], default="clip",
        help="embedder backend (default: clip, falls back to resnet then autoencoder if unavailable)",
    )
    parser.add_argument("--k", type=int, default=15, help="k for the corpus's default candidate-generation neighbor count")
    args = parser.parse_args()

    corpus, images = build_corpus(args.input, args.embedder, args.k)
    corpus.save(args.output)
    print(f"saved corpus ({len(corpus)} items, dim={corpus.embedding_dim}) to {args.output}")

    analysis_dir = args.output / "analysis"
    summary = write_analysis_outputs(
        analysis_dir,
        corpus.embeddings,
        corpus.kmeans_labels,
        corpus.hdbscan_labels,
        corpus.complexity,
        images,
        corpus.ids,
        corpus.manifest["kmeans_k"],
        corpus.manifest["kmeans_silhouette"],
    )

    print(json.dumps(summary, indent=2))
    print(f"analysis outputs written to {analysis_dir}")


if __name__ == "__main__":
    main()
