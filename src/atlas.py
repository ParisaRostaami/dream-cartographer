"""2D dream atlas: MDS projection, agglomerative regions, lexical names.

Clustering is performed in the SVD space (not in 2D) so that the atlas
coordinates remain a visualization of a higher-dimensional neighborhood
structure. Region names are the highest-mean word n-grams inside each
cluster — a transparent, if crude, stand-in for a topic label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

from .embed import RANDOM_STATE

MIN_CLUSTERS = 3
MAX_CLUSTERS = 6


@dataclass
class Atlas:
    coords: np.ndarray  # (n, 2)
    labels: np.ndarray  # (n,)
    region_names: dict[int, str]
    n_clusters: int
    silhouette: float


def _choose_k(X: np.ndarray) -> tuple[np.ndarray, int, float]:
    n = X.shape[0]
    k_max = min(MAX_CLUSTERS, n - 1)
    k_min = min(MIN_CLUSTERS, k_max)
    best_labels = None
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, metric="euclidean"))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        labels = AgglomerativeClustering(n_clusters=k_min, linkage="ward").fit_predict(X)
        return labels, k_min, float("nan")
    return best_labels, best_k, best_score


def _top_word_terms(
    tfidf,
    feature_names: np.ndarray,
    mask: np.ndarray,
    n_terms: int = 3,
) -> list[str]:
    if issparse(tfidf):
        centroid = np.asarray(tfidf[mask].mean(axis=0)).ravel()
    else:
        centroid = np.asarray(tfidf[mask], dtype=np.float64).mean(axis=0)
    order = np.argsort(centroid)[::-1]
    terms: list[str] = []
    for idx in order:
        name = str(feature_names[idx])
        if not name.startswith("word__"):
            continue
        token = name.split("word__", 1)[1].replace("_", " ").strip()
        if len(token) < 3 or token in terms:
            continue
        terms.append(token)
        if len(terms) >= n_terms:
            break
    return terms or ["unlabeled region"]


def name_regions(
    tfidf,
    feature_names: np.ndarray,
    labels: np.ndarray,
) -> dict[int, str]:
    names: dict[int, str] = {}
    for cluster_id in sorted(set(labels.tolist())):
        mask = labels == cluster_id
        terms = _top_word_terms(tfidf, feature_names, mask)
        names[int(cluster_id)] = " / ".join(terms)
    return names


def project_mds(X: np.ndarray) -> np.ndarray:
    """Metric MDS of SVD embeddings into the page plane."""
    n = X.shape[0]
    n_init = 4 if n > 4 else 1
    mds = MDS(
        n_components=2,
        metric_mds=True,
        metric="euclidean",
        n_init=n_init,
        init="random",
        max_iter=400,
        eps=1e-4,
        random_state=RANDOM_STATE,
        normalized_stress="auto",
    )
    return np.asarray(mds.fit_transform(X), dtype=np.float64)


def build_atlas(
    X: np.ndarray,
    tfidf,
    feature_names: np.ndarray,
) -> Atlas:
    labels, n_clusters, sil = _choose_k(X)
    coords = project_mds(X)
    names = name_regions(tfidf, feature_names, labels)
    return Atlas(
        coords=coords,
        labels=np.asarray(labels, dtype=int),
        region_names=names,
        n_clusters=n_clusters,
        silhouette=sil,
    )
