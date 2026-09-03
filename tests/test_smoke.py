"""Behavioral smoke tests for the dream atlas pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atlas import build_atlas
from src.embed import SVD_DIM, embed_dreams, load_dreams
from src.recurrence import motif_graph, motifs_in_text

DATA = ROOT / "data" / "dreams.jsonl"


def test_embedding_dim_in_requested_band() -> None:
    dreams = load_dreams(DATA)
    texts = [d["text"] for d in dreams]
    X, pipe, tfidf, names = embed_dreams(texts, svd_dim=SVD_DIM)
    assert X.shape[0] == len(dreams)
    assert 16 <= X.shape[1] <= 32
    assert X.shape[1] == SVD_DIM
    assert np.isfinite(X).all()
    # L2-normalized rows
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
    assert tfidf.shape[0] == len(dreams)
    assert len(names) == tfidf.shape[1]
    assert pipe.named_steps["svd"].n_components == SVD_DIM


def test_cluster_count_at_least_three() -> None:
    dreams = load_dreams(DATA)
    texts = [d["text"] for d in dreams]
    X, _pipe, tfidf, names = embed_dreams(texts)
    atlas = build_atlas(X, tfidf, names)
    n_clusters = len(set(atlas.labels.tolist()))
    assert n_clusters >= 3
    assert atlas.n_clusters == n_clusters
    assert atlas.coords.shape == (len(dreams), 2)
    assert set(atlas.region_names) == set(range(n_clusters))
    for name in atlas.region_names.values():
        assert isinstance(name, str) and len(name) > 0


def test_graph_has_nodes_and_planted_motifs() -> None:
    dreams = load_dreams(DATA)
    graph = motif_graph(dreams)
    assert graph.number_of_nodes() >= 6
    assert graph.number_of_edges() >= 1
    # Every planted motif should fire at least once on this corpus.
    detected = set()
    for dream in dreams:
        detected |= motifs_in_text(dream["text"])
    assert "flooded library" in detected
    assert "talking moth" in detected
    assert "piano city" in detected
    for node in detected:
        assert node in graph.nodes
