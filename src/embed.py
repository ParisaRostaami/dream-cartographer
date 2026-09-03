"""Lexical embedding of dream reports.

Word 1–2 grams capture named places and motifs; character 3–5 grams add
robustness to inflection and compound imagery (flooded/flooding, moth/moths).
TruncatedSVD then compresses the sparse TF-IDF space into a dense geometry
where cosine neighbors share vocabulary rather than deep semantics.

This is a classical IR pipeline, not a neural dream encoder. On a small
synthetic corpus it is enough to separate recurring locales.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

RANDOM_STATE = 42
SVD_DIM = 24  # within the 16–32 band requested for the atlas latent space


def load_dreams(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records with fields id, date, text."""
    path = Path(path)
    dreams: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("id", "date", "text"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing field {key!r}")
            dreams.append(record)
    if len(dreams) < 3:
        raise ValueError("Need at least 3 dreams to build an atlas")
    return dreams


def build_embedder(svd_dim: int = SVD_DIM) -> Pipeline:
    if not 16 <= svd_dim <= 32:
        raise ValueError(f"svd_dim must be in [16, 32], got {svd_dim}")
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
    )
    union = FeatureUnion([("word", word), ("char", char)])
    return Pipeline(
        [
            ("tfidf", union),
            ("svd", TruncatedSVD(n_components=svd_dim, random_state=RANDOM_STATE)),
            ("l2", Normalizer(norm="l2")),
        ]
    )


def embed_dreams(
    texts: list[str],
    svd_dim: int = SVD_DIM,
) -> tuple[np.ndarray, Pipeline, np.ndarray, np.ndarray]:
    """Fit TF-IDF+SVD on dream texts.

    Returns
    -------
    X : (n, svd_dim) L2-normalized dense embedding
    pipe : fitted sklearn pipeline
    tfidf : (n, vocab) sparse word+char TF-IDF used later for region naming
    feature_names : vocabulary aligned with ``tfidf`` columns
    """
    rng = np.random.default_rng(RANDOM_STATE)
    _ = rng  # documents the global seed contract even for this deterministic fit
    pipe = build_embedder(svd_dim)
    X = pipe.fit_transform(texts)
    tfidf = pipe.named_steps["tfidf"].transform(texts)
    feature_names = pipe.named_steps["tfidf"].get_feature_names_out()
    return np.asarray(X, dtype=np.float64), pipe, tfidf, feature_names
