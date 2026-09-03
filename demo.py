"""Run the full dream-atlas pipeline and write figures under outputs/."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.atlas import build_atlas
from src.embed import SVD_DIM, embed_dreams, load_dreams
from src.recurrence import motif_centrality, motif_graph, night_motif_table
from src.visualize import plot_atlas, plot_motif_graph


def main() -> None:
    data_path = ROOT / "data" / "dreams.jsonl"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dreams = load_dreams(data_path)
    texts = [str(d["text"]) for d in dreams]
    ids = [str(d["id"]) for d in dreams]

    X, _pipe, tfidf, feature_names = embed_dreams(texts, svd_dim=SVD_DIM)
    atlas = build_atlas(X, tfidf, feature_names)
    graph = motif_graph(dreams)
    centrality = motif_centrality(graph)
    nights = night_motif_table(dreams)

    atlas_path = plot_atlas(
        atlas,
        ids,
        out_dir / "atlas.png",
        title="Dream atlas (TF-IDF + SVD + MDS)",
    )
    graph_path = plot_motif_graph(
        graph,
        centrality,
        out_dir / "motif_graph.png",
    )

    points = pd.DataFrame(
        {
            "id": ids,
            "date": [d["date"] for d in dreams],
            "mds_1": atlas.coords[:, 0],
            "mds_2": atlas.coords[:, 1],
            "region_id": atlas.labels,
            "region_name": [atlas.region_names[int(k)] for k in atlas.labels],
        }
    )
    points_path = out_dir / "atlas_points.csv"
    points.to_csv(points_path, index=False)
    central_path = out_dir / "motif_centrality.csv"
    centrality.to_csv(central_path, index=False)
    nights_path = out_dir / "night_motifs.csv"
    nights.to_csv(nights_path, index=False)

    print("=== Dream Cartographer ===")
    print(f"Dreams loaded:           {len(dreams)}")
    print(f"Embedding shape:         {tuple(X.shape)}  (target SVD dim={SVD_DIM})")
    print(f"Agglomerative regions:   {atlas.n_clusters}  (silhouette={atlas.silhouette:.3f})")
    for cid, name in atlas.region_names.items():
        n = int((atlas.labels == cid).sum())
        print(f"  region {cid} (n={n}): {name}")
    print(f"Motif graph:             {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print("Most central motifs (eigenvector):")
    for row in centrality.head(3).itertuples(index=False):
        print(f"  {row.motif:24s}  eigen={row.eigenvector:.3f}  weighted degree={row.degree_weighted:.0f}")
    print("Nights with 2+ motifs:   "
          f"{int((nights['n_motifs'] >= 2).sum())} / {len(nights)}")
    print("Wrote:")
    print(f"  {atlas_path}")
    print(f"  {graph_path}")
    print(f"  {points_path}")
    print(f"  {central_path}")
    print(f"  {nights_path}")


if __name__ == "__main__":
    main()
