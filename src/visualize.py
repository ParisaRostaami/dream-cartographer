"""Atlas scatter with convex-hull regions and a motif co-occurrence graph."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError

from .atlas import Atlas

REGION_COLORS = [
    "#3d5a80",
    "#ee6c4d",
    "#98c1d9",
    "#293241",
    "#e0fbfc",
    "#9b5de5",
]


def plot_atlas(
    atlas: Atlas,
    dream_ids: list[str],
    out_path: str | Path,
    title: str = "Dream atlas",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=140)
    coords = atlas.coords
    labels = atlas.labels

    for cluster_id in sorted(set(labels.tolist())):
        color = REGION_COLORS[cluster_id % len(REGION_COLORS)]
        mask = labels == cluster_id
        pts = coords[mask]
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                ax.add_patch(
                    Polygon(
                        hull_pts,
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.18,
                        linewidth=1.2,
                    )
                )
            except QhullError:
                pass
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[color],
            s=55,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=atlas.region_names.get(cluster_id, f"region {cluster_id}"),
        )
        centroid = pts.mean(axis=0)
        ax.annotate(
            atlas.region_names.get(cluster_id, f"R{cluster_id}"),
            xy=centroid,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#1b1b1b",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=4,
        )

    for i, dream_id in enumerate(dream_ids):
        ax.annotate(
            dream_id,
            xy=coords[i],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="#444444",
            alpha=0.85,
        )

    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_motif_graph(
    graph: nx.Graph,
    centrality,
    out_path: str | Path,
    title: str = "Motif co-occurrence across nights",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=140)

    eigen = dict(zip(centrality["motif"], centrality["eigenvector"]))
    rng = np.random.default_rng(42)
    pos = nx.spring_layout(graph, seed=42, k=1.6, iterations=80)
    # tiny jitter so overlapping labels stay readable on fully connected graphs
    for node in pos:
        pos[node] = pos[node] + rng.normal(0, 0.01, size=2)

    weights = [graph[u][v].get("weight", 1) for u, v in graph.edges]
    max_w = max(weights) if weights else 1
    widths = [1.2 + 3.5 * (w / max_w) for w in weights]
    sizes = [420 + 1600 * eigen.get(n, 0.0) for n in graph.nodes]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color="#8d99ae", alpha=0.75)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=sizes,
        node_color="#3d5a80",
        edgecolors="white",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

    if weights:
        for (u, v), w in zip(graph.edges, weights):
            mid = (pos[u] + pos[v]) / 2
            ax.text(mid[0], mid[1], str(int(w)), fontsize=7, color="#293241", ha="center", va="center")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
