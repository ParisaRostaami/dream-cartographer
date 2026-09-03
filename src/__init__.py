"""Dream Cartographer: geometric atlas of dream-journal text."""

from .embed import SVD_DIM, embed_dreams, load_dreams
from .atlas import build_atlas
from .recurrence import motif_graph, motif_centrality
from .visualize import plot_atlas, plot_motif_graph

__all__ = [
    "SVD_DIM",
    "embed_dreams",
    "load_dreams",
    "build_atlas",
    "motif_graph",
    "motif_centrality",
    "plot_atlas",
    "plot_motif_graph",
]
