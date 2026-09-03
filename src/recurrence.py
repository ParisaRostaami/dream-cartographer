"""Motif co-occurrence graph across nights.

Nodes are a closed set of recurring dream images (the six motifs planted
in the synthetic journal). An undirected weighted edge counts how many
nights a pair of motifs shared a report. Degree and betweenness then
rank which images function as hubs in the dreamer's private geography.

This is co-occurrence, not causation: two motifs can share nights because
the narrative links them, or because the generator over-sampled a pair.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

# Surface forms used both as detectors and as graph node labels.
MOTIF_PATTERNS: dict[str, tuple[str, ...]] = {
    "flooded library": (
        r"flooded library",
        r"flood(?:ed|ing) (?:the )?library",
        r"library(?:'s)? (?:was |is )?(?:underwater|flooded|submerged)",
        r"drowned (?:card )?catalog",
        r"submerged stacks",
        r"water(?:logged)? stacks",
    ),
    "childhood kitchen": (
        r"childhood kitchen",
        r"kitchen of (?:my |our )?childhood",
        r"mother'?s kitchen",
        r"old kitchen",
        r"linoleum (?:floor|tile)",
        r"enamel sink",
    ),
    "airport with no gates": (
        r"airport with no gates",
        r"airport without gates",
        r"ungated (?:airport|terminal)",
        r"terminal without gates",
        r"gates? (?:had )?(?:vanished|disappeared|were gone|were missing)",
        r"no gates (?:left|at all)",
    ),
    "talking moth": (
        r"talking moth",
        r"moth (?:that )?(?:said|spoke|whisper|talk)",
        r"moth'?s voice",
        r"the moth (?:told|asked|answered|replied)",
    ),
    "piano city": (
        r"city that is also a piano",
        r"piano[- ]city",
        r"city (?:was|is) (?:also )?a piano",
        r"streets? (?:were|are) keys",
        r"keyboard of streets",
        r"buildings? (?:were|are) hammers",
    ),
    "sea inside a hospital": (
        r"sea inside a hospital",
        r"hospital .{0,40}sea",
        r"sea .{0,40}hospital",
        r"ward .{0,30}tide",
        r"saltwater corridor",
        r"nurses? .{0,20}tide",
    ),
}

_COMPILED = {
    name: [re.compile(pat, flags=re.IGNORECASE) for pat in pats]
    for name, pats in MOTIF_PATTERNS.items()
}


def motifs_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for name, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            found.add(name)
    return found


def night_motif_table(dreams: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dream in dreams:
        present = motifs_in_text(str(dream["text"]))
        rows.append(
            {
                "id": dream["id"],
                "date": dream["date"],
                "motifs": sorted(present),
                "n_motifs": len(present),
            }
        )
    return pd.DataFrame(rows)


def motif_graph(dreams: Iterable[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(MOTIF_PATTERNS.keys())
    for dream in dreams:
        present = sorted(motifs_in_text(str(dream["text"])))
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                    graph[a][b]["nights"].append(dream["id"])
                else:
                    graph.add_edge(a, b, weight=1, nights=[dream["id"]])
    return graph


def motif_centrality(graph: nx.Graph) -> pd.DataFrame:
    degree = dict(graph.degree(weight="weight"))
    between = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    if graph.number_of_edges() == 0:
        eigen = {n: 0.0 for n in graph.nodes}
    else:
        try:
            eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
        except (nx.NetworkXError, np.linalg.LinAlgError):
            eigen = nx.pagerank(graph, weight="weight")
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "motif": node,
                "degree_weighted": float(degree.get(node, 0.0)),
                "betweenness": float(between.get(node, 0.0)),
                "eigenvector": float(eigen.get(node, 0.0)),
                "degree_unweighted": int(graph.degree(node)),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("eigenvector", ascending=False).reset_index(drop=True)
