# Dream Cartographer

**Parisa Rostami** · Wichita State University · portfolio prototype, 2026

A small computational phenomenology tool: it turns a dream journal into a 2D **dream atlas** (places that recur sit near each other) and a **motif graph** (which images co-occur across nights). It is not a clinical sleep instrument, not a large-language-model dream interpreter, and not an analysis of anyone's real nights. The bundled corpus is synthetic, written so that six planted images recur and intertwine.

## What it actually computes

1. **Lexical geometry.** Each report is a TF-IDF vector of word 1–2 grams plus character 3–5 grams (`char_wb`), then **TruncatedSVD to 24 dimensions** and L2-normalized. Neighbors share vocabulary (flooded stacks, enamel sink, jetways), not latent Freudian content.
2. **Atlas.** Agglomerative clustering (Ward) runs in that 24-D space; *k* is chosen by silhouette in {3,…,6}. **Metric MDS** then places nights on the page. Region labels are the highest-mean word n-grams of each cluster — a readable tag, not a grounded topic model.
3. **Recurrence.** Six motifs are detected with regular expressions: flooded library, childhood kitchen, airport with no gates, talking moth, city-as-piano, sea inside a hospital. Nights become a weighted co-occurrence graph; degree, betweenness, and eigenvector centrality rank hub images.

Hall/Van de Castle coding and later computational dream studies treat reports as structured text. This prototype borrows that stance (text in, geometry out) and stops there. Convex hulls on the scatterplot are a visualization of cluster support, not territorial claims about a psyche.

## Planted corpus

`data/dreams.jsonl` holds 42 reports (`id`, `date`, `text`) spanning 2026-01-08 to 2026-03-14. Motifs are written in on purpose, including mixed nights, so the graph is connected and the atlas has more than one region. Dates are fictional study labels, not a diary.

## How to run

Python 3.10+. No network, no model downloads. `numpy` random seed **42** (and sklearn `random_state=42`) for SVD, MDS, and graph layout.

```text
python -m pip install -r requirements.txt
python demo.py
python -m pytest tests/test_smoke.py -q
```

`demo.py` writes:

| file | contents |
| --- | --- |
| `outputs/atlas.png` | MDS scatter, cluster hulls, lexical region names |
| `outputs/motif_graph.png` | co-occurrence graph; node size ~ eigenvector centrality |
| `outputs/atlas_points.csv` | coordinates and region id per night |
| `outputs/motif_centrality.csv` | degree / betweenness / eigenvector |
| `outputs/night_motifs.csv` | motifs detected in each report |

## Limits (read these)

- TF-IDF cannot see metaphor that does not share strings. A kitchen described only as "the room with the kettle" will not join the kitchen cluster.
- Six motifs are a closed vocabulary. New imagery is invisible to the graph.
- Silhouette-selected *k* is unstable on tiny *n*; 42 points is a toy.
- MDS stress is not reported as a scientific result, only as a layout.

If you swap in a real journal, keep the pipeline, replace `data/dreams.jsonl`, and treat region names as hypotheses to check against the pages.

## Layout

```text
demo.py                 full pipeline + printed summary
src/embed.py            load JSONL, TF-IDF union, TruncatedSVD
src/atlas.py            Ward clusters, MDS, top-term names
src/recurrence.py       motif detectors, NetworkX graph, centrality
src/visualize.py        atlas hulls + graph figure (matplotlib Agg)
tests/test_smoke.py     embedding rank, k ≥ 3, graph has nodes
```

MIT License · Copyright 2026 Parisa Rostami
