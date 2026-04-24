# attn2graph

A research tool for extracting and analyzing graph-theoretic properties of attention matrices from pretrained transformer models.

## Overview

`attn2graph` bridges sublinear graph algorithm research with large language model internals. It loads attention weight matrices from pretrained transformers, interprets them as weighted graphs, and computes structural properties — with a focus on **arboricity** and related parameters.

This project is part of ongoing research in collaboration with **Talya Eden's sublinear algorithms group at Bar-Ilan University**.

## Motivation

Transformer attention matrices define implicit graph structures over token positions. Understanding the combinatorial properties of these graphs — such as arboricity, degeneracy, and triangle density — may shed light on the sparsification and approximation behavior of attention mechanisms.

Key research questions:
- What is the arboricity α(G) of attention graphs in practice?
- How does α_k(G) scale with layer depth, head index, and model size?
- Can sublinear graph algorithms exploit low-arboricity structure for efficient attention approximation?

## Features

- Load attention matrices from HuggingFace pretrained models
- Convert attention weight matrices to graphs (thresholded or weighted)
- Compute **arboricity** via exact and approximate algorithms (including the 2-approximation via modified degeneracy ordering)
- Compute **degeneracy**, **triangle counts**, and **density**
- Batch analysis across layers and attention heads
- Export results to CSV for downstream analysis

## Installation

```bash
git clone https://github.com/aperelmane/attn2graph.git
cd attn2graph
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- `transformers` (HuggingFace)
- `torch`
- `numpy`
- `igraph`
- `networkx` (optional, for comparison)

## Usage

```bash
python attn2graph.py --model bert-base-uncased --layer 6 --head 4
```

Options:
- `--model` — HuggingFace model name or local path
- `--layer` — Transformer layer index (or `all`)
- `--head` — Attention head index (or `all`)
- `--threshold` — Edge weight threshold for graph construction (default: 0.01)
- `--output` — Output CSV file for results

## Research Background

The arboricity of a graph G is the minimum number of spanning forests needed to cover all edges. It is tightly related to degeneracy and max subgraph density, and serves as a key parameter in sublinear algorithms for triangle counting and graph sparsification.

This project empirically validates theoretical predictions from:
- Eden, Ron, Seshadhri — importance sampling for triangle estimation
- The formal correctness proof connecting IS approximation to arboricity bounds (`attention_IS_proof.md`)

## Related Projects

- [GraphAnalyzer](https://github.com/aperelmane/GraphAnalyzer) — C++/Qt6 desktop tool for sublinear graph analysis

## License

MIT

## Acknowledgments

Research conducted in collaboration with Talya Eden's sublinear algorithms group, Bar-Ilan University.
