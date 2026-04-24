# attn2graph

A research tool for extracting and analyzing graph-theoretic properties of attention matrices from pretrained transformer models.

## Overview

`attn2graph` bridges sublinear graph algorithm research with large language model internals. It extracts attention weight matrices from pretrained transformers via a containerized HuggingFace inference pipeline, interprets them as weighted graphs, and computes structural properties — with a focus on **arboricity** and related parameters.

Attention matrix extraction runs in an isolated **Docker container** (Python/PyTorch/HuggingFace), decoupling the inference environment from the analysis pipeline. The extracted matrices are native **PyTorch tensors**, making them directly compatible with any PyTorch-based downstream workflow. Graph analysis is performed by **[GraphAnalyzer](https://github.com/aperelman/GraphAnalyzer)**, which will be able to trigger extractions on demand (future feature), (currently, file can loaded manually) and load the resulting SNAP edge lists for interactive analysis.

## Architecture

```
┌─────────────────────────────────┐
│   Docker Container (Python)     │
│   HuggingFace transformers      │
│   PyTorch + CUDA                │
│                                 │
│   pretrained LLM                │
│   → attention matrices          │
│     (native PyTorch tensors)    │
│   → SNAP edge list files        │
└──────────────┬──────────────────┘
               │  ./data/*.txt
               ▼
┌─────────────────────────────────┐
│   GraphAnalyzer (C++/Qt6)       │
│                                 │
│   threshold + symmetrize        │
│   arboricity α(G)               │
│   degeneracy d(G)               │
│   triangle count                │
│   + extensible algorithm set    │
└─────────────────────────────────┘
```

## Motivation

Transformer attention matrices define implicit graph structures over token positions. Understanding the combinatorial properties of these graphs — such as arboricity, degeneracy, and triangle density — may shed light on the sparsification and approximation behavior of attention mechanisms.

Key research questions:
- What is the arboricity α(G) of attention graphs in practice?
- How does α_k(G) scale with layer depth, head index, and model size?
- Can sublinear graph algorithms exploit low-arboricity structure for efficient attention approximation?

## Features

- **HuggingFace inference container** — extracts raw attention matrices from any pretrained model, outputs SNAP edge list files
- **Native PyTorch tensors** — extracted matrices are directly compatible with any PyTorch-based downstream workflow
- **GraphAnalyzer integration** — trigger extractions on demand, load SNAP edge lists directly into interactive analysis
- Compute **arboricity** via exact and approximate algorithms (including the 2-approximation via modified degeneracy ordering)
- Compute **degeneracy**, **triangle counts**, and **density**
- Batch analysis across all layers and attention heads
- Extensible algorithm set — add new graph algorithms directly in GraphAnalyzer

## Repository Structure

```
attn2graph/
├── docker/               # HuggingFace inference container
│   ├── Dockerfile
│   ├── extract.py        # attention extraction script
│   └── requirements.txt
├── data/                 # SNAP edge list output (gitignored)
└── README.md
```

## Usage

### Step 1 — Extract attention matrices (Docker)

```bash
docker build -t attn2graph-extract ./docker
docker run --gpus all \
  -v $(pwd)/data:/data \
  attn2graph-extract \
  --model gpt2 --layer all --head all --threshold 0.01
```

This produces SNAP edge list files in `./data/`:
```
gpt2_L3_H7_tau0.01_sym-max.txt
gpt2_L3_H7_tau0.02_sym-max.txt
...
```

### Step 2 — Analyze graphs (GraphAnalyzer)

Load any of the generated SNAP edge list files into **[GraphAnalyzer](https://github.com/aperelman/GraphAnalyzer)** for interactive analysis — arboricity, degeneracy, triangle counting, and more. GraphAnalyzer can also trigger the Docker extraction directly from its UI.

## Building

### Prerequisites

- Docker (with NVIDIA runtime for GPU support)
- [GraphAnalyzer](https://github.com/aperelman/GraphAnalyzer) for graph analysis

## Research Background

The arboricity of a graph G is the minimum number of spanning forests needed to cover all edges. It is tightly related to degeneracy and max subgraph density, and serves as a key parameter in sublinear algorithms for triangle counting and graph sparsification.

This project empirically validates theoretical predictions from:
- Eden, Ron, Seshadhri — importance sampling for triangle estimation
- The formal correctness proof connecting IS approximation to arboricity bounds (`attention_IS_proof.md`)

## Related Projects

- [GraphAnalyzer](https://github.com/aperelman/GraphAnalyzer) — C++/Qt6 desktop tool for sublinear graph analysis

## License

MIT

## Acknowledgments

Special thanks to **Dr. Talya Eden** (Bar-Ilan University, sublinear algorithms group) for her foundational work on sublinear graph algorithms and arboricity, which directly inspired the research questions behind this project.
