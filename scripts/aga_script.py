#!/usr/bin/env python3
"""
AGA-script — Attention → SNAP edge list
========================================
Extracts attention matrices from a pretrained transformer and writes one SNAP
edge list per (layer, head) for loading into GraphAnalyzer.

Output files: aga_L{layer}_H{head}.txt
SNAP format:  # comment header, then "src dst" per line (undirected, thresholded)

Usage:
    python aga_script.py                          — run with defaults (gpt2)
    python aga_script.py --list-models            — show supported models and exit
    python aga_script.py --model bert-base        — choose a different model
    python aga_script.py --model gpt2 --tau 0.02  — custom tau threshold

Requirements:
    pip install torch transformers
"""

import argparse
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Keys are the short names accepted by --model.
# "hf_id"    : HuggingFace model identifier
# "family"   : for display grouping
# "params"   : approximate parameter count (human-readable)
# "layers"   : number of attention layers
# "heads"    : attention heads per layer
# "vram_gb"  : approximate VRAM needed for FP32 inference (rough guide)
# "cpu_ok"   : comfortable to run on CPU (inference time < ~30s)
# ---------------------------------------------------------------------------
MODELS = {
    # ── GPT-2 family ──────────────────────────────────────────────────────
    "gpt2": {
        "hf_id":   "gpt2",
        "family":  "GPT-2",
        "params":  "117M",
        "layers":  12,
        "heads":   12,
        "vram_gb": 0.5,
        "cpu_ok":  True,
    },
    "gpt2-medium": {
        "hf_id":   "gpt2-medium",
        "family":  "GPT-2",
        "params":  "345M",
        "layers":  24,
        "heads":   16,
        "vram_gb": 1.5,
        "cpu_ok":  True,
    },
    "gpt2-large": {
        "hf_id":   "gpt2-large",
        "family":  "GPT-2",
        "params":  "774M",
        "layers":  36,
        "heads":   20,
        "vram_gb": 3.0,
        "cpu_ok":  False,
    },
    "gpt2-xl": {
        "hf_id":   "gpt2-xl",
        "family":  "GPT-2",
        "params":  "1.5B",
        "layers":  48,
        "heads":   25,
        "vram_gb": 6.0,
        "cpu_ok":  False,
    },
    # ── BERT family ───────────────────────────────────────────────────────
    "bert-base": {
        "hf_id":   "bert-base-uncased",
        "family":  "BERT",
        "params":  "110M",
        "layers":  12,
        "heads":   12,
        "vram_gb": 0.5,
        "cpu_ok":  True,
    },
    "bert-large": {
        "hf_id":   "bert-large-uncased",
        "family":  "BERT",
        "params":  "340M",
        "layers":  24,
        "heads":   16,
        "vram_gb": 1.5,
        "cpu_ok":  True,
    },
    # ── DistilBERT ────────────────────────────────────────────────────────
    "distilbert": {
        "hf_id":   "distilbert-base-uncased",
        "family":  "DistilBERT",
        "params":  "66M",
        "layers":  6,
        "heads":   12,
        "vram_gb": 0.3,
        "cpu_ok":  True,
    },
    # ── RoBERTa ───────────────────────────────────────────────────────────
    "roberta-base": {
        "hf_id":   "roberta-base",
        "family":  "RoBERTa",
        "params":  "125M",
        "layers":  12,
        "heads":   12,
        "vram_gb": 0.5,
        "cpu_ok":  True,
    },
    "roberta-large": {
        "hf_id":   "roberta-large",
        "family":  "RoBERTa",
        "params":  "355M",
        "layers":  24,
        "heads":   16,
        "vram_gb": 1.5,
        "cpu_ok":  True,
    },
    # ── GPT-Neo ───────────────────────────────────────────────────────────
    "gpt-neo-125m": {
        "hf_id":   "EleutherAI/gpt-neo-125m",
        "family":  "GPT-Neo",
        "params":  "125M",
        "layers":  12,
        "heads":   12,
        "vram_gb": 0.5,
        "cpu_ok":  True,
    },
    "gpt-neo-1.3b": {
        "hf_id":   "EleutherAI/gpt-neo-1.3B",
        "family":  "GPT-Neo",
        "params":  "1.3B",
        "layers":  24,
        "heads":   16,
        "vram_gb": 5.5,
        "cpu_ok":  False,
    },
}


def list_models():
    """Print a formatted table of all supported models and exit."""
    has_cuda = torch.cuda.is_available()
    device_label = "GPU available ✓" if has_cuda else "CPU only (no GPU detected)"
    print(f"\nDevice: {device_label}\n")

    col = "{:<16} {:<12} {:<8} {:<8} {:<8} {:<9} {}"
    print(col.format("NAME", "FAMILY", "PARAMS", "LAYERS", "HEADS", "VRAM(GB)", "CPU OK?"))
    print("-" * 72)

    current_family = None
    for name, m in MODELS.items():
        if m["family"] != current_family:
            if current_family is not None:
                print()
            current_family = m["family"]

        cpu_flag = "✓" if m["cpu_ok"] else "slow"
        gpu_warn = "" if has_cuda or m["cpu_ok"] else "  ⚠ GPU recommended"
        print(col.format(
            name,
            m["family"],
            m["params"],
            m["layers"],
            m["heads"],
            f"~{m['vram_gb']}",
            cpu_flag + gpu_warn,
        ))

    print(f"\nUsage:  python aga_script.py --model <NAME>\n")


def extract(
    model_name: str = "gpt2",
    prompt: str = "The quick brown fox jumps over the lazy dog",
    tau: float = 0.01,
    outdir: str = "./aga_out",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if model_name not in MODELS:
        print(f"[AGA] Unknown model '{model_name}'. Run --list-models to see options.")
        sys.exit(1)

    meta = MODELS[model_name]
    hf_id = meta["hf_id"]

    if device == "cpu" and not meta["cpu_ok"]:
        print(f"[AGA] Warning: {model_name} ({meta['params']}) is slow on CPU. "
              "Consider a smaller model or use --list-models to compare.")

    os.makedirs(outdir, exist_ok=True)

    print(f"[AGA] Loading {model_name} ({hf_id}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, output_attentions=True)
    model.eval().to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_tokens = inputs["input_ids"].shape[1]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    print(f"[AGA] {n_tokens} tokens: {tokens}")

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # tuple: (batch, heads, N, N) per layer
    n_layers = len(attentions)
    n_heads  = attentions[0].shape[1]
    print(f"[AGA] {n_layers} layers x {n_heads} heads | tau={tau}")

    total_edges = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attentions[layer][0, head].cpu().numpy()  # (N, N)

            # Threshold and write SNAP edge list (undirected: i < j only)
            path = os.path.join(outdir, f"aga_L{layer:02d}_H{head:02d}.txt")
            n_edges = 0
            lines = []
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    w = max(attn[i, j], attn[j, i])
                    if w > tau:
                        lines.append(f"{i}\t{j}")
                        n_edges += 1

            with open(path, "w") as f:
                f.write(f"# AGA: model={model_name} layer={layer} head={head}\n")
                f.write(f"# prompt: {prompt}\n")
                f.write(f"# tokens={n_tokens} tau={tau} edges={n_edges}\n")
                f.write("\n".join(lines))
                f.write("\n")

            total_edges += n_edges
            print(f"  L{layer:02d} H{head:02d} -> {n_edges:5d} edges -> {path}")

    print(f"\n[AGA] Done. {n_layers * n_heads} files written to {outdir}/ "
          f"(total edges: {total_edges})")


def main():
    parser = argparse.ArgumentParser(
        description="AGA-script: attention -> SNAP edge lists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run --list-models to see all supported models.",
    )
    parser.add_argument("--list-models", action="store_true",
                        help="Print supported models and exit")
    parser.add_argument("--model",  default="gpt2",
                        help=f"Model name (default: gpt2). One of: {', '.join(MODELS)}")
    parser.add_argument("--tau",    type=float, default=0.01,
                        help="Attention threshold (default: 0.01)")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog",
                        help="Input prompt for the model")
    parser.add_argument("--outdir", default="./aga_out",
                        help="Output directory for SNAP files (default: ./aga_out)")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    extract(
        model_name=args.model,
        prompt=args.prompt,
        tau=args.tau,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
