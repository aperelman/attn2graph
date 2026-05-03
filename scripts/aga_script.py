"""
AGA-script — Attention → SNAP edge list
========================================
Extracts attention matrices from GPT-2 and writes one SNAP edge list
per (layer, head) for loading into GraphAnalyzer.

Output files: aga_L{layer}_H{head}.txt
SNAP format:  # comment header, then "src dst" per line (undirected, thresholded)

Usage:
    python aga_script.py [--model gpt2] [--tau 0.01] [--prompt "..."] [--outdir ./aga_out]

Requirements:
    pip install torch transformers
"""

import argparse
import os

import torch
from transformers import GPT2Model, GPT2Tokenizer


def extract(
    model_name: str = "gpt2",
    prompt: str = "The quick brown fox jumps over the lazy dog",
    tau: float = 0.01,
    outdir: str = "./aga_out",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(outdir, exist_ok=True)

    print(f"[AGA] Loading {model_name} on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
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

            print(f"  L{layer:02d} H{head:02d} -> {n_edges:5d} edges -> {path}")

    print(f"\n[AGA] Done. {n_layers * n_heads} files written to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="AGA-script: attention -> SNAP edge lists")
    parser.add_argument("--model",  default="gpt2")
    parser.add_argument("--tau",    type=float, default=0.01)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--outdir", default="./aga_out")
    args = parser.parse_args()

    extract(
        model_name=args.model,
        prompt=args.prompt,
        tau=args.tau,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
