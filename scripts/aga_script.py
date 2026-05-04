#!/usr/bin/env python3
"""
AGA-script — Attention → SNAP edge list
========================================
Extracts attention matrices from any HuggingFace causal LM that exposes
output_attentions=True, and writes one SNAP edge list per (layer, head)
for loading into GraphAnalyzer.

Output files: aga_L{layer}_H{head}.txt
SNAP format:  # comment header, then "src dst" per line (undirected, thresholded)

Usage:
    python aga_script.py [--model gpt2] [--tau 0.01] [--prompt "..."] [--outdir ./aga_out]
    python aga_script.py --list-models [--search qwen] [--limit 20]

Requirements:
    pip install torch transformers accelerate huggingface_hub
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.0f}M"
    return str(n)


def _vram_gb(n_params: int) -> str:
    gb = n_params * 2 * 1.2 / 1e9
    return f"~{gb:.1f}"


def _cpu_ok(n_params: int) -> str:
    gb = n_params * 2 * 1.2 / 1e9
    if gb <= 2:
        return "✓"
    if gb <= 8:
        return "slow"
    return "✗"


def _fetch_config(model_id: str) -> dict:
    """Fetch config.json from HF Hub. Returns {} on failure."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _layers_heads(cfg: dict) -> tuple:
    """Extract (n_layers, n_heads) from a config dict."""
    layers = (cfg.get("num_hidden_layers")
              or cfg.get("n_layer")
              or cfg.get("num_layers"))
    heads  = (cfg.get("num_attention_heads")
              or cfg.get("n_head")
              or cfg.get("num_heads"))
    return (layers or "?", heads or "?")


def list_models(search: str = "", limit: int = 20):
    try:
        from huggingface_hub import list_models as hf_list_models
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"[AGA] Querying HuggingFace Hub (pipeline_tag=text-generation, limit={limit}"
          + (f", search='{search}'" if search else "") + ")...")

    kwargs = dict(
        pipeline_tag="text-generation",
        sort="downloads",
        limit=limit,
        expand=["safetensors"],
    )
    if search:
        kwargs["search"] = search

    models = list(hf_list_models(**kwargs))
    if not models:
        print("No models found.")
        return

    print(f"[AGA] Fetching configs for {len(models)} models...\n")

    # Fetch configs concurrently
    configs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_config, m.id): m.id for m in models}
        for fut in as_completed(futures):
            configs[futures[fut]] = fut.result()

    # Derive short NAME and FAMILY from model id  (e.g. "Qwen/Qwen2.5-7B" -> "Qwen2.5-7B", "Qwen")
    def _name_family(model_id: str) -> tuple:
        parts  = model_id.split("/")
        name   = parts[-1]
        org    = parts[0] if len(parts) > 1 else ""
        # Map known orgs/prefixes to friendly family names
        family_map = [
            ("Qwen",        "Qwen"),
            ("meta-llama",  "LLaMA"),
            ("mistralai",   "Mistral"),
            ("openai-community", "GPT"),
            ("openai",      "GPT"),
            ("google",      "Gemma"),
            ("microsoft",   "Phi"),
            ("facebook",    "OPT"),
            ("EleutherAI",  "GPT-Neo"),
            ("deepseek-ai", "DeepSeek"),
            ("trl-internal-testing", "Test"),
        ]
        family = org  # fallback
        for key, label in family_map:
            if key.lower() in org.lower() or key.lower() in name.lower():
                family = label
                break
        return name, family

    print(f"{'NAME':<30} {'FAMILY':<12} {'PARAMS':>7}  {'LAYERS':>6}  {'HEADS':>5}  {'VRAM(GB)':>8}  {'CPU OK?'}")
    print("-" * 80)
    for m in models:
        st       = getattr(m, "safetensors", None)
        n_params = st.total if st else 0
        params_s = _fmt_params(n_params) if n_params else "?"
        vram_s   = _vram_gb(n_params)    if n_params else "?"
        cpu_s    = _cpu_ok(n_params)     if n_params else "?"
        cfg      = configs.get(m.id, {})
        layers, heads = _layers_heads(cfg)
        name, family  = _name_family(m.id)
        print(f"{name:<30} {family:<12} {params_s:>7}  {layers:>6}  {heads:>5}  {vram_s:>8}  {cpu_s}")

    print(f"\n{len(models)} models listed.")
    print("\nUsage example:")
    print(f"  python aga_script.py --model {models[0].id}")


def extract(
    model_name: str = "gpt2",
    prompt: str = "The quick brown fox jumps over the lazy dog",
    tau: float = 0.01,
    outdir: str = "./aga_out",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    print(f"[AGA] Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Some tokenizers (e.g. Qwen) don't set a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",  # required for output_attentions on flash-attn models
    )
    model.eval()
    if device != "cuda":
        model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_tokens = inputs["input_ids"].shape[1]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    print(f"[AGA] {n_tokens} tokens: {tokens}")

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # tuple: (batch, heads, N, N) per layer
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. "
            "Make sure the model supports output_attentions=True "
            "and attn_implementation='eager'."
        )

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    print(f"[AGA] {n_layers} layers x {n_heads} heads | tau={tau}")

    total_edges = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attentions[layer][0, head].float().cpu().numpy()  # (N, N)

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

    print(f"\n[AGA] Done. {n_layers * n_heads} files written to {outdir}/")
    print(f"[AGA] Total edges across all graphs: {total_edges}")


def main():
    parser = argparse.ArgumentParser(description="AGA-script: attention -> SNAP edge lists")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model ID (default: gpt2)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Attention threshold (default: 0.01)")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog",
                        help="Input prompt")
    parser.add_argument("--outdir", default="./aga_out",
                        help="Output directory (default: ./aga_out)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda / cpu (default: auto-detect)")
    parser.add_argument("--list-models", action="store_true",
                        help="Query HuggingFace Hub for top text-generation models and exit")
    parser.add_argument("--search", default="",
                        help="Filter --list-models by name (e.g. 'qwen', 'mistral')")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of models to show with --list-models (default: 20)")
    parser.add_argument("--no-trust-remote-code", action="store_true",
                        help="Disable trust_remote_code (may break Qwen)")
    args = parser.parse_args()

    if args.list_models:
        list_models(search=args.search, limit=args.limit)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    extract(
        model_name=args.model,
        prompt=args.prompt,
        tau=args.tau,
        outdir=args.outdir,
        device=device,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
