# Copyright © 2026 Paradigma / MLX Community
# Weight conversion tool for Limite 1B - Violetto -> MLX

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download


def convert(
    hf_repo: str = "paradigma-inc/limite-1b-violetto",
    mlx_path: str = "mlx_model",
    dtype: str = "bfloat16",
):
    mlx_dir = Path(mlx_path)
    mlx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading / resolving Hugging Face repository: {hf_repo}")
    if Path(hf_repo).exists():
        src_dir = Path(hf_repo)
    else:
        src_dir = Path(
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=[
                    "*.json",
                    "*.jinja",
                    "*.safetensors",
                    "*.txt",
                ],
            )
        )

    print(f"[*] Loading raw checkpoint weights from {src_dir}...")
    raw_weights = {}
    for sf in sorted(src_dir.glob("*.safetensors")):
        print(f"    Loading {sf.name}...")
        raw_weights.update(mx.load(str(sf)))

    print(f"[*] Loaded {len(raw_weights)} tensors.")

    # 1. Collect learned projection scales
    qkv_scales = {}
    o_scales = {}
    for k, v in raw_weights.items():
        if "self_attn.qkv_scale" in k:
            layer_idx = int(k.split(".layers.")[1].split(".")[0])
            qkv_scales[layer_idx] = v
        elif "self_attn.o_scale" in k:
            layer_idx = int(k.split(".layers.")[1].split(".")[0])
            o_scales[layer_idx] = v

    print(f"[*] Folding projection scales for {len(qkv_scales)} layers...")
    target_dtype = getattr(mx, dtype, mx.bfloat16)
    converted_weights = {}

    for k, v in raw_weights.items():
        # Skip projection scale scalars once folded
        if "self_attn.qkv_scale" in k or "self_attn.o_scale" in k:
            continue

        # Skip lm_head if present (word embeddings are tied)
        if k == "lm_head.weight":
            continue

        # Fold qkv_scale into Q, K, V projections
        if any(proj in k for proj in [".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"]):
            layer_idx = int(k.split(".layers.")[1].split(".")[0])
            if layer_idx in qkv_scales:
                v = v * qkv_scales[layer_idx].astype(v.dtype)

        # Fold o_scale into O projection
        elif ".self_attn.o_proj.weight" in k:
            layer_idx = int(k.split(".layers.")[1].split(".")[0])
            if layer_idx in o_scales:
                v = v * o_scales[layer_idx].astype(v.dtype)

        # Cast floating point tensors to target dtype (keep float32 for mixing/gating weights if needed)
        if v.dtype in (mx.bfloat16, mx.float16, mx.float32):
            if any(term in k for term in ["dense1", "dense2", "bias", "xsa_alpha", "ve_gate", "attn_gate", "lambda"]):
                # Keep float32 high-precision internal coefficients in float32
                v = v.astype(mx.float32)
            else:
                v = v.astype(target_dtype)

        converted_weights[k] = v

    # Save converted weights
    out_weights_path = mlx_dir / "model.safetensors"
    print(f"[*] Saving converted MLX weights to {out_weights_path}...")
    mx.save_safetensors(str(out_weights_path), converted_weights)

    # Copy tokenizer and configuration files
    print("[*] Copying configuration and tokenizer assets...")
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ]:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, mlx_dir / filename)

    # Process and save config.json with model_file specified
    config_file = src_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
        config["model_file"] = "model.py"
        config["model_type"] = "limite"
        with open(mlx_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Copy model.py into target directory for self-contained loading
    current_dir = Path(__file__).parent
    shutil.copy2(current_dir / "model.py", mlx_dir / "model.py")

    print(f"[✓] Conversion complete! Model ready at: {mlx_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Convert Limite 1B Violetto weights to MLX")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="paradigma-inc/limite-1b-violetto",
        help="Hugging Face repository or local path",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Destination directory for MLX model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Target dtype for weights",
    )
    args = parser.parse_args()

    convert(hf_repo=args.hf_repo, mlx_path=args.mlx_path, dtype=args.dtype)


if __name__ == "__main__":
    main()
