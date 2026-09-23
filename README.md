# MLX Implementation of Limite 1B - Violetto

This repository provides the native **Apple Silicon (MLX)** implementation and serving tools for **Limite 1B - Violetto**, developed by [Paradigma](https://paradigma.inc).

**Limite 1B - Violetto** is an open-weights (Apache 2.0) 1-billion parameter dense autoregressive transformer specifically engineered for high-throughput, competition-level mathematical reasoning.

---

## Benchmark Highlights

Despite having only 1B parameters and being trained from scratch on less than 300B tokens, Limite 1B matches or surpasses models 10× to 30× its size on elite mathematical benchmarks:

| Model | Total Params | AIME 2026 | HMMT Feb. 2026 | BeyondAIME | APEX Shortlist | AIME 2025 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Limite 1B - Violetto** | **1B** | **94.01%** | **83.62%** | **74.25%** | **50.80%** | **90.21%** |
| VibeThinker-1.5B | 1.5B | 70.94% | 47.25% | 48.06% | 10.84% | 72.92% |
| MiniCPM5-2B | 2B | 90.21% | 67.80% | 60.59% | 26.86% | 86.98% |
| VibeThinker-3B | 3B | 93.85% | 78.98% | 72.00% | 46.41% | 92.19% |
| Qwen3.5-4B | 4B | 89.66% | 72.86% | 61.25% | 32.51% | 82.19% |
| Qwen3.5-9B | 9B | 90.42% | 70.36% | 65.56% | 30.72% | 88.75% |
| MUSE-Glimmer-30B | 30B | 93.44% | 80.87% | 70.00% | 51.66% | 92.71% |
| Gemma-4-31B-IT | 31B | 89.17% | 77.08% | 71.00% | 40.16% | 87.81% |

*Data sourced from Paradigma evaluations and MathArena.*

### Limite 1B vs. Qwen2.5-Math-1.5B
* **Architecture:** Qwen2.5-Math uses standard LLaMA-style dense layers. Limite 1B uses speedrun architectural inductive biases (MUDD dynamic residuals, Value Embeddings, XSA, attention gating, custom RoPE).
* **Reasoning Paradigm:** Qwen2.5-Math relies heavily on Tool-Integrated Reasoning (TIR / Python code execution) to score ~79.7% on MATH. Limite 1B operates via pure single-turn symbolic & algebraic Chain-of-Thought (CoT), achieving 94.01% on AIME 2026.
* **Intended Use:** Qwen2.5-Math is bilingual and conversational. Limite 1B is **hyper-specialized for single-turn mathematical problem solving** in English; it deliberately lacks conversational chit-chat capabilities.

---

## Architectural Features Implemented in MLX

1. **MUDD (Multi-tap Dynamic Dense Residuals):** At layers 24 and 47, attention inputs and residual bases are computed via dynamic learned MLP projections across historical checkpoints ($h_0$ embedding, $h_{12}$, $h_{23}$, $h_{24}$, $h_{47}$).
2. **Value Embeddings (VE):** Designated attention layers (every 3 layers: 1, 4, 7, ...) look up token IDs in a secondary embedding table (`value_embeds`), gating and injecting them directly into Value $V$ before updating the KV cache.
3. **Cross-Subspace Attention (XSA):** Attention heads project out normalized Value projections ($y \leftarrow y - \alpha \cdot (y \cdot \hat{v}) \cdot \hat{v}$).
4. **Attention Gating:** Multi-head attention outputs pass through a per-head sigmoid gating mechanism scaled by 2.0 before output projection $W_o$.
5. **Weightless RMSNorm & Pre-RoPE QK-Norm:** RMSNorm without learnable weights.
6. **Partial Rotary with Odd-Lane Sign Flip:** Rotates only the first 64 of 128 head dimensions; flips odd sine signs (`sin[1::2] *= -1`) and reverses adjacent pairs. RoPE is completely disabled on global layers (`global_nope`).
7. **Interleaved Sliding Window / Global Attention:** 36 layers with a 1024-token sliding window; every 4th layer is an unwindowed global layer.
8. **Sigmoid Logit Softcapping:** Logits computed via $23.0 \cdot \sigma\left(\frac{\text{raw} + 5.0}{7.5}\right)$.

---

## Quickstart

### 1. Installation

Using [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/gabrielribeiro/mlx-Limite-1B-Violetto.git
cd mlx-Limite-1B-Violetto
uv sync
```

Or using standard pip:

```bash
pip install -r pyproject.toml
```

### 2. Convert Checkpoint to MLX Format

Download and convert the official Hugging Face weights:

```bash
uv run python convert.py --hf-repo paradigma-inc/limite-1b-violetto --mlx-path ./mlx_model
```

This step:
* Downloads the weights from Hugging Face.
* Folds learned projection scales (`qkv_scale`, `o_scale`) into the projection weights.
* Prepares the model directory with `model.py` and `config.json` configured for `mlx-lm`.

### 3. Run Inference

#### Via CLI Generator:

```bash
uv run python generate.py --model ./mlx_model --prompt "Find the number of positive integers n <= 100 such that gcd(n, 20) = 1."
```

Recommended sampling parameters are set by default (`temperature=0.6`, `top_p=0.95`).

#### In Python via `mlx-lm`:

Because the converted model includes `"model_file": "model.py"` in its `config.json`, standard `mlx-lm` loads and runs it seamlessly:

```python
import mlx_lm

# Load converted MLX model and tokenizer
model, tokenizer = mlx_lm.load("./mlx_model")

# Format problem using canonical ChatML math template
prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.\n"
    "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    "<|im_start|>user\n"
    "Find all real solutions to x^3 - 3x + 1 = 0.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response = mlx_lm.generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    temp=0.6,
    top_p=0.95,
)
print(response)
```

---

## Testing & Verification

Run the test suite to verify numerical parity against PyTorch reference equations and KV cache stepping:

```bash
uv run python test_limite_mlx.py
```

All tests verify:
* Weightless RMSNorm output parity.
* Rotary embedding with odd-lane sign flip parity.
* MUDD mixer dynamic combination parity.
* Sigmoid softcapping parity.
* KV Cache stepping and multi-token prefill equivalence.
* Direct integration with `mlx_lm.load_model`.

---

## License

* **Model Weights:** Apache 2.0 (Paradigma)
* **MLX Code:** Apache 2.0
