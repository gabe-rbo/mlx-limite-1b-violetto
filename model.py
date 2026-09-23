# Copyright © 2026 Paradigma / MLX Community
# Implemented for MLX & Apple Silicon.

from __future__ import annotations

import inspect
import math
import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Fix for Python 3.12 dataclasses when loaded via importlib.util.module_from_spec
if __name__ not in sys.modules:
    sys.modules[__name__] = types.ModuleType(__name__)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache, RotatingKVCache, create_causal_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "limite"
    hidden_size: int = 1280
    num_hidden_layers: int = 48
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 3328
    vocab_size: int = 151680
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True
    attention_softmax_scale: float = 0.1
    sliding_window: int = 1024
    global_layers: list[int] = field(
        default_factory=lambda: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    )
    global_nope: bool = True
    attn_gate_channels: int = 128
    attn_gate_scale: float = 2.0
    pos_mode: str = "rope"
    rope_frac: float = 0.5
    rope_base_local: float = 1024.0
    rope_base_global: float = 1024.0
    rope_n_pairs: int = 32
    ve_dim: int = 128
    ve_layers: list[int] = field(
        default_factory=lambda: [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]
    )
    ve_gate_channels: int = 12
    ve_gate_scale: float = 2.0
    ve_stored_heads: int = 2
    xsa: bool = True
    xsa_layers: list[int] = field(default_factory=lambda: list(range(48)))
    xsa_normalize_eps: float = 0.0001
    mudd: bool = True
    mudd_layers: list[int] = field(default_factory=lambda: [24, 47])
    mudd_taps: int = 3
    mudd_inter: int = 32
    mudd_mlp: bool = True
    mudd_tap_idx: dict[str, list[int]] = field(
        default_factory=lambda: {"24": [0, 12, 24], "47": [0, 23, 47]}
    )
    mlp_type: str = "swiglu"
    softcap_logits: dict[str, Any] = field(
        default_factory=lambda: {"a": 23.0, "b": 5.0, "c": 7.5, "kind": "sigmoid"}
    )
    final_softcap: float = 0.0
    rms_norm_eps_mode: str = "torch_finfo_default"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        valid_keys = set(inspect.signature(cls).parameters.keys())
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)


def rms_norm(x: mx.array, eps: Optional[float] = None) -> mx.array:
    """Weightless RMSNorm matching Limite reference.

    Uses finfo(dtype).eps when eps is not specified.
    """
    if eps is None:
        eps = mx.finfo(x.dtype).eps
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class LimiteRotary:
    """Partial, adjacent-pair-interleaved rotary with an odd-lane sign flip.

    Rotates only the first 2 * rope_n_pairs dimensions (64 out of 128).
    Remaining dimensions pass through unrotated.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.head_dim
        self.n_pairs = min(args.rope_n_pairs, self.head_dim // 2)
        base = args.rope_base_local

        freq = (1.0 / base) ** mx.linspace(0, 1, self.n_pairs, dtype=mx.float32)
        freq = mx.repeat(freq, 2)
        pad_size = max(0, self.head_dim - 2 * self.n_pairs)
        pad_zeros = mx.zeros((pad_size,), dtype=mx.float32)
        self.freq = mx.concatenate([freq, pad_zeros])

        sign = mx.ones((self.head_dim,), dtype=mx.float32)
        sign[1::2] = -1.0
        self.sin_sign = sign

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x is [B, num_heads, S, head_dim]
        S = x.shape[2]
        positions = mx.arange(offset, offset + S, dtype=mx.float32)
        theta = positions[:, None] * self.freq[None, :]
        cos = mx.cos(theta).astype(x.dtype)
        sin = (mx.sin(theta) * self.sin_sign).astype(x.dtype)

        # Broadcast across batch and head dims
        cos = mx.expand_dims(cos, (0, 1))
        sin = mx.expand_dims(sin, (0, 1))

        # Reverse adjacent pairs in the head dimension: [a, b] -> [b, a]
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_flip = x_pairs[..., ::-1].reshape(x.shape)

        return cos * x + sin * x_flip


class LimiteMuddMixer(nn.Module):
    """Dynamic dense residual mixing (MUDD).

    At tapped layers (24 and 47), attention input and residual streams are a
    learned, input-dependent weighted sum of earlier residual-stream states.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        num_layers = args.num_hidden_layers
        taps = args.mudd_taps
        inter = args.mudd_inter
        hidden = args.hidden_size

        self.dense1 = mx.zeros((inter, hidden), dtype=mx.float32)
        self.dense2 = mx.zeros((num_layers, taps, inter), dtype=mx.float32)
        self.bias = mx.zeros((num_layers, taps), dtype=mx.float32)

        self.uses_r_way = args.mudd_mlp
        if self.uses_r_way:
            self.dense2_mlp = mx.zeros((num_layers, taps, inter), dtype=mx.float32)
            self.bias_mlp = mx.zeros((num_layers, taps), dtype=mx.float32)

    def combine(
        self,
        values: list[mx.array],
        x_cur: mx.array,
        layer_idx: int,
        *,
        r_way: bool = False,
    ) -> mx.array:
        count = len(values)
        inner = nn.gelu(rms_norm(x_cur) @ self.dense1.astype(x_cur.dtype).T)

        dense2 = self.dense2_mlp if r_way else self.dense2
        bias = self.bias_mlp if r_way else self.bias

        weights = inner @ dense2[layer_idx, :count].astype(inner.dtype).T
        weights = weights + bias[layer_idx, :count].astype(weights.dtype)

        out = weights[..., 0:1].astype(values[0].dtype) * values[0]
        for idx in range(1, count):
            out = out + weights[..., idx : idx + 1].astype(values[idx].dtype) * values[idx]
        return out


class LimiteMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class LimiteAttention(nn.Module):
    """Limite Attention with Value Embeddings, Pre-RoPE QK-Norm, XSA, and Attn Gating."""

    def __init__(self, args: ModelArgs, layer_idx: int, rotary: LimiteRotary):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.gqa_groups = self.num_heads // self.num_kv_heads
        self.scale = args.attention_softmax_scale

        self.ve_dim = args.ve_dim
        self.ve_stored_heads = args.ve_stored_heads
        self.ve_gate_scale = args.ve_gate_scale
        self.attn_gate_channels = args.attn_gate_channels
        self.attn_gate_scale = args.attn_gate_scale
        self.xsa_eps = args.xsa_normalize_eps

        self.uses_ve = layer_idx in set(args.ve_layers)
        self.uses_xsa = bool(args.xsa) and layer_idx in set(args.xsa_layers)
        self.is_global = layer_idx in set(args.global_layers)
        self.uses_rope = not (bool(args.global_nope) and self.is_global)
        self.rotary = rotary

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.xsa_alpha = mx.zeros((self.num_heads,), dtype=mx.float32)
        if self.uses_ve:
            self.ve_gate = mx.zeros(
                (self.ve_stored_heads, args.ve_gate_channels), dtype=mx.float32
            )
        if self.attn_gate_channels > 0:
            self.attn_gate = mx.zeros(
                (self.num_heads, self.attn_gate_channels), dtype=mx.float32
            )

    def _apply_value_embeddings(
        self,
        input_ids: mx.array,
        attn_in: mx.array,
        v: mx.array,
        value_embeds: nn.Embedding,
    ) -> mx.array:
        # input_ids: [B, S]
        B, S, _, _ = v.shape
        ve = value_embeds(input_ids).astype(v.dtype).reshape(B, S, self.ve_stored_heads, self.ve_dim)
        if self.head_dim > self.ve_dim:
            ve = mx.pad(ve, [(0, 0), (0, 0), (0, 0), (0, self.head_dim - self.ve_dim)])

        gate_w = self.ve_gate
        if self.ve_stored_heads > self.num_kv_heads:
            ve = ve[:, :, : self.num_kv_heads]
            gate_w = gate_w[: self.num_kv_heads]

        gate = self.ve_gate_scale * mx.sigmoid(
            attn_in[..., : gate_w.shape[-1]] @ gate_w.astype(attn_in.dtype).T
        )
        return v + mx.expand_dims(gate, -1) * ve

    def __call__(
        self,
        input_ids: mx.array,
        attn_in: mx.array,
        value_embeds: nn.Embedding,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = attn_in.shape

        q = self.q_proj(attn_in).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(attn_in).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(attn_in).reshape(B, S, self.num_kv_heads, self.head_dim)

        if self.uses_ve:
            v = self._apply_value_embeddings(input_ids, attn_in, v, value_embeds)

        # Pre-RoPE QK-Norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Transpose to [B, num_heads, S, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        if self.uses_rope:
            q = self.rotary(q, offset=offset)
            k = self.rotary(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # GQA expansion
        if self.gqa_groups > 1:
            k_exp = mx.repeat(k, self.gqa_groups, axis=1)
            v_exp = mx.repeat(v, self.gqa_groups, axis=1)
        else:
            k_exp = k
            v_exp = v

        # Attention computation
        q_scaled = q * self.scale
        scores = q_scaled @ k_exp.swapaxes(-1, -2)

        if mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(
                    mask, scores, mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype)
                )
            else:
                scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1, precise=True)
        y = attn_weights @ v_exp  # [B, num_heads, S, head_dim]
        y = y.transpose(0, 2, 1, 3)  # [B, S, num_heads, head_dim]

        # XSA (Cross-Subspace Attention)
        if self.uses_xsa:
            # v_exp is [B, num_heads, S_cached, head_dim]
            # When S > 1 during prefill, align v_exp with current sequence length S
            v_curr = v_exp[..., -S:, :].transpose(0, 2, 1, 3)
            vn = v_curr / (mx.linalg.norm(v_curr, axis=-1, keepdims=True) + self.xsa_eps)
            proj = mx.sum(y * vn, axis=-1, keepdims=True)
            alpha = mx.tanh(self.xsa_alpha).reshape(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        # Attention Gating
        if self.attn_gate_channels > 0:
            gate = self.attn_gate_scale * mx.sigmoid(
                attn_in[..., : self.attn_gate_channels] @ self.attn_gate.astype(attn_in.dtype).T
            )
            y = y * mx.expand_dims(gate, -1)

        y = y.reshape(B, S, self.hidden_size)
        return self.o_proj(y)


class LimiteDecoderLayer(nn.Module):
    """Transformer decoder block with learned residual multipliers."""

    def __init__(self, args: ModelArgs, layer_idx: int, rotary: LimiteRotary):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = layer_idx in set(args.global_layers)
        self.self_attn = LimiteAttention(args, layer_idx, rotary)
        self.mlp = LimiteMLP(args)

        # Learned residual scalars
        self.resid_lambda_attn = mx.array(1.0, dtype=mx.float32)
        self.post_lambda_attn = mx.array(1.0, dtype=mx.float32)
        self.resid_lambda_mlp = mx.array(1.0, dtype=mx.float32)
        self.post_lambda_mlp = mx.array(1.0, dtype=mx.float32)

    def __call__(
        self,
        input_ids: mx.array,
        x: mx.array,
        attn_in: mx.array,
        value_embeds: nn.Embedding,
        attn_residual: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attn_out = self.self_attn(
            input_ids=input_ids,
            attn_in=attn_in,
            value_embeds=value_embeds,
            mask=mask,
            cache=cache,
        )

        # First residual join with learned lambdas
        x = (
            self.resid_lambda_attn.astype(x.dtype) * attn_residual
            + self.post_lambda_attn.astype(x.dtype) * attn_out
        )

        mlp_out = self.mlp(rms_norm(x))

        # Second residual join with learned lambdas
        x = (
            self.resid_lambda_mlp.astype(x.dtype) * x
            + self.post_lambda_mlp.astype(x.dtype) * mlp_out
        )
        return x


class LimiteModel(nn.Module):
    """Full Limite transformer trunk."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.sliding_window = args.sliding_window

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.value_embeds = nn.Embedding(
            args.vocab_size, args.ve_stored_heads * args.ve_dim
        )
        self.rotary = LimiteRotary(args)
        self.mudd = LimiteMuddMixer(args)

        self.layers = [
            LimiteDecoderLayer(args, i, self.rotary)
            for i in range(args.num_hidden_layers)
        ]

        self.mudd_tap_idx: dict[int, list[int]] = {
            int(layer): [int(idx) for idx in taps]
            for layer, taps in args.mudd_tap_idx.items()
        }
        self.retained_history = {
            idx for taps in self.mudd_tap_idx.values() for idx in taps
        }
        self.final_softcap = args.final_softcap

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        x = rms_norm(self.embed_tokens(input_ids))
        B, S = input_ids.shape

        # Construct masks if S > 1 (prefill)
        mask_global = None
        mask_local = None
        if S > 1:
            offset = cache[0].offset if cache is not None and cache[0] is not None else 0
            mask_global = create_causal_mask(S, offset=offset)
            mask_local = create_causal_mask(
                S, offset=offset, window_size=self.sliding_window + 1
            )

        history: dict[int, mx.array] = {0: x} if self.mudd_tap_idx else {}

        for index, layer in enumerate(self.layers):
            if index in self.mudd_tap_idx:
                values = [history[i] for i in self.mudd_tap_idx[index]]
                attn_in = rms_norm(self.mudd.combine(values, x, index))
                residual_base = (
                    self.mudd.combine(values, x, index, r_way=True)
                    if self.mudd.uses_r_way
                    else x
                )
            else:
                attn_in = rms_norm(x)
                residual_base = x

            layer_cache = cache[index] if cache is not None else None
            layer_mask = mask_global if layer.is_global else mask_local

            x = layer(
                input_ids=input_ids,
                x=x,
                attn_in=attn_in,
                value_embeds=self.value_embeds,
                attn_residual=residual_base,
                mask=layer_mask,
                cache=layer_cache,
            )

            if index + 1 in self.retained_history:
                history[index + 1] = x

        if self.final_softcap > 0:
            cap = self.final_softcap
            x = cap * mx.tanh(x / cap)

        return rms_norm(x)


class Model(nn.Module):
    """Top-level Limite model wrapper compatible with mlx_lm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LimiteModel(args)

        softcap = dict(args.softcap_logits)
        self.softcap_a = float(softcap.get("a", 23.0))
        self.softcap_b = float(softcap.get("b", 5.0))
        self.softcap_c = float(softcap.get("c", 7.5))

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        hidden_states = self.model(inputs, cache=cache)
        return self.compute_logits(hidden_states)

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits with Limite's sigmoid softcapping."""
        raw = self.model.embed_tokens.as_linear(hidden_states)
        return self.softcap_a * mx.sigmoid(
            (raw.astype(mx.float32) + self.softcap_b) / self.softcap_c
        ).astype(raw.dtype)

    def make_cache(self) -> List[Any]:
        """Construct hybrid cache: Rotating for sliding window, Standard for global."""
        caches = []
        for layer in self.model.layers:
            if layer.is_global:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window + 1))
        return caches

    @property
    def layers(self) -> List[LimiteDecoderLayer]:
        return self.model.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights and fold projection scales."""
        sanitized = {}

        # First pass: collect scales and projections
        qkv_scales: dict[int, mx.array] = {}
        o_scales: dict[int, mx.array] = {}

        for k, v in weights.items():
            if "self_attn.qkv_scale" in k:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                qkv_scales[layer_idx] = v
            elif "self_attn.o_scale" in k:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                o_scales[layer_idx] = v

        for k, v in weights.items():
            # Skip projection scale parameters (they get folded)
            if "self_attn.qkv_scale" in k or "self_attn.o_scale" in k:
                continue

            # Remove lm_head if present (tied embeddings)
            if k == "lm_head.weight":
                continue

            # Fold scales into QKV and O projections
            if ".self_attn.q_proj.weight" in k or ".self_attn.k_proj.weight" in k or ".self_attn.v_proj.weight" in k:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                if layer_idx in qkv_scales:
                    v = v * qkv_scales[layer_idx].astype(v.dtype)
            elif ".self_attn.o_proj.weight" in k:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                if layer_idx in o_scales:
                    v = v * o_scales[layer_idx].astype(v.dtype)

            sanitized[k] = v

        return sanitized
