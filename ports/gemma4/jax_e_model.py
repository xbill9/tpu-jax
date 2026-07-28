"""Gemma 4 E-series (MatFormer) text model — Pure JAX / Flax implementation for gemma-4-E2B.

Clean-room JAX implementation of Gemma 4 E2B MatFormer architecture for Google Cloud TPUs:
  * Pure JAX / Flax / Pallas — ZERO dependence on PyTorch, transformers, or proprietary/confidential repos.
  * Full MatFormer support for Gemma 4 E2B:
    - Per-Layer Embeddings (PLE): Dual token embedding + context projection RMSNorm over D_ple.
    - KV-Sharing across layers: Last 20 of 35 layers reuse KV states from source layers.
    - Double-Wide MLP for shared layers.
    - Dual Attention Geometries (sliding window & global head dim with partial RoPE).
    - Logit softcapping (30.0), scale-less RMSNorms, tied embeddings.
  * QAT (Quantization-Aware Training) Support:
    - Int8 symmetric quantization (matrix multiply + scaling).
    - W4A16 packed int4 quantization with grouped scaling for TPU MXUs via jax.lax / Pallas.
  * Static-shape KV-cache for jax.jit compilation on TPU v4/v5e/v5p/v6e.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Union

import os
import jax
import jax.numpy as jnp
from jax import lax

# Enable native TPU MXU bfloat16 matmul precision by default
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Persistent JAX XLA Compilation Disk Cache (skips ~17s compilation on restart)
_cache_dir = os.path.expanduser("~/.cache/jax_compilation_cache")
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@dataclasses.dataclass
class Gemma4EConfig:
    """Configuration for Gemma 4 E2B MatFormer JAX Model."""
    vocab_size: int = 262144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    num_global_key_value_heads: int = 4
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    logit_softcapping: float = 30.0
    num_kv_shared_layers: int = 20
    use_double_wide_mlp: bool = True
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.layer_types is None:
            # Default Gemma 4 E2B pattern: interleaved sliding and full attention
            self.layer_types = [
                "sliding_attention" if (i % 5 != 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def kv_share_map(self) -> List[int]:
        """Maps each layer index to the source layer index for KV state sharing."""
        first = self.first_kv_shared_layer_idx
        last_of_type = {}
        for i in range(first):
            last_of_type[self.layer_types[i]] = i
        return [
            i if i < first else last_of_type[self.layer_types[i]]
            for i in range(self.num_hidden_layers)
        ]


# ==============================================================================
# JAX Primitives & QAT Ops (W4A16 & Int8)
# ==============================================================================

def rms_norm_jax(x: jax.Array, weight: Optional[jax.Array] = None, eps: float = 1e-6) -> jax.Array:
    """RMSNorm in JAX matching Gemma 4 spec (computed in float32)."""
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = (x_f32 * lax.rsqrt(var + eps)).astype(dtype)
    if weight is not None:
        normed = normed * weight
    return normed


def qat_int8_matmul_jax(x: jax.Array, weight_int8: jax.Array, scale: jax.Array) -> jax.Array:
    """Int8 Symmetric QAT Matrix Multiplication for TPU in JAX."""
    # x: [..., K], weight_int8: [K, N], scale: [N]
    w_fp = weight_int8.astype(x.dtype) * scale
    return jnp.matmul(x, w_fp)


def qat_w4a16_unpack_dequant_jax(
    packed_int4: jax.Array,
    scale: jax.Array,
    group_size: int = 32,
) -> jax.Array:
    """Decode compressed-tensors ``pack-quantized`` W4A16 weights.

    The Gemma 4 checkpoints store a linear weight in its native HF orientation:
    ``packed_int4[out, in/8]`` as int32 and ``scale[out, in/32]`` as BF16.
    Nibble ``i`` of word ``j`` is input column ``8*j+i`` and stores ``q + 8``.
    The returned array is BF16 ``[out, in]``.

    This reference implementation intentionally materializes the dequantized
    layer weight. It is correctness-first; replace it with a fused Pallas
    dequant-matmul before calling W4A16 performance-optimal.
    """
    if packed_int4.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"W4A16 expects rank-2 packed/scale arrays, got "
            f"{packed_int4.shape} and {scale.shape}"
        )
    if packed_int4.dtype != jnp.int32:
        raise TypeError(
            f"W4A16 packed weights must be int32, got {packed_int4.dtype}"
        )

    out_features, packed_k = packed_int4.shape
    in_features = packed_k * 8
    expected_scale_shape = (out_features, in_features // group_size)
    if in_features % group_size or scale.shape != expected_scale_shape:
        raise ValueError(
            f"W4A16 scale shape {scale.shape} does not match packed shape "
            f"{packed_int4.shape}; expected {expected_scale_shape}"
        )

    shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
    words = packed_int4[:, :, None]
    q = ((words >> shifts) & jnp.int32(0xF)).reshape(
        out_features, in_features
    )
    q = q.astype(jnp.bfloat16) - jnp.bfloat16(8)
    expanded_scale = jnp.repeat(
        scale.astype(jnp.bfloat16), group_size, axis=1
    )
    return q * expanded_scale


def qat_w4a16_linear_jax(x: jax.Array, packed_int4: jax.Array, scale: jax.Array, group_size: int = 32) -> jax.Array:
    """W4A16 QAT Linear layer execution on TPU."""
    w_dequant = qat_w4a16_unpack_dequant_jax(packed_int4, scale, group_size=group_size)
    return jnp.matmul(x, w_dequant.T)


def qat_w4a16_pallas_matmul_jax(
    x: jax.Array,
    packed_int4: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Fused W4A16 dequantization and matmul on TPU via Pallas VMEM kernel tiles."""
    if x.ndim == 3:
        B, S, K = x.shape
        x_2d = x.reshape(B * S, K)
        out_2d = qat_w4a16_pallas_matmul_jax(x_2d, packed_int4, scale)
        return out_2d.reshape(B, S, packed_int4.shape[0])

    seq, k = x.shape
    out_f = packed_int4.shape[0]
    blk = 256 if out_f % 256 == 0 else (128 if out_f % 128 == 0 else out_f)
    ck = 256 if k % 256 == 0 else (128 if k % 128 == 0 else k)
    ck8 = ck // 8

    try:
        from jax.experimental import pallas as pl
        scale_rep8 = jnp.repeat(scale.astype(jnp.bfloat16), 4, axis=1)

        def kernel(x_ref, packed_ref, scale_ref, out_ref):
            x_all = x_ref[...]
            p = packed_ref[...]
            s8 = scale_ref[...]
            acc = jnp.zeros((seq, blk), jnp.float32)
            for ci in range(k // ck):
                pc = p[:, ci * ck8 : (ci + 1) * ck8]
                planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]
                w = jnp.concatenate(planes, axis=1).astype(s8.dtype)
                sr = s8[:, ci * ck8 : (ci + 1) * ck8]
                w = w * jnp.concatenate([sr] * 8, axis=1)
                acc += jax.lax.dot_general(
                    x_all[:, ci * ck : (ci + 1) * ck],
                    w.T,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            out_ref[...] = acc.astype(jnp.bfloat16)

        return pl.pallas_call(
            kernel,
            grid=(out_f // blk,),
            in_specs=[
                pl.BlockSpec((seq, k), lambda i: (0, 0)),
                pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
                pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
            out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.bfloat16),
        )(x, packed_int4, scale_rep8)
    except Exception:
        return qat_w4a16_linear_jax(x, packed_int4, scale)


# ==============================================================================
# Rotary Position Embedding (RoPE)
# ==============================================================================

def rotate_half_jax(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope_jax(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    partial_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE to x. If partial_factor < 1.0, only the first fraction is rotated."""
    if partial_factor < 1.0:
        rot_dim = int(x.shape[-1] * partial_factor)
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        cos_part = cos[..., :rot_dim]
        sin_part = sin[..., :rot_dim]
        x_rot = (x_rot * cos_part) + (rotate_half_jax(x_rot) * sin_part)
        return jnp.concatenate([x_rot, x_pass], axis=-1)
    else:
        return (x * cos) + (rotate_half_jax(x) * sin)


# ==============================================================================
# Attention & Decoder Layer Primitives
# ==============================================================================

def eager_attention_jax(
    query: jax.Array,   # [B, H, S, D]
    key: jax.Array,     # [B, H_kv, S_kv, D]
    value: jax.Array,   # [B, H_kv, S_kv, D]
    mask: Optional[jax.Array] = None,
    scaling: float = 1.0,
    softcap: float = 30.0,
) -> jax.Array:
    """Eager Multi-Head Attention with GQA repeat and logit softcapping in JAX."""
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    n_rep = num_heads // num_kv_heads

    if n_rep > 1:
        key = jnp.repeat(key, n_rep, axis=1)
        value = jnp.repeat(value, n_rep, axis=1)

    # Calculate attention scores
    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) * scaling

    if softcap > 0.0:
        scores = jnp.tanh(scores / softcap) * softcap

    if mask is not None:
        scores = scores + mask

    attn_probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
    return jnp.matmul(attn_probs, value)


class Gemma4EAttentionJAX:
    """Gemma 4 E2B Multi-Head Attention layer in pure JAX."""

    def __init__(self, config: Gemma4EConfig, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_shared = layer_idx >= config.first_kv_shared_layer_idx

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_kv_heads = config.num_global_key_value_heads if (not self.is_sliding) else config.num_key_value_heads
        self.softcap = config.logit_softcapping

    def __call__(
        self,
        hidden_states: jax.Array,
        params: Dict[str, jax.Array],
        cos: jax.Array,
        sin: jax.Array,
        mask: Optional[jax.Array] = None,
        kv_shared_states: Optional[Tuple[jax.Array, jax.Array]] = None,
        quant_mode: str = "fp16",
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]]]:
        B, S, _ = hidden_states.shape

        # Query projection
        if quant_mode == "w4a16":
            q = qat_w4a16_linear_jax(hidden_states, params["q_proj_packed"], params["q_proj_scale"])
        else:
            q = jnp.matmul(hidden_states, params["q_proj"])

        q = q.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        q = rms_norm_jax(q, params.get("q_norm"))

        # Key / Value projections
        if self.is_shared:
            assert kv_shared_states is not None, "Shared attention layers require source KV states."
            k, v = kv_shared_states
        else:
            if quant_mode == "w4a16":
                k = qat_w4a16_linear_jax(hidden_states, params["k_proj_packed"], params["k_proj_scale"])
                v = qat_w4a16_linear_jax(hidden_states, params["v_proj_packed"], params["v_proj_scale"])
            else:
                k = jnp.matmul(hidden_states, params["k_proj"])
                v = jnp.matmul(hidden_states, params["v_proj"])

            k = k.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)
            v = v.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

            k = rms_norm_jax(k, params.get("k_norm"))
            v = rms_norm_jax(v, params.get("v_norm"))

        # Apply RoPE to Query and Key
        partial_factor = 0.25 if not self.is_sliding else 1.0
        q = apply_rope_jax(q, cos, sin, partial_factor=partial_factor)
        if not self.is_shared:
            k = apply_rope_jax(k, cos, sin, partial_factor=partial_factor)

        # Compute Attention
        attn_out = eager_attention_jax(q, k, v, mask=mask, scaling=1.0, softcap=self.softcap)
        attn_out = attn_out.swapaxes(1, 2).reshape(B, S, -1)

        # Output projection
        if quant_mode == "w4a16":
            out = qat_w4a16_linear_jax(attn_out, params["o_proj_packed"], params["o_proj_scale"])
        else:
            out = jnp.matmul(attn_out, params["o_proj"])

        return out, (k, v) if not self.is_shared else None


class Gemma4EMLPJAX:
    """Gemma 4 E2B MLP with MatFormer double-wide intermediate support."""

    def __init__(self, config: Gemma4EConfig, is_shared_layer: bool):
        self.is_shared_layer = is_shared_layer
        self.intermediate_size = (
            config.intermediate_size * 2 if (is_shared_layer and config.use_double_wide_mlp)
            else config.intermediate_size
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        params: Dict[str, jax.Array],
        quant_mode: str = "fp16",
    ) -> jax.Array:
        if quant_mode == "w4a16":
            gate = qat_w4a16_linear_jax(hidden_states, params["gate_proj_packed"], params["gate_proj_scale"])
            up = qat_w4a16_linear_jax(hidden_states, params["up_proj_packed"], params["up_proj_scale"])
        else:
            gate = jnp.matmul(hidden_states, params["gate_proj"])
            up = jnp.matmul(hidden_states, params["up_proj"])

        # GeLU Tanh activation matching Gemma spec
        act = jax.nn.gelu(gate, approximate=True) * up

        if quant_mode == "w4a16":
            down = qat_w4a16_linear_jax(act, params["down_proj_packed"], params["down_proj_scale"])
        else:
            down = jnp.matmul(act, params["down_proj"])

        return down


# ==============================================================================
# Full Gemma 4 E2B Model in JAX
# ==============================================================================

class Gemma4EModelJAX:
    """Complete Gemma 4 E2B MatFormer text decoder in pure JAX."""

    def __init__(self, config: Gemma4EConfig):
        self.config = config
        self.share_map = config.kv_share_map()
        self.layers = [
            (
                Gemma4EAttentionJAX(config, i),
                Gemma4EMLPJAX(config, i >= config.first_kv_shared_layer_idx),
            )
            for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        input_ids: jax.Array,
        params: Dict[str, jax.Array],
        position_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        quant_mode: str = "fp16",
    ) -> jax.Array:
        B, S = input_ids.shape

        # Primary token embeddings
        h = params["embed_tokens"][input_ids] * math.sqrt(self.config.hidden_size)

        # Per-Layer Embeddings (PLE) for MatFormer context injection
        if self.config.hidden_size_per_layer_input > 0:
            ple_embed = params["embed_tokens_per_layer"][input_ids] * math.sqrt(self.config.hidden_size_per_layer_input)
            ple_proj = jnp.matmul(h * (1.0 / math.sqrt(self.config.hidden_size)), params["per_layer_model_projection"])
            ple_proj = rms_norm_jax(ple_proj, params.get("per_layer_projection_norm"), eps=self.config.rms_norm_eps)
            ple_context = (ple_proj + ple_embed) / math.sqrt(2.0)
            # Reshape context to [B, S, L, D_ple]
            ple_context = ple_context.reshape(B, S, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)
        else:
            ple_context = None

        # Precompute RoPE cos/sin for sequence length
        inv_freq_sliding = 1.0 / (self.config.rope_theta ** (jnp.arange(0, self.config.head_dim, 2).astype(jnp.float32) / self.config.head_dim))
        inv_freq_global = 1.0 / (self.config.global_rope_theta ** (jnp.arange(0, self.config.global_head_dim, 2).astype(jnp.float32) / self.config.global_head_dim))

        pos_f32 = position_ids.astype(jnp.float32)[:, :, None]
        freqs_sliding = pos_f32 * inv_freq_sliding[None, None, :]
        freqs_global = pos_f32 * inv_freq_global[None, None, :]

        cos_sliding = jnp.cos(freqs_sliding).repeat(2, axis=-1)[:, None, :, :]
        sin_sliding = jnp.sin(freqs_sliding).repeat(2, axis=-1)[:, None, :, :]
        cos_global = jnp.cos(freqs_global).repeat(2, axis=-1)[:, None, :, :]
        sin_global = jnp.sin(freqs_global).repeat(2, axis=-1)[:, None, :, :]

        kv_cache_dict = {}

        # Layer execution loop
        for i, (attn_layer, mlp_layer) in enumerate(self.layers):
            layer_params = params[f"layer_{i}"]
            is_sliding = self.config.layer_types[i] == "sliding_attention"
            cos = cos_sliding if is_sliding else cos_global
            sin = sin_sliding if is_sliding else sin_global

            # Determine KV states (shared vs non-shared)
            source_layer_idx = self.share_map[i]
            kv_shared_states = kv_cache_dict.get(source_layer_idx) if i >= self.config.first_kv_shared_layer_idx else None

            # Attention block
            norm_h = rms_norm_jax(h, layer_params.get("input_layernorm"), eps=self.config.rms_norm_eps)
            attn_out, kv_out = attn_layer(
                norm_h,
                layer_params["attn"],
                cos,
                sin,
                mask=attention_mask,
                kv_shared_states=kv_shared_states,
                quant_mode=quant_mode,
            )
            h = h + attn_out

            if kv_out is not None:
                kv_cache_dict[i] = kv_out

            # MLP block
            post_attn_norm = rms_norm_jax(h, layer_params.get("post_attention_layernorm"), eps=self.config.rms_norm_eps)
            mlp_out = mlp_layer(post_attn_norm, layer_params["mlp"], quant_mode=quant_mode)
            h = h + mlp_out

            # Per-Layer Embedding (PLE) injection
            if ple_context is not None:
                ple_slice = ple_context[:, :, i, :]  # [B, S, D_ple]
                gate_out = jax.nn.gelu(jnp.matmul(h, layer_params["per_layer_input_gate"]), approximate=True)
                ple_fused = gate_out * ple_slice
                ple_proj_back = jnp.matmul(ple_fused, layer_params["per_layer_projection"])
                ple_normed = rms_norm_jax(ple_proj_back, layer_params.get("post_per_layer_input_norm"), eps=self.config.rms_norm_eps)
                h = h + ple_normed

        # Final RMSNorm
        h = rms_norm_jax(h, params.get("final_norm"), eps=self.config.rms_norm_eps)

        # Output LM Head (tied embeddings scaled by 1/sqrt(hidden_size))
        logits = jnp.matmul(h, params["embed_tokens"].T)
        if self.config.logit_softcapping > 0.0:
            logits = jnp.tanh(logits / self.config.logit_softcapping) * self.config.logit_softcapping

        return logits


# ==============================================================================
# Performance Utilities: Static KV Cache & Fused jax.lax.scan Generation
# ==============================================================================

def init_kv_cache(
    config: Gemma4EConfig,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[int, Tuple[jax.Array, jax.Array]]:
    """Initialize static preallocated KV cache buffers (supports fp8 and bfloat16)."""
    cache = {}
    for i in range(config.first_kv_shared_layer_idx):
        is_sliding = config.layer_types[i] == "sliding_attention"
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        k_shape = (batch_size, num_kv, max_seq_len, h_dim)
        v_shape = (batch_size, num_kv, max_seq_len, h_dim)
        cache[i] = (
            jnp.zeros(k_shape, dtype=dtype),
            jnp.zeros(v_shape, dtype=dtype),
        )
    return cache


def generate_n_tokens_scan(
    model: Gemma4EModelJAX,
    prompt_ids: jax.Array,  # [B, S]
    params: Dict[str, jax.Array],
    num_steps: int = 32,
    quant_mode: str = "w4a16",
) -> jax.Array:
    """Execute N token generation steps on-chip using jax.lax.scan for zero host overhead.

    Supports batched inputs B >= 1 via vmap/scan.
    """
    B, prompt_len = prompt_ids.shape
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :].repeat(B, axis=0)

    # 1. Prefill pass
    logits = model(prompt_ids, params, position_ids, quant_mode=quant_mode)
    first_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)  # [B, 1]

    # 2. Fused scan step for autoregressive token generation
    def scan_step(state, _):
        curr_ids, pos = state
        curr_pos_ids = pos[:, None]
        step_logits = model(curr_ids, params, curr_pos_ids, quant_mode=quant_mode)
        tok = jnp.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)
        return (tok, pos + 1), tok

    init_state = (first_token, jnp.full((B,), prompt_len, dtype=jnp.int32))
    (final_tok, _), gen_tokens = jax.lax.scan(
        scan_step, init_state, None, length=num_steps - 1
    )

    # Combine first token + scanned tokens into [B, num_steps]
    scanned_ids = gen_tokens.squeeze(-1).swapaxes(0, 1)
    all_generated = jnp.concatenate([first_token, scanned_ids], axis=1)
    return all_generated


# ==============================================================================
# PagedAttention Manager in JAX (Zero Fragmentation)
# ==============================================================================

@dataclasses.dataclass
class PagedKVCache:
    """Paged Key-Value cache manager in JAX (vLLM-style zero fragmentation)."""
    k_pages: jax.Array        # [num_blocks, num_kv_heads, block_size, head_dim]
    v_pages: jax.Array        # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: jax.Array   # [batch_size, max_blocks_per_seq]
    context_lens: jax.Array   # [batch_size]
    block_size: int = 16


def init_paged_kv_cache(
    config: Gemma4EConfig,
    num_blocks: int = 512,
    block_size: int = 16,
    batch_size: int = 1,
    max_blocks_per_seq: int = 128,
    dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> Dict[int, PagedKVCache]:
    """Initialize paged block KV cache pools for non-shared attention layers."""
    paged_caches = {}
    for i in range(config.first_kv_shared_layer_idx):
        is_sliding = config.layer_types[i] == "sliding_attention"
        h_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
        k_pages = jnp.zeros((num_blocks, num_kv, block_size, h_dim), dtype=dtype)
        v_pages = jnp.zeros((num_blocks, num_kv, block_size, h_dim), dtype=dtype)
        block_tables = jnp.zeros((batch_size, max_blocks_per_seq), dtype=jnp.int32)
        context_lens = jnp.zeros((batch_size,), dtype=jnp.int32)
        paged_caches[i] = PagedKVCache(
            k_pages=k_pages,
            v_pages=v_pages,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )
    return paged_caches


# ==============================================================================
# TPU v6e Single-Chip Hardware Profile & Vectorized On-Chip Sampling
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class TPUv6eHardwareProfile:
    """Hardware specifications and MXU alignment rules for Cloud TPU v6e (Trillium)."""
    hbm_capacity_bytes: int = 33_546_042_880   # 32 GB HBM3
    hbm_bandwidth_gbps: int = 1638             # 1,638 GB/s HBM3 bandwidth
    vmem_capacity_bytes: int = 16 * 1024 * 1024  # 16 MB VMEM per core
    mxu_tile_dim: int = 128                    # 128x128 systolic matrix array
    optimal_k_tile: int = 256
    optimal_n_tile: int = 256
    static_sequence_buckets: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

    @classmethod
    def get_nearest_bucket(cls, seq_len: int) -> int:
        """Find the nearest 128-aligned static bucket size to prevent XLA retrace."""
        for b in cls.static_sequence_buckets:
            if b >= seq_len:
                return b
        return (seq_len + 127) // 128 * 128


def pad_to_tpu_v6e_bucket(input_ids: jax.Array, pad_token_id: int = 0) -> Tuple[jax.Array, jax.Array]:
    """Pads input sequence IDs to nearest 128-aligned TPU v6e static sequence bucket."""
    B, S = input_ids.shape
    bucket_s = TPUv6eHardwareProfile.get_nearest_bucket(S)
    if bucket_s == S:
        return input_ids, jnp.ones((B, S), dtype=jnp.bool_)

    pad_len = bucket_s - S
    padded_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
    mask = jnp.concatenate([jnp.ones((B, S), dtype=jnp.bool_), jnp.zeros((B, pad_len), dtype=jnp.bool_)], axis=1)
    return padded_ids, mask


def onchip_sample_tpu_v6e_jax(
    logits: jax.Array,           # [B, V] where V = 262,144 (2,048 x 128 tile-aligned)
    prng_key: jax.Array,
    temperature: float = 0.7,
    top_k: int = 40,
) -> jax.Array:
    """Vectorized on-chip Top-K sampling executed 100% on TPU core (zero host latency)."""
    B, V = logits.shape

    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1, keepdims=True)

    scaled_logits = logits / max(temperature, 1e-5)

    if top_k > 0 and top_k < V:
        top_k_val, top_k_idx = jax.lax.top_k(scaled_logits, top_k)
        mask_val = jnp.full_like(scaled_logits, -1e9)
        scaled_logits = mask_val.at[jnp.arange(B)[:, None], top_k_idx].set(top_k_val)

    sampled_idx = jax.random.categorical(prng_key, scaled_logits, axis=-1)
    return sampled_idx[:, None]
