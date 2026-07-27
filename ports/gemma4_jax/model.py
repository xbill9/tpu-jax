"""Gemma 4 DENSE text model (gemma-4-31b architecture) — JAX-native port.

Pure-functional transcription of the parity-proven PyTorch port in
`ports/gemma4/model.py` (which is itself validated against HF transformers
5.12.1). No Flax / NNX / Haiku — params are a plain pytree of `jnp` arrays,
and every entry point is a pure function, so everything composes with
`jax.jit`, `jax.vmap`, `jax.lax.scan`, donation, and sharding annotations.

Params pytree schema
--------------------
A nested dict-of-dicts that mirrors the torch `state_dict()` key structure
1:1 — each dotted torch key becomes a nesting path (layer indices are STRING
keys), so conversion from a torch state dict is purely mechanical
(see `convert.py`):

    params["model"]["embed_tokens"]["weight"]                    (V, H)
    params["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"]
    params["model"]["layers"]["0"]["self_attn"]["q_norm"]["weight"]
    params["model"]["layers"]["0"]["mlp"]["gate_proj"]["weight"]
    params["model"]["layers"]["0"]["input_layernorm"]["weight"]
    params["model"]["layers"]["0"]["layer_scalar"]               (1,)
    params["model"]["norm"]["weight"]
    params["lm_head"]["weight"]        (tied: aliases embed_tokens weight)

Linear weights stay in the torch (out_features, in_features) layout; matmuls
are `x @ W.T` via einsum. Full-attention (`attention_k_eq_v`) layers have no
`v_proj` subtree, and `v_norm` has no weight (scale-less) — both exactly as
in the torch port. Non-persistent torch buffers (RoPE `inv_freq`,
`embed_scale`) are NOT in the pytree; they are recomputed from the config.

Static KV cache
---------------
`init_cache()` returns a tuple (one entry per layer) of
`{"k": [B, n_kv, MAX_SEQ, head_dim], "v": ...}` arrays using that LAYER's
geometry (sliding: num_key_value_heads x head_dim; full/k_eq_v:
num_global_key_value_heads x global_head_dim). Contents match the torch
port's documented choice: K is cached POST-k_norm and POST-RoPE, V
POST-v_norm (for k_eq_v layers: v_norm of the RAW k_proj output). Writes go
through `jax.lax.dynamic_update_slice` at a (possibly traced) position, and
decode masks are built from `arange(MAX_SEQ)` + the position scalar, so
`decode_step` jits to a single graph with no retrace across steps.
`prefill()` / `decode_step()` are pure: they RETURN the updated cache.

Gemma4 quirks reproduced (see the torch port's docstring for the full list):
RMSNorm scales by plain `weight` (not 1+w) with fp32 `pow(mean+eps, -0.5)`;
embeddings scaled by sqrt(hidden); dual attention geometry; attention_k_eq_v
on full layers (V = scale-less v_norm of the RAW k_proj output, no v_proj);
q/k-norm before RoPE; RoPE applied in (B, S, H, D) layout; attention scaling
1.0; proportional partial RoPE on full layers (angles =
int(0.25 * head_dim // 2), frequency exponents over the FULL head_dim,
inv_freq zero-padded); EXCLUSIVE sliding window (0 <= q - kv < window);
sandwich norms; trailing layer_scalar multiply; final logit softcap 30.0
via tanh; tied lm_head.

Numerics note: softmax is computed in float32 and cast back to the working
dtype (as in torch). For fp32-vs-fp32 parity runs set
`jax.config.update("jax_default_matmul_precision", "highest")` so XLA does
not downgrade fp32 matmul accumulation (relevant on TPU; CPU fp32 is exact
either way). In bf16 serving, prefer `jax.default_matmul_precision` /
`preferred_element_type` policy at the call site — the model code itself
stays dtype-polymorphic (it computes in the params' dtype, norms/softmax in
fp32).
"""

import dataclasses
import functools
import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Params = Dict[str, Any]
Cache = Tuple[Dict[str, jnp.ndarray], ...]


@dataclasses.dataclass(frozen=True)
class Gemma4DenseConfig:
  """Mirror of the torch port's Gemma4DenseConfig (frozen => hashable/static)."""

  vocab_size: int = 262_144
  hidden_size: int = 5376
  intermediate_size: int = 21504
  num_hidden_layers: int = 60
  num_attention_heads: int = 32
  num_key_value_heads: int = 16
  head_dim: int = 256
  num_global_key_value_heads: int = 4
  global_head_dim: int = 512
  attention_k_eq_v: bool = True
  attention_bias: bool = False
  sliding_window: int = 1024
  max_position_embeddings: int = 262_144
  rms_norm_eps: float = 1e-6
  final_logit_softcapping: Optional[float] = 30.0
  tie_word_embeddings: bool = True
  pad_token_id: int = 0
  rope_theta_full: float = 1_000_000.0
  partial_rotary_factor_full: float = 0.25
  rope_theta_sliding: float = 10_000.0
  # None -> the default 5:1 pattern ([5x sliding, full] repeated).
  layer_types: Optional[Tuple[str, ...]] = None

  def __post_init__(self):
    if self.layer_types is None:
      object.__setattr__(
          self,
          "layer_types",
          tuple(
              "sliding_attention" if bool((i + 1) % 6) else "full_attention"
              for i in range(self.num_hidden_layers)
          ),
      )
    else:
      object.__setattr__(self, "layer_types", tuple(self.layer_types))
    if len(self.layer_types) != self.num_hidden_layers:
      raise ValueError(
          f"layer_types has {len(self.layer_types)} entries, expected"
          f" {self.num_hidden_layers}"
      )

  @classmethod
  def from_text_config(cls, text_config: dict) -> "Gemma4DenseConfig":
    """Builds a config from an HF `text_config` dict (e.g. config.json)."""
    if text_config.get("enable_moe_block"):
      raise ValueError("MoE text models are not supported by this port")
    if text_config.get("hidden_size_per_layer_input"):
      raise ValueError("Per-layer-input models are not supported by this port")
    if text_config.get("num_kv_shared_layers"):
      raise ValueError("KV-shared layers are not supported by this port")
    rope = text_config.get("rope_parameters") or {}
    rope_full = rope.get("full_attention") or {}
    rope_sliding = rope.get("sliding_attention") or {}
    layer_types = text_config.get("layer_types")
    return cls(
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        intermediate_size=text_config["intermediate_size"],
        num_hidden_layers=text_config["num_hidden_layers"],
        num_attention_heads=text_config["num_attention_heads"],
        num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["head_dim"],
        num_global_key_value_heads=text_config["num_global_key_value_heads"],
        global_head_dim=text_config["global_head_dim"],
        attention_k_eq_v=text_config.get("attention_k_eq_v", False),
        attention_bias=text_config.get("attention_bias", False),
        sliding_window=text_config["sliding_window"],
        max_position_embeddings=text_config["max_position_embeddings"],
        rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
        final_logit_softcapping=text_config.get("final_logit_softcapping"),
        tie_word_embeddings=text_config.get("tie_word_embeddings", True),
        pad_token_id=text_config.get("pad_token_id", 0),
        rope_theta_full=rope_full.get("rope_theta", 1_000_000.0),
        partial_rotary_factor_full=rope_full.get("partial_rotary_factor", 0.25),
        rope_theta_sliding=rope_sliding.get("rope_theta", 10_000.0),
        layer_types=tuple(layer_types) if layer_types is not None else None,
    )

  # Per-layer-type attention geometry (mirrors Gemma4DenseAttention.__init__).
  def layer_geometry(self, layer_type: str) -> Tuple[int, int]:
    """Returns (num_kv_heads, head_dim) for a layer type."""
    is_sliding = layer_type == "sliding_attention"
    head_dim = self.head_dim if is_sliding else self.global_head_dim
    use_alternative = self.attention_k_eq_v and not is_sliding
    num_kv = (
        self.num_global_key_value_heads
        if use_alternative
        else self.num_key_value_heads
    )
    return num_kv, head_dim


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _linear(x: jnp.ndarray, proj: Dict[str, jnp.ndarray]) -> jnp.ndarray:
  """x @ W.T (+ bias) with W in the torch (out, in) layout."""
  y = jnp.einsum("...i,oi->...o", x, proj["weight"])
  if "bias" in proj:
    y = y + proj["bias"]
  return y


def rms_norm(
    x: jnp.ndarray, weight: Optional[jnp.ndarray], eps: float
) -> jnp.ndarray:
  """Gemma4 RMSNorm: fp32 `x * pow(mean(x^2)+eps, -0.5)`, scale by plain
  `weight` (NOT 1+weight); `weight=None` is the scale-less v_norm."""
  hidden = x.astype(jnp.float32)
  mean_squared = jnp.mean(hidden * hidden, axis=-1, keepdims=True) + eps
  normed = hidden * jnp.power(mean_squared, -0.5)  # pow, matching torch/HF
  if weight is not None:
    normed = normed * weight.astype(jnp.float32)
  return normed.astype(x.dtype)


def _mlp(mlp_params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
  """gate/up/down MLP with gelu_pytorch_tanh (== jax approximate gelu)."""
  gate = jax.nn.gelu(_linear(x, mlp_params["gate_proj"]), approximate=True)
  return _linear(gate * _linear(x, mlp_params["up_proj"]), mlp_params["down_proj"])


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _default_inv_freq(head_dim: int, theta: float) -> np.ndarray:
  """Standard RoPE inverse frequencies over the full head_dim (fp32)."""
  exponent = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
  return (1.0 / (theta ** (exponent / head_dim))).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _proportional_inv_freq(
    head_dim: int, theta: float, partial_rotary_factor: float
) -> np.ndarray:
  """"proportional" RoPE inv freqs (HF modeling_rope_utils semantics).

  Only the first `int(partial * head_dim // 2)` angles rotate; their
  frequency exponents run over the FULL head_dim (the first quarter of a
  full-spectrum RoPE, not a compressed one). The rest of inv_freq is
  zero-padded => cos=1 / sin=0 => identity rotation on those dims.
  """
  rope_angles = int(partial_rotary_factor * head_dim // 2)
  exponent = np.arange(0, 2 * rope_angles, 2, dtype=np.int64).astype(np.float32)
  inv_freq = (1.0 / (theta ** (exponent / head_dim))).astype(np.float32)
  nope_angles = head_dim // 2 - rope_angles
  if nope_angles > 0:
    inv_freq = np.concatenate(
        [inv_freq, np.zeros(nope_angles, dtype=np.float32)]
    )
  return inv_freq


def _rope_cos_sin(
    config: Gemma4DenseConfig,
    position_ids: jnp.ndarray,  # (B, S) int
    layer_type: str,
    dtype,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """cos/sin of shape (B, S, head_dim) for the given layer type."""
  if layer_type == "sliding_attention":
    inv_freq = _default_inv_freq(config.head_dim, config.rope_theta_sliding)
  else:
    inv_freq = _proportional_inv_freq(
        config.global_head_dim,
        config.rope_theta_full,
        config.partial_rotary_factor_full,
    )
  freqs = position_ids.astype(jnp.float32)[..., None] * jnp.asarray(
      inv_freq
  )[None, None, :]
  emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, S, head_dim)
  return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
  half = x.shape[-1] // 2
  return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> jnp.ndarray:
  """RoPE on a (B, S, H, D) tensor; cos/sin (B, S, D) unsqueezed at dim 2."""
  cos = cos[:, :, None, :]
  sin = sin[:, :, None, :]
  return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _eager_attention(
    query: jnp.ndarray,  # (B, H, S, D)
    key: jnp.ndarray,  # (B, n_kv, T, D)
    value: jnp.ndarray,  # (B, n_kv, T, D)
    attention_mask: jnp.ndarray,  # additive, broadcastable to (B, 1, S, T)
    num_key_value_groups: int,
) -> jnp.ndarray:
  """Plain eager attention, scaling 1.0; softmax in fp32. -> (B, S, H, D)."""
  key = jnp.repeat(key, num_key_value_groups, axis=1)
  value = jnp.repeat(value, num_key_value_groups, axis=1)
  attn_weights = jnp.einsum("bhsd,bhtd->bhst", query, key)  # scaling == 1.0
  attn_weights = attn_weights + attention_mask
  attn_weights = jax.nn.softmax(
      attn_weights.astype(jnp.float32), axis=-1
  ).astype(query.dtype)
  attn_output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, value)
  return jnp.swapaxes(attn_output, 1, 2)  # (B, S, H, D)


def _compute_qkv(
    config: Gemma4DenseConfig,
    attn_params: Dict[str, Any],
    layer_type: str,
    hidden_states: jnp.ndarray,  # (B, S, hidden)
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Q/K/V exactly as in the torch port:

    q: (B, H, S, D)     post q_norm, post RoPE
    k: (B, n_kv, S, D)  post k_norm, post RoPE   <- what gets cached
    v: (B, n_kv, S, D)  post v_norm              <- what gets cached
                        (k_eq_v layers: v_norm of the RAW k_proj output)
  """
  _, head_dim = config.layer_geometry(layer_type)
  batch, seq_len, _ = hidden_states.shape
  eps = config.rms_norm_eps
  is_sliding = layer_type == "sliding_attention"
  k_eq_v = config.attention_k_eq_v and not is_sliding

  # Q: proj -> per-head RMSNorm -> RoPE (in B,S,H,D layout) -> (B,H,S,D).
  query = _linear(hidden_states, attn_params["q_proj"])
  query = query.reshape(batch, seq_len, -1, head_dim)
  query = rms_norm(query, attn_params["q_norm"]["weight"], eps)
  query = _apply_rope(query, cos, sin)
  query = jnp.swapaxes(query, 1, 2)

  # K/V: V branches off the RAW k_proj output when v_proj is absent.
  key_raw = _linear(hidden_states, attn_params["k_proj"])
  key_raw = key_raw.reshape(batch, seq_len, -1, head_dim)
  if k_eq_v:
    value = key_raw  # attention_k_eq_v: pre-norm, pre-RoPE K
  else:
    value = _linear(hidden_states, attn_params["v_proj"])
    value = value.reshape(batch, seq_len, -1, head_dim)

  key = rms_norm(key_raw, attn_params["k_norm"]["weight"], eps)
  key = _apply_rope(key, cos, sin)
  key = jnp.swapaxes(key, 1, 2)

  value = rms_norm(value, None, eps)  # scale-less v_norm
  value = jnp.swapaxes(value, 1, 2)
  return query, key, value


def _attention(
    config: Gemma4DenseConfig,
    attn_params: Dict[str, Any],
    layer_type: str,
    hidden_states: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    attention_mask: jnp.ndarray,
) -> jnp.ndarray:
  """No-cache attention (the parity reference path)."""
  num_kv, _ = config.layer_geometry(layer_type)
  batch, seq_len, _ = hidden_states.shape
  query, key, value = _compute_qkv(
      config, attn_params, layer_type, hidden_states, cos, sin
  )
  attn_output = _eager_attention(
      query, key, value, attention_mask, config.num_attention_heads // num_kv
  )
  attn_output = attn_output.reshape(batch, seq_len, -1)
  return _linear(attn_output, attn_params["o_proj"])


def _cached_attention(
    config: Gemma4DenseConfig,
    attn_params: Dict[str, Any],
    layer_type: str,
    hidden_states: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    attention_mask: jnp.ndarray,
    k_cache: jnp.ndarray,  # (B, n_kv, MAX_SEQ, D)
    v_cache: jnp.ndarray,
    pos,  # scalar int (traced ok): first cache row this call writes
    is_decode: bool,  # Python bool: static
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Attention writing the cache at rows [pos, pos+S).

  Prefill (is_decode=False, pos=0): attends over the fresh in-call K/V with a
  (S, S) mask (identical math to attending over the cache rows just written).
  Decode (is_decode=True, S=1, pos traced): attends over the FULL static
  cache with a (1, 1, 1, MAX_SEQ) additive mask that zeroes future rows and
  the never-written padding. Returns (attn_out, new_k_cache, new_v_cache).
  """
  num_kv, _ = config.layer_geometry(layer_type)
  batch, seq_len, _ = hidden_states.shape
  query, key, value = _compute_qkv(
      config, attn_params, layer_type, hidden_states, cos, sin
  )
  k_cache = jax.lax.dynamic_update_slice(k_cache, key, (0, 0, pos, 0))
  v_cache = jax.lax.dynamic_update_slice(v_cache, value, (0, 0, pos, 0))

  if is_decode:
    key, value = k_cache, v_cache  # attend over the full static cache
  attn_output = _eager_attention(
      query, key, value, attention_mask, config.num_attention_heads // num_kv
  )
  attn_output = attn_output.reshape(batch, seq_len, -1)
  return _linear(attn_output, attn_params["o_proj"]), k_cache, v_cache


# ---------------------------------------------------------------------------
# Decoder layer (sandwich norms + trailing layer_scalar)
# ---------------------------------------------------------------------------


def _layer_body(
    layer_params: Dict[str, Any],
    hidden_states: jnp.ndarray,
    attn_fn,
    eps: float,
) -> jnp.ndarray:
  """Sandwich-norm residual block; `attn_fn(normed_hidden) -> attn_out`."""
  residual = hidden_states
  hidden_states = rms_norm(
      hidden_states, layer_params["input_layernorm"]["weight"], eps
  )
  hidden_states = attn_fn(hidden_states)
  hidden_states = rms_norm(
      hidden_states, layer_params["post_attention_layernorm"]["weight"], eps
  )
  hidden_states = residual + hidden_states

  residual = hidden_states
  hidden_states = rms_norm(
      hidden_states, layer_params["pre_feedforward_layernorm"]["weight"], eps
  )
  hidden_states = _mlp(layer_params["mlp"], hidden_states)
  hidden_states = rms_norm(
      hidden_states, layer_params["post_feedforward_layernorm"]["weight"], eps
  )
  hidden_states = residual + hidden_states

  return hidden_states * layer_params["layer_scalar"]


# ---------------------------------------------------------------------------
# Masks (additive, finfo.min like the torch port)
# ---------------------------------------------------------------------------


def _big_neg(dtype) -> jnp.ndarray:
  return jnp.asarray(jnp.finfo(dtype).min, dtype=dtype)


def _create_causal_mask(seq_len: int, dtype) -> jnp.ndarray:
  """(1, 1, S, S) additive causal mask (min where kv_idx > q_idx)."""
  q_idx = jnp.arange(seq_len)[:, None]
  kv_idx = jnp.arange(seq_len)[None, :]
  keep = kv_idx <= q_idx
  mask = jnp.where(keep, jnp.asarray(0.0, dtype), _big_neg(dtype))
  return mask[None, None, :, :]


def _create_sliding_window_causal_mask(
    seq_len: int, sliding_window: int, dtype
) -> jnp.ndarray:
  """(1, 1, S, S) additive EXCLUSIVE sliding-window causal mask:
  allowed iff 0 <= q_idx - kv_idx < sliding_window (self included)."""
  q_idx = jnp.arange(seq_len)[:, None]
  kv_idx = jnp.arange(seq_len)[None, :]
  diff = q_idx - kv_idx
  keep = (diff >= 0) & (diff < sliding_window)
  mask = jnp.where(keep, jnp.asarray(0.0, dtype), _big_neg(dtype))
  return mask[None, None, :, :]


def _create_decode_masks(
    pos, max_seq_len: int, sliding_window: int, dtype
) -> Dict[str, jnp.ndarray]:
  """(1, 1, 1, MAX_SEQ) additive masks for one decode step at (traced) `pos`.

    full_attention:    keep iff kv_idx <= pos                     (causal)
    sliding_attention: keep iff kv_idx <= pos AND pos - kv_idx < window
                       (the EXCLUSIVE Gemma4 window, self included)

  Future rows AND the never-written cache padding get finfo.min -> exact 0
  after softmax.
  """
  kv_idx = jnp.arange(max_seq_len)
  keep_causal = kv_idx <= pos
  keep_sliding = keep_causal & ((pos - kv_idx) < sliding_window)
  zero = jnp.asarray(0.0, dtype)
  neg = _big_neg(dtype)
  return {
      "full_attention": jnp.where(keep_causal, zero, neg)[None, None, None, :],
      "sliding_attention": jnp.where(keep_sliding, zero, neg)[
          None, None, None, :
      ],
  }


def _apply_padding_mask(
    mask: jnp.ndarray, attention_mask: jnp.ndarray
) -> jnp.ndarray:
  """Merges a 2D (B, S) padding mask (1 = keep) into an additive 4D mask."""
  dtype = mask.dtype
  padding = (
      1.0 - attention_mask[:, None, None, :].astype(dtype)
  ) * _big_neg(dtype)
  return mask + padding


# ---------------------------------------------------------------------------
# Model-level pure functions
# ---------------------------------------------------------------------------


def _model_dtype(params: Params):
  return params["model"]["embed_tokens"]["weight"].dtype


def _embed(config: Gemma4DenseConfig, params: Params, input_ids) -> jnp.ndarray:
  """Token embeddings scaled by sqrt(hidden_size)."""
  weight = params["model"]["embed_tokens"]["weight"]
  scale = jnp.asarray(math.sqrt(config.hidden_size), dtype=weight.dtype)
  return weight[input_ids] * scale


def _position_embeddings(
    config: Gemma4DenseConfig, position_ids: jnp.ndarray, dtype
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  """cos/sin per layer type; both computed unconditionally (static)."""
  return {
      layer_type: _rope_cos_sin(config, position_ids, layer_type, dtype)
      for layer_type in ("sliding_attention", "full_attention")
  }


def _cap_logits(config: Gemma4DenseConfig, logits: jnp.ndarray) -> jnp.ndarray:
  if config.final_logit_softcapping is not None:
    cap = config.final_logit_softcapping
    logits = jnp.tanh(logits / cap) * cap
  return logits


def forward(
    config: Gemma4DenseConfig,
    params: Params,
    input_ids: jnp.ndarray,  # (B, S) int
    attention_mask: Optional[jnp.ndarray] = None,  # (B, S), 1 = attend
    return_hidden_states: bool = False,
):
  """No-cache full-sequence forward -> (B, S, vocab) softcapped logits.

  With `return_hidden_states`, also returns the list
  [embeddings_out, layer_0_out, ..., layer_{L-1}_out] (pre-final-norm), for
  per-layer parity localization.
  """
  dtype = _model_dtype(params)
  seq_len = input_ids.shape[1]
  position_ids = jnp.arange(seq_len)[None, :]

  masks = {
      "full_attention": _create_causal_mask(seq_len, dtype),
      "sliding_attention": _create_sliding_window_causal_mask(
          seq_len, config.sliding_window, dtype
      ),
  }
  if attention_mask is not None:
    masks = {k: _apply_padding_mask(v, attention_mask) for k, v in masks.items()}

  hidden_states = _embed(config, params, input_ids)
  position_embeddings = _position_embeddings(config, position_ids, dtype)

  all_hidden = [hidden_states] if return_hidden_states else None
  layers = params["model"]["layers"]
  for layer_idx, layer_type in enumerate(config.layer_types):
    layer_params = layers[str(layer_idx)]
    cos, sin = position_embeddings[layer_type]
    attn_fn = lambda h, lp=layer_params, lt=layer_type, c=cos, s=sin: _attention(
        config, lp["self_attn"], lt, h, c, s, masks[lt]
    )
    hidden_states = _layer_body(
        layer_params, hidden_states, attn_fn, config.rms_norm_eps
    )
    if return_hidden_states:
      all_hidden.append(hidden_states)

  hidden_states = rms_norm(
      hidden_states, params["model"]["norm"]["weight"], config.rms_norm_eps
  )
  logits = _cap_logits(config, _linear(hidden_states, params["lm_head"]))
  if return_hidden_states:
    return logits, all_hidden
  return logits


# ----- static-KV-cache path -------------------------------------------------


def init_cache(
    config: Gemma4DenseConfig,
    batch_size: int,
    max_seq_len: int,
    dtype=jnp.float32,
) -> Cache:
  """Zeroed static cache: per layer {"k","v"} of [B, n_kv, MAX_SEQ, head_dim]
  in that layer's geometry. K stored post-k_norm/post-RoPE, V post-v_norm."""
  cache = []
  for layer_type in config.layer_types:
    num_kv, head_dim = config.layer_geometry(layer_type)
    shape = (batch_size, num_kv, max_seq_len, head_dim)
    cache.append({
        "k": jnp.zeros(shape, dtype=dtype),
        "v": jnp.zeros(shape, dtype=dtype),
    })
  return tuple(cache)


def _cached_stack(
    config: Gemma4DenseConfig,
    params: Params,
    cache: Cache,
    hidden_states: jnp.ndarray,
    position_embeddings: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    masks: Dict[str, jnp.ndarray],
    pos,
    is_decode: bool,
) -> Tuple[jnp.ndarray, Cache]:
  """Runs all decoder layers through the cached attention; final norm at end."""
  layers = params["model"]["layers"]
  new_cache = []
  for layer_idx, layer_type in enumerate(config.layer_types):
    layer_params = layers[str(layer_idx)]
    cos, sin = position_embeddings[layer_type]
    layer_cache = cache[layer_idx]
    result = {}

    def attn_fn(h, lp=layer_params, lt=layer_type, c=cos, s=sin,
                lc=layer_cache, out=result):
      attn_out, k_cache, v_cache = _cached_attention(
          config, lp["self_attn"], lt, h, c, s, masks[lt],
          lc["k"], lc["v"], pos, is_decode,
      )
      out["k"], out["v"] = k_cache, v_cache
      return attn_out

    hidden_states = _layer_body(
        layer_params, hidden_states, attn_fn, config.rms_norm_eps
    )
    new_cache.append(result)

  hidden_states = rms_norm(
      hidden_states, params["model"]["norm"]["weight"], config.rms_norm_eps
  )
  return hidden_states, tuple(new_cache)


def prefill(
    config: Gemma4DenseConfig,
    params: Params,
    cache: Cache,
    input_ids: jnp.ndarray,  # (B, S) int, unpadded prompt at positions 0..S-1
) -> Tuple[jnp.ndarray, Cache]:
  """Prompt forward: fills cache rows 0..S-1, returns
  (last-position softcapped logits (B, vocab), updated cache). Pure."""
  dtype = _model_dtype(params)
  seq_len = input_ids.shape[1]
  position_ids = jnp.arange(seq_len)[None, :]

  masks = {
      "full_attention": _create_causal_mask(seq_len, dtype),
      "sliding_attention": _create_sliding_window_causal_mask(
          seq_len, config.sliding_window, dtype
      ),
  }
  hidden_states = _embed(config, params, input_ids)
  position_embeddings = _position_embeddings(config, position_ids, dtype)

  hidden_states, cache = _cached_stack(
      config, params, cache, hidden_states, position_embeddings, masks,
      pos=0, is_decode=False,
  )
  logits = _cap_logits(
      config, _linear(hidden_states[:, -1, :], params["lm_head"])
  )
  return logits, cache


def decode_step(
    config: Gemma4DenseConfig,
    params: Params,
    cache: Cache,
    input_ids: jnp.ndarray,  # (B, 1) int: the token at absolute position pos
    pos,  # scalar int32 (traced): absolute position of input_ids
) -> Tuple[jnp.ndarray, Cache]:
  """One cached greedy-decode step -> (next-token logits (B, vocab), cache).

  Pure and fully static-shape: cache row `pos` is written via
  dynamic_update_slice, masks come from arange(MAX_SEQ) + pos, so
  `jax.jit(decode_step, static_argnums=0)` (or a config-closing lambda)
  traces exactly once and never retraces across steps.
  """
  dtype = _model_dtype(params)
  max_seq_len = cache[0]["k"].shape[2]
  pos = jnp.asarray(pos, dtype=jnp.int32)
  position_ids = pos.reshape(1, 1)

  masks = _create_decode_masks(
      pos, max_seq_len, config.sliding_window, dtype
  )
  hidden_states = _embed(config, params, input_ids)
  position_embeddings = _position_embeddings(config, position_ids, dtype)

  hidden_states, cache = _cached_stack(
      config, params, cache, hidden_states, position_embeddings, masks,
      pos=pos, is_decode=True,
  )
  logits = _cap_logits(
      config, _linear(hidden_states[:, -1, :], params["lm_head"])
  )
  return logits, cache
