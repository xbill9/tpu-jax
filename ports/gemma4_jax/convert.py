"""torch state_dict / safetensors -> Gemma4 JAX params pytree.

Produces the nested dict-of-dicts pytree documented in `model.py`: each
dotted torch state-dict key becomes a nesting path (layer indices stay STRING
keys), e.g.

    "model.layers.3.self_attn.q_proj.weight"
        -> params["model"]["layers"]["3"]["self_attn"]["q_proj"]["weight"]

Key mapping follows the torch port's `load_hf_state_dict()` conventions:

  * `Gemma4ForConditionalGeneration` checkpoints: "model.language_model.*"
    -> "model.*"; vision/audio/projector keys are dropped.
  * Text-only state dicts ("model.*" / "lm_head.*") pass through unchanged.
  * A missing "lm_head.weight" is filled by ALIASING the embed_tokens weight
    (tied embeddings) — no copy is made; the pytree simply references the
    same jnp array twice.

Differences vs. the torch loader (documented per the port spec):

  * The torch port loads into an nn.Module whose non-persistent buffers
    (RoPE `inv_freq`, `embed_scale`) are constructed by the module; here
    those are NOT params at all — `model.py` recomputes them from the
    config, so they never appear in the pytree.
  * The persistent per-layer "layer_scalar" buffers ARE model state in the
    torch state dict, and are kept in the pytree (shape (1,)).
  * torch `load_state_dict(strict=True)` validates key coverage against the
    module; here there is no module, so `state_dict_to_params` converts
    whatever text-model keys survive the remap. Validate shapes downstream
    (the parity test does this implicitly by exact numeric comparison).

dtype: preserved by default — torch bfloat16 tensors become jnp.bfloat16
(via a lossless bf16 -> fp32 -> bf16 round trip, since numpy has no native
bf16). Pass `dtype=` to cast everything (e.g. jnp.float32 for parity runs).
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp

_DROP_PREFIXES = (
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
    "model.multi_modal_projector.",
    "vision_tower.",
    "audio_tower.",
)


def remap_hf_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Applies the torch port's load_hf_state_dict key remap (no tensor work)."""
  remapped = {}
  for key, value in state_dict.items():
    if key.startswith(_DROP_PREFIXES):
      continue
    if key.startswith("model.language_model."):
      key = "model." + key[len("model.language_model."):]
    elif key.startswith("language_model."):
      key = "model." + key[len("language_model."):]
    remapped[key] = value
  return remapped


def _to_jnp(tensor, dtype) -> jnp.ndarray:
  """torch tensor (or array-like) -> jnp array, preserving bf16."""
  try:
    import torch  # deferred: only needed when converting torch tensors
  except ImportError:
    torch = None
  if torch is not None and isinstance(tensor, torch.Tensor):
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
      # numpy has no bf16; bf16 -> fp32 -> jnp.bfloat16 is value-lossless.
      arr = jnp.asarray(t.to(torch.float32).numpy()).astype(jnp.bfloat16)
    else:
      arr = jnp.asarray(t.numpy())
  else:
    arr = jnp.asarray(tensor)
  if dtype is not None:
    arr = arr.astype(dtype)
  return arr


def state_dict_to_params(
    state_dict: Dict[str, Any],
    dtype=None,
    tie_lm_head: bool = True,
) -> Dict[str, Any]:
  """torch state_dict (HF Gemma4 or this repo's torch port) -> params pytree.

  Args:
    state_dict: flat {dotted key: tensor} mapping. HF multimodal prefixes are
      remapped/dropped per `remap_hf_keys`.
    dtype: optional jnp dtype to cast every array to (default: preserve).
    tie_lm_head: fill a missing "lm_head.weight" by aliasing
      "model.embed_tokens.weight".

  Returns:
    Nested dict-of-dicts pytree of jnp arrays (see module docstring).
  """
  remapped = remap_hf_keys(state_dict)
  params: Dict[str, Any] = {}
  for key, tensor in remapped.items():
    node = params
    parts = key.split(".")
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = _to_jnp(tensor, dtype)

  if tie_lm_head and "lm_head" not in params:
    embed = params["model"]["embed_tokens"]["weight"]
    params["lm_head"] = {"weight": embed}  # alias, not a copy
  return params


def load_safetensors_params(
    path: str,
    dtype=None,
    tie_lm_head: bool = True,
) -> Dict[str, Any]:
  """Loads a .safetensors file into the params pytree (needs `safetensors`)."""
  from safetensors import torch as st_torch  # deferred optional dependency

  return state_dict_to_params(
      st_torch.load_file(path), dtype=dtype, tie_lm_head=tie_lm_head
  )
