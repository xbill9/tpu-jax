"""JAX PyTree Parameter Loader for Gemma 4 E2B QAT Checkpoints.

Loads Hugging Face safetensors checkpoints (e.g. google/gemma-4-E2B-it-qat-w4a16-ct or
google/gemma-4-E2B-it-qat-q4_0-unquantized) directly into JAX PyTree dictionaries.

Pure Python & JAX — ZERO PyTorch, transformers, or confidential repository dependencies.
"""

from typing import Dict, Any
import jax.numpy as jnp


def convert_safetensors_to_jax_params(
    raw_weights: Dict[str, Any],
    num_layers: int = 35,
    first_kv_shared_idx: int = 15,
) -> Dict[str, Any]:
    """Converts a dict of safetensors numpy/jax arrays into Gemma4EModelJAX parameter PyTree.

    Handles key remapping for:
      - Primary and Per-Layer Embeddings (PLE)
      - Non-shared layers (0..14) with q/k/v/o attention projections
      - Shared layers (15..34) with q/o attention projections (no k/v)
      - MatFormer Double-Wide MLP weights
      - QAT packed W4A16 weights and scales (_packed, _scale suffix mapping)
    """
    jax_params: Dict[str, Any] = {}

    def get_arr(key: str, default_dtype=jnp.bfloat16):
        if key in raw_weights:
            arr = raw_weights[key]
            if not isinstance(arr, jnp.ndarray):
                arr = jnp.array(arr)
            if jnp.issubdtype(arr.dtype, jnp.integer):
                return arr
            return arr.astype(default_dtype)
        return None

    def get_linear(key: str):
        """Convert HF ``[out, in]`` dense weights to JAX ``[in, out]``."""
        arr = get_arr(key)
        return None if arr is None else arr.T

    def get_quantized(prefix: str, destination: Dict[str, Any], name: str):
        packed_key = f"{prefix}.{name}.weight_packed"
        scale_key = f"{prefix}.{name}.weight_scale"
        if packed_key not in raw_weights:
            return False
        if scale_key not in raw_weights:
            raise ValueError(f"Missing W4A16 scale tensor: {scale_key}")
        packed = get_arr(packed_key)
        scale = get_arr(scale_key)
        if packed.dtype != jnp.int32:
            raise TypeError(
                f"{packed_key} must be compressed-tensors int32, got "
                f"{packed.dtype}"
            )
        if packed.ndim != 2 or scale.ndim != 2:
            raise ValueError(
                f"{packed_key}/{scale_key} must be rank 2, got "
                f"{packed.shape}/{scale.shape}"
            )
        expected_scale = (packed.shape[0], packed.shape[1] // 4)
        if scale.shape != expected_scale:
            raise ValueError(
                f"{scale_key} has shape {scale.shape}; expected "
                f"{expected_scale} for group_size=32"
            )
        destination[f"{name}_packed"] = packed
        destination[f"{name}_scale"] = scale.astype(jnp.bfloat16)
        return True

    # Global Embeddings & Norms
    jax_params["embed_tokens"] = get_arr("model.embed_tokens.weight")
    jax_params["final_norm"] = get_arr("model.norm.weight")

    if "model.embed_tokens_per_layer.weight" in raw_weights:
        jax_params["embed_tokens_per_layer"] = get_arr("model.embed_tokens_per_layer.weight")
        jax_params["per_layer_model_projection"] = get_linear("model.per_layer_model_projection.weight")
        jax_params["per_layer_projection_norm"] = get_arr("model.per_layer_projection_norm.weight")

    # Layer Weights Loop
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        is_shared = i >= first_kv_shared_idx
        layer_p: Dict[str, Any] = {}

        # Layernorms
        layer_p["input_layernorm"] = get_arr(f"{prefix}.input_layernorm.weight")
        layer_p["post_attention_layernorm"] = get_arr(f"{prefix}.post_attention_layernorm.weight")

        # PLE per-layer weights
        if f"{prefix}.per_layer_input_gate.weight" in raw_weights:
            layer_p["per_layer_input_gate"] = get_linear(f"{prefix}.per_layer_input_gate.weight")
            layer_p["per_layer_projection"] = get_linear(f"{prefix}.per_layer_projection.weight")
            layer_p["post_per_layer_input_norm"] = get_arr(f"{prefix}.post_per_layer_input_norm.weight")

        # Attention weights
        attn_p: Dict[str, Any] = {}
        for proj in ["q_proj", "o_proj"]:
            proj_prefix = f"{prefix}.self_attn"
            if not get_quantized(proj_prefix, attn_p, proj):
                dense_key = f"{proj_prefix}.{proj}.weight"
                if dense_key in raw_weights:
                    attn_p[proj] = get_linear(dense_key)

        if f"{prefix}.self_attn.q_norm.weight" in raw_weights:
            attn_p["q_norm"] = get_arr(f"{prefix}.self_attn.q_norm.weight")

        if not is_shared:
            for proj in ["k_proj", "v_proj"]:
                proj_prefix = f"{prefix}.self_attn"
                if not get_quantized(proj_prefix, attn_p, proj):
                    dense_key = f"{proj_prefix}.{proj}.weight"
                    if dense_key in raw_weights:
                        attn_p[proj] = get_linear(dense_key)

            if f"{prefix}.self_attn.k_norm.weight" in raw_weights:
                attn_p["k_norm"] = get_arr(f"{prefix}.self_attn.k_norm.weight")
            if f"{prefix}.self_attn.v_norm.weight" in raw_weights:
                attn_p["v_norm"] = get_arr(f"{prefix}.self_attn.v_norm.weight")

        layer_p["attn"] = attn_p

        # MLP weights
        mlp_p: Dict[str, Any] = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            proj_prefix = f"{prefix}.mlp"
            if not get_quantized(proj_prefix, mlp_p, proj):
                dense_key = f"{proj_prefix}.{proj}.weight"
                if dense_key in raw_weights:
                    mlp_p[proj] = get_linear(dense_key)

        layer_p["mlp"] = mlp_p
        jax_params[f"layer_{i}"] = layer_p

    return jax_params
