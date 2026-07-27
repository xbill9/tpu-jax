"""JAX PyTree Parameter Loader for Gemma 4 E2B QAT Checkpoints.

Loads Hugging Face safetensors checkpoints (e.g. google/gemma-4-E2B-it-qat-w4a16-ct or
google/gemma-4-E2B-it-qat-q4_0-unquantized) directly into JAX PyTree dictionaries.

Pure Python & JAX — ZERO PyTorch, transformers, or confidential repository dependencies.
"""

from typing import Dict, Any, Tuple
import jax
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

    # Helper to get array as jnp bfloat16 or int8
    def get_arr(key: str, default_dtype=jnp.bfloat16):
        if key in raw_weights:
            arr = raw_weights[key]
            if not isinstance(arr, jnp.ndarray):
                arr = jnp.array(arr)
            return arr if arr.dtype in (jnp.int8, jnp.uint8) else arr.astype(default_dtype)
        return None

    # Global Embeddings & Norms
    jax_params["embed_tokens"] = get_arr("model.embed_tokens.weight")
    jax_params["final_norm"] = get_arr("model.norm.weight")

    if "model.embed_tokens_per_layer.weight" in raw_weights:
        jax_params["embed_tokens_per_layer"] = get_arr("model.embed_tokens_per_layer.weight")
        jax_params["per_layer_model_projection"] = get_arr("model.per_layer_model_projection.weight")
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
            layer_p["per_layer_input_gate"] = get_arr(f"{prefix}.per_layer_input_gate.weight")
            layer_p["per_layer_projection"] = get_arr(f"{prefix}.per_layer_projection.weight")
            layer_p["post_per_layer_input_norm"] = get_arr(f"{prefix}.post_per_layer_input_norm.weight")

        # Attention weights
        attn_p: Dict[str, Any] = {}
        for proj in ["q_proj", "o_proj"]:
            if f"{prefix}.self_attn.{proj}.weight" in raw_weights:
                attn_p[proj] = get_arr(f"{prefix}.self_attn.{proj}.weight")
            elif f"{prefix}.self_attn.{proj}.weight_packed" in raw_weights:
                attn_p[f"{proj}_packed"] = get_arr(f"{prefix}.self_attn.{proj}.weight_packed")
                attn_p[f"{proj}_scale"] = get_arr(f"{prefix}.self_attn.{proj}.weight_scale")

        if f"{prefix}.self_attn.q_norm.weight" in raw_weights:
            attn_p["q_norm"] = get_arr(f"{prefix}.self_attn.q_norm.weight")

        if not is_shared:
            for proj in ["k_proj", "v_proj"]:
                if f"{prefix}.self_attn.{proj}.weight" in raw_weights:
                    attn_p[proj] = get_arr(f"{prefix}.self_attn.{proj}.weight")
                elif f"{prefix}.self_attn.{proj}.weight_packed" in raw_weights:
                    attn_p[f"{proj}_packed"] = get_arr(f"{prefix}.self_attn.{proj}.weight_packed")
                    attn_p[f"{proj}_scale"] = get_arr(f"{prefix}.self_attn.{proj}.weight_scale")

            if f"{prefix}.self_attn.k_norm.weight" in raw_weights:
                attn_p["k_norm"] = get_arr(f"{prefix}.self_attn.k_norm.weight")
            if f"{prefix}.self_attn.v_norm.weight" in raw_weights:
                attn_p["v_norm"] = get_arr(f"{prefix}.self_attn.v_norm.weight")

        layer_p["attn"] = attn_p

        # MLP weights
        mlp_p: Dict[str, Any] = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if f"{prefix}.mlp.{proj}.weight" in raw_weights:
                mlp_p[proj] = get_arr(f"{prefix}.mlp.{proj}.weight")
            elif f"{prefix}.mlp.{proj}.weight_packed" in raw_weights:
                mlp_p[f"{proj}_packed"] = get_arr(f"{prefix}.mlp.{proj}.weight_packed")
                mlp_p[f"{proj}_scale"] = get_arr(f"{prefix}.mlp.{proj}.weight_scale")

        layer_p["mlp"] = mlp_p
        jax_params[f"layer_{i}"] = layer_p

    return jax_params
