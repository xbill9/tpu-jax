"""Compile a small Gemma-like decoder block with the JAX Neuron backend.

This is a compiler/runtime compatibility probe, not a performance benchmark.
It deliberately uses fixed shapes and functional KV-cache updates, matching
the constraints of the planned Gemma 4 decode path.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp


BATCH = 1
QUERY_HEADS = 8
KV_HEADS = 4
HEAD_DIM = 64
HIDDEN = QUERY_HEADS * HEAD_DIM
INTERMEDIATE = 1024
MAX_LENGTH = 128


def rms_norm(x: jax.Array, scale: jax.Array) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + 1e-6).astype(x.dtype)
    return normalized * scale


def rotate_half(x: jax.Array) -> jax.Array:
    left, right = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-right, left), axis=-1)


def decoder_step(
    token: jax.Array,
    position: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    weights: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One fixed-shape decoder step with grouped-query attention."""
    x = rms_norm(token, weights["norm"])
    query = (x @ weights["q"]).reshape(BATCH, QUERY_HEADS, HEAD_DIM)
    key = (x @ weights["k"]).reshape(BATCH, KV_HEADS, HEAD_DIM)
    value = (x @ weights["v"]).reshape(BATCH, KV_HEADS, HEAD_DIM)

    angle = position.astype(x.dtype) * weights["rope"]
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    query = query * cosine + rotate_half(query) * sine
    key = key * cosine[:, :KV_HEADS] + rotate_half(key) * sine[:, :KV_HEADS]

    key_cache = jax.lax.dynamic_update_slice(
        key_cache, key[:, :, None, :], (0, 0, position[0], 0)
    )
    value_cache = jax.lax.dynamic_update_slice(
        value_cache, value[:, :, None, :], (0, 0, position[0], 0)
    )

    repeat = QUERY_HEADS // KV_HEADS
    keys = jnp.repeat(key_cache, repeat, axis=1)
    values = jnp.repeat(value_cache, repeat, axis=1)
    scores = jnp.einsum("bhd,bhld->bhl", query, keys) / jnp.sqrt(HEAD_DIM)
    valid = jnp.arange(MAX_LENGTH)[None, None, :] <= position[:, None, None]
    probs = jax.nn.softmax(jnp.where(valid, scores, -1e30), axis=-1)
    attended = jnp.einsum("bhl,bhld->bhd", probs, values).reshape(BATCH, HIDDEN)

    residual = token + attended @ weights["o"]
    y = rms_norm(residual, weights["post_norm"])
    gated = jax.nn.gelu(y @ weights["gate"], approximate=True) * (y @ weights["up"])
    return residual + gated @ weights["down"], key_cache, value_cache


def inputs() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    key = jax.random.key(7)
    keys = jax.random.split(key, 9)

    def normal(k: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return (jax.random.normal(k, shape, dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)

    weights = {
        "norm": jnp.ones((HIDDEN,), dtype=jnp.bfloat16),
        "post_norm": jnp.ones((HIDDEN,), dtype=jnp.bfloat16),
        "q": normal(keys[0], (HIDDEN, QUERY_HEADS * HEAD_DIM)),
        "k": normal(keys[1], (HIDDEN, KV_HEADS * HEAD_DIM)),
        "v": normal(keys[2], (HIDDEN, KV_HEADS * HEAD_DIM)),
        "o": normal(keys[3], (HIDDEN, HIDDEN)),
        "gate": normal(keys[4], (HIDDEN, INTERMEDIATE)),
        "up": normal(keys[5], (HIDDEN, INTERMEDIATE)),
        "down": normal(keys[6], (INTERMEDIATE, HIDDEN)),
        "rope": normal(keys[7], (BATCH, QUERY_HEADS, HEAD_DIM)),
    }
    token = normal(keys[8], (BATCH, HIDDEN))
    position = jnp.array([0], dtype=jnp.int32)
    cache_shape = (BATCH, KV_HEADS, MAX_LENGTH, HEAD_DIM)
    key_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
    value_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
    return token, position, key_cache, value_cache, weights


def main() -> None:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")

    compiled = jax.jit(decoder_step)
    args = inputs()
    started = time.perf_counter()
    hidden, key_cache, value_cache = compiled(*args)
    jax.block_until_ready(hidden)
    first_call_s = time.perf_counter() - started

    started = time.perf_counter()
    hidden, key_cache, value_cache = compiled(
        hidden, jnp.array([1], dtype=jnp.int32), key_cache, value_cache, args[-1]
    )
    jax.block_until_ready(hidden)
    warm_call_s = time.perf_counter() - started

    finite = bool(jnp.all(jnp.isfinite(hidden)))
    if not finite:
        raise RuntimeError("decoder probe produced non-finite output")

    print(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "platform": devices[0].platform,
                "device_count": len(devices),
                "devices": [str(device) for device in devices],
                "first_call_s": round(first_call_s, 4),
                "warm_call_s": round(warm_call_s, 4),
                "output_shape": list(hidden.shape),
                "finite": finite,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

