"""Windowed KV cache must be a memory optimization, not a behaviour change.

Sliding-attention layers can never attend outside their window, so allocating
`max_seq_len` slots for them is dead memory. These tests assert the ring-buffer
version produces the SAME tokens as the full-length cache, including when the
prompt is longer than the window (the case the ring exists for).

Run: python3 -m unittest tests.test_windowed_kv
"""

import dataclasses
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ports.gemma4.jax_e_model import (  # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    make_cached_decode_step,
    make_prefill_causal_mask,
    prefill_with_kv_cache,
)
from test_kv_cache_parity import DTYPE, build_tiny_params  # noqa: E402

WINDOW = 4


def windowed_config() -> Gemma4EConfig:
    """Tiny config with a window small enough that prompts exceed it."""
    return Gemma4EConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96,
        num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_global_key_value_heads=2, global_head_dim=32,
        num_kv_shared_layers=4, use_double_wide_mlp=True,
        hidden_size_per_layer_input=16, vocab_size_per_layer_input=128,
        sliding_window=WINDOW,
    )


def generate(model, params, prompt, n, window_kv, prompt_len=None):
    """Prefill + greedy decode.

    prompt_len: number of REAL tokens; the rest of `prompt` is padding. Serving
    always pads to a bucket, so leaving this None (fully valid) is the case that
    hid a live bug for the windowed path — see ShortGenerationWindowedKVTest.
    """
    B, S = prompt.shape
    if prompt_len is None:
        valid_prompt = jnp.ones((B, S), dtype=jnp.bool_)
    else:
        valid_prompt = (jnp.arange(S)[None, :] < prompt_len).repeat(B, axis=0)
    last, caches, valid = prefill_with_kv_cache(
        model, prompt, valid_prompt, params, n,
        quant_mode="fp16", cache_dtype=DTYPE, window_kv=window_kv,
    )
    step = jax.jit(make_cached_decode_step(model, quant_mode="fp16", window_kv=window_kv))
    # Logical position advances from the REAL prompt length while the cache slot
    # advances from the padded bucket edge — exactly how JaxGemmaEngine drives it.
    lens = jnp.full((B,), S if prompt_len is None else int(prompt_len), dtype=jnp.int32)
    tok = jnp.argmax(last, axis=-1, keepdims=True)
    out, logits = [tok], [last]
    for t in range(n - 1):
        caches, valid, last = step(params, caches, valid, tok, lens + t, jnp.int32(S + t))
        logits.append(last)
        tok = jnp.argmax(last, axis=-1, keepdims=True)
        out.append(tok)
    return jnp.concatenate(out, axis=1), logits


class WindowedKVTest(unittest.TestCase):
    def setUp(self):
        self.config = windowed_config()
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    def test_sliding_layers_allocate_only_the_window(self):
        full = init_kv_cache(self.config, 1, 64, DTYPE, window_kv=False)
        win = init_kv_cache(self.config, 1, 64, DTYPE, window_kv=True)
        sliding = [i for i in range(self.config.first_kv_shared_layer_idx)
                   if self.config.layer_types[i] == "sliding_attention"]
        full_att = [i for i in range(self.config.first_kv_shared_layer_idx)
                    if self.config.layer_types[i] == "full_attention"]
        self.assertTrue(sliding and full_att, "config must exercise both layer types")
        for i in sliding:
            self.assertEqual(win[i][0].shape[2], WINDOW, f"layer {i} should hold only the window")
            self.assertEqual(full[i][0].shape[2], 64)
        for i in full_att:
            self.assertEqual(win[i][0].shape[2], 64, f"full-attention layer {i} must keep full length")

    def test_saves_memory(self):
        def total(c):
            return sum(k.size * k.dtype.itemsize + v.size * v.dtype.itemsize for k, v in c.values())
        full = total(init_kv_cache(self.config, 1, 64, DTYPE, window_kv=False))
        win = total(init_kv_cache(self.config, 1, 64, DTYPE, window_kv=True))
        self.assertLess(win, full)

    def test_same_tokens_when_prompt_fits_in_window(self):
        prompt = jax.random.randint(jax.random.PRNGKey(1), (1, WINDOW - 1), 1, self.config.vocab_size)
        a, _ = generate(self.model, self.params, prompt, 4, window_kv=False)
        b, _ = generate(self.model, self.params, prompt, 4, window_kv=True)
        self.assertEqual(a.tolist(), b.tolist())

    def test_same_tokens_when_prompt_exceeds_window(self):
        """The case the ring buffer exists for: prompt longer than the window."""
        prompt = jax.random.randint(jax.random.PRNGKey(2), (2, 3 * WINDOW + 1), 1, self.config.vocab_size)
        a, la = generate(self.model, self.params, prompt, 5, window_kv=False)
        b, lb = generate(self.model, self.params, prompt, 5, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            d = float(jnp.max(jnp.abs(x - y)))
            self.assertLess(d, 1e-4, f"step {i} logits diverge by {d:.2e} under windowed KV")
        self.assertEqual(a.tolist(), b.tolist())

    def test_decode_wraps_past_the_window(self):
        """Generate more tokens than the window so the ring wraps during decode."""
        prompt = jax.random.randint(jax.random.PRNGKey(3), (1, 2), 1, self.config.vocab_size)
        n = 2 * WINDOW + 2
        a, la = generate(self.model, self.params, prompt, n, window_kv=False)
        b, lb = generate(self.model, self.params, prompt, n, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            self.assertLess(float(jnp.max(jnp.abs(x - y))), 1e-4, f"step {i} diverges after wrap")
        self.assertEqual(a.tolist(), b.tolist())


class ShortGenerationWindowedKVTest(unittest.TestCase):
    """The whole sequence fits inside the window.

    Every test above runs a sequence at least as long as the window, so the
    `min(max_seq_len, sliding_window)` clamp in `init_kv_cache` never fires and
    the ring buffer is always exactly `window` slots wide. Below the window the
    buffer is NARROWER, and the decode step used to build its mask from
    `config.sliding_window` regardless:

        add got incompatible shapes for broadcasting:
        (1, 8, 1, 88), (1, 1, 1, 512)

    This is not an exotic configuration — it is what serving a short prompt with
    a modest token budget does on real E2B geometry, which is how it surfaced.
    """

    WINDOW = 64

    def setUp(self):
        cfg = windowed_config()
        self.config = dataclasses.replace(cfg, sliding_window=self.WINDOW)
        self.model = Gemma4EModelJAX(self.config)
        self.params = build_tiny_params(self.config)

    def test_sliding_buffer_is_clamped_to_the_sequence(self):
        cache = init_kv_cache(self.config, 1, 8, DTYPE, window_kv=True)
        sliding = [i for i in sorted(cache)
                   if self.config.layer_types[i] == "sliding_attention"]
        self.assertTrue(sliding, "config has no sliding layers to check")
        self.assertEqual(int(cache[sliding[0]][0].shape[2]), 8)

    def test_decode_runs_when_the_sequence_is_shorter_than_the_window(self):
        prompt = jax.random.randint(jax.random.PRNGKey(11), (1, 2), 1, self.config.vocab_size)
        tokens, _ = generate(self.model, self.params, prompt, 3, window_kv=True)
        self.assertEqual(tokens.shape[1], 3)

    def test_windowing_is_a_noop_when_nothing_can_leave_the_window(self):
        """A window wider than the whole sequence must change no token."""
        prompt = jax.random.randint(jax.random.PRNGKey(12), (1, 3), 1, self.config.vocab_size)
        a, la = generate(self.model, self.params, prompt, 4, window_kv=False)
        b, lb = generate(self.model, self.params, prompt, 4, window_kv=True)
        for i, (x, y) in enumerate(zip(la, lb)):
            d = float(jnp.max(jnp.abs(x - y)))
            self.assertLess(d, 1e-4, f"step {i} logits diverge by {d:.2e}")
        self.assertEqual(a.tolist(), b.tolist())

    def test_padded_prompt_matches_unwindowed(self):
        """The case that shipped broken: a bucket-padded prompt under windowing.

        Serving pads every prompt up to a bucket, so the cache holds zeroed K/V
        between the real prompt length and the bucket edge. The ring mask ignored
        `valid` and attended to all of it, and greedy decode drifted into an
        endless repeat a few tokens in — fluent, wrong, and silent.
        """
        S, real = 16, 3
        prompt = jax.random.randint(jax.random.PRNGKey(13), (1, S), 1, self.config.vocab_size)
        a, la = generate(self.model, self.params, prompt, 6, window_kv=False, prompt_len=real)
        b, lb = generate(self.model, self.params, prompt, 6, window_kv=True, prompt_len=real)
        for i, (x, y) in enumerate(zip(la, lb)):
            d = float(jnp.max(jnp.abs(x - y)))
            self.assertLess(d, 1e-4, f"step {i} logits diverge by {d:.2e} on a padded prompt")
        self.assertEqual(a.tolist(), b.tolist())

    def test_padded_prompt_matches_unwindowed_past_the_window(self):
        """Same, but long enough that the ring actually wraps."""
        cfg = dataclasses.replace(windowed_config(), sliding_window=WINDOW)
        model = Gemma4EModelJAX(cfg)
        params = build_tiny_params(cfg)
        S, real, n = 12, 5, 2 * WINDOW + 2
        prompt = jax.random.randint(jax.random.PRNGKey(14), (1, S), 1, cfg.vocab_size)
        a, la = generate(model, params, prompt, n, window_kv=False, prompt_len=real)
        b, lb = generate(model, params, prompt, n, window_kv=True, prompt_len=real)
        self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()
