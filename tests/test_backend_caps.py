"""The engine's Neuron branches, exercised on a CPU host.

`JAX_E_PLATFORM=neuron` makes `ports.gemma4.backend` report Neuron capabilities
without an Inferentia device, so the branches the Inf2 port added can be tested
here. This proves the branches are *taken* and that they are numerically sane —
it proves nothing about what neuronx-cc accepts. That is what
`jax_neuron/compile_probe.py` is for, and its findings are what the capability
table encodes.

Each test therefore does two things: pin the TPU path as unchanged, and check
the Neuron substitute agrees with it.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402

from ports.gemma4 import backend                        # noqa: E402


class CapabilityTableTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(backend._PLATFORM_ENV, None)
        backend.reset_caps_cache()

    def _caps_for(self, platform):
        os.environ[backend._PLATFORM_ENV] = platform
        backend.reset_caps_cache()
        return backend.caps()

    def test_tpu_keeps_every_measured_capability(self):
        caps = self._caps_for("tpu")
        self.assertTrue(caps.is_tpu)
        for field in ("pallas", "float8_kv", "buffer_donation",
                      "device_top_k", "default_bf16_matmul"):
            self.assertTrue(getattr(caps, field), f"TPU lost {field}")

    def test_neuron_disables_what_neuronx_cc_rejects(self):
        caps = self._caps_for("neuron")
        self.assertTrue(caps.is_neuron)
        self.assertFalse(caps.pallas)         # Mosaic is TPU/GPU only
        self.assertFalse(caps.device_top_k)   # NCC_EVRF001, measured
        self.assertFalse(caps.float8_kv)      # NCC_EVRF051, measured

    def test_no_scatter_capability_exists(self):
        """Pinned deliberately: the sampler's scatter compiles for inf2.

        Re-adding a `scatter` field would re-assert an inherited claim that was
        measured false. See the note on `device_top_k` in backend.py.
        """
        self.assertFalse(hasattr(backend.BackendCaps, "scatter"))
        self.assertNotIn("scatter",
                         {f.name for f in __import__("dataclasses").fields(backend.BackendCaps)})

    def test_cpu_keeps_the_fused_kernel_reachable_via_the_interpreter(self):
        """Neuron is the only platform where the fused kernel is unreachable.

        CPU must keep it selectable, because that is how its numerics are tested
        off-device in tests/test_perf_optimizations.py.
        """
        caps = self._caps_for("cpu")
        self.assertTrue(caps.pallas)
        self.assertTrue(caps.pallas_interpret)

    def test_tpu_lowers_pallas_natively(self):
        caps = self._caps_for("tpu")
        self.assertTrue(caps.pallas)
        self.assertFalse(caps.pallas_interpret)

    def test_unknown_platform_is_rejected_not_guessed(self):
        os.environ[backend._PLATFORM_ENV] = "inferentia"
        backend.reset_caps_cache()
        with self.assertRaises(ValueError):
            backend.caps()


class TopKWithoutDeviceTopKTests(unittest.TestCase):
    """`_kth_largest` / `_top_k_mask` must agree with lax.top_k."""

    @staticmethod
    def _logits(seed=0, batch=3, vocab=4096):
        return jax.random.normal(jax.random.PRNGKey(seed), (batch, vocab),
                                 dtype=jnp.float32)

    def test_iterative_threshold_matches_lax_top_k(self):
        import ports.gemma4.jax_e_model as M

        logits = self._logits()
        for top_k in (1, 2, 8, 40):
            expected = jax.lax.top_k(logits, top_k)[0][:, -1:]
            got = self._iterative(M, logits, top_k)
            self.assertTrue(
                bool(jnp.all(got == expected)),
                f"k={top_k}: iterative threshold {got} != lax.top_k {expected}",
            )

    @staticmethod
    def _iterative(M, logits, top_k):
        """Force the no-device-top_k branch regardless of the detected platform."""
        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM["neuron"]
        try:
            return M._kth_largest(logits, top_k)
        finally:
            M._CAPS = original

    def test_masked_rows_keep_at_least_k_survivors(self):
        import ports.gemma4.jax_e_model as M

        logits = self._logits(seed=7)
        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM["neuron"]
        try:
            masked = M._top_k_mask(logits, 40)
        finally:
            M._CAPS = original
        survivors = jnp.sum(masked > -1e9, axis=-1)
        # Exactly k unless the k-th value is tied, which cannot truncate below k.
        self.assertTrue(bool(jnp.all(survivors >= 40)), f"survivors={survivors}")

    def test_ties_widen_rather_than_truncate(self):
        """The documented tie behaviour, pinned so it cannot regress silently."""
        import ports.gemma4.jax_e_model as M

        # Six values tied at the k=3 boundary.
        logits = jnp.array([[5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32)
        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM["neuron"]
        try:
            masked = M._top_k_mask(logits, 3)
        finally:
            M._CAPS = original
        survivors = int(jnp.sum(masked > -1e9))
        self.assertEqual(survivors, 6, "all tied values at the threshold must survive")

    def test_large_top_k_is_refused_not_silently_unrolled(self):
        import ports.gemma4.jax_e_model as M

        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM["neuron"]
        try:
            with self.assertRaises(ValueError):
                M._kth_largest(self._logits(), M._MAX_UNROLLED_TOP_K + 1)
        finally:
            M._CAPS = original


class SamplerEndToEndTests(unittest.TestCase):
    def test_greedy_is_identical_on_both_paths(self):
        """temperature<=0 is argmax and must not depend on the platform at all."""
        import ports.gemma4.jax_e_model as M

        logits = jax.random.normal(jax.random.PRNGKey(3), (2, 1024), jnp.float32)
        key = jax.random.PRNGKey(0)
        results = []
        original = M._CAPS
        try:
            for platform in ("tpu", "neuron"):
                M._CAPS = backend._CAPS_BY_PLATFORM[platform]
                results.append(M.onchip_sample_tpu_v6e_jax(
                    logits, key, temperature=0.0, top_k=40))
        finally:
            M._CAPS = original
        self.assertTrue(bool(jnp.all(results[0] == results[1])))

    def test_sampled_token_is_inside_the_top_k(self):
        import ports.gemma4.jax_e_model as M

        logits = jax.random.normal(jax.random.PRNGKey(11), (1, 2048), jnp.float32)
        allowed = set(jax.lax.top_k(logits, 40)[1][0].tolist())
        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM["neuron"]
        try:
            for seed in range(20):
                tok = M.onchip_sample_tpu_v6e_jax(
                    logits, jax.random.PRNGKey(seed), temperature=0.8, top_k=40)
                self.assertIn(int(tok[0, 0]), allowed)
        finally:
            M._CAPS = original


class CacheDtypeGuardTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(backend._PLATFORM_ENV, None)
        backend.reset_caps_cache()

    def test_float8_kv_is_refused_on_neuron_with_a_usable_message(self):
        import jax_engine

        os.environ[backend._PLATFORM_ENV] = "neuron"
        backend.reset_caps_cache()
        for name in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            with self.assertRaises(ValueError) as ctx:
                jax_engine.resolve_cache_dtype(name)
            self.assertIn("int8", str(ctx.exception))

    def test_int8_and_bf16_still_resolve_on_neuron(self):
        import jax_engine

        os.environ[backend._PLATFORM_ENV] = "neuron"
        backend.reset_caps_cache()
        self.assertEqual(jax_engine.resolve_cache_dtype("int8"), jnp.int8)
        self.assertEqual(jax_engine.resolve_cache_dtype("bf16"), jnp.bfloat16)

    def test_float8_still_resolves_on_tpu(self):
        import jax_engine

        os.environ[backend._PLATFORM_ENV] = "tpu"
        backend.reset_caps_cache()
        self.assertEqual(jax_engine.resolve_cache_dtype("fp8"), jnp.float8_e4m3fn)


class W4A16ImplGuardTests(unittest.TestCase):
    def tearDown(self):
        import ports.gemma4.jax_e_model as M
        M.set_w4a16_impl("reference", "plane")

    def _with_platform(self, platform, fn):
        import ports.gemma4.jax_e_model as M

        original = M._CAPS
        M._CAPS = backend._CAPS_BY_PLATFORM[platform]
        try:
            return fn(M)
        finally:
            M._CAPS = original

    def test_fused_is_refused_on_neuron(self):
        def check(M):
            with self.assertRaises(RuntimeError) as ctx:
                M.set_w4a16_impl("fused")
            self.assertIn("Pallas", str(ctx.exception))
        self._with_platform("neuron", check)

    def test_auto_degrades_to_reference_on_neuron(self):
        def check(M):
            M.set_w4a16_impl("auto")
            self.assertEqual(M._W4A16_IMPL, "reference")
        self._with_platform("neuron", check)

    def test_fused_is_still_selectable_on_tpu(self):
        def check(M):
            M.set_w4a16_impl("fused")
            self.assertEqual(M._W4A16_IMPL, "fused")
        self._with_platform("tpu", check)


if __name__ == "__main__":
    unittest.main()
