"""Tests for the two JAX-Neuron probes.

`probe.py` needs an Inf2 host, so it is only checked statically here.

`compile_probe.py` needs neither hardware nor `neuronx-cc` to *lower*, so the
parts that do not shell out to the compiler are exercised for real. That matters:
its configs and parameter tree are hand-built, and a mistake there compiles a
graph the server never runs — which looks exactly like a pass.
"""

import ast
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "jax_neuron" / "probe.py"
COMPILE_PROBE = ROOT / "jax_neuron" / "compile_probe.py"

os.environ.setdefault("JAX_PLATFORMS", "cpu")


class JaxProbeTests(unittest.TestCase):
    def test_probe_parses_without_importing_optional_jax(self):
        ast.parse(PROBE.read_text())

    def test_probe_is_pure_jax_and_has_static_kv_update(self):
        source = PROBE.read_text()
        self.assertNotIn("import torch", source)
        self.assertIn("jax.lax.dynamic_update_slice", source)
        self.assertIn("jax.jit(decoder_step)", source)
        self.assertIn("jax.block_until_ready", source)


class CompileProbeTests(unittest.TestCase):
    def test_compile_probe_parses(self):
        ast.parse(COMPILE_PROBE.read_text())

    def test_tiny_config_preserves_the_kv_sharing_invariant(self):
        """Both attention types must exist before the KV-sharing boundary.

        `kv_share_map` resolves each shared layer to the last unshared layer of
        its own type and raises KeyError otherwise. The first --tiny config had
        5 layers and 2 shared, which put every full-attention layer past the
        boundary; it failed instantly, but a config that merely dropped a
        feature would have compiled and proven nothing.
        """
        from jax_neuron.compile_probe import e2b_config, tiny_config

        for config in (tiny_config(), e2b_config()):
            share_map = config.kv_share_map()                # must not raise
            self.assertEqual(len(share_map), config.num_hidden_layers)
            types = set(config.layer_types[:config.first_kv_shared_layer_idx])
            self.assertEqual(types, {"sliding_attention", "full_attention"})

    def test_tiny_config_keeps_every_architectural_feature(self):
        """Extents may shrink; features may not. A tiny config that quietly
        dropped KV sharing or the double-wide MLP would report a PASS for a
        graph the real model does not have."""
        from jax_neuron.compile_probe import e2b_config, tiny_config

        tiny, e2b = tiny_config(), e2b_config()
        self.assertTrue(tiny.use_double_wide_mlp, "lost the double-wide MLP")
        self.assertGreater(tiny.num_kv_shared_layers, 0, "lost KV sharing")
        self.assertGreater(tiny.hidden_size_per_layer_input, 0, "lost PLE")
        self.assertNotEqual(tiny.head_dim, tiny.global_head_dim,
                            "lost the dual attention geometry")
        self.assertEqual(set(tiny.layer_types), set(e2b.layer_types))

    def test_abstract_params_cover_every_parameter_the_model_reads(self):
        """The compiled graph must have the same parameters as the served one.

        `layer_scalar` is the specific trap: `build_benchmark_params` omits it,
        the real safetensors loader supplies it, and the decoder multiplies the
        whole residual stream by it. Compiling without it would silently probe a
        different model.
        """
        from jax_neuron.compile_probe import abstract_params, tiny_config

        config = tiny_config()
        params = abstract_params(config)
        for i in range(config.num_hidden_layers):
            layer = params[f"layer_{i}"]
            self.assertIn("layer_scalar", layer, f"layer_{i} missing layer_scalar")
            self.assertIn("attn", layer)
            self.assertIn("mlp", layer)
        self.assertIn("embed_tokens", params)
        self.assertIn("embed_tokens_per_layer", params)

    def test_abstract_params_allocate_nothing(self):
        """The whole point: a 7.29 GB tree described, not materialized."""
        import jax

        from jax_neuron.compile_probe import abstract_params, e2b_config

        params = abstract_params(e2b_config())
        leaves = jax.tree_util.tree_leaves(params)
        self.assertTrue(all(isinstance(x, jax.ShapeDtypeStruct) for x in leaves))
        total = sum(int(x.size) * x.dtype.itemsize for x in leaves)
        self.assertGreater(total, 7e9, "E2B parameter tree should be ~7.3 GB")

    def test_every_stage_lowers_to_hlo(self):
        """Lowering needs no compiler, so it is checked for real, not by regex."""
        from jax_neuron import compile_probe as cp

        config = cp.tiny_config()
        params = cp.abstract_params(config)
        import jax.numpy as jnp

        for lowered in (
            cp.lower_decode(config, params, 1, 128, jnp.bfloat16, "w4a16", False),
            cp.lower_prefill(config, params, 1, 128, jnp.bfloat16, "w4a16", False),
            cp.lower_sample(config, 1),
        ):
            module = lowered.compiler_ir(dialect="hlo")
            self.assertGreater(len(module.as_serialized_hlo_module_proto()), 0)

    def test_id_normalization_is_a_no_op_without_the_compiler_schema(self):
        """It must pass the proto through untouched rather than guess."""
        from jax_neuron.compile_probe import normalize_instruction_ids

        try:
            import neuronxcc  # noqa: F401
        except ImportError:
            payload = b"not-a-real-proto"
            self.assertEqual(normalize_instruction_ids(payload), (payload, False))
        else:
            self.skipTest("neuronx-cc installed; renumbering is exercised by the probe")


if __name__ == "__main__":
    unittest.main()
