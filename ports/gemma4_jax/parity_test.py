"""Parity test: ports/gemma4_jax (JAX) vs ports/gemma4 (parity-proven torch).

Builds the MINI Gemma4 dense config EXACTLY as the torch parity test does
(imports MINI_TEXT_CONFIG and fuzz_weights from ports/gemma4/parity_test.py),
instantiates the TORCH port with fuzzed fp32 weights, converts its state_dict
to the JAX params pytree, and checks:

  1. Full-model forward parity (no padding + right padding): final logits
     max abs diff < 1e-4, with per-layer hidden-state localization on failure.
  2. Cached-decode parity: greedy 16 tokens via prefill + decode_step in both
     frameworks (batch 1 and 2) — identical token sequences, per-step logits
     max abs diff < 1e-4.
  3. jit staticness: jax.jit(decode_step) traces exactly ONCE per (batch,
     MAX_SEQ) shape and never retraces across steps (trace counter).

Everything runs in float32 on CPU with
jax_default_matmul_precision="highest" so XLA fp32 matmuls match torch's
fp32 accumulation.

Run:  cd /home/xbill/tpu-jax/ports/gemma4_jax && python3 parity_test.py
"""

import importlib.util
import os
import sys

import numpy as np

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TORCH_PORT_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, "gemma4"))

TOLERANCE = 1e-4
PROMPT_LEN = 12
NEW_TOKENS = 16
MAX_SEQ = 64


def _load_module(name: str, path: str):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


# Distinct module names avoid the model.py name collision between the two
# ports; the torch parity_test's `import model` is pointed at the torch port.
jax_model = _load_module("gemma4_jax_model", os.path.join(_HERE, "model.py"))
convert = _load_module("gemma4_jax_convert", os.path.join(_HERE, "convert.py"))
torch_model = _load_module(
    "gemma4_torch_model", os.path.join(_TORCH_PORT_DIR, "model.py")
)
_saved = sys.modules.get("model")
sys.modules["model"] = torch_model
torch_parity = _load_module(
    "gemma4_torch_parity", os.path.join(_TORCH_PORT_DIR, "parity_test.py")
)
if _saved is None:
  del sys.modules["model"]
else:
  sys.modules["model"] = _saved

MINI_TEXT_CONFIG = torch_parity.MINI_TEXT_CONFIG


def build_models():
  """Torch port (fuzzed fp32 weights) + JAX config/params from its state dict."""
  torch_cfg = torch_model.Gemma4DenseConfig.from_text_config(
      dict(MINI_TEXT_CONFIG)
  )
  tport = torch_model.Gemma4DenseForCausalLM(torch_cfg).eval()
  torch_parity.fuzz_weights(tport)  # same seeded fuzzer as the torch tests

  jax_cfg = jax_model.Gemma4DenseConfig.from_text_config(dict(MINI_TEXT_CONFIG))
  assert tuple(torch_cfg.layer_types) == jax_cfg.layer_types
  params = convert.state_dict_to_params(tport.state_dict())  # fp32 preserved

  n_params = sum(
      x.size for x in jax.tree_util.tree_leaves(params)
  ) - params["lm_head"]["weight"].size  # lm_head aliases embed (tied)
  n_torch = sum(p.numel() for p in tport.parameters()) + sum(
      b.numel() for b in tport.buffers() if b.ndim and b.shape == (1,)
  )  # + layer_scalar buffers
  print(f"converted params: {n_params} leaves-elements "
        f"(torch params+layer_scalars: {n_torch})")
  assert n_params == n_torch, (n_params, n_torch)
  return tport, jax_cfg, params


def compare_forward(name, tport, jax_cfg, params, input_ids, attention_mask):
  with torch.no_grad():
    t_logits, t_hidden = tport(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
  j_ids = jnp.asarray(input_ids.numpy())
  j_mask = None if attention_mask is None else jnp.asarray(
      attention_mask.numpy()
  )
  j_logits, j_hidden = jax_model.forward(
      jax_cfg, params, j_ids, attention_mask=j_mask, return_hidden_states=True
  )

  max_diff = float(np.abs(t_logits.numpy() - np.asarray(j_logits)).max())
  ok = max_diff < TOLERANCE
  print(f"[forward {name}] final logits max abs diff: {max_diff:.3e} "
        f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE:g})")
  if not ok:
    for i, (th, jh) in enumerate(zip(t_hidden, j_hidden)):
      d = float(np.abs(th.numpy() - np.asarray(jh)).max())
      label = "embeddings" if i == 0 else f"layer {i - 1}"
      print(f"  hidden[{i}] ({label}): max abs diff {d:.3e}")
  return ok, max_diff


def jax_generate_cached(jax_cfg, params, prompt_np, new_tokens, max_seq):
  """Greedy prefill + jitted decode loop; returns (tokens, logits, traces)."""
  batch, prompt_len = prompt_np.shape
  trace_count = {"n": 0}

  def _counting_decode(params, cache, token, pos):
    trace_count["n"] += 1  # Python side effect: runs only when tracing
    return jax_model.decode_step(jax_cfg, params, cache, token, pos)

  decode_jit = jax.jit(_counting_decode)

  cache = jax_model.init_cache(jax_cfg, batch, max_seq, dtype=jnp.float32)
  logits, cache = jax_model.prefill(
      jax_cfg, params, cache, jnp.asarray(prompt_np)
  )
  step_logits = [np.asarray(logits)]
  next_token = jnp.argmax(logits, axis=-1)[:, None].astype(jnp.int32)
  tokens = [prompt_np, np.asarray(next_token)]

  for step in range(new_tokens - 1):
    pos = jnp.asarray(prompt_len + step, dtype=jnp.int32)
    logits, cache = decode_jit(params, cache, next_token, pos)
    step_logits.append(np.asarray(logits))
    next_token = jnp.argmax(logits, axis=-1)[:, None].astype(jnp.int32)
    tokens.append(np.asarray(next_token))
  return np.concatenate(tokens, axis=1), step_logits, trace_count["n"]


def compare_decode(name, tport, jax_cfg, params, prompt):
  with torch.no_grad():
    t_tokens, t_logits = tport.generate_cached(prompt, NEW_TOKENS, MAX_SEQ)
  j_tokens, j_logits, traces = jax_generate_cached(
      jax_cfg, params, prompt.numpy(), NEW_TOKENS, MAX_SEQ
  )

  tokens_ok = bool((t_tokens.numpy() == j_tokens).all())
  max_diff = max(
      float(np.abs(t.numpy() - j).max()) for t, j in zip(t_logits, j_logits)
  )
  jit_ok = traces == 1
  ok = tokens_ok and max_diff < TOLERANCE and jit_ok
  print(f"[decode {name}] tokens {'identical' if tokens_ok else 'DIFFER'}, "
        f"per-step logits max abs diff {max_diff:.3e}, "
        f"decode_step traces: {traces} "
        f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE:g})")
  if not tokens_ok:
    print(f"  torch: {t_tokens.tolist()}")
    print(f"  jax:   {j_tokens.tolist()}")
  return ok, max_diff, traces


def main():
  torch.manual_seed(0)
  tport, jax_cfg, params = build_models()

  results = []

  # 1. Full-model forward parity (same inputs as the torch parity test).
  batch, seq_len = 2, 48  # seq_len > sliding_window=16 to exercise the window
  input_ids = torch.randint(
      0, MINI_TEXT_CONFIG["vocab_size"], (batch, seq_len)
  )
  results.append(
      compare_forward("no padding", tport, jax_cfg, params, input_ids, None)
  )
  attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
  attention_mask[1, 44:] = 0  # right padding, shorter than the sliding window
  results.append(
      compare_forward(
          "right padding", tport, jax_cfg, params, input_ids, attention_mask
      )
  )

  # 2 + 3. Cached greedy decode parity + jit no-retrace, batch 1 and 2.
  gen = torch.Generator().manual_seed(42)
  trace_counts = []
  for b in (1, 2):
    prompt = torch.randint(
        1, MINI_TEXT_CONFIG["vocab_size"], (b, PROMPT_LEN), generator=gen
    )
    ok, diff, traces = compare_decode(
        f"batch={b} greedy {NEW_TOKENS} tokens MAX_SEQ={MAX_SEQ}",
        tport, jax_cfg, params, prompt,
    )
    results.append((ok, diff))
    trace_counts.append(traces)

  worst = max(diff for _, diff in results)
  if all(ok for ok, _ in results):
    print(f"JAX PARITY PASS (worst-case max abs diff {worst:.3e}, "
          f"decode_step trace counts {trace_counts})")
    return 0
  print(f"JAX PARITY FAIL (worst-case max abs diff {worst:.3e}, "
        f"decode_step trace counts {trace_counts})")
  return 1


if __name__ == "__main__":
  sys.exit(main())
