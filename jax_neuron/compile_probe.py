#!/usr/bin/env python3
"""Compile the real Gemma 4 E-series graphs for Inferentia2 with neuronx-cc.

`probe.py` answers a smaller question: can the JAX Neuron runtime execute a
Gemma-*shaped* decoder block on a live Inf2 host. This answers the question that
actually gates the port: does `neuronx-cc` accept the graphs this repository's
engine emits — the same `prefill_with_kv_cache`, `make_cached_decode_step`, and
sampler that serve on TPU.

It needs no Inferentia hardware and no model weights.

  * No hardware, because `neuronx-cc` is an ahead-of-time compiler. It reads an
    HLO module and emits a NEFF; the device is only needed to *run* one. The
    compiler is an ordinary x86-64 Linux wheel on the AWS Neuron pip index.
  * No weights, because `jax.eval_shape` builds the parameter tree as
    `ShapeDtypeStruct`s and `jax.jit(...).lower(...)` accepts those directly.
    A full-size E2B tree is 7.3 GB of weights that are never allocated.

What a PASS means: the graph is expressible on Neuron — every op lowered, the
shapes fit the compiler's memory model, and a NEFF exists. It is NOT a claim
about numerics or speed. Those need the device, the real checkpoint, and the
parity tests named in `deployments/aws-inf2/README.md`.

What a FAIL means: read the error. `neuronx-cc` reports the offending HLO
instruction, which maps back to a line in `ports/gemma4/jax_e_model.py`.

One caveat this script reports on rather than hides: the HLO is produced by
lowering on the CPU backend, so a *custom call* in the module may be an artifact
of CPU lowering rather than something the Neuron plugin would emit. Custom calls
are listed per stage so a failure that names one can be judged accordingly.

Usage:

    python3 jax_neuron/compile_probe.py --tiny            # structural, minutes
    python3 jax_neuron/compile_probe.py --stage decode    # full E2B decode step
    python3 jax_neuron/compile_probe.py --lower-only      # HLO, skip the compiler
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Both must be set before jax_e_model is imported.
#   JAX_PLATFORMS=cpu   — lowering only; do not look for an accelerator.
#   JAX_E_PLATFORM      — make the engine take its Neuron branches (no Pallas,
#                         no fp8 KV, no lax.top_k) even though the process is
#                         running on a CPU host. See ports/gemma4/backend.py.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_E_PLATFORM", "neuron")

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from ports.gemma4 import backend                              # noqa: E402
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params  # noqa: E402
from ports.gemma4.jax_e_model import (                        # noqa: E402
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    make_cached_decode_step,
    onchip_sample_tpu_v6e_jax,
    prefill_with_kv_cache,
)

STAGES = ("decode", "prefill", "sample")


# ---------------------------------------------------------------- configuration

def e2b_config() -> Gemma4EConfig:
    """The shipped gemma-4-E2B-it geometry."""
    return Gemma4EConfig(
        vocab_size=262144,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
    )


def tiny_config() -> Gemma4EConfig:
    """Same structure, small enough to compile in minutes.

    Every architectural feature that could trip the compiler is preserved:
    interleaved sliding/full attention, KV sharing over the tail layers, the
    double-wide MLP on shared layers, per-layer embeddings, and the W4A16 packed
    int4 weights. Only the extents shrink. A tiny PASS is evidence about op
    coverage; a full-size PASS is additionally evidence about capacity.

    The layer count is not free to choose. `kv_share_map` resolves each shared
    layer to the last unshared layer of its own attention type, so both types
    must appear before `first_kv_shared_layer_idx` or it raises KeyError. With
    the default `i % 5 != 4` interleave, 10 layers with 4 shared puts the
    boundary at 6 and layer 4 is the full-attention source — the same shape as
    E2B's 35/20 split (boundary 15, sources at 14 and 9).
    """
    return Gemma4EConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_global_key_value_heads=2,
        global_head_dim=128,
        num_kv_shared_layers=4,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=2048,
    )


def abstract_params(config: Gemma4EConfig) -> dict[str, Any]:
    """The W4A16 parameter tree as ShapeDtypeStructs — nothing is allocated.

    `build_benchmark_params` is the same synthetic tree the TPU sweep uses, so
    the compiled graph matches what is benchmarked there. `layer_scalar` is added
    on top because the real safetensors loader supplies it and the decoder
    multiplies the residual stream by it; omitting it would compile a graph the
    server never runs.
    """
    tree = jax.eval_shape(lambda: build_benchmark_params(config))
    for i in range(config.num_hidden_layers):
        tree[f"layer_{i}"]["layer_scalar"] = jax.ShapeDtypeStruct((), jnp.bfloat16)
    return tree


# ------------------------------------------------------------------- lowering

def _abstract(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def lower_decode(config, params, batch, context, cache_dtype, quant_mode, window_kv):
    """One cached decode step: the graph that runs once per generated token."""
    model = Gemma4EModelJAX(config)
    step = make_cached_decode_step(model, quant_mode=quant_mode, window_kv=window_kv)
    caches = jax.eval_shape(
        lambda: init_kv_cache(config, batch_size=batch, max_seq_len=context,
                              dtype=cache_dtype, window_kv=window_kv)
    )
    args = (
        params,
        caches,
        _abstract((batch, context), jnp.bool_),      # valid
        _abstract((batch, 1), jnp.int32),            # tok
        _abstract((batch,), jnp.int32),              # logical_pos
        _abstract((), jnp.int32),                    # slot
    )
    return jax.jit(step).lower(*args)


def lower_prefill(config, params, batch, context, cache_dtype, quant_mode, window_kv,
                  max_new_tokens=1):
    """The prefill pass: prompt in, KV cache and last-token logits out."""
    model = Gemma4EModelJAX(config)
    fn = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype",
                         "window_kv"),
    )
    return fn.lower(
        model=model,
        prompt_ids=_abstract((batch, context), jnp.int32),
        prompt_valid=_abstract((batch, context), jnp.bool_),
        params=params,
        max_new_tokens=max_new_tokens,
        quant_mode=quant_mode,
        cache_dtype=cache_dtype,
        window_kv=window_kv,
    )


def lower_sample(config, batch, top_k=40, temperature=0.7):
    """On-accelerator top-k sampling over the full vocabulary.

    Compiled separately because it is the one stage whose implementation the
    port actually changed: `_kth_largest` replaces `lax.top_k`, which neuronx-cc
    rejects, with an iterative masked-max on backends where `caps.device_top_k`
    is False.
    """
    def sample(logits, key):
        return onchip_sample_tpu_v6e_jax(logits, key, temperature=temperature,
                                         top_k=top_k)

    key = jax.eval_shape(lambda: jax.random.PRNGKey(0))
    return jax.jit(sample).lower(
        _abstract((batch, config.vocab_size), jnp.float32), key
    )


# ------------------------------------------------------------------ HLO dumping

def normalize_instruction_ids(proto: bytes) -> tuple[bytes, bool]:
    """Renumber HLO instruction ids so an older XLA schema can read them.

    Not a model concern — a toolchain skew, and worth spelling out because the
    crash it causes looks like a compiler bug:

        Check failed: unique_id_ < (2147483647) (4294967297 vs. 2147483647)

    Current XLA (the one inside jaxlib) packs an instruction's unique id as
    ``(computation_id << 32) | index``. neuronx-cc 2.26 embeds an older XLA whose
    ``HloInstruction::unique_id()`` is an int32, so every instruction past the
    first computation overflows its CHECK and the compiler aborts before it has
    looked at a single operator.

    The packing is only a numbering scheme: what the module needs is that ids be
    unique within it and that every reference agree. So renumber densely from
    zero and rewrite the three fields that hold instruction ids — ``operand_ids``,
    ``control_predecessor_ids``, and each computation's ``root_id``. Computation
    ids are a separate namespace and are already small, so they are left alone.

    Returns (proto, changed). If the schema is unavailable the input is passed
    through untouched rather than guessed at.
    """
    try:
        from neuronxcc.thirdparty_libs.xla.service import hlo_pb2
    except ImportError:
        return proto, False

    module = hlo_pb2.HloModuleProto()
    module.ParseFromString(proto)

    int32_max = (1 << 31) - 1
    if not any(instr.id > int32_max
               for comp in module.computations for instr in comp.instructions):
        return proto, False

    remap: dict[int, int] = {}
    for comp in module.computations:
        for instr in comp.instructions:
            remap.setdefault(instr.id, len(remap))
    for comp in module.computations:
        for instr in comp.instructions:
            instr.id = remap[instr.id]
            instr.operand_ids[:] = [remap[o] for o in instr.operand_ids]
            instr.control_predecessor_ids[:] = [
                remap[o] for o in instr.control_predecessor_ids
            ]
        comp.root_id = remap[comp.root_id]
    return module.SerializeToString(), True


def dump_hlo(lowered, path: Path) -> tuple[int, bool, list[str]]:
    """Write the serialized HLO module proto neuronx-cc reads with --framework XLA.

    Returns (bytes written, ids renumbered, custom-call targets).
    """
    module = lowered.compiler_ir(dialect="hlo")
    proto = module.as_serialized_hlo_module_proto()
    proto, renumbered = normalize_instruction_ids(proto)
    path.write_bytes(proto)
    return len(proto), renumbered, custom_calls(module)


def custom_calls(module) -> list[str]:
    """Custom-call targets in the HLO the compiler will actually read.

    Scanned on the HLO module rather than `lowered.as_text()`, which is
    StableHLO and spells these differently — a scan of the wrong dialect finds
    nothing and reads as "no custom calls", which is the worst kind of wrong for
    a diagnostic.

    These are the instructions most likely to be an artifact of lowering on the
    CPU backend rather than something the Neuron plugin would emit for the same
    JAX code, so a failure naming one deserves a second look on real hardware.
    """
    try:
        text = module.to_string()
    except Exception:
        return []
    targets = set()
    for line in text.splitlines():
        marker = "custom_call_target="
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        if tail.startswith('"'):
            targets.add(tail[1:].split('"', 1)[0])
    return sorted(targets)


# -------------------------------------------------------------------- compiling

def find_compiler(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    return shutil.which("neuronx-cc")


def compile_neff(compiler: str, hlo: Path, out: Path, target: str, optlevel: str,
                 model_type: str, timeout_s: int) -> dict[str, Any]:
    cmd = [
        compiler, "compile",
        "--framework", "XLA",
        "--target", target,
        "--model-type", model_type,
        "--optlevel", optlevel,
        "--output", str(out),
        str(hlo),
    ]
    # cwd=the artifact directory, NOT the caller's. neuronx-cc scatters working
    # state through the current directory as it runs — hash-named kernel dirs,
    # `neuronxcc-*` scratch, `log-neuron-cc.txt`, `global_metric_store.json` —
    # and cleans up none of it. Run from the repo root and the repo root fills
    # with dozens of untracked directories.
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                              cwd=str(out.parent))
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "command": " ".join(cmd), "seconds": timeout_s,
            "error": f"neuronx-cc exceeded the {timeout_s}s timeout",
        }
    seconds = round(time.perf_counter() - started, 1)
    output = (proc.stdout or "") + (proc.stderr or "")
    result: dict[str, Any] = {
        "ok": proc.returncode == 0 and out.exists(),
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "seconds": seconds,
    }
    if result["ok"]:
        result["neff_bytes"] = out.stat().st_size
    else:
        result["error"] = _first_error(output)
        result["log_tail"] = output.strip().splitlines()[-25:]
    return result


def _first_error(output: str) -> str:
    """The most specific line in a neuronx-cc log, for the summary table."""
    for line in output.splitlines():
        stripped = line.strip()
        if any(marker in stripped for marker in
               ("ERROR", "Error", "error:", "Exception", "NCC_", "Unsupported",
                "not supported", "Failed")):
            return stripped[:400]
    tail = output.strip().splitlines()
    return tail[-1][:400] if tail else "neuronx-cc failed without output"


# ------------------------------------------------------------------------ main

def run_stage(name: str, args, config, params, outdir: Path, compiler: str | None,
              cache_dtype) -> dict[str, Any]:
    record: dict[str, Any] = {"stage": name}
    started = time.perf_counter()
    try:
        if name == "decode":
            lowered = lower_decode(config, params, args.batch, args.context,
                                   cache_dtype, args.quant_mode, args.window_kv)
        elif name == "prefill":
            lowered = lower_prefill(config, params, args.batch, args.context,
                                    cache_dtype, args.quant_mode, args.window_kv,
                                    max_new_tokens=args.max_new_tokens)
        elif name == "sample":
            lowered = lower_sample(config, args.batch, top_k=args.top_k)
        else:
            raise ValueError(f"unknown stage {name!r}")
    except Exception as exc:
        record["lowered"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:600]
        record["seconds"] = round(time.perf_counter() - started, 1)
        return record

    record["lowered"] = True
    record["lower_seconds"] = round(time.perf_counter() - started, 1)

    hlo_path = outdir / f"{name}.hlo"
    try:
        size, renumbered, calls = dump_hlo(lowered, hlo_path)
        record["hlo_bytes"] = size
        record["hlo_ids_renumbered"] = renumbered
        if calls:
            record["custom_calls"] = calls
    except Exception as exc:
        record["error"] = f"HLO export failed: {type(exc).__name__}: {exc}"[:400]
        return record

    if compiler is None or args.lower_only:
        record["compiled"] = None
        return record

    result = compile_neff(
        compiler, hlo_path, outdir / f"{name}.neff", args.target, args.optlevel,
        args.model_type, args.timeout,
    )
    record["compiled"] = result.pop("ok")
    record.update(result)
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", action="append", choices=STAGES,
                   help="Stage to probe; repeatable. Default: all of them.")
    p.add_argument("--tiny", action="store_true",
                   help="Small config with the same structure; compiles in minutes.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=None,
                   help="Static cache length. Default 512 (128 for --tiny).")
    p.add_argument("--max-new-tokens", type=int, default=1,
                   help="Prefill only: extra cache slots beyond the prompt.")
    p.add_argument("--quant-mode", default="w4a16", choices=("w4a16", "int8", "fp16"))
    p.add_argument("--kv-dtype", default="bf16",
                   help="KV cache dtype. fp8_* are here to be TESTED, not used: "
                        "the engine refuses them on Neuron, and this is how that "
                        "refusal is checked against the compiler rather than assumed.")
    p.add_argument("--window-kv", action="store_true",
                   help="Ring-buffer the sliding-attention layers' KV.")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--target", default="inf2",
                   choices=("inf2", "trn1", "trn1n", "trn2", "trn2n"))
    p.add_argument("--optlevel", default="1", choices=("1", "2", "3"),
                   help="neuronx-cc -O level. 1 compiles fastest; 2 is the serving default.")
    p.add_argument("--model-type", default="transformer",
                   choices=("transformer", "generic", "unet-inference"))
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-stage neuronx-cc timeout in seconds.")
    p.add_argument("--neuronx-cc", default=None,
                   help="Path to the neuronx-cc binary if it is not on PATH.")
    p.add_argument("--lower-only", action="store_true",
                   help="Emit HLO and stop; do not invoke the compiler.")
    p.add_argument("--out", default=None,
                   help="Where HLO modules and NEFFs go, and where neuronx-cc "
                        "is run so its scratch lands there too. "
                        "Default: .neuron_compile_probe/ at the repo root (gitignored).")
    p.add_argument("--json", default=None, help="Write the summary to this path.")
    args = p.parse_args()

    stages = args.stage or list(STAGES)
    config = tiny_config() if args.tiny else e2b_config()
    if args.context is None:
        args.context = 128 if args.tiny else 512

    from ports.gemma4.jax_e_model import _KV_SCALE_DTYPE  # noqa: F401  (import check)
    cache_dtypes = {"bf16": jnp.bfloat16, "bfloat16": jnp.bfloat16,
                    "fp16": jnp.float16, "float16": jnp.float16, "int8": jnp.int8,
                    "fp8_e4m3": jnp.float8_e4m3fn, "fp8_e5m2": jnp.float8_e5m2}
    if args.kv_dtype not in cache_dtypes:
        p.error(f"--kv-dtype must be one of {sorted(cache_dtypes)}")
    cache_dtype = cache_dtypes[args.kv_dtype]

    outdir = Path(args.out or (_REPO_ROOT / ".neuron_compile_probe"))
    outdir.mkdir(parents=True, exist_ok=True)

    compiler = find_compiler(args.neuronx_cc)
    caps = backend.caps()

    params = abstract_params(config)
    param_bytes = sum(int(x.size) * jnp.dtype(x.dtype).itemsize
                      for x in jax.tree_util.tree_leaves(params))

    summary: dict[str, Any] = {
        "jax_version": jax.__version__,
        "engine_platform": caps.platform,
        "engine_caps": {
            "pallas": caps.pallas, "float8_kv": caps.float8_kv,
            "device_top_k": caps.device_top_k,
            "buffer_donation": caps.buffer_donation,
        },
        "config": "tiny" if args.tiny else "gemma-4-E2B",
        "target": args.target,
        "batch": args.batch,
        "context": args.context,
        "quant_mode": args.quant_mode,
        "kv_dtype": args.kv_dtype,
        "window_kv": args.window_kv,
        "param_bytes": param_bytes,
        "num_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "neuronx_cc": compiler,
        "artifacts": str(outdir),
        "stages": [],
    }
    if compiler is None and not args.lower_only:
        summary["warning"] = (
            "neuronx-cc not found; lowering to HLO only. Install with: pip install "
            "neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com"
        )

    print(f"config={summary['config']} target={args.target} "
          f"batch={args.batch} context={args.context} "
          f"quant={args.quant_mode} kv={args.kv_dtype} "
          f"params={param_bytes / 1e9:.2f} GB (abstract)", flush=True)

    for name in stages:
        print(f"\n=== {name} ===", flush=True)
        record = run_stage(name, args, config, params, outdir, compiler, cache_dtype)
        summary["stages"].append(record)
        if not record.get("lowered"):
            print(f"  LOWER FAIL  {record.get('error')}", flush=True)
            continue
        print(f"  lowered in {record['lower_seconds']}s, "
              f"HLO {record.get('hlo_bytes', 0) / 1e6:.1f} MB", flush=True)
        if record.get("custom_calls"):
            print(f"  custom calls: {', '.join(record['custom_calls'])}", flush=True)
        if record.get("compiled") is None:
            print("  compile skipped", flush=True)
        elif record["compiled"]:
            print(f"  COMPILE PASS  {record['seconds']}s, "
                  f"NEFF {record['neff_bytes'] / 1e6:.1f} MB", flush=True)
        else:
            print(f"  COMPILE FAIL  after {record['seconds']}s", flush=True)
            print(f"  {record.get('error')}", flush=True)

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n")
        print(f"\nsummary written to {args.json}", flush=True)
    else:
        print("\n" + text, flush=True)

    failed = [s for s in summary["stages"]
              if not s.get("lowered") or s.get("compiled") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
