#!/usr/bin/env python3
"""Configure JAX NeuronX and run the shared Gemma 4 OpenAI server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def configure_neuron() -> None:
    # Must be set before importing JAX. Neuron documents RBG as the supported
    # PRNG implementation; sampling otherwise risks unsupported Threefry ops.
    os.environ.setdefault("JAX_PLATFORMS", "neuron,cpu")
    os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "rbg")
    os.environ.setdefault("NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU", "1")
    os.environ.setdefault("NEURON_CC_FLAGS", "--model-type=transformer")
    # The engine's only Pallas kernel lowers through Mosaic, which is TPU-only,
    # so it can never compile natively here. INTERPRET=1 keeps that path on the
    # portable interpreter instead of hard-failing; the default W4A16 impl is
    # "reference" anyway, so this only matters if set_w4a16_impl() is called.
    os.environ["JAX_E_PALLAS_INTERPRET"] = "1"
    # Buffer donation is the engine's largest decode win on TPU (1.62x bf16),
    # but Neuron compiles donated inputs as must-alias and then aborts the
    # request if the runtime cannot donate, rather than warning and copying:
    #   INVALID_ARGUMENT: An input was configured to be must-alias at compile
    #   time but not donated at runtime
    # Off by default here, and a setdefault so it can be re-enabled from the
    # environment once donation is shown to work on this plugin. Expect the
    # decode regression the TPU measurements predict until then.
    os.environ.setdefault("JAX_E_DONATE_CACHE", "0")


def verify_neuron() -> list[object]:
    import jax

    devices = list(jax.devices())
    neuron = [device for device in devices if device.platform == "neuron"]
    if not neuron:
        found = ", ".join(f"{d.platform}:{d}" for d in devices) or "none"
        raise RuntimeError(f"No JAX Neuron device found; discovered {found}")
    print(f"JAX Neuron ready: {len(neuron)} device(s): {neuron}", flush=True)
    return neuron


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true")
    known, server_args = parser.parse_known_args()

    configure_neuron()
    verify_neuron()
    if known.check_only:
        return

    root = Path(__file__).resolve().parents[2]
    server = root / "jax_openai_server.py"
    if not server.exists():
        raise FileNotFoundError(f"Shared server not found at {server}")

    # Keep the shared server CLI authoritative. The reference W4A16 path is its
    # default and is required because the optional fused kernel targets TPU.
    sys.path.insert(0, str(root))
    sys.argv = [str(server), *server_args]
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()
