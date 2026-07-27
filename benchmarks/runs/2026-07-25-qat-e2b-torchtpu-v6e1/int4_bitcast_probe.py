#!/usr/bin/env python3.12
"""Feasibility probe: can Mosaic lower int32 -> int4 bitcast + native convert?

Trick: stored nibble = q + 8 (offset-8). XOR each nibble's top bit
(packed ^ 0x88888888) turns it into two's-complement int4, so
bitcast_convert_type(..., int4) yields q directly; one native convert to
bf16 replaces the shift/and/sub chain (5 VPU ops -> ~2).

Checks correctness vs the shift/mask reference on one 12B-sized layer,
then times both kernel variants back to back.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

GROUP = 32
S = 256
O, K = 15360, 3840
BLK, CK = 256, 960
CK8 = CK // 8
XOR = np.int32(np.uint32(0x88888888))


def _perm(k, ck):
    j = np.arange(k)
    c, r = j // ck, j % ck
    return c * ck + 8 * (r % (ck // 8)) + r // (ck // 8)


def _make(variant):
    def kernel(x_ref, packed_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        p = packed_ref[...]
        s4 = scale_ref[...]
        acc = jnp.zeros((S, BLK), jnp.float32)
        for ci in range(K // CK):
            pc = p[:, ci * CK8:(ci + 1) * CK8]
            if variant == "bitcast":
                q = jax.lax.bitcast_convert_type(pc ^ XOR, jnp.int4)  # [BLK, CK8, 8]
                w = q.reshape(BLK, CK).astype(s4.dtype)
            else:
                planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]
                w = jnp.concatenate(planes, axis=1).astype(s4.dtype)
            sr = s4[:, ci * CK8:(ci + 1) * CK8]
            w = w * jnp.concatenate([sr] * 8, axis=1)
            acc += jax.lax.dot_general(
                x_all[:, ci * CK:(ci + 1) * CK], w,
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc.astype(jnp.bfloat16)

    def fn(x: jax.Array, packed: jax.Array, scale_rep4: jax.Array) -> jax.Array:
        if variant == "shift":  # nibble-plane order needs the x permutation
            x = x[:, _perm(K, CK)]
        # bitcast [CK8, 8] -> [CK] is plane-INTERLEAVED order... check both
        # orders for correctness; the perm needed for bitcast is the identity
        # within int32 (little-endian nibbles are consecutive orig columns).
        return pl.pallas_call(
            kernel,
            grid=(O // BLK,),
            in_specs=[
                pl.BlockSpec((S, K), lambda i: (0, 0)),
                pl.BlockSpec((BLK, K // 8), lambda i: (i, 0)),
                pl.BlockSpec((BLK, K // 8), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((S, BLK), lambda i: (0, i)),
            out_shape=jax.ShapeDtypeStruct((S, O), jnp.bfloat16),
        )(x, packed, scale_rep4)

    return fn


shift_op = pallas.jax_op("probe::shift", _make("shift"))
bitcast_op = pallas.jax_op("probe::bitcast", _make("bitcast"))


def main() -> int:
    device = torch.device("tpu")
    torch.manual_seed(0)
    packed = torch.randint(-(2**31), 2**31 - 1, (O, K // 8), dtype=torch.int32).to(device)
    scale = (torch.randn(O, K // GROUP).abs() * 0.01 + 0.001).to(torch.bfloat16)
    scale_rep4 = scale.repeat_interleave(4, dim=1).to(device)
    x = torch.randn(S, K, dtype=torch.bfloat16).to(device)

    # reference: torch in-graph dequant
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)
    w = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(O, K) - 8
    w = (w.to(scale.dtype).reshape(O, -1, GROUP).cpu() * scale.unsqueeze(-1)).reshape(O, K)
    y_ref = (x.cpu().float() @ w.float().T)

    y_shift = shift_op(x, packed, scale_rep4).cpu().float()
    print(f"MARKER shift variant rel diff: {((y_shift - y_ref).abs().max() / y_ref.abs().max()).item():.3e}")

    try:
        y_bit = bitcast_op(x, packed, scale_rep4).cpu().float()
        # bitcast order: nibble i of int32 m = orig col 8m+i (consecutive),
        # scale col m matches sc[m//4] only in plane order -> if diff is big,
        # the needed fix is scale/x ordering, not the bitcast itself.
        rel = ((y_bit - y_ref).abs().max() / y_ref.abs().max()).item()
        print(f"MARKER bitcast variant rel diff (identity order): {rel:.3e}")
        print("MARKER bitcast COMPILES on Mosaic")
    except Exception as e:
        print(f"MARKER bitcast FAILED: {str(e)[:300]}")
        return 0

    for name, op in [("shift", shift_op), ("bitcast", bitcast_op)]:
        with torch.no_grad():
            op(x, packed, scale_rep4).cpu()
            t0 = time.monotonic()
            for _ in range(100):
                out = op(x, packed, scale_rep4)
            out.cpu()
            print(f"MARKER {name}: {(time.monotonic() - t0) / 100 * 1000:.3f} ms/call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
