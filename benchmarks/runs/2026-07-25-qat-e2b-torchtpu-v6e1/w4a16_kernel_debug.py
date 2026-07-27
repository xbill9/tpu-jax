#!/usr/bin/env python3.12
"""Differential microbench: price each component of the fused w4a16 kernel.

Four kernels with IDENTICAL grid/tiling/dot structure, differing only in how
the weight block reaches the MXU:
  A fused-int4   : shift/mask 8 nibble planes + scale mul   (the production kernel)
  B int8-stored  : int8 weights in the same plane-permuted layout; convert+scale
                   (the "no unpack" escape — 2x memory of int4, half of bf16)
  C control-bf16 : pre-materialized bf16 weights, straight to dot
                   (Pallas floor for this tiling)
  D dense-XLA    : plain x @ w.T under torch.compile (no Pallas at all)

  A - C = true unpack+scale cost      B - C = convert+scale cost
  C - D = Pallas per-call/tiling overhead vs XLA's own matmul

Shapes: the 12B extremes (gate_proj 15360x3840, down_proj 3840x15360).
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


def _blk_for(n: int, target: int) -> int:
    blk = target
    while n % blk:
        blk //= 2
    return blk


def _perm(k: int, ck: int) -> np.ndarray:
    ck8 = ck // 8
    j = np.arange(k)
    c, r = j // ck, j % ck
    return c * ck + 8 * (r % ck8) + r // ck8


def w4a16_fused_jax(x: jax.Array, packed: jax.Array, scale_rep4: jax.Array) -> jax.Array:
    seq, k = x.shape
    out_f = packed.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _blk_for(k, 960)
    ck8 = ck // 8
    x = x[:, _perm(k, ck)]

    def kernel(x_ref, packed_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        p = packed_ref[...]
        s4 = scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            pc = p[:, ci * ck8:(ci + 1) * ck8]
            planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]
            w = jnp.concatenate(planes, axis=1).astype(s4.dtype)
            sr = s4[:, ci * ck8:(ci + 1) * ck8]
            w = w * jnp.concatenate([sr] * 8, axis=1)
            acc += jax.lax.dot_general(
                x_all[:, ci * ck:(ci + 1) * ck], w,
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc

    out = pl.pallas_call(
        kernel,
        grid=(out_f // blk,),
        in_specs=[
            pl.BlockSpec((seq, k), lambda i: (0, 0)),
            pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
            pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.float32),
    )(x, packed, scale_rep4)
    return out.astype(jnp.bfloat16)


def w4a16_int8_jax(x: jax.Array, w8: jax.Array, scale_rep4: jax.Array) -> jax.Array:
    """w8: [O, K] int8 already in the plane-permuted column order (host-side),
    so the same concat([sr]*8) scale trick applies and no relayout is needed."""
    seq, k = x.shape
    out_f = w8.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _blk_for(k, 960)
    ck8 = ck // 8
    x = x[:, _perm(k, ck)]

    def kernel(x_ref, w8_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        wq = w8_ref[...]
        s4 = scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            w = wq[:, ci * ck:(ci + 1) * ck].astype(s4.dtype)
            sr = s4[:, ci * ck8:(ci + 1) * ck8]
            w = w * jnp.concatenate([sr] * 8, axis=1)
            acc += jax.lax.dot_general(
                x_all[:, ci * ck:(ci + 1) * ck], w,
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc

    out = pl.pallas_call(
        kernel,
        grid=(out_f // blk,),
        in_specs=[
            pl.BlockSpec((seq, k), lambda i: (0, 0)),
            pl.BlockSpec((blk, k), lambda i: (i, 0)),
            pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.float32),
    )(x, w8, scale_rep4)
    return out.astype(jnp.bfloat16)


def bf16_control_jax(x: jax.Array, w: jax.Array) -> jax.Array:
    """Same tiling and chunked dot as the fused kernel, weights pre-made bf16."""
    seq, k = x.shape
    out_f = w.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _blk_for(k, 960)

    def kernel(x_ref, w_ref, out_ref):
        x_all = x_ref[...]
        wb = w_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            acc += jax.lax.dot_general(
                x_all[:, ci * ck:(ci + 1) * ck], wb[:, ci * ck:(ci + 1) * ck],
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc

    out = pl.pallas_call(
        kernel,
        grid=(out_f // blk,),
        in_specs=[
            pl.BlockSpec((seq, k), lambda i: (0, 0)),
            pl.BlockSpec((blk, k), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.float32),
    )(x, w)
    return out.astype(jnp.bfloat16)


op_fused = pallas.jax_op("w4dbg::fused", w4a16_fused_jax)
op_int8 = pallas.jax_op("w4dbg::int8", w4a16_int8_jax)
op_ctrl = pallas.jax_op("w4dbg::ctrl", bf16_control_jax)


def torch_dequant(packed, scale, out_f, in_f):
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    w = (packed.unsqueeze(-1) >> shifts) & 0xF
    w = w.reshape(out_f, in_f) - 8
    w = w.to(scale.dtype).reshape(out_f, -1, GROUP) * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f).to(torch.bfloat16)


def int8_plane_permuted(packed, out_f, in_f):
    """int8 weights (q-8, unscaled) with columns in the kernel's plane order."""
    ck = _blk_for(in_f, 960)
    ck8 = ck // 8
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF) - 8        # [O, K/8, 8] (m, i)
    chunks = []
    for c in range(in_f // ck):
        blk = q[:, c * ck8:(c + 1) * ck8, :]                # [O, ck8, 8]
        chunks.append(blk.permute(0, 2, 1).reshape(out_f, ck))  # plane-major
    return torch.cat(chunks, dim=1).to(torch.int8)


def bench(fn, *args, iters=50):
    with torch.no_grad():
        fn(*args).cpu()
        t0 = time.monotonic()
        for _ in range(iters):
            out = fn(*args)
        out.cpu()
        return (time.monotonic() - t0) / iters * 1000


def main() -> int:
    device = torch.device("tpu")
    torch.manual_seed(0)
    for out_f, in_f in [(15360, 3840), (3840, 15360)]:
        packed = torch.randint(-(2**31), 2**31 - 1, (out_f, in_f // 8), dtype=torch.int32)
        scale = (torch.randn(out_f, in_f // GROUP, dtype=torch.float32).abs() * 0.01 + 0.001).to(torch.bfloat16)
        x = torch.randn(S, in_f, dtype=torch.bfloat16)
        scale_rep4 = scale.repeat_interleave(4, dim=1)
        w_ref = torch_dequant(packed, scale, out_f, in_f)
        w8 = int8_plane_permuted(packed, out_f, in_f)

        packed_d, scale4_d, x_d = packed.to(device), scale_rep4.to(device), x.to(device)
        w_ref_d, w8_d = w_ref.to(device), w8.to(device)

        y_ref = (x @ w_ref.T).float()
        for name, y in [
            ("fused", op_fused(x_d, packed_d, scale4_d).cpu().float()),
            ("int8", op_int8(x_d, w8_d, scale4_d).cpu().float()),
            ("ctrl", op_ctrl(x_d, w_ref_d).cpu().float()),
        ]:
            rel = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
            print(f"MARKER [{out_f}x{in_f}] {name} max rel diff: {rel:.3e}")
            assert rel < 2e-2, f"{name} numerics off"

        dense_c = torch.compile(lambda a, w: a @ w.T, backend="tpu", dynamic=False)
        t_a = bench(op_fused, x_d, packed_d, scale4_d)
        t_b = bench(op_int8, x_d, w8_d, scale4_d)
        t_c = bench(op_ctrl, x_d, w_ref_d)
        t_d = bench(dense_c, x_d, w_ref_d)
        print(f"MARKER [{out_f}x{in_f}] A fused-int4 {t_a:.3f} ms | B int8 {t_b:.3f} ms | "
              f"C ctrl-bf16 {t_c:.3f} ms | D dense-XLA {t_d:.3f} ms")
        print(f"MARKER [{out_f}x{in_f}] unpack+scale (A-C) {t_a - t_c:.3f} ms | "
              f"convert+scale (B-C) {t_b - t_c:.3f} ms | pallas overhead (C-D) {t_c - t_d:.3f} ms")
    print("MARKER debug shootout done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
