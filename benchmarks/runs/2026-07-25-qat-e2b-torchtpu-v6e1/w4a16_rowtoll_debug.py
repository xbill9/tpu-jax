#!/usr/bin/env python3.12
"""The 1-row toll, dissected: time the int8 kernel vs activation row count.

Cached decode feeds 1 row/stream through every Linear; no-cache fed 256.
Same weight bytes either way, yet cached b1 runs ~20% slower at the model
level (12B: 27.1 vs 22.6 ms/step). This sweeps rows in {1, 8, 32, 64, 128,
256} on the 12B extreme shapes to show where the time goes: per-call
overhead, sub-8-row block padding, or MXU under-fill.

Also times the same sweep on plain bf16 dense XLA matmul as the control:
if dense shows the same row-independence, the toll is kernel-specific;
if dense also charges ~flat time per call, it's fundamental bandwidth.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

GROUP = 32


def _blk_for(n: int, target: int) -> int:
    blk = target
    while n % blk:
        blk //= 2
    return blk


def _ck_for(k: int) -> int:
    for cand in range(1024, 0, -128):
        if k % cand == 0:
            return cand
    return k


def make_int8_op(rows: int):
    """Row-count-specific registration (jax_op fns are shape-annotated by
    tracing; distinct names per row count keep the registry clean)."""

    def int8_matmul(x: jax.Array, w8: jax.Array, scale_rep4: jax.Array) -> jax.Array:
        seq, k = x.shape
        out_f = w8.shape[0]
        blk = 256 if out_f % 256 == 0 else (128 if out_f % 128 == 0 else out_f)
        ck = _ck_for(k)
        ck8 = ck // 8

        j = np.arange(k)
        c, r = j // ck, j % ck
        perm = c * ck + 8 * (r % ck8) + r // ck8
        x = x[:, perm]

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
            out_ref[...] = acc.astype(jnp.bfloat16)

        return pl.pallas_call(
            kernel,
            grid=(out_f // blk,),
            in_specs=[
                pl.BlockSpec((seq, k), lambda i: (0, 0)),
                pl.BlockSpec((blk, k), lambda i: (i, 0)),
                pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
            out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.bfloat16),
        )(x, w8, scale_rep4)

    return pallas.jax_op(f"rowtoll::int8_r{rows}", int8_matmul)


def int8_plane_permuted(packed, out_f, in_f):
    ck = _ck_for(in_f)
    ck8 = ck // 8
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF) - 8
    chunks = []
    for c in range(in_f // ck):
        blk = q[:, c * ck8:(c + 1) * ck8, :]
        chunks.append(blk.permute(0, 2, 1).reshape(out_f, ck))
    return torch.cat(chunks, dim=1).to(torch.int8).contiguous()


def bench(fn, *args, iters=100):
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
    rows_sweep = [1, 8, 32, 64, 128, 256]
    for out_f, in_f in [(15360, 3840), (3840, 15360)]:
        packed = torch.randint(-(2**31), 2**31 - 1, (out_f, in_f // 8), dtype=torch.int32)
        scale = (torch.randn(out_f, in_f // GROUP).abs() * 0.01 + 0.001).to(torch.bfloat16)
        w8 = int8_plane_permuted(packed, out_f, in_f).to(device)
        s4 = scale.repeat_interleave(4, dim=1).contiguous().to(device)
        w_dense = torch.randn(out_f, in_f, dtype=torch.bfloat16).to(device)
        dense_c = torch.compile(lambda a, w: a @ w.T, backend="tpu", dynamic=False)

        for rows in rows_sweep:
            x = torch.randn(rows, in_f, dtype=torch.bfloat16).to(device)
            op = make_int8_op(rows)
            t_k = bench(op, x, w8, s4)
            t_d = bench(dense_c, x, w_dense)
            print(f"MARKER [{out_f}x{in_f}] rows={rows:>3}  int8-kernel {t_k:.3f} ms | "
                  f"dense-XLA {t_d:.3f} ms")
    print("MARKER rowtoll done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
