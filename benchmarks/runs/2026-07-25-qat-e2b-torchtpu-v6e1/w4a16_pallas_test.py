#!/usr/bin/env python3.12
"""Fused w4a16 dequant-matmul Pallas kernel for TorchTPU — single-layer test.

y = x @ dequant(packed, scale).T computed inside one Pallas kernel: packed int4
weights (compressed-tensors pack-quantized: nibble i of int32 j = weight j*8+i,
stored value = q+8) are unpacked and scaled per group of 32 in VMEM, fed
straight to the MXU. The bf16 weight tensor never exists in HBM.

Phases: correctness vs the unpack-in-graph torch reference (12B layer shapes),
then a timing shootout: fused kernel vs in-graph dequant vs dense bf16 matmul.
"""

import time

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

GROUP = 32
S = 256  # static sequence buffer, matches the decode bench


def _blk_for(n: int, target: int) -> int:
    blk = target
    while n % blk:
        blk //= 2
    return blk


def w4a16_matmul_jax(x: jax.Array, packed: jax.Array, scale_rep4: jax.Array) -> jax.Array:
    """x: [S, K] bf16; packed: [O, K//8] int32; scale_rep4: [O, K//8] bf16
    (= group scales repeat_interleave(4) along dim 1, done host-side)."""
    import numpy as np

    seq, k = x.shape
    out_f = packed.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _blk_for(k, 960)
    ck8 = ck // 8

    # Nibble-plane order: kernel column (chunk c, plane i, int32 m) holds
    # original weight column c*ck + 8m + i. Permute x to match (cheap: [S, K]).
    j = np.arange(k)
    c, r = j // ck, j % ck
    perm = c * ck + 8 * (r % ck8) + r // ck8
    x = x[:, perm]

    def kernel(x_ref, packed_ref, scale_ref, out_ref):
        x_all = x_ref[...]                                     # [S, K] bf16 (permuted)
        p = packed_ref[...]                                    # [BLK, K//8] int32
        s4 = scale_ref[...]                                    # [BLK, K//8] bf16
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            pc = p[:, ci * ck8:(ci + 1) * ck8]                 # [BLK, ck8]
            planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]  # elementwise only
            w = jnp.concatenate(planes, axis=1).astype(s4.dtype)      # [BLK, ck]
            sr = s4[:, ci * ck8:(ci + 1) * ck8]                # scale for every plane: sc[m//4]
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


w4a16_matmul = pallas.jax_op("w4a16::matmul", w4a16_matmul_jax)


def torch_dequant(packed, scale, out_f, in_f):
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    w = (packed.unsqueeze(-1) >> shifts) & 0xF
    w = w.reshape(out_f, in_f) - 8
    w = w.to(scale.dtype).reshape(out_f, -1, GROUP) * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f).to(torch.bfloat16)


def bench(fn, *args, iters=50):
    with torch.no_grad():
        fn(*args).cpu()  # warmup / compile
        t0 = time.monotonic()
        for _ in range(iters):
            out = fn(*args)
        out.cpu()
        return (time.monotonic() - t0) / iters * 1000  # ms


def main() -> int:
    device = torch.device("tpu")
    torch.manual_seed(0)
    # 12B gate_proj and down_proj shapes — the two extremes
    for out_f, in_f in [(15360, 3840), (3840, 15360)]:
        packed = torch.randint(-(2**31), 2**31 - 1, (out_f, in_f // 8), dtype=torch.int32).to(device)
        scale = (torch.randn(out_f, in_f // GROUP, dtype=torch.float32).abs() * 0.01 + 0.001).to(torch.bfloat16).to(device)
        x = torch.randn(S, in_f, dtype=torch.bfloat16).to(device)

        scale_rep4 = scale.repeat_interleave(4, dim=1)         # [O, K//8]
        y_fused = w4a16_matmul(x, packed, scale_rep4).cpu().float()
        w_ref = torch_dequant(packed, scale, out_f, in_f)
        y_ref = (x @ w_ref.T).cpu().float()
        rel = ((y_fused - y_ref).abs().max() / y_ref.abs().max()).item()
        print(f"MARKER [{out_f}x{in_f}] fused vs reference max rel diff: {rel:.3e}")
        assert rel < 2e-2, "fused kernel numerics off"

        ref_c = torch.compile(lambda x, p, s: x @ torch_dequant(p, s, out_f, in_f).T, backend="tpu")
        w_dense = w_ref.to(device)
        dense_c = torch.compile(lambda x, w: x @ w.T, backend="tpu")

        t_fused = bench(w4a16_matmul, x, packed, scale_rep4)
        t_graph = bench(ref_c, x, packed, scale)
        t_dense = bench(dense_c, x, w_dense)
        gb = out_f * in_f / 2 / 1e9
        print(f"MARKER [{out_f}x{in_f}] fused {t_fused:.3f} ms | in-graph dequant {t_graph:.3f} ms | "
              f"dense bf16 {t_dense:.3f} ms | packed bytes/iter {gb:.2f} GB -> fused {gb / t_fused * 1000:.0f} GB/s eff")
    print("MARKER all shapes pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
