#!/usr/bin/env python3.12
"""Compact-scale kernel redesign: kill the scale_rep4 4x storage tax.

Trick: extend the column permutation one level. Within each K-chunk, order
int32 columns as (r, q) where m = 4q + r — i.e. the 4 int32s of each scale
group become lane-strided instead of adjacent. Then for kernel column
(plane i, r, q) the original weight column is c*ck + 32q + 8r + i, whose
group index is exactly q — independent of BOTH i and r. So the compact
scale block [blk, ck/32] expands with two nested lane-concats:
    per_plane = concat([sc]*4)   # over r
    full      = concat([per_plane]*8)  # over i
No repeat_interleave, no relayout, scales stored at their natural
[O, K/32] size (0.0625 B/elem instead of rep4's 0.25 B/elem — 5.5 GB
saved on 31B).

Validates numerics vs the natural-order reference and times compact vs
rep4 variants for both int4-fused and int8-stored kernels.
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


def _ck_for(k: int) -> int:
    for cand in range(1024, 0, -128):
        if k % cand == 0:
            return cand
    return k


def _perm_compact(k: int, ck: int) -> np.ndarray:
    """x permutation: kernel col j -> original col (for x gather)."""
    ck8 = ck // 8
    qq = ck8 // 4
    j = np.arange(k)
    c, t = j // ck, j % ck
    i, u = t // ck8, t % ck8
    r, q = u // qq, u % qq
    return c * ck + 32 * q + 8 * r + i


def pack_compact(packed: torch.Tensor, in_f: int) -> torch.Tensor:
    """Reorder int32 columns chunk-wise from m=4q+r order to (r, q) order."""
    ck = _ck_for(in_f)
    ck8, qq = ck // 8, ck // 32
    out = []
    for c in range(in_f // ck):
        pc = packed[:, c * ck8:(c + 1) * ck8]                # [O, ck8], m = 4q+r
        out.append(pc.reshape(-1, qq, 4).permute(0, 2, 1).reshape(-1, ck8))
    return torch.cat(out, dim=1).contiguous()


def int8_compact(packed: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    """int8 (q-8) weights in the compact-scale column order."""
    ck = _ck_for(in_f)
    ck8, qq = ck // 8, ck // 32
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q8 = ((packed.unsqueeze(-1) >> shifts) & 0xF) - 8        # [O, K/8, 8] (m, i)
    out = []
    for c in range(in_f // ck):
        blk = q8[:, c * ck8:(c + 1) * ck8, :]                # [O, ck8, 8]
        blk = blk.reshape(out_f, qq, 4, 8)                   # [O, q, r, i]
        out.append(blk.permute(0, 3, 2, 1).reshape(out_f, ck))  # (i, r, q)
    return torch.cat(out, dim=1).to(torch.int8).contiguous()


def fused_compact_jax(x: jax.Array, packed: jax.Array, scale: jax.Array) -> jax.Array:
    """packed: compact-ordered [O, K/8] int32; scale: [O, K/32] bf16."""
    seq, k = x.shape
    out_f = packed.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _ck_for(k)
    ck8, cg = ck // 8, ck // 32
    x = x[:, _perm_compact(k, ck)]

    def kernel(x_ref, packed_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        p = packed_ref[...]
        s = scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            pc = p[:, ci * ck8:(ci + 1) * ck8]
            planes = [((pc >> (4 * i)) & 0xF) - 8 for i in range(8)]
            w = jnp.concatenate(planes, axis=1).astype(s.dtype)
            sc = s[:, ci * cg:(ci + 1) * cg]
            per_plane = jnp.concatenate([sc] * 4, axis=1)
            w = w * jnp.concatenate([per_plane] * 8, axis=1)
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
            pl.BlockSpec((blk, k // 32), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.float32),
    )(x, packed, scale)
    return out.astype(jnp.bfloat16)


def int8_compact_jax(x: jax.Array, w8: jax.Array, scale: jax.Array) -> jax.Array:
    """w8: compact-ordered [O, K] int8; scale: [O, K/32] bf16."""
    seq, k = x.shape
    out_f = w8.shape[0]
    blk = _blk_for(out_f, 128)
    ck = _ck_for(k)
    cg = ck // 32
    x = x[:, _perm_compact(k, ck)]

    def kernel(x_ref, w8_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        wq = w8_ref[...]
        s = scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            w = wq[:, ci * ck:(ci + 1) * ck].astype(s.dtype)
            sc = s[:, ci * cg:(ci + 1) * cg]
            per_plane = jnp.concatenate([sc] * 4, axis=1)
            w = w * jnp.concatenate([per_plane] * 8, axis=1)
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
            pl.BlockSpec((blk, k // 32), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.float32),
    )(x, w8, scale)
    return out.astype(jnp.bfloat16)


op_fused_c = pallas.jax_op("wcs::fused", fused_compact_jax)
op_int8_c = pallas.jax_op("wcs::int8", int8_compact_jax)


def torch_dequant(packed, scale, out_f, in_f):
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    w = (packed.unsqueeze(-1) >> shifts) & 0xF
    w = w.reshape(out_f, in_f) - 8
    w = w.to(scale.dtype).reshape(out_f, -1, GROUP) * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f).to(torch.bfloat16)


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
    for out_f, in_f in [(15360, 3840), (3840, 15360), (5376, 21504), (21504, 5376)]:
        packed = torch.randint(-(2**31), 2**31 - 1, (out_f, in_f // 8), dtype=torch.int32)
        scale = (torch.randn(out_f, in_f // GROUP).abs() * 0.01 + 0.001).to(torch.bfloat16)
        x = torch.randn(S, in_f, dtype=torch.bfloat16)
        w_ref = torch_dequant(packed, scale, out_f, in_f)
        y_ref = (x @ w_ref.T).float()

        pc = pack_compact(packed, in_f).to(device)
        w8c = int8_compact(packed, out_f, in_f).to(device)
        s_d, x_d = scale.to(device), x.to(device)

        for name, y in [
            ("fused-compact", op_fused_c(x_d, pc, s_d).cpu().float()),
            ("int8-compact", op_int8_c(x_d, w8c, s_d).cpu().float()),
        ]:
            rel = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
            status = "PASS" if rel < 2e-2 else "FAIL"
            print(f"MARKER [{out_f}x{in_f}] {name} max rel {rel:.3e} {status}")

        t_f = bench(op_fused_c, x_d, pc, s_d)
        t_8 = bench(op_int8_c, x_d, w8c, s_d)
        print(f"MARKER [{out_f}x{in_f}] fused-compact {t_f:.3f} ms | int8-compact {t_8:.3f} ms")
    print("MARKER compact debug done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
