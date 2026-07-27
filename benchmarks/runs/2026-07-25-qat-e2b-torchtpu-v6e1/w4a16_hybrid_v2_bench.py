#!/usr/bin/env python3.12
"""Hybrid v2: compact-scale kernels (scales stored [O, K/32] — no rep4 tax).

int8-compact Linears up to an HBM byte budget, fused int4-compact beyond.
Frees ~5.5 GB on 31B vs rep4, roughly doubling the int8 budget.
Usage: python3.12 w4a16_hybrid_v2_bench.py [model_id] [batch] [int8_budget_gb]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-31B-it-qat-w4a16-ct"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
INT8_BUDGET_GB = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
PROMPT = "Explain in two sentences why TPUs are fast."
GROUP = 32
SEQ = 256
WARMUP_STEPS = 4
BENCH_STEPS = 64

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




class Int8CompactLinear(torch.nn.Module):
    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("w8", int8_compact(packed, out_f, in_f))
        self.register_buffer("scale", scale.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        y = op_int8_c(x.reshape(-1, self.in_f), self.w8, self.scale)
        y = y.reshape(*x.shape[:-1], self.out_f)
        return y + self.bias if self.bias is not None else y


class FusedCompactLinear(torch.nn.Module):
    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("packed", pack_compact(packed, in_f))
        self.register_buffer("scale", scale.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        y = op_fused_c(x.reshape(-1, self.in_f), self.packed, self.scale)
        y = y.reshape(*x.shape[:-1], self.out_f)
        return y + self.bias if self.bias is not None else y


class W4A16Linear(torch.nn.Module):
    """In-graph dequant fallback for non-tileable layers."""

    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("shifts", torch.arange(0, 32, 4, dtype=torch.int32))
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = (self.packed.unsqueeze(-1) >> self.shifts) & 0xF
        w = w.reshape(self.out_f, self.in_f) - 8
        w = w.to(self.scale.dtype).reshape(self.out_f, -1, GROUP) * self.scale.unsqueeze(-1)
        w = w.reshape(self.out_f, self.in_f).to(torch.bfloat16)
        y = x @ w.T
        return y + self.bias if self.bias is not None else y

def main() -> int:
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    print(f"Loading {MODEL} packed (run_compressed=True, CPU)...")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    t0 = time.monotonic()
    int8_n = fallback = 0
    int8_bytes = 0
    for name, mod in list(model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        out_f, in_f = (int(v) for v in mod.weight_shape)
        packed = mod.weight_packed.data
        scale = mod.weight_scale.data
        if scale.shape[0] != out_f:
            scale = scale.T.contiguous()
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        tileable = out_f % 128 == 0 and in_f % 8 == 0 and _ck_for(in_f) is not None
        tileable = tileable and in_f % 32 == 0 and (_ck_for(in_f) or 0) % 32 == 0
        want_int8 = int8_bytes + out_f * in_f < INT8_BUDGET_GB * 1e9
        cls = (Int8CompactLinear if want_int8 else FusedCompactLinear) if tileable else W4A16Linear
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        new = cls(packed, scale, out_f, in_f, bias)
        setattr(parent, name.rsplit(".", 1)[-1], new)
        int8_n += tileable and want_int8
        fallback += not tileable
        if isinstance(new, Int8CompactLinear):
            int8_bytes += new.w8.numel() + new.scale.numel() * 2
    print(f"MARKER swapped: {int8_n} int8 within {INT8_BUDGET_GB} GB budget, {fallback} fallback; "
          f"int8+scale = {int8_bytes / 1e9:.2f} GB; converted in {time.monotonic() - t0:.1f}s")

    model = model.to(device).eval()

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((BATCH, SEQ), pad_id, dtype=torch.long)
    tokens[:, :n_prompt] = ids[0]
    tokens = tokens.to(device)
    mask = torch.zeros((BATCH, SEQ), dtype=torch.long)
    mask[:, :n_prompt] = 1
    mask = mask.to(device)
    one = torch.ones((BATCH, 1), dtype=torch.long).to(device)
    pos = torch.tensor([n_prompt], dtype=torch.long).to(device)

    def step(tokens, mask, last_idx):
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        return logits.index_select(1, last_idx).argmax(-1)

    step_c = torch.compile(step, backend="tpu", dynamic=False)

    def decode_one():
        nonlocal pos, tokens, mask
        nxt = step_c(tokens, mask, pos - 1)
        tokens = tokens.index_copy(1, pos, nxt)
        mask = mask.index_copy(1, pos, one)
        pos = pos + 1

    print(f"Warmup ({WARMUP_STEPS} steps, includes compile)...")
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            decode_one()
        tokens[0, :1].cpu()
    print(f"MARKER warmup done in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(BENCH_STEPS):
            decode_one()
        tokens[0, :1].cpu()
    elapsed = time.monotonic() - t0

    n_total = n_prompt + WARMUP_STEPS + BENCH_STEPS
    text = tokenizer.decode(tokens[0, n_prompt:n_total].cpu(), skip_special_tokens=True)
    print("--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    print(f"MARKER HYBRIDV2 w4a16 compiled decode (SEQ={SEQ}, batch={BATCH}, no KV cache): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s/stream, "
          f"{BATCH * BENCH_STEPS / elapsed:.1f} tok/s aggregate "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
