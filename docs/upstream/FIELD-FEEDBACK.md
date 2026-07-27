# TorchTPU field feedback — Gemma 4 w4a16 on v6e-1

**Context:** engineering notes from building an end-to-end w4a16
compressed-tensors execution path for Gemma 4 (E2B→31B dense + 26B-A4B MoE)
on torch_tpu
`0.1.1.dev20260725090141` / torch 2.11.0+cpu / Python 3.12, single v6e-1.
Each item: symptom → root cause as we understand it → suggested improvement.
Full numbers in RESULTS.md.

## (a) Pallas / Mosaic

**1. Sub-word unpack requires a nibble-plane layout.**
- *Symptom:* the natural int4 unpack `[K/8, 8] → [K]` reshape inside a kernel
  either lowers very slowly (2.8 ms/layer) or is rejected outright with
  Mosaic's `Unsupported reshape`.
- *Cause:* it is a cross-lane interleave; Mosaic has no fast lowering.
- *Workaround:* keep each of the 8 shift/mask planes elementwise and
  concatenate planes; permute the *activation* columns to match (a negligible
  gather on [256, K]) and make group-32 scales plane-independent
  (`scale[m//4]`, host-side `repeat_interleave(4)` + in-kernel tile).
- *Suggestion:* document this pattern (it is the make-or-break detail for any
  packed-weight kernel), or give Mosaic a native lowering for the interleave.

**2. int4 bitcast is a dead end on current Mosaic.**
- *Symptom:* `lax.bitcast_convert_type(packed ^ 0x88888888, int4)` (XOR flips
  the offset-8 nibble to two's-complement) fails to compile:
  `Changing bitwidths not supported.`
- *Suggestion:* support bitwidth-changing bitcasts, or state the limitation in
  the Pallas-on-TPU docs. This would remove most of the VPU unpack cost that
  currently makes fused int4 kernels VPU-bound (38–57% of kernel time —
  RESULTS.md).

**3. Scoped VMEM is carved out of the pipeline-buffer pool — even unused.**
- *Symptom:* controlled experiment (31B fused kernel, batch 4, *only* the
  limit varied via `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib`):
  64 MB → 166.9 ms/step; 96 MB → 194.1 ms (+16%) for the identical workload
  (the kernel needs only ~52 MB either way). Batch 8 needs 97.75 MB scoped
  (`CompileTimeScopedVmemOom` at 96 MB) and at 104 MB compiles but regresses
  to 398.8 ms.
- *Cause:* the scoped limit is reserved from the ~128 MB physical VMEM
  whether used or not; every MB granted comes out of pipelining buffers.
- *Suggestion:* document the tradeoff prominently (the default is 32 MB and
  raising it looks free); ideally size the reservation from actual scoped
  usage. Design rule we derived: keep the limit at the minimum that fits and
  scale batch via grid tiling, not larger whole-activation blocks. (The hard
  wall is real: at batch 4 × 1024 ctx the no-cache whole-x block needs
  139.4 MB > 127.9 MB physical VMEM — uncompilable.)

**4. Block last-dim constraints.** Pallas TPU block last dims must be ×128 or
full-size. Documented behavior, but the error surfaces late; an early
shape-check with a clear message would help.

## (b) SDPA / attention

**5. Fused FLASH_ATTENTION is unreachable from stock HF models.**
- *Symptom:* compiled E2B decode at SEQ=512 still skipped the fused kernel;
  the reported failing condition is `attn_mask is None (current: present)`,
  not sequence length.
- *Cause:* HF forward always passes an explicit mask, so the
  `attn_mask=None` + `is_causal` fast path can never trigger from
  transformers code. (The seq-divisible-by-512 condition also applies.)
- *Suggestion:* accept an explicit causal mask (or pattern-match the standard
  causal/sliding masks) in the fused path; otherwise document clearly that
  fused SDPA requires hand-written mask-free attention. Our port's prefill
  uses `attn_mask=None, is_causal=True` on full-attention layers for this
  reason.
- *Note for Gemma 4 specifically:* attention scale is 1.0 (qk-norms make
  logits scale-free) and head dims are 256/512 — the fused path must honor
  SDPA's explicit `scale=` argument.

## (c) Execution modes

**6. `DEFER_AND_FUSE` is a 7× regression on autoregressive generate.**
- *Symptom:* `TPU_DEFER_AND_FUSE=1` on eager HF `generate()` (E2B): 0.5 tok/s
  vs 3.4 plain eager — 10,398 compile requests for 55 tokens (84% cache hits,
  still losing).
- *Cause:* generate's shapes change every step, so each fused group is a
  fresh compile fingerprint.
- *Suggestion:* docs advertise 2.3× on elementwise chains; add an explicit
  warning that it only pays on static-shape eager chains, and/or a heuristic
  that falls back when the fingerprint-miss rate stays high.

## (d) Compile / cache

**7. Tier-2 cache silently off at world size 1.** Runtime logs "Tier-2
compilation cache is disabled for world size 1" when the env var is absent —
easy to misread as a defect on single-chip dev VMs. Suggest a doc note.

**8. Tier-3 fingerprinting includes input scalar values.** Graph fingerprints
cover shapes + dtypes + op sequence **+ input scalar values**, so a changing
Python scalar argument recompiles per value. Keeping loop state in device
tensors avoids it; worth a prominent doc callout (we found it by reading
cache stats). Positive result worth advertising: the tier-3 GCS cache halved
our cold warmup between runs (84.3 s → 36.6 s) and persists across VMs.

## (e) Telemetry

**9. `torch.tpu._hbm_usage_summary()` reports only compile-cache HBM.**
It shows "0B for 0 executables" pre-compile and never the model footprint;
byte-count estimates remained our only memory source (and were off once —
scale expansion added an unaccounted +5.5 GB on 31B, found via transfer OOM).
A true resident-HBM breakdown would have saved a session.

## (f) Documentation gaps

**10. torch version lag.** Nightlies pair with torch 2.11.0+cpu while parts
of the docs/tooling assume 2.12 — e.g. the "PrivateUse1ProfilerRegistry not
found / native TPU profiling disabled" warning on torch < 2.12 is benign
(traces still capture) but reads as a failure. A compatibility note per
nightly would help.

**11. venv assumption.** The quickstart assumes a venv; installing straight
into a dedicated `python3.12`'s site-packages (system python is 3.10) works
identically and suits VM startup scripts. Worth stating that the requirement
is really "Python 3.12 + the authenticated index", not the venv itself.

## (g) Model / checkpoint findings worth a doc note

**12. Gemma 4 E is exactly three features — and the ct checkpoints encode
one of them.**
- E = dense + (1) KV-shared layers 15–34, (2) per-layer embeddings,
  (3) double-wide MLP on shared layers — NO altup/laurel/sparsity. Porting
  E on top of a dense Gemma 4 port is far cheaper than the family's
  reputation suggests.
- The `w4a16-ct` checkpoints ship **no k/v/k_norm on shared layers by
  construction** — which resolves the known vLLM-TPU load failure
  ("`k_norm.weight` not initialized, layers 15–34") as a *loader* bug, not
  an export bug. Worth relaying to the vLLM-TPU side; a shared-layer note in
  any Gemma 4 example would save others the same triage.

**13. MoE expert traffic grows with batch — the sparse advantage is a
low-batch advantage.**
- *Symptom:* 26B-A4B on one v6e-1: b1 122.5 tok/s (8.2 ms/step), b4 181.4
  aggregate (22.1 ms), b8 HBM-OOM — HLO temporaries 42.13 G > 31.24 G
  (gathered dequantized expert slices materialize as bf16 temps).
- *Cause:* each stream routes independently to its own top-8 experts, so
  expert bytes touched per step grow with batch; measured dense 12B already
  beats the 26B MoE per-step at b4 (18.4 vs 22.1 ms).
- *Suggestion:* a fused dequant-inside-matmul expert kernel (our v2) stops
  the temp materialization; until then, single-chip MoE serving should be
  sized for low batch — b1 is the 26B sweet spot.
