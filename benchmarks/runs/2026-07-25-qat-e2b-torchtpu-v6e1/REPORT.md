# Gemma 4 E2B QAT checkpoints on TorchTPU — steps 1, 2 & 3

**Run:** 2026-07-25 (evening session) · `torchtpu-vm` (ct6e-standard-1t, flex-start, europe-west4-a, project `comglitn`)
**Goal:** validate Google's QAT checkpoints on TorchTPU (vLLM-on-TPU cannot load them — see SKILL.md known-broken note). Steps 1–2 of the plan in project memory; step 3 (packed int4 Pallas kernel) is next.
**Stack:** torch_tpu `0.1.1.dev20260725090141`, torch 2.11.0+cpu, transformers 5.14.1, compressed-tensors 0.17.1, Python 3.12.

## Results

| Checkpoint | Load path | Eager | Compiled (SEQ=256 loop) | Output quality |
| :--- | :--- | ---: | ---: | :--- |
| `gemma-4-E2B-it-qat-q4_0-unquantized` | plain transformers | 3.1 tok/s | **227.6 tok/s** (4.4 ms/step) | coherent |
| `gemma-4-E2B-it-qat-w4a16-ct` | `CompressedTensorsConfig(run_compressed=False)` | 3.3 tok/s | — (see finding 2) | coherent |
| stock `gemma-4-E2B-it` (baseline, earlier run) | plain transformers | 3.4 tok/s | 227.3 tok/s | — |

Both QAT checkpoints are **ungated** on HF. Compiled throughput with QAT weights is
identical to stock — QAT costs nothing at bf16 execution, as expected.

## Findings

1. **The unquantized QAT export loads cleanly in transformers** — no missing-weight
   warnings. vLLM's "k_norm.weight not initialized (layers 15–34)" failure is a
   vLLM weight-mapping problem, not a checkpoint defect.
2. **`w4a16-ct` with default `run_compressed=True` cannot be `torch.compile`d on TPU:**
   the on-the-fly decompression graph-breaks on `Tensor.item()` and then the TPU
   backend rejects the resulting dynamic shapes (`does not support dynamic shape`).
   Compressed execution on TPU therefore requires the step-3 custom kernel.
3. **Dequantized load works end to end:** `run_compressed=False` decompresses the
   int4/group-32 weights to bf16 at load (~2 s for 276 modules), `.to("tpu")` is
   NOT blocked by the quantizer, and generation matches the unquantized QAT model.
   Since the checkpoint is w4a16 (activations stay 16-bit), this dequantized
   forward is numerically the target quantized model.
4. Same infra papercuts as the morning run: startup pip 403s on the wheel registry
   (fixed over IAP SSH with a user token); MCP SSH tools can't reach the VM
   (no IAP) — `verify_pytorch_tpu` and friends need manual `--tunnel-through-iap`
   equivalents on this network.

## Step 3: true packed int4 execution (`w4a16_packed_bench.py`)

No Pallas needed for v1: `W4A16Linear` keeps `weight_packed` (int32) + group-32
scales in HBM and dequantizes inside the compiled forward with traceable ops only
— `(packed >> 4*i) & 0xF`, `- 8`, `* scale` (arithmetic shift is safe: the mask
kills sign-extension bits). All 276 CompressedLinears swapped; static shapes, no
graph breaks, single compile.

| Metric | Packed w4a16 | bf16 baseline |
| :--- | ---: | ---: |
| Quantized-Linear weight bytes in HBM | **1.06 GB** | 3.75 GB (3.56×) |
| Compiled decode (SEQ=256 loop) | **157.5 tok/s** (6.3 ms/step) | 227.6 tok/s (4.4 ms/step) |
| Warmup incl. compile | 57.1 s | 19.3 s |
| Dequant vs compressed-tensors decompress | **bit-exact (0.0)** CPU and TPU | — |
| Greedy output | identical text to dequantized reference | — |

The ~1.9 ms/step overhead is the per-forward dequant (XLA materializes the bf16
weight per layer; unpack runs on the VPU). On E2B this is a memory win, not a
speed win.

### Same script across the QAT family (all ungated, all bit-exact)

| Model | Quantized Linears packed | vs bf16 | Packed decode | bf16 decode |
| :--- | ---: | ---: | ---: | ---: |
| E2B (276 modules) | 1.06 GB | 3.75 GB | 157.5 tok/s | 227.6 |
| E4B (343 modules) | 2.23 GB | 7.95 GB | 72.4 tok/s | 114.1 |
| 12B (328 modules) | 6.13 GB | 21.80 GB | 19.3 tok/s | 45.2 |

All 3.56× smaller; 12B drops from ~26 GB total HBM (barely fit) to ~10 GB. The +29.7 ms/step overhead is consistent with
the dequantized bf16 weights (21.8 GB) round-tripping through HBM each step
(~2 × 21.8 GB ÷ ~1.6 TB/s ≈ 27 ms) — i.e. the cost is exactly the
materialization traffic a **fused Pallas dequant-matmul** (weights never leave
HBM as bf16) would eliminate. That kernel is the clear next optimization.
Meanwhile the packed path frees ~16 GB of HBM on 12B — room for longer buffers,
lockstep batch, or a KV cache. Roadmap: E4B next, maybe the ~27B MoE (which
bf16 cannot fit on one v6e-1 at all — packed int4 is the only way it fits).

Logs: `logs/w4a16_run.log` (E2B), `logs/w4a16_12b.log` (12B).

## Step 3b: fused Pallas dequant-matmul kernel (`w4a16_fused_model_bench.py`)

`pallas.jax_op` (torch_tpu's JAX/Pallas bridge; auto-registers a
torch.library custom op with fake impls, composes with
`torch.compile(backend="tpu")`; requires `pip install jax`, type-annotated
JAX fn). Kernel: grid over 256-row weight blocks, full-K blocks, in-kernel
K chunking (~1024), fp32 accumulator. The make-or-break detail is
**nibble-plane layout**: naive `[K/8, 8] → [K]` unpack is a cross-lane
interleave that Mosaic lowers slowly (2.8 ms/layer) or rejects ("Unsupported
reshape"). Instead each of the 8 shift/mask planes stays elementwise and
planes are concatenated; the *activation* columns are permuted to match
(gather on [256, K] — negligible), and group-32 scales become
plane-independent (`scale[m//4]`, host-side `repeat_interleave(4)` +
in-kernel tile). Also learned: Pallas TPU block last-dims must be ×128 or
full-size; scoped VMEM limit is 32 MB (raise via
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536`).

| 12B decode (SEQ=256 loop) | ms/step | tok/s |
| :--- | ---: | ---: |
| bf16 dense | 22.1 | 45.2 |
| packed, in-graph dequant | 51.8 | 19.3 |
| **packed, fused Pallas kernel** | **35.4** | **28.3** (328/328 layers fused) |

E2B fused: 112.9 tok/s (8.9 ms/step) — **slower than in-graph** (157.5).
The fused custom call carries a fixed per-invocation cost (~276–328 calls
per step); on E2B's small layers that overhead exceeds the bandwidth
savings, on 12B's big layers the savings win. **Crossover rule: in-graph
dequant for small models, fused kernel for big ones.** Single-layer
numerics vs reference: ~2e-03 max rel (reduction order); model output text
identical to the in-graph packed run for both models.

**Remaining gap analysis:** ~11 G weights × ~5 VPU ops (shift/and/sub/
convert/mul) ≈ 14 ms/step of unpack on 12B — matches the 13 ms gap to bf16.
The kernel is VPU-bound on nibble extraction, not bandwidth-bound.

**Escapes, with the first already probed** (`int4_bitcast_probe.py`,
`logs/int4_probe.log`):

- ❌ **int4 bitcast — DEAD END on current Mosaic.**
  `lax.bitcast_convert_type(packed ^ 0x88888888, int4)` (XOR flips the
  offset-8 nibble to two's-complement int4) fails to compile:
  *"Changing bitwidths not supported."* Re-probe on future torch_tpu
  nightlies; do not re-derive.
- **Lockstep batch (next best, guaranteed):** unpack cost is per-step, not
  per-stream — batch 4 amortizes it 4×. Needs an S-dimension grid tile in
  the kernel (x [1024, 15360] bf16 = 31 MB no longer fits VMEM whole).
- **int8 middle point:** store weights as int8 (dequant int4→int8 at load):
  no unpack at all, just native convert+mul (~2 VPU ops); 2× memory of
  packed int4 but still 2× under bf16 (~13 GB for 12B) — likely near-parity
  speed. Worth benching against batch amortization.

Even as-is: 28.3 tok/s at ~10 GB HBM beats 19.3, and the freed ~16 GB
enables the KV-cache/batch work.

## Step 4 — zero-code-change sweep (2026-07-26, same VM + fresh vm2)

All three doc-derived free knobs measured on E2B qat-unquantized. **None
helps; the wins are structural.**

| Experiment | Result | Baseline | Verdict |
|---|---|---|---|
| `TPU_DEFER_AND_FUSE=1` on eager HF generate | 0.5 tok/s (10,398 compile reqs / 55 tok, 84% hits) | 3.4 tok/s plain eager | ❌ 7× WORSE — generate's shapes change every step; each fused group is a fresh fingerprint. Static-shape eager chains only. |
| `--seq 512` compiled decode (fused-SDPA attempt) | 194.2 tok/s (5.1 ms/step) | 227.6 @ SEQ=256 | ❌ Fused FLASH_ATTENTION still skipped — failing condition is **`attn_mask is None (current: present)`**, not seq length. HF forward always passes a mask ⇒ fused SDPA is unreachable from stock HF models. Mask-free `is_causal` attention goes on the gemma4-port feature list. |
| `TORCH_TPU_INTERNAL_XLA_OPTIONS=xla_optimization_level=O3` | 226.9 tok/s (4.4 ms/step) | 227.6 | ➖ No change; O2 default already saturates this graph. |

**Startup template first live boot (torchtpu-vm2):** whole startup **1 m 35 s**
(vs ~10 min AR-path), mirror install + extras + smoke test green, and the
tier-3 compile cache confirmed writing `.bin` entries to
`gs://comglitn-torchtpu-wheels/compile-cache/8a5b7822779ee13a/` from an SSH
workload. (Startup smoke test itself runs cache-cold — env vars are written
to /etc/environment after it.)

## Step 5 — lockstep scaling + the 31B single-chip fit (2026-07-26)

**Lockstep batch=4, E2B bf16 (SEQ=256):** 450.5 tok/s aggregate at
8.9 ms/step vs 227.6 tok/s / 4.4 ms at batch 1 — 4 streams for 2.0× the
step cost (1.98× aggregate). Confirms the amortization thesis: per-step,
batch-independent costs (the fused kernel's ~14 ms/step unpack on 12B)
divide by the stream count.

**gemma-4-31B-it-qat-w4a16-ct FITS AND RUNS on one v6e-1** (fresh vm2,
in-graph dequant W4A16Linear, first try):

| metric | value |
|---|---|
| Linears swapped | 410 |
| packed+scale HBM | **16.47 GB** (bf16 would be 58.57 GB — impossible on a 32 GB chip) |
| dequant parity | max abs diff **0.0** vs compressed-tensors reference (CPU and TPU) |
| warmup (incl. compile) | 77.4 s |
| compiled decode | 4.6 tok/s (217.6 ms/step) |

In-graph dequant is the wrong side of the crossover at this size (it
re-materializes ~29 G weights in bf16 every step); the fused Pallas kernel
run is the follow-up. The headline stands regardless: the biggest dense
Gemma 4 QAT checkpoint is single-chip-servable on TorchTPU.

**Fused Pallas kernel on 31B (same night, vm2):** 410/410 layers fused,
warmup 71.5 s, **9.2 tok/s (109.0 ms/step)** — 2.0× over in-graph (217.6 ms).
Output text coherent. Consistent with the VPU-bound model: ~29 G weights /
11 G × the 12B kernel's ~35 ms ≈ 93 ms predicted, 109 observed. With
lockstep batch 4 (unpack amortizes, per the 1.98×-at-2×-cost E2B data)
a ~25-35 tok/s aggregate 31B single-chip server is the projected next stop.

**Fused 31B + lockstep batch 4 (host-loop patch, kernel unchanged):
24.0 tok/s aggregate** (6.0/stream, 166.9 ms/step, 64 MB scoped VMEM).
4 streams for 1.53× the step cost = 2.6× aggregate over single-stream
fused (9.2). Cost model from the two points: ~90 ms fixed (unpack +
per-call overhead) + ~19 ms/stream → batch 8 ≈ 33 tok/s if the down_proj
activation block (~88 MB at batch 8) clears the 96 MB scoped-VMEM notch.
The bench script now takes batch as argv[2].

**Batch scaling curve, fused 31B** (whole-x kernel, scoped VMEM as noted):

| batch | scoped VMEM | ms/step | tok/s aggregate | note |
|---|---|---|---|---|
| 1 | 64 MB | 109.0 | 9.2 | |
| 4 | 64 MB | 166.9 | **24.0** | sweet spot so far |
| 8 | 96 MB | — | — | CompileTimeScopedVmemOom: needs 97.75 MB (x block 88 MB) |
| 6 | 96 MB | 302.2 | 19.9 | also regresses |
| 8 | 104 MB | 398.8 | 20.1 | compiles but REGRESSES — no VMEM left for pipelining |

Batches 6 and 8 break the linear cost model (~90 ms fixed + ~19 ms/stream):
the scoped-VMEM limit must grow to hold the whole x block, and every MB
given to scoped allocation comes out of the pipeline-buffer pool (128 MB
physical − scoped limit). Batch 4 at the default-ish 64 MB split is the
whole-x design's peak on 31B: **24.0 tok/s aggregate is the number**.
Pushing further needs the S-tiled grid (tile x rows, accept re-unpacking
per S-tile) or int8-stored weights.

**Fused 12B + lockstep batch 4 (closing run): 56.5 tok/s aggregate**
(14.1/stream, 70.8 ms/step, 64 MB scoped VMEM — x block 31 MB, full
pipelining headroom). Marginal cost ~11.8 ms/stream over the 35.4 ms
single-stream base. **Packed int4 12B at batch 4 out-throughputs bf16
single-stream (45.2 tok/s) by 25% at ~1/3 the HBM (~10 GB vs 26 GB).**
The memory-win and the throughput-win now point the same direction.

**E4B fused cell closed (last open matrix cell):** fused batch 1 =
54.6 tok/s (18.3 ms/step) vs 72.4 in-graph — crossover rule holds, E4B
stays on the in-graph side single-stream. Fused batch 4 = 118.9 tok/s
aggregate (33.7 ms/step, 1.8× step cost for 4 streams). In-graph batch-4
E4B left unmeasured (packed bench has no batch arg); likely higher still.

## Step 6 — kernel component shootout (`w4a16_kernel_debug.py`, fresh VM)

Four kernels, identical grid/tiling/chunked-dot structure, only the weight
path differs. 12B extreme shapes, 50-iter means:

| ms | gate 15360x3840 | down 3840x15360 |
|---|---|---|
| A fused-int4 (production) | 0.415 | 0.352 |
| B int8-stored, plane-permuted | 0.306 | 0.272 |
| C bf16-in-Pallas control | 0.257 | 0.149 |
| D dense XLA matmul | 0.180 | 0.125 |
| unpack+scale = A-C | 0.158 (38%) | 0.202 (57%) |
| convert+scale = B-C | 0.049 | 0.122 |
| Pallas/tiling overhead = C-D | 0.077 | 0.025 |

Confirmed: (1) nibble unpack is 38-57% of fused kernel time — the VPU-bound
diagnosis is now measured, not inferred; (2) the int8-stored escape prices
as predicted: ~25% faster than fused int4 on every shape, same numerics
(int8 kernel reuses the plane-permuted layout so the concat-scale trick
still applies — no relayout); (3) XLA's own tiling beats our fixed
blk=128/ck=960 by 20-40% on the tall shape — the gap Helion autotuning
would target.

**Scoped-VMEM control experiment (31B batch 4, only the limit varied):**
64 MB = 166.9 ms/step, 96 MB = 194.1 (+16%), 104 MB = 194.1 (warm-started
from cache; may be the 96 MB binary — treat 96 as the second point).
The limit alone slows the identical workload: scoped VMEM is carved out of
the pipeline-buffer pool whether used or not (the batch-4 kernel needs only
~52 MB). Confirms the batch-6/8 regression mechanism and the design rule:
keep the scoped limit at the minimum that fits; scale batch via an S-tiled
grid at 64 MB, not by growing the whole-x block. Bonus datapoint: tier-3
GCS cache halved warmup between runs (84.3 s cold -> 36.6 s).

## Step 7 — int8-stored weights at model level (`w4a16_int8_model_bench.py`)

Load-time int4→int8 conversion (7.7 s, plane-permuted layout so the kernel
does only convert+scale+dot — zero nibble work). 12B, 328/328 layers:

| 12B config | HBM (Linears) | b1 | b4 aggregate |
|---|---|---|---|
| bf16 | 26 GB | 45.2 tok/s (22.1 ms) | — |
| fused int4 | ~10 GB | 28.3 tok/s (35.4 ms) | 56.5 tok/s (70.8 ms) |
| **int8-stored** | **13.62 GB** | **44.3 tok/s (22.6 ms)** | **66.5 tok/s (60.2 ms)** |

int8-stored reaches bf16-parity speed (−2%) at half the memory; the
kernel-level ~25% prediction understated the model-level win (36%).
New ranking for 12B-class: int8-stored > bf16 (memory) > fused int4
(only when HBM is the binding constraint). 31B pure-int8 doesn't fit
(29.6 GB weights); hybrid int8/int4 split is the follow-up.

**Hybrid int8/int4 on 31B (`w4a16_hybrid_model_bench.py`, budget as argv[3]):**
13 GB int8 budget = HBM transfer OOM — real all-fused residency is ~22 GB
(the 16.47 GB figure counts scale UNexpanded; scale_rep4 adds 0.25 B/elem
= +5.5 GB on 31B). 8 GB budget fits: 91/410 layers int8, **10.2 tok/s
(97.6 ms/step)**, +11% over all-fused (9.2). The 31B single-chip ceiling
with today's kernels: 10.2 b1 / 24.0 aggregate b4. Next memory lever:
store scales unexpanded (frees 5.5 GB on 31B → roughly doubles the int8
budget → ~11 tok/s, and more once the kernel reads compact scales).

## Step 8 — tier-4 validation pass (2026-07-26, `tier4_validation.py` / `tier4_xprof.py`)

**Parity (official harness):** all four extreme shapes (12B + 31B
geometries) × both production kernels (fused-int4, int8-stored) PASS
`torch_tpu._internal.utils.utils.assert_close` STRICT — 8/8, zero failures.

**Perplexity (quality-neutrality):** 12B int8-swapped vs dequantized-bf16
target on identical BOS-led 256-token windows (40 windows, 10,200 scored
tokens, raw literary text): nll/tok **6.86799 vs 6.86737** (PPL 961.0 vs
960.4) — a 0.009% delta. The packed path is quality-neutral end to end.
(Absolute PPL is it-model-on-raw-text scale; the delta is the claim.
Methodology note: Gemma windows MUST start with BOS — without it both
variants score nll ~10.2.)

**xProf:** 50-step trace of int8 12B batch-4 decode captured (110.7 ms/step
under ~1.8x tracing overhead); archived at
`gs://comglitn-torchtpu-wheels/traces/xprof_trace_12b_int8_b4` for
tensorboard analysis.

**Telemetry caveat:** `torch.tpu._hbm_usage_summary()` reports only
compilation-cache HBM ("0B for 0 executables" pre-compile), not total
model footprint — byte-count estimates remain our memory source.

**Tier-1 int8 IN-GRAPH variant (`w4a16_int8_ingraph_bench.py`) — NEGATIVE:**
E2B 116.5 tok/s (8.6 ms) vs int4 in-graph 157.5; E4B 61.8 (16.2 ms) vs
72.4. In-graph is bandwidth-bound: XLA materializes the bf16 temp either
way, so int8's 2x-larger weight reads only add traffic. Completes the
symmetric rule: **in-graph path -> int4 (bandwidth-bound); Pallas kernel
path -> int8 (VPU-bound).** Final per-model best configs:
E2B int4-in-graph 157.5 | E4B int4-in-graph 72.4 |
12B int8-fused 44.3 (66.5 agg b4) | 31B hybrid 10.2 (24.0 agg b4 fused).

## Step 9 — compact-scale kernels + hybrid v2 (`w4a16_compact_debug.py`, `w4a16_hybrid_v2_bench.py`)

Scales stored at natural [O, K/32] (0.0625 B/elem) instead of rep4's
0.25 B/elem: the column permutation gains one more level — within each
chunk, int32 columns reorder from m=4q+r to (r, q), making the group index
q independent of both plane i and r, so the compact scale block expands
with two nested lane-concats (concat x4 over r, then x8 over i). No
repeat_interleave, no relayout. 8/8 numerics PASS; per-layer timing mixed
vs rep4 (up to +30% on some shapes) — but model-level coverage wins:

**31B hybrid v2 (17 GB int8 budget, 227/410 layers int8-compact, rest
fused-int4-compact): 11.4 tok/s (87.7 ms/step) — new single-chip record**
(9.2 all-fused rep4 -> 10.2 hybrid rep4 8 GB -> 11.4 hybrid compact 17 GB).
Fit math with compact scales: resident ≈ 16.5 + 0.47 x budget_GB;
17 GB -> ~24.5 GB Linears, loads clean.

**Hybrid v2 at batch 4: stays behind all-fused.** 96 MB limit: 15.9 agg
(252.3 ms); corrected to the 64 MB limit (compact blocks fit): 17.9 agg
(223.8 ms) — vs all-fused rep4's 24.0. Closing rule of the whole design
space: **batch amortizes unpack, so batch >= 4 is bandwidth-bound and
packed int4 (fewest bytes) wins; batch 1 is VPU-bound and int8 wins.**
Untested cell: all-fused COMPACT at b4 (might edge 24.0 via scale-byte
savings; next session).

FINAL 31B single-chip: **11.4 tok/s b1** (hybrid v2 compact, 17 GB budget)
/ **24.0 tok/s aggregate b4** (all-fused rep4).

**Last open cell — all-fused-COMPACT 31B (budget 0): loses to rep4 at both
batches** (b1 116.3 vs 109.0 ms; b4 226.5 vs 166.9). The two-level scale
expansion (concat x4 then x8 = 32 narrow lane-concats vs rep4's 8 wide)
costs more VPU than the 0.19 B/elem byte saving returns. Compact scales
are a MEMORY unlock (they bought the int8 coverage behind the 11.4 b1
record), not a speed play. Benchmark matrix now fully closed:
**31B b1 crown = 11.4 (hybrid compact), b4 crown = 24.0 (all-fused rep4).**

## Step 10 — the KV-cached port on real TPU (session C, `ports/gemma4/tpu_bench.py`)

The clean-room port (bitwise HF parity) + static KV cache + real 12B ct
weights (328/328 Linears -> Int8W4A16Linear; ZERO port changes needed for
real weights) compiled with backend="tpu", dynamic=False:

| 12B config | b1 | b4 aggregate | prefill (128 tok) |
|---|---|---|---|
| no-cache int8 (old record) | 44.3 tok/s (22.6 ms) | 66.5 tok/s | n/a (in-loop) |
| **cached port, int8** | 36.8 (27.1 ms) | **217.4 (18.4 ms/step)** | **46 ms compiled** |

- Cached b1 sits BELOW no-cache: at MAX_SEQ=256 decode is weight-bandwidth
  bound (13.6 GB read/step either way) and 1-row matmuls waste tiling the
  256-row full-buffer got free. steps_per_call=8: no change (not
  launch-bound).
- Cached b4 is the payoff: weights are read once per step regardless of
  batch, so 4 streams cost 18.4 ms total — **3.3x the aggregate record**,
  and per-stream (54.4) beats even bf16 no-cache b1 (45.2).
- Prefill/decode split: 46 ms compiled prefill vs the old design's
  in-loop prompt recompute.

## Step 11 — context scaling: the cache's raison d'etre (12B int8, fresh VM)

| batch 1 | 256 ctx | 1024 ctx |
|---|---|---|
| no-cache | 44.3 tok/s (22.6 ms) | 17.9 tok/s (55.9 ms) — 2.5x worse |
| cached port | 36.8 (27.1 ms) | **36.1 (27.7 ms) — flat** |

| batch 4 | 256 ctx | 1024 ctx |
|---|---|---|
| no-cache | 66.5 agg | **UNCOMPILABLE** — whole-x VMEM 139.4 M > 127.9 M physical |
| cached port | 217.4 agg | 178.4 agg (22.4 ms/step) |

No-cache cost grows linearly-plus with context (and its whole-x kernel
block grows past physical VMEM at batch x context = 4096 rows); cached
decode is flat (+0.6 ms for 4x context = the attention read over the
longer cache). Crossover at batch 1 sits just past 256 ctx; at any real
context or batch the cache is not an optimization but a requirement.

## Step 12 — the E-series port on TPU: E2B cached (`tpu_bench_e.py`)

First TPU run of the E-port (KV-shared layers aliasing cache slots +
per-layer embeddings + double-wide MLPs), real E2B ct weights, 276/276
Linears on the int8 kernel, 2.35 GB weights+scale — compiled first try:

| E2B | b1 | b4 |
|---|---|---|
| best no-cache (int4 in-graph) | 157.5 tok/s | — |
| bf16 no-cache (decode bench) | 227.6 | 450.5 agg |
| **cached E-port, int8** | 116.6 (8.6 ms) | **809.9 agg / 202.5 per stream (4.9 ms/step)** |

Same shape as 12B: cached b1 pays the 1-row toll on small layers, cached
b4 rewrites the book — per-stream beats every prior E2B config, aggregate
is 1.8x the bf16 b4 record at ~40% of its memory. Full port family now
TPU-proven: dense (12B measured), E-series (E2B measured), MoE
(parity-proven locally, TPU assembly pending).

**Cached E4B (old VM's last hour, `tpu_bench_e.py`):** 343/343 int8
Linears, 4.97 GB. **b1 = 73.9 tok/s (13.5 ms) — beats the old E4B record
(72.4) even single-stream**, the first model where cached wins at b1
(KV-sharing trims per-step cache traffic). **b4 = 427.2 tok/s aggregate**
(106.8/stream, 9.4 ms/step) — 3.6x the old fused-b4 record (118.9).
Cached matrix now measured for E2B/E4B/12B; 31B cached remains the one
open cell.

## Step 13 — 26B-A4B MoE on a single v6e-1 (first ever; `tpu_bench_moe.py`)

No packed checkpoint exists — the bf16 QAT masters
(qat-q4_0-unquantized, 52 GB) were DIY-quantized at load: 30 layers
swapped to W4A16Experts (int4 g32, ct layout), 34 s quantization pass,
peak host RAM 114.9 GB. HBM weight footprint **17.64 GB** (12.85 int4
experts + 4.79 bf16 non-expert). The static gather+bmm expert dispatch
compiled on real TPU (backend="tpu") without modification.

**batch 1: 122.5 tok/s (8.2 ms/step)** — second-fastest single-stream
config in the family (only ~1/16 of expert bytes touched per token).
Compiled prefill exec 711 ms / 128 tok. Decoded text fully coherent —
the DIY quantization + router wiring validated end to end on real
weights. KV cache: 0.06 GB.

With this, all four Gemma 4 architectures (E2B, E4B, 12B, 31B dense,
26B MoE) run w4a16-quantized on single v6e-1 chips through the port.

**26B MoE batch curve:** b1 122.5 (8.2 ms) / b4 181.4 agg (22.1 ms) /
b8 HBM-OOM (HLO temporaries 42.13 G > 31.24 G — gathered dequantized
expert slices materialize as bf16 temps). Two architectural findings:
(1) MoE expert traffic GROWS with batch (each stream routes to its own
top-8), so the sparse advantage is a low-batch advantage — dense 12B
already beats the 26B per-step at b4 (18.4 vs 22.1 ms); (2) scaling MoE
batch needs a fused dequant-inside-matmul expert kernel (v2) to stop
temp materialization. b1 remains the 26B sweet spot: 122.5 tok/s at
17.64 GB is the second-fastest single-stream config in the family.

## Step 14 — deep-dig session (2026-07-27): last cell + long context + row toll

**Cached 31B (int4-fused Linears): b1 = 9.5 tok/s (105.1 ms) — edges
no-cache (9.2); b4 = 44.0 tok/s aggregate (91.0 ms/step) — NEW 31B
RECORD, 1.83x the old 24.0.** At 31B the weight-read floor (~90 ms)
dominates, so cached b4 ~= the floor while no-cache paid 1024-row
activation costs on top. Matrix complete: every model, every cell.

**12B cached long-context:** b4 aggregate 217.4 (256) -> 178.4 (1024) ->
124.9 (2048); ms/step 18.4 -> 22.4 -> 32.0 — flatness bends as attention
over the cache becomes visible, but the regime is unreachable for the
no-cache design at any batch.

**Row-toll microbench (`w4a16_rowtoll_debug.py`):** kernel time is
row-FLAT 1->256 (bandwidth-bound); summed per-layer 1-row calls are
CHEAPER than 256-row. The cached-b1 toll on 12B/E2B is NOT in the
Linears — it lives in the non-Linear decode graph (1-token attention,
index_copy/mask glue). Sub-8-row padding penalty exists only on wide-K
shapes (0.203 vs 0.120 ms at 1 vs 8 rows). Next lever for b1: fuse the
attention/glue path, not the matmuls.
