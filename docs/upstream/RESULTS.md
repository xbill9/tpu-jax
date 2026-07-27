# Gemma 4 QAT w4a16-ct on TorchTPU — benchmark book (condensed)

**Setup:** single v6e-1 (ct6e-standard-1t, 32 GB HBM), torch_tpu
`0.1.1.dev20260725090141`, torch 2.11.0+cpu, transformers 5.14.1,
compressed-tensors 0.17.1. Checkpoints: Google's ungated
`gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct` (int4 symmetric, group-32), plus
26B-A4B MoE DIY-quantized at load from the bf16 QAT masters
(qat-q4_0-unquantized). All decode numbers: compiled
(`backend="tpu", dynamic=False`), SEQ=256 loop, greedy.
"b4 aggregate" = 4 lockstep streams, summed tok/s.

## Final champion board (best b1 / best b4 per model)

| Model | b1 champion | b4 aggregate champion | Quant HBM |
|---|---|---|---|
| E2B | 157.5 tok/s (6.3 ms) — int4 in-graph, no cache | **809.9** (202.5/stream, 4.9 ms) — cached port, int8 | 1.06 GB in-graph / 2.35 GB cached-int8 (bf16 3.75) |
| E4B | **73.9** (13.5 ms) — cached port, int8 | **427.2** (106.8/stream, 9.4 ms) — cached port, int8 | 4.97 GB (bf16 7.95) |
| 12B | 44.3 (22.6 ms) — int8 + w8a16 Pallas kernel, no cache | **217.4** (54.4/stream, 18.4 ms) — cached port, int8 | 13.62 GB (bf16 26) |
| 31B | **11.4** (87.7 ms) — hybrid int8/int4, compact scales, 17 GB int8 budget | **24.0** (166.9 ms) — all-fused packed int4 (rep4), no cache | 16.47 GB packed / ~24.5 GB hybrid (bf16 58.57 — does not fit) |
| cached 31B | 9.5 (105.1 ms) | **44.0 (91.0 ms/step)** | measured 2026-07-27 — new 31B aggregate record (1.83x the no-cache 24.0) |
| 26B MoE | **122.5** (8.2 ms) — cached port, DIY int4 experts | 181.4 (22.1 ms) | 17.64 GB (12.85 int4 experts + 4.79 bf16) |

Notes: 12B cached — compiled 128-token prefill = 46 ms; cached b1 (36.8,
27.1 ms) sits below no-cache b1 because MAX_SEQ=256 decode is
weight-bandwidth-bound either way and 1-row matmuls waste tiling; cached b4
reads weights once per step for 4 streams — 3.3× the no-cache aggregate
(66.5), per-stream 54.4 > bf16 b1's 45.2. Superseded configs kept for the
cost model: E4B int4 in-graph 72.4 b1 / fused-b4 118.9; E4B/E2B bf16
baselines 114.1 / 227.6 (450.5 agg b4); 31B all-fused b1 9.2 (109.0 ms).

## Context scaling (12B int8): the cache's raison d'être

| batch 1 | 256 ctx | 1024 ctx |
|---|---|---|
| no-cache | 44.3 tok/s (22.6 ms) | 17.9 tok/s (55.9 ms) — 2.5× worse |
| cached port | 36.8 (27.1 ms) | **36.1 (27.7 ms) — flat** |

| batch 4 | 256 ctx | 1024 ctx |
|---|---|---|
| no-cache | 66.5 agg | **UNCOMPILABLE** — whole-x VMEM 139.4 MB > 127.9 MB physical |
| cached port | 217.4 agg | 178.4 agg (22.4 ms/step) |

No-cache cost grows linearly-plus with context, and its whole-x kernel block
grows past physical VMEM at batch × context = 4096 rows; cached decode is
flat (+0.6 ms for 4× context = the attention read over the longer cache).
Crossover at batch 1 sits just past 256 ctx; at any real context or batch
the cache is not an optimization but a requirement.

## Cached E-series port (KV-shared layers alias cache slots)

E2B: 276/276 Linears on the int8 kernel, 2.35 GB weights+scale, compiled
first try. b1 = 116.6 tok/s (8.6 ms); b4 = **809.9 aggregate / 202.5 per
stream (4.9 ms/step)** — per-stream beats every prior E2B config, aggregate
is 1.8× the bf16 b4 record (450.5) at ~40% of its memory.

E4B: 343/343 int8 Linears, 4.97 GB. b1 = **73.9 tok/s (13.5 ms) — beats the
old E4B record (72.4) even single-stream**, the first model where cached wins
at b1 (KV-sharing trims per-step cache traffic). b4 = **427.2 aggregate**
(106.8/stream, 9.4 ms/step) — 3.6× the old fused-b4 record (118.9).

## 26B-A4B MoE on one v6e-1 (first single-chip run)

No packed checkpoint exists — the bf16 QAT masters (qat-q4_0-unquantized,
52 GB) were DIY-quantized at load: 30 layers swapped to W4A16Experts (int4
g32, ct layout), 34 s quantization pass, peak host RAM 114.9 GB. HBM weight
footprint **17.64 GB** (12.85 int4 experts + 4.79 bf16 non-expert). Compiled
prefill exec 711 ms / 128 tok; KV cache 0.06 GB; decoded text fully coherent.

| batch | result |
|---|---|
| 1 | **122.5 tok/s (8.2 ms/step)** — second-fastest single-stream config in the family (~1/16 of expert bytes touched per token) |
| 4 | 181.4 aggregate (22.1 ms) |
| 8 | HBM OOM — HLO temporaries 42.13 G > 31.24 G (gathered dequantized expert slices materialize as bf16 temps) |

Expert traffic grows with batch (each stream routes to its own top-8), so
the sparse advantage is a low-batch advantage — dense 12B already beats the
26B per-step at b4 (18.4 vs 22.1 ms). Scaling MoE batch needs a fused
dequant-inside-matmul expert kernel (v2) to stop temp materialization.

## Cost model: bandwidth-bound vs VPU-bound

Kernel component shootout (12B extreme shapes, identical grid/tiling, 50-iter
means, ms):

| | gate 15360×3840 | down 3840×15360 |
|---|---:|---:|
| A fused int4 (production) | 0.415 | 0.352 |
| B int8-stored, plane-permuted | 0.306 | 0.272 |
| C bf16-in-Pallas control | 0.257 | 0.149 |
| D dense XLA matmul | 0.180 | 0.125 |
| nibble unpack+scale (A−C) | 0.158 (38%) | 0.202 (57%) |
| int8 convert+scale (B−C) | 0.049 | 0.122 |
| Pallas/tiling overhead (C−D) | 0.077 | 0.025 |

Four rules fall out, each confirmed at model level:

1. **In-graph dequant path → store int4.** XLA materializes the bf16 weight
   temp regardless, so the path is bandwidth-bound; int8-stored in-graph only
   adds read traffic (E2B: 116.5 vs int4's 157.5 tok/s; E4B: 61.8 vs 72.4).
2. **Pallas kernel path, batch 1 → store int8.** The fused-int4 kernel is
   VPU-bound on nibble unpack (38–57% above); load-time int4→int8 removes it.
   12B model level: 44.3 vs 28.3 tok/s (−2% vs bf16 at half the memory).
3. **Batch ≥ 4 → packed int4 wins.** Lockstep batch amortizes the per-step
   unpack (E2B bf16 b4: 1.98× aggregate at 2.0× step cost), so the path turns
   bandwidth-bound and fewest-bytes wins: 31B b4 all-fused int4 24.0 vs
   hybrid-v2 17.9 aggregate.
4. **Compact scales are a memory unlock, not a speed play.** Natural
   [O, K/32] scales (0.0625 B/elem vs rep4's 0.25) cost extra VPU (32 narrow
   lane-concats vs 8 wide): all-fused-compact loses at both batches on 31B
   (b1 116.3 vs 109.0 ms; b4 226.5 vs 166.9). But the freed 5.5 GB bought the
   int8 coverage behind the 11.4 tok/s 31B record.

Crossover between in-graph and fused kernel (single stream): small models
in-graph (E2B fused 112.9 < 157.5; E4B fused 54.6 < 72.4 — per-call custom-op
overhead × ~300 layers dominates), large models fused (12B 28.3 > 19.3;
31B 9.2 > 4.6 — bandwidth savings dominate).

Secondary findings: XLA's own matmul tiling beats our fixed blk=128/ck=960 by
20–40% on the tall shape (autotuning target); scoped-VMEM control experiment
and the batch-6/8 regression are in FIELD-FEEDBACK.md item 3; `steps_per_call`
chaining (8 decode steps per compiled call) changed nothing — decode is not
launch-bound; `xla_optimization_level=O3` changed nothing (227.6 → 226.9).

## Validation receipts

| Check | Result |
|---|---|
| Dequant vs compressed-tensors decompression | bit-exact, max abs diff 0.0 (CPU and TPU; E2B→31B) |
| Kernel numerics, `assert_close` STRICT (official harness) | 8/8 PASS — 4 extreme 12B/31B shapes × {fused-int4, int8-stored} |
| Single-layer fused kernel vs reference | ~2e-03 max rel (reduction order); model output text identical |
| HF parity (port, fp32 CPU, both attention geometries) | final logits within 1e-4, per-layer localization |
| Cached vs no-cache decode (port) | identical token sequences, per-step logits within 1e-4, incl. batch=2 and MAX_SEQ-padding invariance |
| Perplexity, 12B int8 vs dequantized-bf16 target | nll/tok 6.86799 vs 6.86737 (PPL 961.0 vs 960.4) — **0.009% delta**; 40 BOS-led 256-token windows, 10,200 scored tokens (Gemma windows must start with BOS: without it both score nll ~10.2) |
| Greedy outputs (all quantized paths) | text identical to the dequantized reference |
| xProf | 50-step trace of int8 12B b4 decode archived (110.7 ms/step under ~1.8× tracing overhead) |

Warmup incl. compile: 19.3 s (E2B bf16) → 57.1 s (E2B packed) → 77.4 s (31B
in-graph) / 71.5 s (31B fused); tier-3 GCS compile cache halves repeats
(84.3 → 36.6 s).

**Unmeasured cells:**
in-graph b4 E4B; MoE batch ≥ 8 (blocked on the v2 fused expert kernel).
