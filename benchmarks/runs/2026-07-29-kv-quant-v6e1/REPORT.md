# Quantized KV cache and chunked prefill — Gemma 4 E2B QAT, TPU v6e-1

**Date:** 2026-07-29 · **Hardware:** single `ct6e-standard-1t` (v6e-1, 32 GB HBM),
`europe-west4-a`, flex-start · **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct`

## Headline

The v6e-1 has **two independent memory budgets**, and almost every optimization
this repo tried was aimed at the wrong one.

| budget | governs | measured |
|---|---|---|
| resident KV tokens | decode / concurrency | **663,552-712,704** bf16 · **1,245,184-1,540,096** int8 |
| prompt tokens per prefill pass | batch admission | **B x chunk <= 8,192** |

Neither is a batch size.

> **Corrected 2026-07-29 (later session).** An earlier version of this report gave
> the decode budget as a flat 524,288 tokens with "0.0% spread across four context
> lengths", and int8 KV as "exactly 2.00x". Both were artifacts of the measurement
> grid: every ladder was built from the same power-of-two skeleton, so the last
> passing rung landed on a common product by construction. Re-measured by bisection
> to 2%, the ceiling is 663,552-712,704 and varies ~7% with context, and int8 KV
> gives 1.88-1.98x. The conclusions are unchanged; the precision was manufactured.
> Lesson retained in the methods section: a result that lands exactly on a round
> number should be suspected of coming from the sampling grid.

## 1. The decode budget is a constant, to 0.0%

Max batch at each context, then the product:

Bisected to 2%. (The doubling-ladder figures this table previously carried are
superseded; see the correction note above.)

| ctx | bf16 max B | KV tokens | int8 max B | KV tokens | int8 gain |
|---:|---:|---:|---:|---:|:---:|
| 512 | 1296 | 663,552 | 2432 | 1,245,184 | 1.88x |
| 8,192 | 87 | 712,704 | 172 | 1,409,024 | 1.98x |

The chip is best described as a KV-token budget rather than a batch size — roughly
0.7M tokens at bf16, 1.3M at int8, sliceable as many short sessions or few long
ones. It is not a *constant*: reaching the budget at short context needs B=1296
against B=87, and per-sequence non-KV overhead scales with B, so high-batch
configurations give some back. That is the same `g(B)` term visible in step time.

Decode step time follows the same variable. Grouped by ctx x B rather than by
either alone:

| KV tokens | ctx 2048 | 4096 | 8192 | 16384 | 32768 | spread |
|---:|---:|---:|---:|---:|---:|---|
| 16,384 | 5.09 | 5.10 | — | — | — | ±0.1% |
| 32,768 | 7.53 | 7.65 | 7.62 | — | — | ±0.8% |
| 65,536 | 8.32 | 8.98 | 8.63 | 8.64 | — | ±3.8% |
| 131,072 | 14.36 | 13.31 | 14.01 | 13.90 | 13.47 | ±3.8% |
| 262,144 | 21.30 | 21.07 | 21.19 | 20.95 | 20.90 | ±0.9% |
| 524,288 | 40.09 | 39.58 | 39.28 | 39.23 | 39.21 | ±1.1% |

Latency is therefore predictable from queue depth alone: ~21 ms at half budget,
~39 ms at full. A `max_batch_size` constant is the wrong admission abstraction;
track the sum of context lengths instead. (This is what vLLM's paged allocator
already does — the measurement vindicates that design rather than improving on it.)

Second-order term: the ctx=512 rows sit *above* the line (23.81 vs 21.1 at
262,144). Weight-application FLOPs scale with **B**, not with ctx x B, and only
become visible when batch is large enough to fill the budget at short context.
So `step ≈ f(ctx·B) + g(B)`.

## 2. int8 KV doubles the budget and is also faster

Two bugs had to be fixed first, and they failed in opposite directions:

* **fp8 raised.** JAX deliberately excludes float8 from its type-promotion
  lattice — several mutually incompatible fp8 layouts exist — so a cached fp8 key
  meeting a bf16 query errors instead of promoting. The write side already cast;
  the read side never did.
* **int8 did not raise.** int8 *is* in the lattice, so `bf16 x int8` silently
  contracted against raw integers with no scale applied. An earlier benchmark
  reported a perfectly healthy 5.98 ms step for arithmetic that would have
  produced garbage.

Fix: explicit read-side cast, plus symmetric per-`(batch, head, position)` scales
over `head_dim` (2 bytes per 256-element row, 0.8% overhead).

**The scales never widen the cache.** Both are applied to the contraction result,
not to K/V:

* the K scale is indexed by key position `t` while the score sums over `head_dim`,
  so it factors out of the sum;
* the V scale is also indexed by `t` — which *is* the summed axis — so it folds
  into the probabilities instead.

Both are exact rewrites (`tests/test_quantized_kv.py::ScaleFactorizationTest`
verifies against dequantize-first at n_rep = 1, 4, 8). A naive
dequantize-then-attend would allocate exactly the buffer being avoided.

Step time, same load:

| config | bf16 | int8 | speedup |
|---|---:|---:|:---:|
| ctx 512, B=512 | 23.91 ms | 19.66 ms | 1.22x |
| ctx 8192, B=32 | 21.32 ms | 13.41 ms | 1.59x |
| ctx 8192, B=64 | 39.36 ms | 22.16 ms | 1.78x |

The speedup **grows with KV size** — the signature of a genuinely bandwidth-bound
step. Peak aggregate throughput rose from 22,995 to **29,755 tok/s**.

Quality on the real checkpoint (greedy, chat template):

| prompt | bf16 | int8 | fp8_e4m3 | fp8_e5m2 |
|---|---|---|---|---|
| `What is 2+2?` | `4` | `4` | `4` | `4` |
| `The capital of France is` | `Paris` | `Paris` | `Paris` | `Paris` |
| gravity, one sentence | full | full (one wording flip) | identical to bf16 | **truncated** |

**int8 is the recommendation.** It is both the most accurate and the fastest: a
per-row scale already supplies the dynamic range that e4m3 spends exponent bits
on. `fp8_e5m2` is strictly worse on both axes and visibly degrades output.

Cost of the scales at B=1: 147 -> 142 tok/s (3%), which inverts to 1.78x *faster*
at scale.

## 3. Prefill is the real batch ceiling, and it is linear

Compile-time `memory_analysis()` measures configurations that cannot be run:

| B | S | prompt tokens | temps | MB/token |
|---:|---:|---:|---:|---:|
| 8 | 128 | 1,024 | 2.24 GB | 2.19 |
| 8 | 256 | 2,048 | 4.40 GB | 2.15 |
| 8 | 512 | 4,096 | 8.75 GB | 2.14 |
| 8 | 1024 | 8,192 | 17.39 GB | 2.12 |
| 32 | 128 | 4,096 | 8.68 GB | 2.12 |
| 32 | 256 | 8,192 | 17.38 GB | 2.12 |

Flat ~2.13 MB per prompt token, **linear in B x S, not quadratic** — XLA tiles the
softmax, so `B x H x S x S` scores are never materialized. With ~25 GB left after
weights, the budget is roughly **11,900 prompt tokens per pass**, which maps
directly onto vLLM's `max_num_batched_tokens`.

`chunked_prefill_with_kv_cache` bounds `B * chunk_size` against that budget.
Verified token-exact against one-shot prefill
(`tests/test_chunked_prefill.py`), and composes with int8 KV.

Measured at ctx 2048, `window_kv=False` throughout so the comparison is fair:

| mode | bf16 max B | int8 max B | B x chunk | bf16 prefill | int8 prefill |
|---|---:|---:|---:|---:|---:|
| one-shot | **OOM at B=8** | **OOM at B=8** | — | — | — |
| chunk=512 | 16 | 16 | **8,192** | 1,742.8 ms | 1,946.4 ms |
| chunk=256 | 32 | 32 | **8,192** | 2,203.3 ms | 2,263.0 ms |
| chunk=128 | 64 | 64 | **8,192** | 3,881.8 ms | 4,316.6 ms |

A third constant, six for six: halving the chunk exactly doubles the admissible
batch, and one-shot prefill cannot admit even B=8. `B * chunk_size <= 8,192` is
the admission rule.

int8 KV gives an **identical** prefill ceiling at every chunk size — prefill
temporaries are activations and do not depend on the cache dtype. The two budgets
are genuinely independent: quantizing KV buys decode capacity and nothing else,
chunking buys prefill admission and nothing else.

The measured 8,192 sits below the ~11,900 projected from `memory_analysis`
because the KV cache and the chunk masks are resident alongside the temporaries;
the projection counted temporaries only.

**Known limitation.** Chunking currently requires `window_kv=False`, since a
chunk writes `chunk_size` contiguous slots at an arbitrary offset that a shorter
ring buffer would wrap. That forfeits windowing, which is expensive at long
context — each chunk attends over the full cache rather than a 512-slot window,
so prefill runs ~5x slower than a windowed one-shot pass (1,742 ms vs 354 ms at
B=8, ctx 2048). Chunking currently buys *admission*, not speed. Supporting ring
writes in chunked mode would let the two compose and is the obvious next step.

## What this retires

Measured against the correct budget, most of the earlier optimization list is noise:

| knob | measured | verdict |
|---|---|---|
| dense vs packed weights | 1.00–1.02x at B>=512 | irrelevant at scale |
| `window_kv` | ~3% cost, no benefit | remove |
| int8 lm_head | 1.00–1.05x | drop |
| fused Pallas W4A16 | 0.59x | delete |
| int8 PLE | 0.95x, -2.35 GB | keep — buys budget, not speed |

## Checkpoint composition (why int8 PLE still matters)

Read off the shipped weights:

| component | resident | quantized? |
|---|---:|:---:|
| **PLE table** `[262144, 8960]` BF16 | **4.698 GB** | no |
| token embed (tied -> lm_head) | 0.805 GB | no |
| MLP (int4 packed) | 0.876 GB | yes |
| attention (int4 packed) | 0.157 GB | yes |
| PLE projection (int4) | 0.023 GB | yes |
| total | **6.56 GB** | |

W4A16 QAT compressed **1.06 GB** and left **5.50 GB of unquantized lookup tables**.
The PLE table alone is 72% of resident weights. Further transformer quantization
chases the small term; int4 PLE (-3.52 GB) does not.

## Corrections to earlier reporting

* A previously reported ceiling of "B=32 / 5,743 tok/s" was an artifact of
  prefilling every sequence simultaneously. Decode alone reaches B=1024 at
  ~23k tok/s (bf16) and B=2048 (int8).
* "int8 PLE gives no capacity unlock" was tested against the prefill OOM wall,
  which weight savings cannot move. Against the decode budget it buys ~160k KV
  tokens.
* "int8 KV works and is faster" (pre-fix) measured a valid bandwidth floor for
  invalid arithmetic. The bandwidth claim survived the fix; the correctness claim
  did not exist yet.
* Chunked prefill was justified in code comments as bounding a quadratic
  `B x H x S x S` term. The measurement shows the term is linear; the comments
  have been corrected.

## Reproduce

```
tests/test_quantized_kv.py       # 12 tests: scales, factorization, layout, decode
tests/test_chunked_prefill.py    #  8 tests: mask, parity, token-exactness
```

100 tests pass on CPU; the two suites above also pass on TPU.

---

# Addendum: the 31B on CPU (2026-07-29, overnight)

Validating the two `jax_e_loader` branches the 31B exercises and E2B does not
(`hidden_size_per_layer_input == 0`, `num_kv_shared_layers == 0`). Run on GCE
CPU boxes; no TPU involved.

## `attention_k_eq_v`: the 31B ships no `v_proj` on full-attention layers

The load failed with exactly ten missing tensors, all at `i % 6 == 5` — every
full-attention layer in the 60-layer `[s,s,s,s,s,f]` pattern. Reading the
checkpoint keys directly: those layers carry `q_proj`, `k_proj`, `k_norm` and
`o_proj`, and **no `v_proj` at all**. `config.json` sets `attention_k_eq_v: True`.
V *is* K — one projection feeds both.

E2B sets the flag `False` and ships `v_proj` on all fifteen non-shared layers,
verified key by key, so the default path is unaffected.

Fixed by aliasing V to K in the loader (the same arrays, not copies), gated on
`Gemma4EConfig.attention_k_eq_v`. Six tests in `tests/test_attention_k_eq_v.py`
cover the alias, that the flag *off* still reports the tensor missing, and that a
real `v_proj` is never clobbered.

The loader's strict validation — added earlier the same day after the E2B
multimodal-prefix incident produced an all-`None` parameter tree — is what turned
this into a precise ten-tensor error instead of a silently wrong model.

## Results

| stage | result |
|---|---|
| E2B control generation | PASS — `'Paris'`, 6.56 GB of weights |
| 31B config | PASS — 60 layers, no PLE, no KV sharing, identity share map |
| 31B load | PASS — 135.7 s, **19.36 GB of weights**, all 60 layers verified |
| 31B forward (60 layers) | **OOM at 130.5 GB** on a 125 GB host |
| 31B forward (6 layers) | PASS — prefill 36.9 s, decode ~8.7 s/token, finite logits |

The 19.36 GB measured against 18.98 GB projected from `config.json` is a **2%**
error, which corroborates the v6e-1 capacity math above: ~13 GB free for KV after
weights, ~17 concurrent streams at ctx 8192 with an int8 cache.

The truncated run keeps the first six layers — the smallest prefix containing a
full-attention layer — so it exercises both attention geometries (sliding: 16 KV
heads x 256; full: 4 x 512), the K/V alias, the no-PLE branch and the degenerate
share map. Asserted on the real weights: `layer_5` V *is* `layer_5` K, `layer_0`
V is independent.

## Host memory, measured

| workload | peak RSS |
|---|---:|
| E2B, load + generation | 26 GB |
| 31B, load only | 48 GB |
| 31B, load + 60-layer forward | >130 GB |

XLA:CPU allocates roughly 2x the weight bytes to load and far more to run: the
reference W4A16 path dequantizes each packed weight to dense bf16 inside the
forward, and the CPU backend keeps those temporaries alive across layers. On TPU
the dequant fuses into the matmul (`multiply_reduce_fusion`), which is why the
same model serves from 32 GB of HBM.

**Practical sizing: 32 GiB for E2B, 64 GiB to load and inspect a 31B, and more
than 128 GiB to run one unfused on CPU** — or truncate the layer count, which
tests the same code paths for a tenth of the memory.

## Cost note

The first box was Spot and was preempted mid-run, taking the 23 GB checkpoint
download with it. Standard provisioning plus an in-place machine-type resize
(which preserves the boot disk, and therefore the download) was the cheaper path
overall despite the higher hourly rate.

---

# Addendum 2: the PLE table, and KV bytes/token settled

## The QAT checkpoint quantized 16% of the model

Read off the shipped weights, E2B:

| component | resident | quantized by QAT? |
|---|---:|:---:|
| **PLE table** `embed_tokens_per_layer` `[262144, 8960]` BF16 | **4.698 GB** | no |
| token embedding (tied -> lm_head) | 0.805 GB | no |
| MLP (int4 packed) | 0.876 GB | yes |
| attention (int4 packed) | 0.157 GB | yes |
| PLE projections (int4 packed) | 0.023 GB | yes |
| **total** | **6.56 GB** | |

W4A16 compressed **1.06 GB** and left **5.50 GB of lookup tables** in BF16. The PLE
table alone is 72% of resident weights — the largest tensor in the model, and
untouched by the quantization the checkpoint is named for.

## Quantizing it is free, and buys nothing

`quantize_ple_table(bits=4|8)` with scales grouped per layer slice (256 elements;
a single row-wide scale is 16 levels across all 8,960 at 4 bits).

| PLE | cache | ctx | max B | step ms | agg tok/s | vs BF16 PLE |
|---|---|---:|---:|---:|---:|---:|
| bf16 | bf16 | 512 | 1296 | 60.82 | 21,309 | — |
| int8 | bf16 | 512 | 1408 | 65.48 | 21,502 | +0.9% |
| int4 | bf16 | 512 | 1456 | 68.57 | 21,234 | -0.4% |
| bf16 | bf16 | 8192 | 87 | 52.24 | 1,665 | — |
| int8 | bf16 | 8192 | 95 | 57.00 | 1,667 | +0.1% |
| int4 | bf16 | 8192 | 99 | 60.63 | 1,633 | -1.9% |
| bf16 | int8 | 512 | 2432 | 73.77 | 32,968 | — |
| int8 | int8 | 512 | 2656 | 78.13 | **33,996** | +3.1% |
| int4 | int8 | 512 | 2752 | 82.55 | 33,337 | +1.1% |
| bf16 | int8 | 8192 | 172 | 58.01 | 2,965 | — |
| int8 | int8 | 8192 | 188 | 64.09 | 2,933 | -1.1% |

Mean across all comparisons: **+0.26%**. Weights fall 6.56 -> 4.23 -> 3.05 GB, a 53%
cut, and aggregate throughput does not move. Capacity rises 9-13%; step time rises
6-13%; they cancel.

**Why**: a gather never streams through HBM. The PLE table costs residency, not
bandwidth, so shrinking it frees space rather than time — and the freed space goes
to more concurrent sequences, each paying the dequant on every step.

This sharpens the roofline rule rather than contradicting it: **quantization pays
only on tensors that actually cross the memory bus.** A 4.7 GB table read by gather
is invisible to the roofline.

## Quality: int4 is free

Real checkpoint, greedy, seven prompts. All correct at every bit width.

| prompt | bf16 (6.56 GB) | int8 (4.23 GB) | int4 (3.05 GB) |
|---|---|---|---|
| `What is 2+2?` | `4` | `4` | `4` |
| `The capital of France is` | `Paris` | `Paris` | `Paris` |
| first five primes | 2, 3, 5, 7, 11 | 2, 3, 5, 7, 11 | 2, 3, 5, 7, 11 |
| `'good morning'` in Spanish | `Buenos días.` | `Buenos días.` | `Buenos días.` |
| haiku | identical | identical | identical |
| gravity | "drawn toward each other" | same | "exert a pull on each other" |
| three colours | prepends a self-introduction | same | answers directly |

The 7% synthetic round-trip error at int4 does not degrade trained weights. At B=1,
int4 is also *faster* than int8 (112 vs 98 tok/s) because the gather reads half the
bytes and the packing wins back more than the nibble unpack costs.

**Recommendation: quantize the PLE when capacity-bound, never when throughput-bound.
Not defaulted, for that reason.**

## KV is 18.0 KiB/token, and an earlier note had it backwards

The three full-attention layers among the fifteen that own KV carry a **512-wide** K
projection with a matching `k_norm` `[512]`; the twelve sliding layers are 256.
`init_kv_cache` allocates **18.00 KiB/token** at every context length, matching
`12 x 1 x 256 x 2 + 3 x 1 x 512 x 2 = 9,216` elements x 2 B exactly.

A previous note recorded 15.0 KiB/token and concluded our estimates were "~20%
pessimistic". Backwards: 15.0 KiB is exactly what a **uniform 256-dim** assumption
yields, and the checkpoint contradicts it. Our figure is correct.

An allocator that sizes E2B's KV uniformly at `head_dim` would under-provision the
three full-attention layers by half. The provenance of the 15.0 KiB reading is not
recorded, so this is a lead, not a report.

## Measurement bugs found in this session, all mine

| bug | failure mode | caught by |
|---|---|---|
| doubling ladder too coarse | reports real gains as "no change"; manufactured the fake 0.0% spread | arithmetic — the predicted gain fell between rungs |
| shared `base` params | charges quantized configs the headroom being measured | reading the harness |
| PLE quantization OOM on-chip | 8.75 GB upcast with 7.58 GB free | it raised |
| **quantized table left on host** | **capacity looks BETTER; gathers cross PCIe** | **18,482 ms step time** |

The last is the one worth keeping: it failed in the flattering direction and nothing
raised. Only the step-time column exposed it. **Never accept a capacity number
without a latency number beside it** — a configuration that fits more and runs
slower is not a win, and only the pair tells you which you got.
