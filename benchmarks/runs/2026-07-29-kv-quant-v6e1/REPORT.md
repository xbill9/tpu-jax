# Quantized KV cache and chunked prefill — Gemma 4 E2B QAT, TPU v6e-1

**Date:** 2026-07-29 · **Hardware:** single `ct6e-standard-1t` (v6e-1, 32 GB HBM),
`europe-west4-a`, flex-start · **Model:** `google/gemma-4-E2B-it-qat-w4a16-ct`

## Headline

The v6e-1 has **two independent memory budgets**, and almost every optimization
this repo tried was aimed at the wrong one.

| budget | governs | measured |
|---|---|---|
| resident KV tokens | decode / concurrency | **524,288** bf16 · **1,048,576** int8 |
| prompt tokens per prefill pass | batch admission | **~11,900** (~2.13 MB/token) |

Both are flat constants. Neither is a batch size.

## 1. The decode budget is a constant, to 0.0%

Max batch at each context, then the product:

| ctx | bf16 max B | KV tokens | int8 max B | KV tokens | e4m3 KV tokens |
|---:|---:|---:|---:|---:|---:|
| 512 | 1024 | 524,288 | 2048 | 1,048,576 | 1,048,576 |
| 2,048 | 256 | 524,288 | 512 | 1,048,576 | 1,048,576 |
| 8,192 | 64 | 524,288 | 128 | 1,048,576 | 1,048,576 |
| 32,768 | 16 | 524,288 | 32 | 1,048,576 | 1,048,576 |
| | | **spread 0.0%** | | **spread 0.0%** | **spread 0.0%** |

Context varies 64x and batch varies 64x; the product does not move. The chip is a
fixed KV-token budget you may slice any way you like — 1024 chat sessions or 16
novel-length contexts, same silicon.

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
