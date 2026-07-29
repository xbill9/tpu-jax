---
title: "One TPU Chip, Eight Agents: Is Raw JAX a Viable Serving Path for Small Agent Workloads?"
published: false
description: "Gemma 4 E2B QAT can't be loaded by vLLM on TPU, so I built the inference path in pure JAX and worked out what a single v6e chip can actually carry: the memory math for a small agent fleet, what raw JAX gives you, and the three things you'd have to build before it's production."
tags: tpu, llm, jax, agents
---

*Cloud TPU v6e-1 (`ct6e-standard-1t`, one v6e chip, 32 GB HBM), GCE flex-start, europe-west4-a. vLLM baseline measured 2026-07-21.*

## The workload nobody benchmarks

Serving benchmarks optimize for the wrong shape. They report throughput at concurrency 100 with 1,024-token prompts, because that's what a public inference endpoint looks like.

A small agent workload looks nothing like that:

- **Low concurrency, high value per stream.** Two to eight agents, not two hundred. Each one is a person waiting, or a pipeline stage blocking.
- **Latency compounds serially.** An agent turn is *think → call tool → read result → think again*. Ten round trips at 400 ms of model time each is four seconds of wall clock the user feels.

  Worth flagging that this premise is contested. Zimbres' measurement series argues the reverse for agent fleets: agents "exchange messages among themselves, multiplying token volume by an order of magnitude or more per user task, with no human waiting on any individual token. There, per-token latency loses most of its meaning and aggregate throughput becomes the binding constraint" ([DOI 10.5281/zenodo.21221952](https://doi.org/10.5281/zenodo.21221952)). Both are right, for different topologies — a serial agent loop with a human at the end is latency-bound; a parallel machine-to-machine fleet is throughput-bound. The metric that covers both is **goodput**: throughput subject to a per-token latency objective. That's the axis I report below.
- **Context grows monotonically.** Every tool result is appended and resent. A 20-turn agent loop replays a context that started at 500 tokens and ended at 8,000. This costs more than the extra prefill: on a v5e chip, *decode* velocity for Gemma 2B falls from ~8,300 to ~2,900 tok/s as input length approaches the context limit, "governed almost entirely by input length and almost not at all by output length" ([DOI 10.5281/zenodo.21227936](https://doi.org/10.5281/zenodo.21227936)). The generation rate itself sags as history accumulates.
- **Output is structured.** Tool calls are JSON that has to parse. A malformed call isn't a quality regression, it's a crash. This cuts in our favour: greedy decoding is nearly free for a specialized agent, because "fine-tuning is what makes deterministic decoding lossless" — a narrowly fine-tuned model has a sharply peaked distribution, so the gap between sampled and greedy shrinks toward zero as specialization increases ([DOI 10.5281/zenodo.21221952](https://doi.org/10.5281/zenodo.21221952)). A generalist chat model loses real variety at temperature zero; one emitting tool calls loses almost nothing — and gains reproducible, cacheable, testable behaviour.

That workload fits on one chip — if you can serve the model at all. Which brings us to the problem.

## The forcing function: #3225

Quantization-aware-trained exports of Gemma 4 E2B **do not load in vLLM on TPU**:

| Checkpoint | Backend | Failure | Verdict |
| :--- | :--- | :--- | :---: |
| `-qat-w4a16-ct` | tpu-inference (JAX) | `int4 compressed-tensors` scheme unimplemented for `per_layer_model_projection` | ✕ no load |
| `-qat-q4_0-unquantized` | tpu-inference (JAX) | demands `self_attn.k_norm.weight` for layers 15–34 | ✕ no load |
| `-qat-q4_0-unquantized` | torchax (`MODEL_IMPL_TYPE=vllm`) | identical missing-weights error | ✕ no load |
| `google/gemma-4-E2B-it` (BF16) | tpu-inference (JAX) | — | ✓ serves |

**The checkpoint is right and the loader is wrong.** Comparing safetensors headers: the BF16 export ships `self_attn.k_norm` for all 35 layers; the QAT export ships it only for the 15 non-KV-shared layers. Both configs declare `num_kv_shared_layers: 20`.

Layers 15–34 reuse K/V computed by lower layers. They hold no K-side parameters, so a k-norm for them is meaningless — the QAT export is the architecturally honest one, and the plain checkpoint only loads because it happens to carry the unused tensors along. Filed as [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225); the fix is to skip instantiating K/V-side parameters for shared layers rather than demanding them unconditionally.

So: serve BF16 and forgo QAT, or write the inference path yourself. I wrote it.

## The engine

`ports/gemma4/jax_e_model.py` is a pure-JAX Gemma 4 E2B — no PyTorch, no `torch_xla`, nowhere in the path. That's not an aspiration: the checkpoint goes safetensors → JAX PyTree directly (via `safetensors.flax`, which handles bfloat16 natively), and `tests/test_jax_engine.py` asserts `torch` never even enters `sys.modules` during a load. If you're used to "JAX" inference that quietly loads weights through `AutoModelForCausalLM` and converts, this isn't that.

Sharing is encoded in the layer definition instead of patched at load time:

```python
@property
def first_kv_shared_layer_idx(self) -> int:
    return self.num_hidden_layers - self.num_kv_shared_layers   # 35 - 20 = 15

def kv_share_map(self) -> List[int]:
    """Maps each layer index to the source layer index for KV state sharing."""
    first = self.first_kv_shared_layer_idx
    last_of_type = {}
    for i in range(first):
        last_of_type[self.layer_types[i]] = i
    return [
        i if i < first else last_of_type[self.layer_types[i]]
        for i in range(self.num_hidden_layers)
    ]
```

Sharing is per attention *type* — E2B interleaves sliding and full attention, and a shared layer inherits from the last non-shared layer of its own kind. Only `range(first_kv_shared_layer_idx)` ever allocates K/V. W4A16 weights (eight int4 values per int32, BF16 scale per 32-element group) are unpacked on chip, read straight from safetensors into a JAX PyTree.

Decoding runs against a static KV cache written with `dynamic_update_slice`, so each step attends to the full prefix. The way you know a decoder is correct is to compare it against the slow thing that obviously is — `tests/test_kv_cache_parity.py` runs greedy generation both ways, cached decode versus re-running the full model over the growing sequence every step:

```
step | max|dlogit| | ref tok  | cached tok | ref top-2 gap
   0 |    0.000000 | [62, 36] | [62, 36]   | [0.009738, 0.118984]
   ...
   7 |    0.000001 | [97, 95] | [97, 95]   | [0.012065, 0.000438]
```

Maximum divergence across every step: **1e-6**, float32 roundoff. (The test asserts logit agreement within tolerance and token agreement only where the top-2 gap exceeds it — near a 0.0004 tie, an exact argmax assertion is flaky by construction.) A companion test proves padding a prompt to a 128-aligned bucket is invisible to the output: RoPE follows logical position, not cache slot.

## Feasibility, part 1: the memory math

Here is the result that actually settles the "can one chip hold my agent fleet" question, and it needs no benchmark at all — just the architecture.

E2B's KV sharing means only 15 of 35 layers hold KV tensors, with a single 256-dim KV head:

```
15 layers × 1 head × 256 dim × (K+V) × 2 bytes = 15.0 KiB per token (bf16)
                                                  7.5 KiB per token (fp8)
```

That matches the measured allocator exactly (7,680 bytes/token under fp8 in the vLLM boot report). So an agent fleet costs:

| Fleet | Context each | KV total (bf16) | KV total (fp8) |
| :--- | ---: | ---: | ---: |
| 4 agents | 4,096 | 252 MB | 126 MB |
| 8 agents | 4,096 | 503 MB | 252 MB |
| 8 agents | 8,192 | 1.01 GB | 503 MB |
| 32 agents | 8,192 | 4.03 GB | 2.01 GB |

Against 31.24 GiB of usable HBM, with measured BF16 weights at 5.75 GiB (less under W4A16): **eight agents at 8K context spend 0.94 GiB on KV, for 6.69 GiB total — leaving 24.5 GiB idle.** A small agent fleet is not remotely memory-bound on this chip. Anyone sizing hardware by "2B params + long contexts, I'll need more memory" is solving a problem this architecture already solved.

The binding constraint is compute and scheduling. Which is where it gets interesting.

## Feasibility, part 2: what agent workloads need that raw JAX doesn't have

This is the honest core of the assessment. Three gaps, ranked by how much they'd hurt an agent workload specifically:

**1. No prefix caching — and agent loops are the pathological case.** Every agent turn resends the entire conversation plus every tool result so far. With prefix caching, turn 20 reprocesses only the new tokens. Without it, you re-prefill the whole growing context every single turn. Over a 20-turn loop with context growing 500 → 8,000 tokens, that's *quadratic* wasted prefill. vLLM ships this; raw JAX does not. **For agent workloads this is the single biggest gap**, and it isn't exotic to build — but you do have to build it. Measured from the other direction: prefix caching "is at its most effective in agent fleets, because agents share long system prompts and accumulated context, so the prompt-processing cost of inter-agent messages collapses" ([DOI 10.5281/zenodo.21221952](https://doi.org/10.5281/zenodo.21221952)).

**2. No guided decoding.** Tool calls have to be parseable JSON. vLLM constrains generation to a schema or grammar so malformed calls are structurally impossible. In raw JAX you sample freely and hope, or you write the constrained-sampling layer yourself. For an agent that dispatches on tool name, a 1% malformed rate is a 1% crash rate.

**3. Lockstep batching penalizes uneven turns.** A fixed batch runs together and a finished sequence holds its slot until the whole batch drains. Agent turns are wildly variable — one agent emits a 12-token tool call while another writes 400 tokens of reasoning. vLLM's continuous batching swaps in new work per step; here, short turns wait on long ones. At the low concurrency agent fleets run at, this is a real efficiency tax.

Plus the standing costs of the approach: static shapes mean any uncompiled `(batch, seq)` combination stalls on XLA compilation, so you bucket and spend FLOPs on padding; and every new architecture or quantization format is hand-written.

Against that, what raw JAX genuinely buys you: **it runs the checkpoint vLLM can't**, the whole stack is inspectable, and warm restarts are seconds rather than minutes — `jax_compilation_cache_dir` persists compiled HLO and skips ~17 s of XLA compilation, versus vLLM's measured ~8.5 minute time-to-healthy (of which 329 s is compilation). For an agent dev loop where you restart constantly, that difference is felt hourly.

## The vLLM bar

For calibration, vLLM serving the BF16 checkpoint on the same chip, measured with `vllm bench serve` (1,024-token input, 128-token output):

| Concurrency | Output tok/s | Per-stream tok/s | Median TTFT | Median TPOT |
| :---: | ---: | ---: | ---: | ---: |
| 1 | 209 | 213 | 16 ms | 4.7 ms |
| 8 | 1,209 | 161 | 27 ms | 6.2 ms |
| 32 | 1,636 | 57 | 155 ms | 17.5 ms |
| 64 | 2,140 | 39 | 122 ms | 25.3 ms |
| 100 | 2,215 | 27 | 833 ms | 36.8 ms |

Note the shape at agent-scale concurrency: at 8 streams it holds **161 tok/s per stream at 27 ms TTFT**. That is a good bar, and it's what you give up by leaving vLLM. The honest framing is not "raw JAX beats vLLM" — it's "vLLM can't load this checkpoint, and here's what the alternative costs you."

## Measured: the raw JAX engine on the same chip

> **Numbers pending re-measurement.** These were taken before the engine could
> actually generate — against synthetic weights and an architecture with five bugs
> (documented in the run report). The fixes are close to FLOP-neutral so they should
> hold roughly, but this page has already retracted one number twice on exactly that
> kind of reasoning. Treat as provisional until re-run.


Cache-correct decode, jitted prefill and steady-state decode timed separately, warmup discarded, median of 5, against the checkpoint's real architecture (`benchmarks/runs/2026-07-28-jax-e2b-v6e1`):

| agents | ctx | prefill (TTFT) | decode step | per-user | aggregate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 8.9 ms | 6.80 ms | 147.1 tok/s | 147.1 tok/s |
| 2 | 512 | 14.8 ms | 4.86 ms | 206.0 tok/s | 411.9 tok/s |
| 4 | 512 | 24.4 ms | 4.89 ms | 204.6 tok/s | 818.5 tok/s |
| 8 | 512 | 50.6 ms | 4.88 ms | 204.8 tok/s | 1,638.6 tok/s |
| 16 | 512 | 130.5 ms | 5.08 ms | 196.9 tok/s | 3,151.0 tok/s |
| 32 | 512 | 285.7 ms | 5.57 ms | 179.5 tok/s | **5,743.0 tok/s** |

**Eight agents each get ~205 tok/s. Thirty-two still get ~180.** Per-user cost is essentially flat from 2 to 32 concurrent streams — 4.9 to 5.6 ms per token — because at this batch range the step is dominated by reading weights, and those bytes amortize across the batch.

**The right question is not tokens/sec, it's how many streams fit inside a latency objective.** Borrowing the goodput framing from Zimbres' v6e batch-scaling study ([DOI 10.5281/zenodo.21462837](https://doi.org/10.5281/zenodo.21462837)), whose headline is that on a 31B model throughput saturates at 256 concurrent streams while time-per-token doubles with every further doubling — so goodput hits zero exactly where raw throughput looks best: at a 50 ms/token service objective — roughly human reading speed, and generous for an agent — *every* measured point clears it with about 9× headroom, and throughput was still climbing when the sweep ran out of memory at 32 agents.

Their recipe says to find the saturation knee by sweeping batch and watching the throughput multiplier per doubling decay through **1.8 → 1.6 → 1.2 → flat**. Ours reads 2.80 → 1.99 → 2.00 → 1.92 → **1.82** at 32 agents: still on the linear stretch, nowhere near flat. **We never reached the knee — HBM ran out first.**

That is the opposite of what happens with a 31B model on four chips, where goodput collapses at the throughput knee. **For a 2B model on one chip the binding constraint is memory, not latency.** And the memory going is not the KV cache — it's `eager_attention_jax` materializing the full `[batch, heads, query, key]` score matrix in fp32. The unlock for more concurrency is a flash-attention kernel that never materializes it, not anything to do with weights.

### A number I got wrong three times

An earlier version of this post reported a "~2.7× per-user speedup" going from one agent to two. Here is the full history of that one figure, because the way it failed is more useful than the figure:

| measurement | 1 → 2 agents | verdict |
| --- | ---: | --- |
| original sweep (no KV cache, un-jitted prefill) | 2.80× | withdrawn as an artifact |
| rebuilt harness, wrong model config | 2.79× | "retraction reversed" — also wrong |
| rebuilt harness, **real** model config | **1.40×** | current best measurement |

I withdrew it, then reinstated it when a properly rebuilt harness reproduced it to within 0.01×. That agreement was persuasive and meaningless: both runs used the same wrong architecture — `hidden_size` 2048 where the checkpoint says 1536, four KV heads where it has one — so they agreed with each other rather than with the model. **Reproducibility is not validity when both runs share an assumption.** Reading the checkpoint's own `config.json` was what settled it.

A 1.40× single-stream penalty is still there and still not explained by arithmetic intensity, which predicts one and two streams should cost about the same per step. It is a modest implementation effect, not the headline it was written up as.

## The server

`jax_openai_server.py` runs generation on the JAX engine: `/v1/chat/completions`, `/v1/completions`, `/health`, Prometheus `/metrics`, and SSE streaming that emits one chunk per decoded token.

```bash
python3 jax_openai_server.py --port 8000 --max-model-len 4096
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [{"role": "user", "content": "Explain TPU MXU vectorization."}],
    "stream": true
  }'
```

`tests/test_openai_server.py` drives the real endpoints, including an assertion that streaming and non-streaming produce **identical greedy text** — precisely the thing that silently rots when streaming is bolted on separately. `tests/test_jax_engine.py` additionally asserts the load path never imports `torch`.

## Verdict

**Is raw JAX viable for a small agent workload on one TPU chip? Yes, with one caveat that decides it for you.**

The memory case is emphatic: eight agents at 8K context is a rounding error against 32 GB, and E2B's KV sharing is why. The correctness case is settled — the decoder is verified against full re-forward, and the QAT checkpoint that blocks vLLM loads and runs. The dev-loop case is genuinely better: five-second warm restarts versus eight-and-a-half minutes.

The caveat is prefix caching. If your agents hold long conversations and resend growing context — and that is what agents *do* — you will re-prefill quadratically until you build it. Budget that work before choosing this path, or stay on vLLM with the BF16 checkpoint and wait for #3225.

Where this is clearly the right call: you need the QAT checkpoint specifically, you're running a handful of agents rather than a public endpoint, and you want a stack you can read end to end.

### Measure it on your own chip

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8 --contexts 512,2048,8192 --json-out results.json
```

One caveat if you run it: the sweep builds synthetic weights from a config whose KV head count doesn't match the shipped checkpoint (4 heads where E2B has a single 256-dim one), so it allocates 72 KiB/token instead of 15 and its OOM frontier is pessimistic by roughly 4.8×. Serving is unaffected — `JaxGemmaEngine` reads the real `config.json` — but check that number against your checkpoint before trusting the memory column.

### Two optimizations that didn't work

Worth recording, because the reasoning was plausible and the result wasn't. Decode reads the full weight set per token, so 4-bit weights ought to cut traffic ~4× — a fused Pallas kernel that unpacks int4 inside the VMEM tile instead of dequantizing to BF16 in HBM first. Predicted ~3.4× on the memory floor. Measured: **0.59× at B=1, 0.21× at B=2**, and `CompileTimeScopedVmemOom` on five of eight cells because the kernel loads all of `x` as one VMEM block rather than tiling the sequence axis. Quantizing the LM head to int8 — the single largest read in a step — bought **1.00×/1.04×/1.05×** at B=1/2/8, for 0.8% logit error.

Neither failure was novel. The same series reports speculative decoding costing a **factor of six**, a precision change that "looked certain on paper" costing 14% and being reverted, and the general rule that this stack is best modified *above* the compiler (configuration) or *below* it (whole kernels) "rather than through point edits to the compiled middle" — which is exactly what a fused dequant-matmul is ([DOI 10.5281/zenodo.21221952](https://doi.org/10.5281/zenodo.21221952)). It also brackets the ceiling I was chasing: the entire logits pipeline is 18% of a decode step, so an int8 LM head could never have paid much.

Both of mine failed for a related reason: **weight traffic isn't the constraint here.** The pure-bandwidth floor for this model is ~3.6 ms/token at B=1, and the measured step is 20.6 ms — about 6× above it. Cutting bytes read can't speed up a step that isn't waiting on bytes. That 6× gap, and the single-stream penalty, are the same mystery, and they're where the next real work is.

The uncomfortable part is that this is a known failure pattern, not a novel one. The same series ran the fashionable strategies and measured them losing: a team "following the mainstream sequence without profiling would have deployed speculative decoding, measured here at six times slower in this regime; pursued quantization, measured at zero functioning routes on this stack, one failing silently; or hand-optimized a genuine numeric inefficiency, measured at 14 percent slower when removed at the wrong layer." I did the second one, and mine failed silently too.

There's a landmine in that kernel worth flagging if you write one: it originally unpacked nibbles plane-major while contracting activations in natural order — silently wrong — and nobody noticed because a bare `except Exception` fell back to the reference path on any host without a TPU. It only computes garbage where Pallas actually compiles. Verify kernels against a reference on the hardware you ship on, and never let a fallback swallow the exception that would have told you.

Watch per-stream decode latency at batch 2–8 with realistic context — that is the number an agent workload lives or dies on, not peak aggregate throughput.

### Repository

- **Engine:** [`ports/gemma4/jax_e_model.py`](https://github.com/xbill9/tpu-jax/blob/main/ports/gemma4/jax_e_model.py) · **loader:** [`ports/gemma4/jax_e_loader.py`](https://github.com/xbill9/tpu-jax/blob/main/ports/gemma4/jax_e_loader.py)
- **Serving engine:** [`jax_engine.py`](https://github.com/xbill9/tpu-jax/blob/main/jax_engine.py) · **server:** [`jax_openai_server.py`](https://github.com/xbill9/tpu-jax/blob/main/jax_openai_server.py)
- **Tests:** [`tests/test_kv_cache_parity.py`](https://github.com/xbill9/tpu-jax/blob/main/tests/test_kv_cache_parity.py) (cached decode vs full re-forward) · [`tests/test_jax_engine.py`](https://github.com/xbill9/tpu-jax/blob/main/tests/test_jax_engine.py) (load path, torch-free assertion) · [`tests/test_openai_server.py`](https://github.com/xbill9/tpu-jax/blob/main/tests/test_openai_server.py) (endpoints, SSE)
- **Benchmark harness:** [`ports/gemma4/jax_e_benchmark_sweep_v2.py`](https://github.com/xbill9/tpu-jax/blob/main/ports/gemma4/jax_e_benchmark_sweep_v2.py)
- **vLLM baseline data:** [`benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json`](https://github.com/xbill9/tpu-jax/blob/main/benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json)
- **Upstream issue:** [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)

### Prior art

Rubens de Almeida Zimbres' TPU inference measurement series (CC-BY-4.0) is the closest published
work to this, and several of its results are cited above. Notes on how each bears on the engine
here are in [`docs/references/tpu-inference-measurement-series.md`](https://github.com/xbill9/tpu-jax/blob/main/docs/references/tpu-inference-measurement-series.md).

- [10.5281/zenodo.21221952](https://doi.org/10.5281/zenodo.21221952) — *From 1,540 to 19,511 Tokens per Second on a Single TPU v5e Chip* (12.7x; arithmetic-intensity framework; what failed and why)
- [10.5281/zenodo.21227936](https://doi.org/10.5281/zenodo.21227936) — *Token Velocity on a Single TPU v5e Chip* (decode rate vs. input length)
- [10.5281/zenodo.21404155](https://doi.org/10.5281/zenodo.21404155) — *The Decode Block Size Heuristic in TPU Ragged Paged Attention* (28–69% left on the table by one constant)
- [10.5281/zenodo.21462837](https://doi.org/10.5281/zenodo.21462837) — *Batch Scaling and Goodput of a Tuned Attention Kernel on TPU v6e* (the goodput cliff)
