# Gemma 4 E2B: architecture quirks, verified against the reference

Every entry below is checked against `transformers.models.gemma4.modeling_gemma4`
(transformers 5.12+). That module is readable locally — **no torch install, no TPU,
no checkpoint download needed** — and reading it is faster and more reliable than
inferring architecture from tensor shapes. Five of the bugs below were found the
slow way first; the rest were found by diffing.

Status legend: **✅ verified** against the reference · **⚠️ inferred** from shapes or
measurement only.

---

## 1. The checkpoint is multimodal ✅

The text decoder lives under `model.language_model.`, alongside `model.audio_tower.`
(751 tensors) and vision towers (659). A loader that assumes a bare `model.` prefix
finds **nothing** — and `get_arr()` returning `None` for every key produces a
parameter tree that "loads" in seconds and holds zero bytes.

`jax_e_loader.detect_text_prefix()` tries `model.language_model.`, `language_model.`,
`model.` and raises listing the candidates if none matches.

## 2. Sandwich norms ✅

`post_attention_layernorm` normalizes the attention **output**, before the residual
add — it is *not* a pre-norm for the MLP. The feed-forward block is wrapped on both
sides:

```
residual = h
h = attn(input_layernorm(h))
h = residual + post_attention_layernorm(h)

residual = h
h = mlp(pre_feedforward_layernorm(h))
h = residual + post_feedforward_layernorm(h)
```

Getting this wrong still runs and still emits fluent-looking tokens. They are just
the wrong tokens.

## 3. `layer_scalar` scales the whole residual stream ✅

Each decoder layer ends with `hidden_states *= self.layer_scalar` — a single learned
scalar per layer, applied to the **entire** hidden state after every residual add
(including the PLE injection), not to the layer's delta.

It is the counterweight to this checkpoint's large RMSNorm weights (`final_norm`
mean ≈ 14, max 118). Without it the residual stream grows layer over layer and the
output logits pin against the ±30 softcap. Observed values on E2B run 0.02–0.79.

## 4. RMSNorm is `x_normed * weight` ✅

**Not** the `x_normed * (1 + weight)` convention used by earlier Gemma generations.
The weights here are not zero-centred (means of +5 to +19 are normal), which is the
quick way to tell the two conventions apart.

`v_norm` is constructed `with_scale=False` — it normalizes with no weight at all, and
the checkpoint ships no `v_norm` tensor. Passing `weight=None` to an RMSNorm that
multiplies only when a weight is present reproduces this.

## 5. Attention: no score softcap, and `scaling = 1.0` ✅

- The config declares `final_logit_softcapping: 30.0` and **no**
  `attn_logit_softcapping`. Gemma 3+ dropped softcapping of attention *scores*;
  applying it saturates `tanh` and destroys the attention distribution.
- The reference sets `self.scaling = 1.0` — **not** `head_dim ** -0.5`. `q_norm` and
  `k_norm` already normalize query and key before the dot product, so the usual
  `1/sqrt(d)` is not applied on top. (I "fixed" this once by adding it. That was the
  regression, not the fix.)
- `q_norm`/`k_norm` are applied **before** RoPE.

## 6. RoPE: concatenated frequency layout, partial rotary by masking ✅

`rotate_half` pairs channel *i* with *i + d/2*, so the frequency table must be
`cat(freqs, freqs)`. Building it with `repeat_interleave` pairs every channel against
the wrong frequency — the model still generates, but it echoes and repeats.

Per-layer-type RoPE comes from a nested `rope_parameters` dict:

| layer type | `rope_theta` | `rope_type` | `partial_rotary_factor` |
| --- | ---: | --- | ---: |
| `sliding_attention` | 10,000 | `default` | 1.0 (absent) |
| `full_attention` | 1,000,000 | `proportional` | 0.25 |

`proportional` keeps the **full** `head_dim` and zeroes the inverse frequencies past
the factor, rather than slicing channels: with `global_head_dim` 512 and factor 0.25,
the first 64 frequency pairs are rotated and the remaining 192 get `inv_freq = 0`
(i.e. `cos = 1`, `sin = 0`, identity). Masking the frequency table is equivalent and
keeps the `cat(freqs, freqs)` pairing intact.

For text-only inputs `apply_multidimensional_rope` falls through to standard
`apply_rotary_pos_emb` — the "multidimensional" path is for image/audio position dims.

## 7. KV sharing is keyed by layer *type* ✅

`first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers` (35 − 20 = 15
on E2B). Layers at or above it carry **no** `k_proj`, `v_proj` or `k_norm` — the
omission upstream's loader mishandles ([tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)).

The reference stores `shared_kv_states[layer_type] = (k, v)` and every non-shared
layer overwrites its type's entry, so a shared layer reads the **last non-shared
layer of its own type**. Since E2B interleaves sliding and full attention, there are
two independent sources. `Gemma4EConfig.kv_share_map()` computes exactly this.

## 8. Per-Layer Embeddings ✅

Two components, combined:

```
token identity  = embed_tokens_per_layer[ids] * sqrt(D_ple)     -> [B,S,L,D_ple]
context         = per_layer_model_projection(inputs_embeds) * hidden_size**-0.5
                  reshaped to [B,S,L,D_ple], then per_layer_projection_norm
per_layer_input = (context + token_identity) * 2**-0.5
```

Verified constants: `embed_scale = hidden_size**0.5`, PLE `embed_scale =
hidden_size_per_layer_input**0.5`, `per_layer_model_projection_scale =
hidden_size**-0.5`, `per_layer_input_scale = 2**-0.5`.

The trap: `per_layer_projection_norm` has shape **`[D_ple]`**, so the norm applies to
the reshaped `[B,S,L,D_ple]` tensor per layer-slice — *not* across the flattened
`[L*D_ple]` projection.

In the QAT checkpoint all three PLE projections (`per_layer_model_projection`,
`per_layer_input_gate`, `per_layer_projection`) ship **W4A16-packed**, not dense. The
global one is the exact tensor #3225 reports as unimplemented in vLLM's TPU loader.

## 9. Double-wide MLP on shared layers ✅

`use_double_wide_mlp=True` means KV-shared layers use `2 * intermediate_size`
(12288 = 2 × 6144 on E2B). Confirmed in the checkpoint: layer 20's `gate_proj` is
`[12288, 192]` packed.

## 10. The tokenizer does not add BOS ⚠️

`tok("hello")` → `[23391]`, no `<bos>`. Gemma expects one, and without it the model
**echoes the prompt** instead of answering. `JaxGemmaEngine` now prepends it.
(Measured behaviour, not read from the reference.)

## 11. Config values that differ from intuition ✅

| field | value | note |
| --- | ---: | --- |
| `hidden_size` | 1536 | not 2048 |
| `num_key_value_heads` | 1 | a single 256-dim KV head |
| `num_global_key_value_heads` | `null` | falls back to 1 |
| `sliding_window` | 512 | 32 of 35 layers |
| `tie_word_embeddings` | true | `lm_head.weight` is a materialized copy |
| `layer_types` | explicit list | full attention at 4, 9, 14 … (`i % 5 == 4`) |

## 12. `attention_k_eq_v`: V *is* K, and the checkpoint omits `v_proj` ✅

Set `True` on **gemma-4-31B**, `False` on E2B. Where it is set, the affected layers
ship no `v_proj` at all — one projection feeds both K and V.

Verified by reading the 31B checkpoint on a CPU box: all ten **full-attention**
layers (`i % 6 == 5` in its 60-layer `[s,s,s,s,s,f]` pattern) carry `q_proj`,
`k_proj`, `k_norm`, `o_proj` and **no `v_proj`**, while every sliding layer carries
both. Shapes at layer 5 confirm the geometry — `k_proj` is `[2048, 672]` packed,
i.e. `num_global_key_value_heads` (4) × `global_head_dim` (512), and `k_norm` is
`[512]`.

Loading the 31B without handling this yields exactly ten missing tensors and, in a
loader that tolerates `None`, a silently broken model. `jax_e_loader` aliases V to
K (the same arrays, not copies) when `Gemma4EConfig.attention_k_eq_v` is set.

The KV cache still stores K and V separately, which is redundant but correct.
Collapsing it would save one of the two planes on those layers — worth ~4.5% of the
31B's KV, since sliding layers dominate its budget.

**This does not explain E2B's KV-bytes discrepancy below.** E2B sets the flag
`False` and ships `v_proj` on all fifteen non-shared layers, checked key by key.

## 13. `use_bidirectional_attention` is a vision setting ✅

`"vision"` on the 31B, absent/`null` on E2B. It selects bidirectional attention for
image tokens; text decoding is unaffected, so the causal-only text path is correct
for both. `store_full_length_kv` is **not present in either config** — it is a
reference-implementation concept, not a checkpoint field.

## 14. KV is 18.0 KiB/token, and full-attention layers really are 512-dim ✅

Settled by reading the checkpoint. The three full-attention layers among the fifteen
that own KV (`i % 5 == 4`, so L4/L9/L14) carry a **512-wide** K projection, and their
`k_norm` is `[512]` to match:

| layer type | count | `k_proj` out | `k_norm` | KV head dim |
|---|---:|---:|---:|---:|
| `sliding_attention` | 12 | 256 | `[256]` | 256 |
| `full_attention` | 3 | **512** | **`[512]`** | **512** |

So `global_head_dim` is the KV head dim on those layers, not merely the query head
dim. `init_kv_cache` allocates **18.00 KiB/token** at every context length, matching
`12 × 1 × 256 × 2 + 3 × 1 × 512 × 2 = 9,216` elements × 2 B exactly.

An earlier note recorded a 15.0 KiB/token reading and concluded our estimates were
"~20% pessimistic". That was backwards. 15.0 KiB is precisely what a **uniform
256-dim** assumption produces (`15 × 1 × 256 × 2 × 2 B`), and the checkpoint
contradicts it. Our figure is the correct one; nothing needs adjusting.

Worth checking upstream: an allocator that sizes E2B's KV uniformly at `head_dim`
would under-provision the three full-attention layers by half. The provenance of
that 15.0 KiB reading is not recorded here, so this is a lead rather than a report.

## 15. Unresolved ⚠️

- **`store_full_length_kv` behaviour.** The reference marks the last non-shared layer
  of each type as storing full-length KV. Our windowed-KV ring windows every sliding
  layer including the source. Self-consistent (windowed and full-length outputs match
  in `tests/test_windowed_kv.py`), but whether it matches Gemma's intent for shared
  sliding layers is unverified.

## How to check the next one

```python
import transformers.models.gemma4.modeling_gemma4 as m; print(m.__file__)
```

Read it before inferring anything from tensor shapes. Every fix in sections 2–6 came
from that file after hours of guessing produced nothing but plausible-looking garbage.
