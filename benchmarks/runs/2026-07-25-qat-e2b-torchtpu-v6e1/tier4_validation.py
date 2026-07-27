#!/usr/bin/env python3.12
"""Tier-4 validation: official parity harness + perplexity + HBM telemetry.

Modes (run each in its OWN process — one process owns the TPU, and the
bf16 model needs the HBM the int8 model would otherwise hold):

  parity          kernel outputs vs reference dequant matmul through the
                  official torch_tpu assert_close STRICT harness (bf16 bar
                  rtol=1.6e-2/atol=1e-5), 12B extreme shapes
  ppl-int8 FILE   perplexity of the int8-swapped w4a16 model over FILE
  ppl-bf16 FILE   perplexity of the dequantized (run_compressed=False)
                  bf16 model over the same FILE — the reference number
Both ppl modes print torch.tpu._hbm_usage_summary() after model load.

Comparative PPL on identical windows is the claim that matters: int8-vs-
bf16 delta ~0 proves the packed path is quality-neutral end to end.
"""

import sys
import time

import torch

MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
SEQ = 256
N_WINDOWS = 40


def get_assert_close():
    try:
        from torch_tpu._internal.utils.utils import assert_close
        return assert_close, "torch_tpu STRICT"
    except Exception:
        from functools import partial
        return partial(torch.testing.assert_close, rtol=1.6e-2, atol=1e-5), \
            "torch.testing (official bf16 tolerances)"


def parity() -> int:
    from w4a16_int8_model_bench import Int8W4A16Linear, W4A16Linear
    from w4a16_fused_model_bench import FusedW4A16Linear

    device = torch.device("tpu")
    assert_close, harness = get_assert_close()
    print(f"MARKER parity harness: {harness}")
    torch.manual_seed(0)
    failures = 0
    for out_f, in_f in [(15360, 3840), (3840, 15360), (5376, 21504), (21504, 5376)]:
        packed = torch.randint(-(2**31), 2**31 - 1, (out_f, in_f // 8), dtype=torch.int32)
        scale = (torch.randn(out_f, in_f // 32).abs() * 0.01 + 0.001).to(torch.bfloat16)
        x = torch.randn(1, SEQ, in_f, dtype=torch.bfloat16)

        ref_mod = W4A16Linear(packed, scale, out_f, in_f, None).to(device)
        with torch.no_grad():
            y_ref = ref_mod(x.to(device)).cpu()
            for name, cls in [("fused-int4", FusedW4A16Linear), ("int8", Int8W4A16Linear)]:
                mod = cls(packed, scale, out_f, in_f, None).to(device)
                y = mod(x.to(device)).cpu()
                try:
                    assert_close(y, y_ref)
                    print(f"MARKER parity [{out_f}x{in_f}] {name}: PASS")
                except Exception as e:
                    failures += 1
                    print(f"MARKER parity [{out_f}x{in_f}] {name}: FAIL — {str(e)[:200]}")
    print(f"MARKER parity done, {failures} failures")
    return 1 if failures else 0


def load_model(variant: str):
    import transformers
    if variant == "int8":
        from w4a16_int8_model_bench import Int8W4A16Linear, W4A16Linear, _ck_for
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        n = 0
        for name, mod in list(model.named_modules()):
            if not hasattr(mod, "weight_packed"):
                continue
            out_f, in_f = (int(v) for v in mod.weight_shape)
            packed, scale = mod.weight_packed.data, mod.weight_scale.data
            if scale.shape[0] != out_f:
                scale = scale.T.contiguous()
            bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
            tileable = out_f % 128 == 0 and in_f % 8 == 0 and _ck_for(in_f) is not None
            cls = Int8W4A16Linear if tileable else W4A16Linear
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], cls(packed, scale, out_f, in_f, bias))
            n += 1
        print(f"MARKER ppl-{variant}: swapped {n} Linears")
    else:
        try:
            CTConfig = transformers.CompressedTensorsConfig
        except AttributeError:
            from transformers.utils.quantization_config import CompressedTensorsConfig as CTConfig
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16,
            quantization_config=CTConfig(run_compressed=False),
        )
        print("MARKER ppl-bf16: loaded dequantized (numerically the target model)")
    return model


def ppl(variant: str, text_path: str) -> int:
    import transformers
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer(open(text_path, encoding="utf-8", errors="ignore").read(),
                    return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    # Gemma is BOS-sensitive: every window must start with BOS or nll is garbage.
    stride = SEQ - 1
    n_win = min(N_WINDOWS, ids.shape[0] // stride)
    bos = torch.tensor([tokenizer.bos_token_id], dtype=ids.dtype)
    print(f"MARKER ppl-{variant}: {ids.shape[0]} tokens -> {n_win} BOS-led windows of {SEQ}")

    model = load_model(variant).to(device).eval()
    try:
        print(f"MARKER hbm after load: {torch.tpu._hbm_usage_summary()!r}")
    except Exception as e:
        print(f"MARKER hbm summary unavailable: {e}")

    mask = torch.ones((1, SEQ), dtype=torch.long).to(device)

    def nll(window):
        logits = model(input_ids=window, attention_mask=mask, use_cache=False).logits
        lsm = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        return -lsm.gather(-1, window[:, 1:].unsqueeze(-1)).sum()

    nll_c = torch.compile(nll, backend="tpu", dynamic=False)
    total, count = 0.0, 0
    t0 = time.monotonic()
    with torch.no_grad():
        for w in range(n_win):
            chunk = ids[w * stride:(w + 1) * stride]
            window = torch.cat([bos, chunk]).unsqueeze(0).to(device)
            total += nll_c(window).cpu().item()
            count += SEQ - 1
    import math
    print(f"MARKER ppl-{variant}: nll/tok {total / count:.6f}  PPL {math.exp(total / count):.4f}  "
          f"({count} tokens in {time.monotonic() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    text_path = sys.argv[2] if len(sys.argv) > 2 else None
    # The bench modules parse sys.argv at import time — hand them a bare argv.
    sys.argv = sys.argv[:1]
    if mode == "parity":
        raise SystemExit(parity())
    raise SystemExit(ppl(mode.removeprefix("ppl-"), text_path))
