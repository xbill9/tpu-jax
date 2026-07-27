#!/usr/bin/env python3.12
"""Tier-4 xProf capture: trace ~50 int8-12B decode steps for offline analysis.

Writes the trace under ~/xprof_trace; view later with
  tensorboard --logdir ~/xprof_trace   (over IAP port-forward)
Needs the default extras: 'setuptools<81' xprof tensorboard-plugin-profile.
The "native TPU profiling disabled on torch<2.12" warning is benign.
"""

import time

import torch
import transformers

from w4a16_int8_model_bench import Int8W4A16Linear, W4A16Linear, _ck_for  # registers the kernel

MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
BATCH = 4
SEQ = 256
STEPS = 50


def main() -> int:
    from torch_tpu._internal import profiler

    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
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
    model = model.to(device).eval()

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain in two sentences why TPUs are fast."}],
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

    print("Warmup (compile outside the trace)...")
    with torch.no_grad():
        for _ in range(4):
            decode_one()
        tokens[0, :1].cpu()

    print(f"Tracing {STEPS} steps...")
    with torch.no_grad():
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.TPU],
            on_trace_ready=profiler.xprof_trace_handler(dir_name="xprof_trace"),
        ):
            t0 = time.monotonic()
            for _ in range(STEPS):
                decode_one()
            tokens[0, :1].cpu()
    print(f"MARKER xprof: {STEPS} steps in {time.monotonic() - t0:.2f}s "
          f"({1000 * (time.monotonic() - t0) / STEPS:.1f} ms/step), trace in ~/xprof_trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
