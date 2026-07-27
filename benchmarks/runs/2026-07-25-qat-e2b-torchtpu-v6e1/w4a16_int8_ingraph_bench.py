#!/usr/bin/env python3.12
"""Tier-1: int8-stored weights on the IN-GRAPH path (no Pallas) for the
small models where in-graph beats the fused kernel (E2B/E4B).

W4A16Linear's 5-op unpack (shift/mask/sub/convert/mul) becomes 2 ops
(convert/mul): weights stored int8 (q-8) in natural order, dequant traced
into the compiled graph. 2x HBM of packed int4, half of bf16.

Usage: python3.12 w4a16_int8_ingraph_bench.py [model_id] [batch]
"""

import sys
import time

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E2B-it-qat-w4a16-ct"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PROMPT = "Explain in two sentences why TPUs are fast."
GROUP = 32
SEQ = 256
WARMUP_STEPS = 4
BENCH_STEPS = 64


class Int8InGraphLinear(torch.nn.Module):
    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        w = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(out_f, in_f) - 8
        self.register_buffer("w8", w.to(torch.int8).contiguous())
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = self.w8.to(self.scale.dtype).reshape(self.out_f, -1, GROUP) * self.scale.unsqueeze(-1)
        y = x @ w.reshape(self.out_f, self.in_f).to(torch.bfloat16).T
        return y + self.bias if self.bias is not None else y


def main() -> int:
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    print(f"Loading {MODEL} packed (run_compressed=True, CPU)...")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    n = 0
    nbytes = 0
    for name, mod in list(model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        out_f, in_f = (int(v) for v in mod.weight_shape)
        packed, scale = mod.weight_packed.data, mod.weight_scale.data
        if scale.shape[0] != out_f:
            scale = scale.T.contiguous()
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        new = Int8InGraphLinear(packed, scale, out_f, in_f, bias)
        setattr(parent, name.rsplit(".", 1)[-1], new)
        n += 1
        nbytes += new.w8.numel() + new.scale.numel() * 2
    print(f"MARKER swapped {n} modules to int8 in-graph; int8+scale = {nbytes / 1e9:.2f} GB")

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
    print(f"MARKER INT8-INGRAPH compiled decode (SEQ={SEQ}, batch={BATCH}, no KV cache): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s/stream, "
          f"{BATCH * BENCH_STEPS / elapsed:.1f} tok/s aggregate "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
