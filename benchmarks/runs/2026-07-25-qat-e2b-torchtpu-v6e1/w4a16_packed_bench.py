#!/usr/bin/env python3.12
"""Step 3: true packed int4 (w4a16) execution of gemma-4-E2B-it-qat-w4a16-ct on TorchTPU.

Keeps compressed-tensors pack-quantized weights packed (int32, 8 nibbles each) in
HBM and dequantizes inside the compiled forward with traceable ops only:
(packed >> 4*i) & 0xF, minus 8, times per-group-32 scale. No .item(), static
shapes, so torch.compile(backend="tpu") accepts it — unlike CompressedLinear's
on-the-fly decompression.

Phases (MARKER lines): reference dequant load -> packed load + module swap ->
CPU bit-exactness check -> TPU eager dequant check -> compiled decode bench.
"""

import gc
import time

import torch
import transformers
from transformers import CompressedTensorsConfig

MODEL = "google/gemma-4-E2B-it-qat-w4a16-ct"
PROMPT = "Explain in two sentences why TPUs are fast."
GROUP = 32
SEQ = 256
WARMUP_STEPS = 4
BENCH_STEPS = 64
CHECK_LAYER = "model.language_model.layers.0.mlp.gate_proj"


class W4A16Linear(torch.nn.Module):
    """Linear over packed int4 weights; dequantizes per forward with traceable ops."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, out_f: int, in_f: int, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("packed", packed)  # [out, in//8] int32
        self.register_buffer("scale", scale)    # [out, in//GROUP]
        self.register_buffer("shifts", torch.arange(0, 32, 4, dtype=torch.int32))
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def dequant(self) -> torch.Tensor:
        w = (self.packed.unsqueeze(-1) >> self.shifts) & 0xF          # [out, in//8, 8]
        w = w.reshape(self.out_f, self.in_f) - 8                       # int32, [-8, 7]
        w = w.to(self.scale.dtype).reshape(self.out_f, -1, GROUP) * self.scale.unsqueeze(-1)
        return w.reshape(self.out_f, self.in_f).to(torch.bfloat16)

    def forward(self, x):
        y = x @ self.dequant().T
        return y + self.bias if self.bias is not None else y


def main() -> int:
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    print("Phase 1: reference dequantized load (run_compressed=False, CPU)...")
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    ref_w = ref_model.get_submodule(CHECK_LAYER).weight.detach().clone()
    del ref_model
    gc.collect()
    print(f"MARKER ref weight {CHECK_LAYER}: {tuple(ref_w.shape)} {ref_w.dtype}")

    print("Phase 2: packed load (run_compressed=True, CPU) + module swap...")
    t0 = time.monotonic()
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    probe = model.get_submodule(CHECK_LAYER)
    print(f"MARKER compressed module type: {type(probe).__name__}")
    for n, p in list(probe.named_parameters(recurse=False)) + list(probe.named_buffers(recurse=False)):
        print(f"MARKER   {n}: {tuple(p.shape)} {p.dtype}")

    packed_bytes = scale_bytes = bf16_bytes = 0
    swapped = 0
    for name, mod in list(model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        out_f, in_f = (int(x) for x in mod.weight_shape)
        packed = mod.weight_packed.data
        scale = mod.weight_scale.data
        if scale.shape[0] != out_f:
            scale = scale.T.contiguous()
        assert packed.shape == (out_f, in_f // 8), (name, packed.shape, (out_f, in_f))
        assert scale.shape == (out_f, in_f // GROUP), (name, scale.shape)
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], W4A16Linear(packed, scale, out_f, in_f, bias))
        packed_bytes += packed.numel() * 4
        scale_bytes += scale.numel() * scale.element_size()
        bf16_bytes += out_f * in_f * 2
        swapped += 1
    print(f"MARKER swapped {swapped} modules in {time.monotonic() - t0:.1f}s; "
          f"packed+scale = {(packed_bytes + scale_bytes) / 1e9:.2f} GB vs bf16 {bf16_bytes / 1e9:.2f} GB "
          f"({bf16_bytes / (packed_bytes + scale_bytes):.2f}x smaller)")

    print("Phase 3: CPU bit-exactness check vs reference dequant...")
    mine = model.get_submodule(CHECK_LAYER).dequant()
    diff = (mine.float() - ref_w.float()).abs().max().item()
    print(f"MARKER cpu dequant max abs diff vs reference: {diff}")
    assert diff == 0.0, "unpack/scale mismatch vs compressed-tensors decompression"

    print("Phase 4: TPU eager dequant check...")
    model = model.to(device).eval()
    tpu_deq = model.get_submodule(CHECK_LAYER).dequant().cpu()
    tdiff = (tpu_deq.float() - ref_w.float()).abs().max().item()
    print(f"MARKER tpu eager dequant max abs diff: {tdiff}")

    print("Phase 5: compiled static-shape decode bench...")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    assert n_prompt + WARMUP_STEPS + BENCH_STEPS < SEQ

    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((1, SEQ), pad_id, dtype=torch.long)
    tokens[:, :n_prompt] = ids[0]
    tokens = tokens.to(device)
    mask = torch.zeros((1, SEQ), dtype=torch.long)
    mask[:, :n_prompt] = 1
    mask = mask.to(device)
    one = torch.ones((1, 1), dtype=torch.long).to(device)
    pos = torch.tensor([n_prompt], dtype=torch.long).to(device)

    def step(tokens, mask, last_idx):
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        return logits.index_select(1, last_idx).argmax(-1)

    step_c = torch.compile(step, backend="tpu")

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
    print(f"MARKER packed w4a16 compiled decode (SEQ={SEQ}, no KV cache): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
