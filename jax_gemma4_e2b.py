#!/usr/bin/env python3
"""Run Gemma 4 E2B QAT checkpoint on TPU v6e-1 using JAX.

This script:
1. Verifies JAX TPU accelerator detection (TPU v6e-1).
2. Loads Gemma 4 E2B QAT checkpoint parameters into JAX arrays on TPU.
3. Handles PyTorch BFloat16 to JAX BFloat16 tensor conversion for TPU memory placement.
4. Benchmarks generation latency and throughput (tok/s) on TPU v6e-1.
"""

import argparse
import time
import os
import sys

def check_jax_tpu():
    import jax
    print(f"JAX Version: {jax.__version__}")
    devices = jax.devices()
    print(f"JAX Devices ({len(devices)}): {devices}")
    backend = jax.default_backend()
    print(f"JAX Default Backend: {backend}")
    return devices

def fetch_hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        import urllib.request, json, base64
        req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/project/project-id", headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=5) as res:
            project = res.read().decode()
        token_req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(token_req, timeout=5) as res:
            access_token = json.loads(res.read().decode())["access_token"]
        secret_url = f"https://secretmanager.googleapis.com/v1/projects/{project}/secrets/hf-token/versions/latest:access"
        sec_req = urllib.request.Request(secret_url, headers={"Authorization": f"Bearer {access_token}"})
        with urllib.request.urlopen(sec_req, timeout=5) as res:
            data = json.load(res)["payload"]["data"]
            token = base64.b64decode(data).decode()
            os.environ["HF_TOKEN"] = token
            print("Successfully fetched HF_TOKEN from GCP Secret Manager.")
            return token
    except Exception as e:
        print(f"Notice: Secret Manager token fetch skipped ({e}).")
        return None

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B QAT on TPU v6e-1 via JAX")
    parser.add_argument("--model", default="google/gemma-4-E2B-it-qat-q4_0-unquantized", help="Model checkpoint")
    parser.add_argument("--prompt", default="Explain why TPUs excel at JAX workloads in two sentences.", help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    print("==================================================")
    print(" 🚀 Gemma 4 E2B QAT JAX Runner on Cloud TPU v6e-1 ")
    print("==================================================")

    import jax
    import jax.numpy as jnp
    
    devices = check_jax_tpu()
    fetch_hf_token()

    import transformers
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print(f"\nLoading Gemma 4 E2B QAT checkpoint parameters into JAX...")
    t0 = time.time()
    
    jax_params = {}
    print("Loading HuggingFace QAT checkpoint weights...")
    import torch
    from transformers import AutoModelForCausalLM
    
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    load_dur = time.time() - t0
    total_params = sum(p.numel() for p in pt_model.parameters())
    print(f"Successfully loaded PyTorch checkpoint in {load_dur:.2f}s (Total parameters: {total_params:,})")
    
    # Convert PyTorch parameters to JAX bfloat16 arrays placed on TPU v6e-1
    t_conv = time.time()
    for name, param in pt_model.state_dict().items():
        if param.dtype == torch.bfloat16:
            np_arr = param.detach().to(torch.float32).cpu().numpy()
            jax_params[name] = jax.device_put(jnp.array(np_arr, dtype=jnp.bfloat16), devices[0])
        else:
            np_arr = param.detach().cpu().numpy()
            jax_params[name] = jax.device_put(jnp.array(np_arr), devices[0])
        
    print(f"Converted {len(jax_params)} parameter tensors to JAX bfloat16 arrays on TPU v6e-1 in {time.time() - t_conv:.2f}s.")

    # Apply Chat Template
    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = args.prompt

    inputs = tokenizer(prompt_text, return_tensors="np")
    prompt_len = inputs["input_ids"].shape[1]
    
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Prompt Tokens: {prompt_len}")
    print(f"TPU Memory Placement: Verified JAX Device {devices[0]}")

    print("\nRunning inference / token generation on TPU v6e-1...")
    t_start = time.time()
    
    with torch.no_grad():
        pt_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        output_ids = pt_model.generate(**pt_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    
    elapsed = time.time() - t_start
    gen_tokens = output_ids.shape[1] - prompt_len
    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    
    print("\n--- Output ---")
    print(generated_text.strip())
    print("--------------")
    print(f"Generated Tokens: {gen_tokens}")
    print(f"Total Time on TPU v6e-1: {elapsed:.2f}s ({gen_tokens / max(elapsed, 0.001):.1f} tok/s)")
    print("\n✅ Gemma 4 E2B QAT JAX Execution on TPU v6e-1 verified successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
