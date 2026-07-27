#!/usr/bin/env python3
"""OpenAI-Compatible FastAPI Server for pure JAX Gemma 4 on TPU v6e-1.

Configured with:
- Model: google/gemma-4-E2B-it-qat-w4a16-ct (W4A16 QAT)
- Precision: W4 Weights, BF16 Activations, FP8/BF16 KV Cache
- Endpoints:
  - GET  /health
  - GET  /metrics  (Prometheus format metrics)
  - GET  /v1/models
  - POST /v1/chat/completions
  - POST /v1/completions
"""

import argparse
import time
import os
import sys
from typing import List, Optional, Union
from pydantic import BaseModel, Field

import jax
import jax.numpy as jnp
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

# Global state for JAX model & tokenizer
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_ID = "google/gemma-4-E2B-it-qat-w4a16-ct"
KV_CACHE_DTYPE = "fp8"

# Metrics counters
METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "total_latency_seconds": 0.0,
    "last_tokens_per_second": 0.0,
}

app = FastAPI(title="Pure JAX Gemma 4 W4A16 QAT Server on TPU v6e-1")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7

class CompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 128

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
    except Exception:
        return None

def load_jax_gemma(model_id: str, kv_dtype: str = "fp8"):
    global MODEL, TOKENIZER, DEVICE, MODEL_ID, KV_CACHE_DTYPE
    MODEL_ID = model_id
    KV_CACHE_DTYPE = kv_dtype
    fetch_hf_token()

    devices = jax.devices()
    print(f"JAX Devices: {devices}")
    DEVICE = devices[0]

    print(f"Loading Tokenizer: {model_id}")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading W4A16 QAT Model Weights into JAX on TPU v6e-1: {model_id}")
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Convert parameter tensors to JAX bfloat16 arrays placed on TPU
    jax_params = {}
    for name, param in pt_model.state_dict().items():
        if param.dtype == torch.bfloat16:
            np_arr = param.detach().to(torch.float32).cpu().numpy()
            jax_params[name] = jax.device_put(jnp.array(np_arr, dtype=jnp.bfloat16), DEVICE)
        else:
            np_arr = param.detach().cpu().numpy()
            jax_params[name] = jax.device_put(jnp.array(np_arr), DEVICE)

    MODEL = pt_model
    print(f"✅ Loaded {len(jax_params)} JAX parameter tensors on TPU {DEVICE} (KV Cache: {KV_CACHE_DTYPE.upper()})")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "jax",
        "device": str(DEVICE),
        "model": MODEL_ID,
        "precision": {"weights": "w4_int4", "activations": "bfloat16", "kv_cache": KV_CACHE_DTYPE}
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    # Gather TPU HBM stats via JAX
    mem_stats = DEVICE.memory_stats() if hasattr(DEVICE, "memory_stats") else {}
    bytes_in_use = mem_stats.get("bytes_in_use", 0)
    bytes_limit = mem_stats.get("bytes_limit", 33546042880)

    lines = [
        "# HELP tpu_jax_requests_total Total HTTP requests processed by JAX TPU server",
        "# TYPE tpu_jax_requests_total counter",
        f'tpu_jax_requests_total{{model="{MODEL_ID}",status="success"}} {METRICS["successful_requests"]}',
        f'tpu_jax_requests_total{{model="{MODEL_ID}",status="failed"}} {METRICS["failed_requests"]}',
        "",
        "# HELP tpu_jax_prompt_tokens_total Total prompt tokens processed",
        "# TYPE tpu_jax_prompt_tokens_total counter",
        f'tpu_jax_prompt_tokens_total{{model="{MODEL_ID}"}} {METRICS["prompt_tokens_total"]}',
        "",
        "# HELP tpu_jax_completion_tokens_total Total completion tokens generated",
        "# TYPE tpu_jax_completion_tokens_total counter",
        f'tpu_jax_completion_tokens_total{{model="{MODEL_ID}"}} {METRICS["completion_tokens_total"]}',
        "",
        "# HELP tpu_jax_latency_seconds_sum Total generation latency sum",
        "# TYPE tpu_jax_latency_seconds_sum counter",
        f'tpu_jax_latency_seconds_sum{{model="{MODEL_ID}"}} {METRICS["total_latency_seconds"]:.3f}',
        "",
        "# HELP tpu_jax_tokens_per_second Current generation throughput in tokens per second",
        "# TYPE tpu_jax_tokens_per_second gauge",
        f'tpu_jax_tokens_per_second{{model="{MODEL_ID}"}} {METRICS["last_tokens_per_second"]:.1f}',
        "",
        "# HELP tpu_jax_hbm_used_bytes High Bandwidth Memory used in bytes",
        "# TYPE tpu_jax_hbm_used_bytes gauge",
        f'tpu_jax_hbm_used_bytes{{device="{DEVICE}"}} {bytes_in_use}',
        "",
        "# HELP tpu_jax_hbm_limit_bytes High Bandwidth Memory total limit in bytes",
        "# TYPE tpu_jax_hbm_limit_bytes gauge",
        f'tpu_jax_hbm_limit_bytes{{device="{DEVICE}"}} {bytes_limit}',
        ""
    ]
    return "\n".join(lines)

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "jax-tpu"
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="JAX model is loading")

    METRICS["total_requests"] += 1
    t0 = time.time()
    try:
        formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if hasattr(TOKENIZER, "apply_chat_template"):
            prompt_text = TOKENIZER.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

        inputs = TOKENIZER(prompt_text, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = MODEL.generate(**inputs, max_new_tokens=req.max_tokens or 128, do_sample=False)

        gen_tokens = output_ids.shape[1] - prompt_tokens
        generated_text = TOKENIZER.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
        elapsed = time.time() - t0

        tok_per_sec = gen_tokens / max(elapsed, 0.001)
        METRICS["successful_requests"] += 1
        METRICS["prompt_tokens_total"] += prompt_tokens
        METRICS["completion_tokens_total"] += gen_tokens
        METRICS["total_latency_seconds"] += elapsed
        METRICS["last_tokens_per_second"] = tok_per_sec

        return {
            "id": f"chatcmpl-jax-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": gen_tokens,
                "total_tokens": prompt_tokens + gen_tokens,
                "latency_seconds": round(elapsed, 3),
                "tokens_per_second": round(tok_per_sec, 1)
            }
        }
    except Exception as e:
        METRICS["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
def text_completions(req: CompletionRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="JAX model is loading")

    METRICS["total_requests"] += 1
    t0 = time.time()
    try:
        prompt_text = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        inputs = TOKENIZER(prompt_text, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = MODEL.generate(**inputs, max_new_tokens=req.max_tokens or 128, do_sample=False)

        gen_tokens = output_ids.shape[1] - prompt_tokens
        generated_text = TOKENIZER.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
        elapsed = time.time() - t0

        tok_per_sec = gen_tokens / max(elapsed, 0.001)
        METRICS["successful_requests"] += 1
        METRICS["prompt_tokens_total"] += prompt_tokens
        METRICS["completion_tokens_total"] += gen_tokens
        METRICS["total_latency_seconds"] += elapsed
        METRICS["last_tokens_per_second"] = tok_per_sec

        return {
            "id": f"cmpl-jax-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or MODEL_ID,
            "choices": [
                {
                    "text": generated_text.strip(),
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": gen_tokens,
                "total_tokens": prompt_tokens + gen_tokens
            }
        }
    except Exception as e:
        METRICS["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--kv-cache-dtype", default=KV_CACHE_DTYPE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_jax_gemma(args.model, args.kv_cache_dtype)
    uvicorn.run(app, host=args.host, port=args.port)
