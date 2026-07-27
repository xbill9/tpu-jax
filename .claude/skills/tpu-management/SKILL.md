---
name: tpu-management
description: Manage Google Cloud TPU capacity, Gemma 4 vLLM serving, and PyTorch (TorchTPU) workloads on TPU VMs. Use when the user asks about provisioning, finding, listing, or destroying TPUs / queued resources / flex-start VMs, starting or debugging vLLM on TPU (v6e, v5p, v5e), running or writing PyTorch on TPU, TPU quotas and zones, TPU cost estimates, benchmarking TPU serving, or the TPU devops MCP agent. Triggers include "TPU", "queued resource", "flex-start", "v6e", "vLLM on TPU", "TPU quota", "PyTorch on TPU", "TorchTPU", "torch.compile on TPU".
---

# TPU Management

Operate Google Cloud TPU infrastructure: acquire capacity, then either run Gemma 4
vLLM serving (verify health, benchmark) or a PyTorch (TorchTPU) environment for
direct torch workloads, and tear down. Two ways to act:

1. **Preferred — MCP agent tools.** If the `tpu-devops-agent` MCP server is
   connected in this session, use its tools (catalog below). They wrap the correct
   `gcloud` invocations, discovery, and retry/cleanup logic.
2. **Fallback — direct `gcloud`.** If the MCP server is not connected, either offer to
   register the bundled server (see "Registering the MCP server") or run the equivalent
   `gcloud` commands from `references/tpu-guide.md`.

## Bundled files

- `mcp/server.py` — the FastMCP DevOps agent (snapshot of the repo-root `server.py`;
  the live copy at the repo root is authoritative if the two differ).
- `mcp/project-setup.sh` — one-command installer: copies this skill into a target project and
  registers the MCP server (see "Registering the MCP server").
- `mcp/startup_script_template.sh` — the TPU VM startup script the agent injects when
  creating a queued resource (pulls `vllm/vllm-tpu:nightly` and serves the model).
- `mcp/startup_script_pytorch_template.sh` — the startup script for `workload="pytorch"`
  VMs (installs PyTorch + the TorchTPU backend on the bare VM and smoke-tests
  `torch.compile(backend="tpu")` — no docker, no HF token).
- `references/tpu-guide.md` — the TPU getting started guide: prerequisites,
  flex-start capacity zones per TPU family, `gcloud` creation templates for v6e/v5p/v5e,
  persistent-disk + startup-script patterns, quota metrics and request procedure,
  troubleshooting/FAQ. Read it when working without the MCP tools, diagnosing
  provisioning failures, or answering quota/capacity/billing questions.

## Registering the MCP server

Easiest path — run the bundled installer (idempotent; installs this skill into the
target project and writes the `tpu-devops` entry into the project's `.mcp.json`,
using the system `python3` — it warns if the pip deps below are missing but never
creates a venv):

```bash
mcp/project-setup.sh /path/to/project --project <gcp-project-id>   # one project
mcp/project-setup.sh --global                                      # all projects (user scope)
# from the skill repo root: make init TARGET=/path/to/project ARGS='--project <id>'
```

Run `mcp/project-setup.sh --help` for all options (`--model`, `--accelerator`, `--tp`,
`--server-name`, `--skip-deps`). Then restart Claude Code in the target project and
approve the server when prompted; `/mcp` should list `tpu-devops`.

Manual alternative:

```bash
claude mcp add tpu-devops \
  --env GOOGLE_CLOUD_PROJECT=<project-id> \
  --env MODEL_NAME=google/gemma-4-31B-it \
  --env ACCELERATOR_TYPE=v6e-8 \
  --env TENSOR_PARALLEL_SIZE=8 \
  -- python .claude/skills/tpu-management/mcp/server.py
```

Requires: `pip install -r mcp/requirements.txt`, an authenticated
`gcloud` CLI with alpha components (`gcloud components install alpha`), and the TPU API
enabled. The server reads config from env vars: `GOOGLE_CLOUD_PROJECT` (falls back to
the active gcloud config), `GOOGLE_CLOUD_ZONE` (default `europe-west4-a`),
`GOOGLE_CLOUD_REGION`, `MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`.
A Hugging Face token must exist as Secret Manager secret `hf-token` (save one with the
`save_hf_token` tool) before any resource creation.

## Standard lifecycle

1. **Status first.** `get_system_status` (dashboard) or `list_queued_resources` /
   `find_gpu`. Never create before checking what already exists.
2. **Acquire capacity.**
   - Preferred (v6e/v5p): `create_tpu_vm_instance` — GCE flex-start VM with vLLM
     auto-start — or `find_tpu_vm` to sweep zones until one grants capacity (family
     quota is only discoverable by attempting creation). Then `wait_for_vllm_ready`
     polls until serving is up; `get_tpu_vm_serial_log` for manual watching.
     For a PyTorch dev VM instead of serving, pass `workload="pytorch"` to either
     tool and follow with `wait_for_pytorch_ready` (see "PyTorch (TorchTPU) on TPU").
   - Known zone, legacy API: `create_tpu_queued_resource` (non-destructive; skips if
     the resource already exists) or `manage_queued_resource` (destructive — deletes
     every other queued resource in the zone). Flex-start by default: 4h max-run;
     `reserved=True` for reservations.
   - Unknown zone: `get_zones_with_available_quota`, or `find_tpu` which sweeps every
     zone with quota, polls until ACTIVE (3 min, extended to 10 min once PROVISIONING),
     and cleans up failures. It skips zones previously marked failed in
     `~/.cache/tpu-devops/tpu_zones_status.md`.
3. **Wait for ACTIVE.** `describe_queued_resource`.
   Queued resources move QUEUED → PROVISIONING → ACTIVE; FAILED/SUSPENDED means
   delete and retry (the manage tool does this automatically).
4. **Serve.** The creation startup script auto-starts vLLM. Otherwise
   `manage_vllm_docker` with action `start|stop|restart|status|log|rm` (targets the
   queued resource's node by default, or a GCE VM via `instance_name` — same for
   `get_vllm_docker_logs`, `get_tpu_system_logs`, `run_vllm_benchmark`).
   **Switching the model:** call `start` with `model_name` (or any serving param) —
   that replaces the container with the new config; a plain `start` just restarts
   the existing container unchanged. It auto-picks
   load format, max-model-len, and memory utilization from the model size
   (26B/31B → `tpu_streaming_loader`, 16384 ctx, 0.80 util; smaller → `runai_streamer`,
   65536 ctx, 0.90 util). Model load can take many minutes — check
   `get_vllm_docker_logs` for "Application startup complete."
5. **Verify.** `verify_model_health`, `get_vllm_endpoint`, `get_model_details`,
   `query_queued_gemma4` (`include_stats=True` for TTFT/throughput). Health checks,
   queries, and benchmarks auto-target whatever model the server actually loaded
   (via `/v1/models`), so they keep working after a deploy-time `model_name` override.
6. **Benchmark (optional).** `run_vllm_benchmark` (runs `vllm bench serve` in a
   separate container on the VM). Pass `save_result=True` to also get the run's
   metrics as a `throughput.sweep[]` entry for the repo's benchmark report format
   (`benchmarks/serving-report.schema.json`) — use it once per concurrency level
   when building a report.
7. **Tear down.** `destroy_queued_resource`. Flex-start bills until deletion and
   cannot be paused — always confirm teardown of idle resources with the user, and
   remind them a flex-start resource left running expires at max-run-duration.

## MCP tool catalog (by task)

**Capacity & lifecycle (GCE flex-start — recommended for v6e/v5p):**
`create_tpu_vm_instance` (creates the VM with the proven flags: 200GB boot disk,
docker-installing startup script, cloud-platform scopes; `workload="pytorch"` for a
TorchTPU VM), `find_tpu_vm` (zone sweep, same workload choice),
`wait_for_vllm_ready` (poll until serving), `wait_for_pytorch_ready`,
`verify_pytorch_tpu` (rerun the TorchTPU smoke test), `update_torchtpu`
(in-place nightly upgrade or version pin + smoke test), `list_tpu_vm_instances`,
`destroy_tpu_vm_instance`, `get_tpu_vm_serial_log`, `get_tpu_vm_endpoint`

**Capacity & lifecycle (queued resources — legacy, v5e):** `find_tpu`,
`create_tpu_queued_resource` (non-destructive),
`manage_queued_resource` (destructive cleanup), `destroy_queued_resource`,
`list_queued_resources`, `describe_queued_resource`,
`get_zones_with_available_quota`, `find_gpu`, `estimate_deployment_cost`

**Serving:** `manage_vllm_docker`, `get_vllm_endpoint`,
`get_vllm_deployment_config` (gcloud one-liner), `save_hf_token`

**Health, logs & diagnostics:** `get_system_status`, `verify_model_health`,
`get_model_details`, `get_metrics`, `get_vllm_docker_logs`, `get_tpu_system_logs`,
`get_cloud_logging_logs`, `analyze_cloud_logging` (Gemma-4-powered log triage)

**Inference & benchmarking:** `query_queued_gemma4` (`include_stats=True` for
latency/throughput), `run_vllm_benchmark`

Every agent in this repo also exposes `get_help` for its live configuration.

## vLLM on TPU — required flags (Gemma 4)

When composing or reviewing a vLLM serve command for TPU, use:
`--tensor-parallel-size 8` (v6e-8), `--max-model-len 16384`,
`--disable_chunked_mm_input`, `--max_num_batched_tokens 4096`,
`--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`,
and `--limit-mm-per-prompt '{"image":4,"audio":1}'` for multimodal
(the agent uses `{"image":0,"audio":0}` for text-only serving).
Image: `vllm/vllm-tpu:nightly`, run with `--privileged --net=host --shm-size 10gb`
and `HF_HOME=/dev/shm`.

Upstream references: [vLLM TPU docs](https://docs.vllm.ai/projects/tpu/en/latest/),
[Recommended Models & Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)
(the support matrix — check it before serving quantized checkpoints),
[vLLM Recipes](https://recipes.vllm.ai) (per-model deployment guides), and the
[tpu-inference GitHub repo](https://github.com/vllm-project/tpu-inference)
([releases](https://github.com/vllm-project/tpu-inference/releases) track newly
landed quantization/model support).

Known-broken (verified on `vllm-tpu:nightly`, Jul 2026): the Gemma 4 **E2B QAT**
checkpoints do not load on TPU in any form — `-qat-w4a16-ct` fails with
"compressed-tensors scheme for layer 'per_layer_model_projection' is not yet
supported in the JAX path", and `-qat-q4_0-unquantized` fails on both the JAX and
`MODEL_IMPL_TYPE=vllm` (torchax) paths with "weights not initialized from
checkpoint: layers.15-34 self_attn.k_norm.weight" (the export omits k_norm for
the upper KV-sharing layers). Serve the plain `google/gemma-4-E2B-it` instead.

## PyTorch (TorchTPU) on TPU

For direct PyTorch workloads (training, kernels, experiments) instead of vLLM
serving, provision with `workload="pytorch"`:

1. `create_tpu_vm_instance(workload="pytorch")` (default name `torchtpu-vm`) or
   `find_tpu_vm(workload="pytorch")` — the startup script installs a dedicated
   Python 3.12 interpreter (TorchTPU requires 3.12; the upstream quickstart's
   venv is replaced by installing straight into python3.12's site-packages,
   which keeps the same separation from the system python3.10 with no venv) and
   pip-installs `TORCH_TPU_PIP_SPEC` (default `torch_tpu`) with `--pre` from
   the authenticated Artifact Registry index `TORCH_TPU_INDEX` using the VM
   service account's token (no docker, no HF token). On success it writes
   "TorchTPU environment ready." to the serial console. A pip 403 means the
   VM's service account lacks Artifact Registry reader access to the torch_tpu
   registry.
2. `wait_for_pytorch_ready` polls that marker; `verify_pytorch_tpu` reruns the
   smoke test (`/opt/torchtpu_smoke.py`) over SSH any time. `update_torchtpu`
   upgrades the wheels in place to the latest nightly (or `version=` to pin —
   the nightly registry is transient, so treat pins as short-term; archive
   known-good wheels yourself for long-term reproducibility). Recreating the
   VM also lands the latest nightly, since the startup script resolves it at
   boot.
3. Run code over `gcloud compute ssh` (add `--tunnel-through-iap` on networks
   that block port 22) using **`python3.12`** — the system `python3` (3.10)
   has no torch. Tear down with `destroy_tpu_vm_instance` — flex-start bills
   until deletion.

**Rules when writing or reviewing PyTorch code for these TPUs:**

- Use **plain `torch`** — never `import torch_tpu` (installing the package
  registers the TPU backend) and never PyTorch/XLA (`torch_xla`). Flag either
  if you see it.
- **Never pip-install `torch` itself** on these VMs: the torch_tpu index pulls
  the matching CPU build of torch automatically; a manually installed torch
  breaks the dispatcher pairing.
- Target the TPU as a device string: `torch.device("tpu")` / `device="tpu"`;
  bring results back with `.cpu()` to force materialization.
- `torch.compile(model, backend="tpu")` to maximize FLOPs utilization.
- **Avoid graph breaks** inside compiled forward passes: no Python control flow
  on tensor values, no `.item()` / scalar-to-tensor conversions, no prints.
- **Align shapes to the MXU:** batch sizes and tensor dims in multiples of
  128 (or at least 8/16) so they tile the Matrix Multiply Unit cleanly.
- **Prefer `torch.bfloat16`** — native on the MXU; float32 is emulated.
- **Attention:** `F.scaled_dot_product_attention` only has a fast path on TPU
  when the FLASH_ATTENTION conditions are met (measured 2026-07-26: the
  FIRST condition is `attn_mask is None` — stock HF models always pass a
  mask, so the fused kernel is unreachable from them; hand-rolled models
  must call SDPA with `attn_mask=None, is_causal=True`; also seq lengths
  divisible by 512). Otherwise it falls back to slow MATH. The official ViT
  tutorial hand-rolls attention instead — do the same, or shape buffers to
  hit the fast path. (Training with SDPA's fast path wants fp32 — a
  documented exception to the bf16 rule.) When porting a model, check its
  attention scaling: SDPA defaults to 1/sqrt(head_dim), but e.g. Gemma 4
  uses scale=1.0 — pass `scale=` explicitly.
- **Static KV-cache decode pattern (compile-clean, verified in
  ports/gemma4/):** preallocate `[B, n_kv, MAX_SEQ, head_dim]` buffers,
  write with `index_copy_` at a TENSOR position (never a Python int), build
  decode masks from `arange(MAX_SEQ)` + the pos tensor via `torch.where` —
  zero data-dependent control flow, traces `fullgraph=True` first try.
  Note decode matmuls have batch-many rows: at batch 1 they under-fill the
  MXU (12B cached b1 was SLOWER than the no-cache 256-row loop); the cache
  pays off at batch ≥ 4 (weights are read once per step regardless of
  batch — 12B cached b4 = 3.3× the no-cache aggregate).
- **Training works** (backward + optimizers, eager): accumulate loss
  on-device via `loss.detach()` and materialize with ONE `.item()` per epoch
  — per-step host syncs stall the TPU pipeline.
- **`torch.compile` requires `dynamic=False`** (dynamic=True/None unsupported;
  official guide). Any input-shape change triggers a full recompile (minutes)
  — keep every compiled-region tensor dimension constant and pad/slice
  OUTSIDE the compiled region (the static-buffer decode loop pattern).
  Allocate scratch models/tensors directly on-device with
  `with torch.device("tpu"):` instead of CPU-init + transfer.
- **Env vars lock at backend init** (first torch import / `torch.device("tpu")`)
  — set them before, or restart the process. Debug:
  `TORCH_SHOW_CPP_STACKTRACES=1` (C++ frames in RuntimeErrors),
  `TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS=1` (bounds checks — catches silent
  OOB corruption, slow), `TORCH_DISABLE_ADDR2LINE=1` if symbolizing hangs.
  Tuning: `TORCH_TPU_INTERNAL_XLA_OPTIONS=xla_optimization_level=O[0-3]`
  (O2 default; O3 slower compile, maybe faster steps);
  `XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=<dir>"` dumps final HLO —
  the ground truth on what fused. One level deeper, the LLO (hardware-op)
  dump needs the flags in BOTH places:
  `LIBTPU_INIT_ARGS="--xla_jf_dump_to=<dir> --xla_jf_dump_llo_text=true"`
  AND appended to `sys.argv` before torch init. Pipeline for orientation:
  Python → ATen → StableHLO (TorchTPU's hand-off) → HLO (+passes) → LLO
  (+hardware passes). **Compile cache**: tier-2
  (`TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=<name>` under /dev/shm) +
  tier-3 (`..._TIER3_COMPILATION_CACHE_ROOT=gs://...` +
  `..._LOCAL_BACKUP_TASK=1`) persists compiled binaries ACROSS VMs —
  the pytorch startup template wires this to the wheel-mirror bucket
  (SA needs objectAdmin), killing the 19-70s warmup on repeat runs.
  **Verified live 2026-07-26** (first boot of the mirror-first template):
  whole startup 1m35s (vs ~10 min via AR), smoke test green, and SSH
  workloads inherit the /etc/environment vars — tier-3 `.bin` entries
  confirmed landing under `compile-cache/<binary-fingerprint>/` in the
  bucket. Note the startup smoke test itself runs cache-cold (env vars are
  written after it), and the runtime logs "Tier-2 compilation cache is
  disabled for world size 1" when the env var is absent.
  Graph fingerprints cover shapes + dtypes + op sequence **+ input scalar
  values** — a changing Python scalar argument recompiles per value (keep
  loop state in device tensors). Variable shapes: bucket them (N buckets =
  N cold starts, then all warm); DataLoaders need `drop_last=True` or the
  short last batch recompiles every epoch. Verify a loop is warm with
  `torch.tpu._get_cache_stats()` (per-entry read counts + compile durations).
- **Execution modes** (`torch_tpu._internal.execution_mode`): default is
  Strict Eager (`DEFER_NEVER`, async one-op dispatch);
  `DEFER_NEVER_AND_LAUNCH_BLOCKING` (`TPU_LAUNCH_BLOCKING=1`) is the
  CUDA_LAUNCH_BLOCKING equivalent for debugging; **`DEFER_AND_FUSE`**
  (`TPU_DEFER_AND_FUSE=1`) groups ops for cross-op XLA fusion — 2.3× on
  elementwise chains per the docs. **Measured 2026-07-26: do NOT use it on
  HF `generate()`** — autoregressive decode changes shapes every step, so
  every fused group is a fresh compile fingerprint (E2B: 0.5 tok/s vs the
  3.4 tok/s plain-eager baseline, 10k compile requests for 55 tokens). It
  only pays on static-shape eager chains. Materialization
  triggers (force execution even when fused): `.item()`, `.cpu()`,
  `.tolist()`, `print(tensor)`, data-dependent control flow.
- **Debugging traps & tools:** `@torch.inference_mode()` + torch.compile
  crashes — always `@torch.no_grad()`. Only ONE process can hold the TPU
  ("already in use by process with pid ...") — kill stray python3.12 before
  benchmarking. Deferred execution surfaces errors at materialization
  (`.cpu()`), not the faulting line — wrap in
  `execution_mode.eager_mode(EagerMode.DEFER_NEVER)` (slow) to pinpoint
  NaN/OOM. `utils.OpTracer()` audits dispatched ops — high `aten.copy_` /
  `aten._to_copy` counts mean hidden CPU round-trips.
  `torch._logging.set_logs(aot_graphs=True)` dumps the traced forward AND
  backward FX graphs (one level above the HLO dump). Custom backward over
  stock ops → `torch.autograd.Function`; new kernels → `torch.library`
  custom ops (what `pallas.jax_op` wraps) — per upstream, don't build custom
  ops on autograd.Function in PyTorch 2.x. Bug reports upstream:
  include the Python traceback + `TORCH_SHOW_CPP_STACKTRACES=1` output.
- **`torch.tpu` is a drop-in for `torch.cuda`** (is_available, device_count,
  current_device, manual_seed_all); the docs' device-agnostic selector is
  `torch.accelerator.current_accelerator()` (returns the tpu device);
  `torch_tpu._internal.utils.log_utils.log_to_stderr()` routes runtime glog
  to stderr (also silences "Could not open log file /tmp/tpu_logs" noise
  when that dir is root-owned); AMP is `torch.amp.autocast("tpu", ...)`;
  Stream/Event APIs are dummies (XLA/PjRt orders execution). Telemetry
  (unstable): `torch.tpu._hbm_usage_summary()` for real HBM footprint,
  `torch.tpu._get_cache_stats()` for compile-cache hits/latencies.
- **Profiling (xProf):** `from torch_tpu._internal import profiler`, wrap
  ~100 representative steps in `profiler.profile(activities=[CPU, TPU],
  on_trace_ready=profiler.xprof_trace_handler(dir_name=...))` (~5–10%
  overhead), then `tensorboard --logdir=<dir>` (needs `setuptools<81` +
  `tensorboard-plugin-profile`; on headless VMs port-forward with
  `gcloud compute ssh ... --tunnel-through-iap -- -L 6006:localhost:6006`).
  The "PrivateUse1ProfilerRegistry not found / native TPU profiling disabled"
  warning on torch < 2.12 is benign — traces still capture. Triage: gaps
  between TPU ops = host bottleneck; short bursts = graph breaks; ideal is
  long fused blocks and >60% MXU utilization.
- **Numerical parity checks:** use `torch_tpu._internal.utils.utils.assert_close`
  (STRICT mode default; failure reports suggest exact rtol/atol). Sync weights
  via `state_dict` first, compare same dtype to same dtype (bf16-TPU vs
  bf16-CPU — never vs fp32), baseline bf16 tolerance rtol=1.6e-2/atol=1e-5.
  Triage: >70% mismatched elements usually means benign fusion/precision
  drift; sparse large outliers point to a real logic bug.
- **Quantization is DIY:** no turnkey int8/torchao path yet — quantized ops are
  authored as custom ops or Pallas kernels (see `examples/ops_and_kernels/qat_linear.py`
  and the quant notebooks in the torch_tpu repo).
- **Custom Pallas kernels:** `pip install jax`, then wrap a type-annotated JAX
  fn (containing `pl.pallas_call`) with `pallas.jax_op("ns::name", fn)` from
  `torch_tpu._internal.pallas` — upstream says that import will likely move to
  `torch_tpu.experimental`, so try that if it breaks. jax_op auto-registers a
  custom op + fake impl, so it composes with `torch.compile(backend="tpu")`;
  training needs a backward wired via `register_autograd` — a second jax_op,
  or auto-derived with `jax.jvp`/`jax.grad` (upstream
  `tests/pallas/pallas_test.py::test_jax_dot_grad_for_backwards`). Pallas is
  optional: wrapping a plain `jax.lax` fn exposes StableHLO ops ATen lacks
  (`population_count` — inference-only; `approx_max_k`). Caution: HLO/LLO
  dumps explode fast (60k+ files per session) — dump one small run at a time.

Upstream references: [torch_tpu repo](https://github.com/google-pytorch/torch_tpu)
(private — GDE program access), [docs](http://google-pytorch.github.io/torch_tpu/)
(runnable Marimo notebooks; `marimo edit docs/notebooks/` on a TPU VM),
[overview video](https://www.youtube.com/watch?v=H8SjVNB7YhM),
[TPU Developers Hub](https://cloud.google.com/products/tpu/tpu-developer),
["How to think about TPUs"](https://jax-ml.github.io/scaling-book/tpus/)
(scaling book ch. 2 — MXU/VMEM/HBM mental model), and
[Helion on TPU](https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/)
(PyTorch kernel DSL compiling to Pallas with autotuning). TorchTPU also runs on
**v5e** and in **Colab** TPU runtimes (per the GDE quickstart) — fallbacks when
v6e flex capacity is tight.

**Field notes (verified on a live v6e-1, Jul 2026):**

- **The VM's default compute SA has no access to the torch_tpu registry** (it
  lives in Google's `ml-oss-artifacts-transient` project — no IAM grant
  possible from your side). Fixed by the **GCS wheel mirror**: the startup
  script first installs from `TORCH_TPU_WHEELS_GCS` (default
  `gs://<project>-torchtpu-wheels`; grant the compute SA
  `roles/storage.objectViewer`), and only falls back to the authenticated
  index. Refresh the mirror with a user-token
  `pip download --pre --no-deps --only-binary=:all: --python-version 3.12
  --platform manylinux_2_31_x86_64` of `torch_tpu==<nightly>` (torch_tpu wheels
  are tagged `manylinux_2_31`; the torch base wheel is `manylinux_2_28`; an
  unpinned request resolves a useless 919-byte `0.0.0` stub) plus
  `torch==2.11.0+cpu`, then `gcloud storage cp` both to the bucket. If neither
  path works, the last resort is the manual install over IAP SSH with a
  user-identity token piped via stdin.
- **The torch_tpu wheel's `torch` dependency is unpinned**, so pip resolves the
  newest CPU torch (2.13) and the extension fails to load with
  `undefined symbol: ...torch...autograd...deleteNode...`. The Jul 2026
  nightlies are built against **torch 2.11.0+cpu** — install
  `torch==2.11.0+cpu` from the same index (the default `TORCH_TPU_PIP_SPEC`
  pins this; drop the pin once upstream constrains its dependency).
- **HF `generate()` cannot be torch.compiled on TPU today** — it feeds a scalar
  CPU tensor into the graph and TorchTPU compiled mode requires every arg
  on-device. Run generate eager (upstream's HF example does the same), or use a
  hand-rolled static-shape decode loop for compiled speed — 67× faster than
  eager and ≈ vLLM single-stream in our v6e-1 measurement
  (`benchmarks/runs/2026-07-25-torchtpu-e2b-v6e1/REPORT.md`).
- Small papercuts: the image's jinja2 (3.0.3) is too old for
  `apply_chat_template` (needs ≥3.1); the fused SDPA kernel needs sequence
  lengths divisible by 512 (else silent fallback to the unfused path).

**References:** [torch_tpu repo](https://github.com/google-pytorch/torch_tpu) ·
[TorchTPU docs](http://google-pytorch.github.io/torch_tpu/) ·
[TPU Developers Hub](https://cloud.google.com/products/tpu/tpu-developer) ·
["How to think about TPUs" (scaling book §2)](https://jax-ml.github.io/scaling-book/tpus/) ·
[Helion on TPU blog post](https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/)

## Field notes — GCE flex-start path (`gcloud compute instances create`)

Verified on a live v6e-1 deployment (Jul 2026). When creating TPU VMs as GCE
instances (the reference guide's template) rather than queued resources, the
guide's command as written will fail; apply all of these:

- **Boot disk:** the default is only 10 GB (hyperdisk-balanced) — `vllm/vllm-tpu:nightly`
  overflows it during layer extraction ("no space left on device"). Add
  `--boot-disk-size=200GB`. If already created, recover without losing flex-start
  capacity: `gcloud compute disks resize <name> --size=200GB` then
  `gcloud compute instances reset <name>` (never delete/recreate — that forfeits
  the capacity grant and restarts the max-run clock).
- **Docker:** not preinstalled on the `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`
  image (unlike TPU runtime images). The bundled startup script template now
  installs `docker.io` when missing; custom scripts must do the same.
- **Secrets at boot:** add `--scopes=cloud-platform` at creation and grant the
  default compute SA `roles/secretmanager.secretAccessor` on `hf-token`
  (`gcloud secrets add-iam-policy-binding hf-token --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor`).
  The bundled startup template fetches the token at boot via the metadata server +
  Secret Manager REST API with a retry loop (~30 min) so an IAM grant applied after
  creation still lands — the token is never written into instance metadata. Custom
  scripts should do the same. Symptom of a missing grant/scope: the fetch 403s forever.
- **Watch boot via serial console, not SSH:** SSH is often blocked by firewall
  policy. The startup template mirrors its log to `/dev/console`; follow it with
  `gcloud compute instances get-serial-port-output <name>`. Grep for the final
  "vLLM application startup complete." line — the earlier "Waiting for
  'Application startup complete.'" echo is a false-positive match.
- **When direct SSH times out, tunnel through IAP:** even with a VPC rule
  allowing tcp:22, an org policy or the client network may drop direct port-22
  traffic (symptom: `gcloud compute ssh` hangs then "Connection timed out").
  `gcloud compute ssh <name> --tunnel-through-iap` rides over HTTPS instead and
  needs only the standard IAP firewall rule (source `35.235.240.0/20`) plus
  `roles/iap.tunnelResourceAccessor`. Note the MCP agent's SSH-based tools
  (`manage_vllm_docker`, `get_vllm_docker_logs`, `run_vllm_benchmark`,
  `get_tpu_system_logs`) do not use IAP; on such networks run their documented
  equivalents manually with `--tunnel-through-iap`.
- **Quota is per region AND per TPU family:** creation fails immediately with
  `Quota 'TPUS_PER_TPU_FAMILY' exceeded. Limit: 0.0` in regions without CT6E
  quota (observed: us-east5 = 0, europe-west4 OK). This dimensioned quota is not
  visible via `gcloud compute regions describe` — attempt creation (fails fast)
  or check the console. Failure sequence for a v6e-1: boot ~1 min → docker
  install ~1 min → image pull ~5 min → model download/compile ~5-10 min.

## Cautions

- `destroy_queued_resource` and `manage_queued_resource` delete infrastructure —
  `manage_queued_resource` deletes ALL queued resources in the zone other than the
  named primary. Confirm with the user before invoking against a zone that may hold
  resources they want kept.
- Flex-start requests expire (`--valid-until-duration`) and instances self-delete at
  `--max-run-duration`; data on the VM is lost. Persist data on a separate disk or GCS
  (see the reference guide).
- Stuck in `WAITING_FOR_RESOURCES`/`PROVISIONING` or `STOCKOUT`: usually the
  `GPUS_ALL_REGIONS` global quota is 0 — see the Troubleshooting section of
  `references/tpu-guide.md` before retrying other zones.
- v5e uses the legacy queued-resources API and separate quota metrics; v6e/v5p use GCE
  machine types (`ct6e-standard-4t`, `ct5p-hightpu-4t`). Zone/family table is in the
  reference guide.
