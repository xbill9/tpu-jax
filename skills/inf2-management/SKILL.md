---
name: inf2-management
description: Provision and operate AWS EC2 Inferentia2 instances through the inf2-devops MCP server — Neuron vLLM, the Gemma-4 Option-B container, or this repository's pure-JAX engine.
---

# AWS Inferentia2 management

Use the `inf2-devops` MCP tools for AWS Inf2 lifecycle work.

## Workflow

1. Call `get_help` and `check_inf2_quotas`.
2. Use `get_deployment_config` for a read-only review of launch settings.
3. Confirm the subnet, security group, and IAM instance profile.
4. Call `create_inf2_instance`; creation starts billing. Launches default
   to spot capacity — pass `spot=False` only when the user asks for
   on-demand. Spot capacity can be interrupted or unavailable; on an
   `InsufficientInstanceCapacity` / spot error, report it and offer
   on-demand instead of retrying silently.
5. Poll `list_inf2_instances`, then call `verify_neuron_health`.
6. Resolve the API with `get_endpoint`; call `query_model` after health passes.

   `verify_neuron_health`, `get_vllm_logs`, and `get_endpoint` take a `serving`
   argument. Pass the mode the host was launched with — the launch tags it
   `Serving=<mode>`, so `list_inf2_instances` is not enough; read the tag or
   remember what you launched. Defaulting to `vllm` against a JAX host probes
   for a docker container that does not exist and reports a healthy host as
   broken.
7. Prefer `stop_inf2_instance` when preserving storage is useful.
   `terminate_inf2_instance` is permanent and requires explicit approval.

## Serving Gemma-4 (`serving="optb"`)

The Neuron vLLM DLC cannot serve Gemma-4 (`optimum-neuron` has no Gemma-4
model class; the endpoint comes up healthy but generates gibberish). For
`google/gemma-4-E2B-it`, pass `serving="optb"` to `create_inf2_instance` or
`get_deployment_config`. This deploys a prebuilt `torch_neuronx` container
(default `docker.io/xbill9/gemma4-optb:slim`, override with `OPTB_IMAGE`)
with compiled neffs and weights baked in — no Hugging Face token required.
It is a single-device build: only `inf2.xlarge` or `inf2.8xlarge`. Cloud-init
adds a 16 GB swapfile because the one-time neff load peaks ~14.5 GB of host
RAM on the 16 GB `inf2.xlarge`. Expect a long first start (image pull +
neff load) before health passes.

## Serving pure JAX (`serving="jax"`)

This repository's own engine — the same `ports/gemma4/` model code that serves on
TPU v6e — reached through the `jax-neuronx` PJRT plugin. No docker, no vLLM, no
`torch_neuronx`, no NxD.

Pass `serving="jax"` plus `source_uri` (an S3 tarball of this repo) to
`create_inf2_instance` or `get_deployment_config`. Unlike the container modes
nothing is baked into an image, so the bundle is mandatory; the tools refuse the
launch without it rather than start a billing instance that cannot serve.

The cloud-init is `deployments/aws-inf2/user_data.sh`, read at render time —
the same file `deployments/aws-inf2/deploy.py` uses, never a copy. Use
`deploy.py` directly when you want the plan/apply workflow with a persistent
compile-cache volume; use these MCP tools for the conversational path.

Constraints, all enforced:

- **Single device only** (`inf2.xlarge` / `inf2.8xlarge`). The compiled graph is
  `--logical-nc-config=1`, so the budget is one NeuronCore's 16 GiB. Measured
  occupancy for E2B decode at context 512 is 8.16 GB.
- **The API binds to `127.0.0.1:8000` by design.** There is no public listener;
  `get_endpoint(serving="jax")` returns the SSM port-forwarding command instead
  of reporting a false "not reachable".
- **First start is slow.** `neuronx-cc` compiles the decode graph before health
  passes — about 20 minutes at `-O1` for full E2B, measured off-device. The
  persistent cache volume is what makes the second start fast.

Status: the graphs are known to compile for `inf2`
(`benchmarks/runs/2026-07-30-neuron-compile-e2b/REPORT.md`), but numerics and
throughput have **not** been validated on a device. Treat it as a porting
target, and do not quote TPU numbers as Inf2 results.

## Guardrails

- Accept only supported `inf2.*` types; never silently fall back to GPU.
- Never place Hugging Face tokens in user data or output. Use Secrets Manager.
- Use Systems Manager for remote commands. Do not expose SSH to the internet.
- Scope discovery to `ManagedBy=inf2-devops`.
- Restrict port 8000 to trusted networks or a private load balancer.
- Match tensor parallelism to NeuronCore count; the server derives it.
- Upgrade the Neuron DLC as a tested SDK/container compatibility set.

The caller needs EC2, SSM, Secrets Manager, and Service Quotas permissions.
The instance profile needs SSM core permissions, ECR Public read access, and
read access to the configured Hugging Face secret.
