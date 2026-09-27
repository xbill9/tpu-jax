# Gemma 4 pure-JAX on AWS Inferentia2

This is the AWS deployment scaffold for the same pure-JAX engine used on TPU.
It targets an `inf2.xlarge` through the JAX NeuronX PJRT plugin; it does not use
vLLM, PyTorch, `optimum-neuron`, or NxD Inference.

The model math, safetensors loader, cached decode, OpenAI API, and corrected
benchmark methodology remain shared with the TPU path. The platform-specific
pieces are isolated here:

- `neuron_entrypoint.py` configures and verifies the Neuron JAX backend before
  importing the server.
- `user_data.sh` mounts the cache volume, installs the Inf2-compatible JAX
  NeuronX stack, and creates a systemd service.
- `deploy.py` plans, launches, or terminates one tagged EC2 host with SSM access
  and a persistent EBS cache volume.

## Support boundary

AWS documents JAX NeuronX on Inf2 as beta. This scaffold is therefore a porting
target, not a measured-performance claim. In particular:

- W4A16 uses the JAX reference dequantize-and-matmul path, which is the engine
  default. The optional fused kernel lowers through Mosaic and cannot compile
  on Neuron at all, so the entrypoint forces it onto the Pallas interpreter
  rather than letting it fail at trace time.
- Neuron supports JAX buffer donation, but the complete Gemma graph must still
  compile and pass parity tests on real Inf2 hardware.
- `jax.debug` callbacks/checkify, dynamic `while_loop`, integer dot products,
  and several other JAX features are unsupported on Neuron. The serving path
  must remain static-shaped.
- The bootstrap pins the JAX NeuronX component to `0.10.0.1.0.*`, the newest
  build on the Neuron index — there is no 0.8, 0.9, or 0.11 plugin, so 0.10 is
  the ceiling and server-side parity with a 0.11 dev environment is not
  purchasable. Pin the CPU oracle to the same 0.10 line instead; a reference
  that runs a different JAX than the device is a weak oracle.
- `jax` and `jaxlib` are **not** on the Neuron index (only `jax-neuronx`,
  `libneuronxla`, and `neuronx-cc` are), so the install must use
  `--extra-index-url`. `--index-url` replaces PyPI and cannot resolve jaxlib.
- jax 0.10 requires Python >=3.11, so this pairs with an **Ubuntu 24.04** image;
  every 22.04 DLAMI ships Python 3.10 and is disqualified. Use the Base Neuron
  AMI: packages install into the system interpreter with
  `--break-system-packages` (24.04 is PEP 668 externally-managed, and this is a
  single-purpose host), so the image supplies the driver, runtime, and Python
  while pip supplies the framework. A framework DLAMI would ship a preinstalled
  environment that goes unused. If you move the pin, move the AMI with it.

## Known Neuron risk sites in the shared engine

A sibling port of the same Gemma 4 family to the same hardware (through
PyTorch/NxD rather than JAX) hit a specific set of walls. That stack is not this
stack, so none of these are confirmed here — but they are the places to look
first when something fails on device, and each has a known-cheap mitigation.

| Risk | Where in this repo | Mitigation if it bites |
|---|---|---|
| `tanh` logit softcap over the 262144-token vocab overflowed the 196608 B/partition SBUF (`NCC_INLA001`) | `ports/gemma4/jax_e_model.py:963` | Softcap is monotonic, so `argmax(softcap(x)) == argmax(x)`: greedy decode is unaffected by dropping it. Set `logit_softcapping=0.0` for the device graph and re-apply host-side only when `temperature > 0`. |
| Data-dependent scatter — `neuronx-cc` prefers plain arithmetic over scatter/dynamic indexing | top-k mask `…at[…].set(…)` at `ports/gemma4/jax_e_model.py:1549`; the `dynamic_update_slice` KV write | Replace with a one-hot masked write (`buf*(1-oh) + new*oh`), which is pure arithmetic and trace-safe. |
| Fused/SDPA attention overflowed SBUF at larger sliding windows | not applicable — `attention_jax` is already eager | Already the portable choice; keep it. |
| Param-only checkpoint loads silently skip the `layer_scalar` **buffer**, over-scaling every layer ~16x into `cos ≈ 0` garbage with no error | already handled at `ports/gemma4/jax_e_model.py:946` | Covered; the safetensors loader reads it explicitly. Do not regress this. |
| Host RAM, not the accelerator, is the binding constraint on `inf2.xlarge` | `user_data.sh` swapfile | Already provisioned (`--swap-gib`, default 32). |

Two capacity notes carried over from that port: `inf2.xlarge` and `inf2.8xlarge`
carry the *identical* 2-core / 32 GB-HBM accelerator and differ only in host
vCPU and RAM, so the cheap box is cheaper per token once swap is in place; and
"E2B" is a MatFormer *effective* parameter count — the real device footprint is
~5B, which is why capacity must be planned from real parameters. This scaffold
serves the W4A16 QAT checkpoint, which is smaller again.

## Prerequisites

1. An AWS account with Inf2 quota, an existing VPC/subnet/security group, and
   an EC2 instance profile that includes `AmazonSSMManagedInstanceCore`.
2. The instance profile may read the Hugging Face token from Secrets Manager
   secret `hf-token`. Store it as a plain string, not JSON — the bootstrap
   passes `SecretString` straight through as `HF_TOKEN`.
3. A source bundle uploaded to S3. It should unpack with this repository at its
   root. The instance role needs `s3:GetObject` for that object.
4. A Deep Learning **Base** Neuron AMI on **Ubuntu 24.04**. Resolve it from
   `/aws/service/neuron/dlami/base/ubuntu-24.04/latest/image_id` and pass the ID
   with `--ami-id`; automatic name discovery is convenient for development but
   does not guarantee the SDK line or the Python version jax 0.10 needs.

No inbound SSH is required. The API binds to `127.0.0.1:8000`; reach it through
SSM port forwarding or put a private load balancer in front of it.

## Plan and launch

Install only the local control-plane dependency:

```bash
python3 -m pip install boto3
```

Generate a launch plan (read-only; this is the default):

```bash
python3 deployments/aws-inf2/deploy.py plan \
  --region us-east-1 \
  --subnet-id subnet-... \
  --security-group-id sg-... \
  --instance-profile-name gemma4-inf2 \
  --source-uri s3://my-bucket/tpu-jax-inf2.tar.gz
```

Launch only after inspecting the plan:

```bash
python3 deployments/aws-inf2/deploy.py launch --apply \
  --region us-east-1 \
  --subnet-id subnet-... \
  --security-group-id sg-... \
  --instance-profile-name gemma4-inf2 \
  --source-uri s3://my-bucket/tpu-jax-inf2.tar.gz
```

The launcher refuses to create a second pending or running instance with the
same `Project` tag in the region. Spot is opt-in with `--market-type spot`.

## Storage and teardown

Two volumes are attached. The root volume is disposable and deletes on
termination. A second gp3 volume on `/dev/sdf` holds `/opt/gemma4/cache`
(Hugging Face weights, the XLA compilation cache, and the Neuron compile cache)
and is **retained** on termination, because recompiling the Gemma graph from
cold costs far more than the idle volume does.

Teardown plans first, like launch:

```bash
python3 deployments/aws-inf2/deploy.py terminate --region us-east-1   # plan
python3 deployments/aws-inf2/deploy.py terminate --apply --region us-east-1
```

The output lists the retained cache volume ID per host. That volume keeps
billing until you delete it. `deploy.py` does not reattach it — to reuse the
caches, attach it to the new instance as `/dev/sdf` in the same Availability
Zone before the service starts; `user_data.sh` mounts any already-formatted
non-root disk it finds and skips `mkfs`.

## Validate on the host

```bash
sudo journalctl -u gemma4-jax-inf2 -f
sudo -u ubuntu /usr/bin/python3 \
  /opt/gemma4/app/deployments/aws-inf2/neuron_entrypoint.py --check-only
curl http://127.0.0.1:8000/health
```

Before publishing any Inf2 result, run cached-decode parity, HTTP smoke tests,
and a corrected v2 sweep on the device. Do not reuse TPU throughput or memory
numbers as Inf2 claims.

### If the output is garbage

Two rules from the sibling port, both earned by losing hours to them:

**Run the CPU oracle before you suspect the device.** Load the same checkpoint
on the same box with `JAX_PLATFORMS=cpu` and run one greedy forward. If the CPU
reference emits the *same* garbage, the accelerator is exonerated and the bug is
upstream — tokenizer, inputs, or weights. In that port every single garbage-output
incident was innocent silicon: a missing `tokenizer.json` that mapped every prompt
to `<unk>`, a mis-restored weight reload, an unloaded scalar buffer, and a
driver/SDK mismatch. Sanity-check `tok("hello world").input_ids` first; it is a
two-minute check that has replaced multi-hour compiler hunts.

**Validate the serving path, not the trace.** Parity that passes in-process on a
freshly built model does not exercise the code path a fresh server process uses.
Check the exact artifact, loaded the exact way the service loads it, against an
*independent* float reference. A green test against the wrong oracle — that port
had an auto-port report "100% PASS" against a golden built from a PLE-stripped
checkpoint — is worse than a red one.
