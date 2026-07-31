"""AWS Inferentia2 lifecycle and inference MCP server.

The server intentionally uses boto3 rather than shelling out to the AWS CLI so
it works with profiles, environment credentials, IAM roles, and AWS SSO-backed
credential processes. Remote administration uses Systems Manager Run Command;
no inbound SSH rule or private key is required.
"""

import asyncio
import base64
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from openai import AsyncOpenAI
from pydantic import Field

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # Keep offline schema/tests usable before `pip install -r`.
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""

    class ClientError(Exception):
        """Fallback used only when the optional AWS dependency is absent."""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inf2-devops")

mcp = FastMCP("inf2-devops-agent")
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "inf2.xlarge")
SERVICE_NAME = os.getenv("SERVICE_NAME", "vllm-inf2")
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_IMAGE = os.getenv(
    "VLLM_IMAGE",
    "public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:"
    "0.16.0-neuronx-py312-sdk2.31.0-ubuntu24.04",
)
NEURON_AMI_NAME = os.getenv(
    "NEURON_AMI_NAME", "Deep Learning AMI Neuron*Ubuntu*"
)
# Prebuilt Gemma-4 E2B Option-B container: torch_neuronx-traced graph with the
# compiled neffs and weights baked in, serving OpenAI routes on port 8080.
# The vLLM DLC path cannot serve Gemma-4 (optimum-neuron has no model class).
OPTB_IMAGE = os.getenv("OPTB_IMAGE", "docker.io/xbill9/gemma4-optb:slim")

# ---------------------------------------------------------------- JAX serving
# serving="jax" is the pure-JAX path: no docker, no vLLM, no torch_neuronx. It
# runs `deployments/aws-inf2/` — the same engine that serves on TPU v6e, reached
# through the jax-neuronx PJRT plugin. Its graphs are known to compile for inf2;
# see benchmarks/runs/2026-07-30-neuron-compile-e2b/REPORT.md.
#
# Unlike the two container modes, this one is NOT self-contained: the host pulls
# a source bundle of this repository from S3, so `source_uri` is mandatory.
JAX_SERVICE_NAME = "gemma4-jax-inf2"
# user_data.sh binds the API to loopback deliberately. Reach it with SSM port
# forwarding or a private load balancer; there is no public listener to probe.
JAX_PORT = int(os.getenv("JAX_PORT", "8000"))
JAX_MAX_MODEL_LEN = int(os.getenv("JAX_MAX_MODEL_LEN", "4096"))
JAX_SWAP_GIB = int(os.getenv("JAX_SWAP_GIB", "32"))
JAX_NEURON_CC_FLAGS = os.getenv("JAX_NEURON_CC_FLAGS", "--model-type=transformer")
# Override when the skill is installed away from the repo checkout.
JAX_DEPLOY_DIR = os.getenv("INF2_JAX_DEPLOY_DIR")


def _jax_deploy_dir() -> Path:
    """Locate `deployments/aws-inf2`, which owns the JAX cloud-init template.

    Searched for rather than computed from `__file__`, because this server lives
    at two different depths — `.claude/skills/inf2-management/mcp/` in the repo
    and `skills/inf2-management/mcp/` in the plugin copy — and may be installed
    to `~/.claude/skills/` entirely outside the tree.

    The template is READ, never duplicated. `deployments/aws-inf2/deploy.py`
    renders the same file with the same markers; a second copy inside this
    server would drift from it silently, and the failure mode of a stale
    bootstrap is a host that comes up healthy serving the wrong thing.
    """
    if JAX_DEPLOY_DIR:
        candidate = Path(JAX_DEPLOY_DIR).expanduser()
        if (candidate / "user_data.sh").is_file():
            return candidate
        raise RuntimeError(
            f"INF2_JAX_DEPLOY_DIR={JAX_DEPLOY_DIR!r} has no user_data.sh"
        )
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "deployments" / "aws-inf2"
        if (candidate / "user_data.sh").is_file():
            return candidate
    raise RuntimeError(
        "serving='jax' needs deployments/aws-inf2/user_data.sh from the "
        "tpu-jax-inf2 checkout, which was not found above this server. Set "
        "INF2_JAX_DEPLOY_DIR to point at it."
    )


def _jax_user_data(
    model: str,
    instance_type: str,
    source_uri: Optional[str],
    max_model_len: int = JAX_MAX_MODEL_LEN,
    swap_gib: int = JAX_SWAP_GIB,
    neuron_cc_flags: str = JAX_NEURON_CC_FLAGS,
    hf_secret_id: str = HF_SECRET_ID,
) -> str:
    """Render deployments/aws-inf2/user_data.sh — same markers as deploy.py."""
    if not source_uri:
        raise ValueError(
            "serving='jax' requires source_uri (s3://bucket/tpu-jax-inf2.tar.gz). "
            "The host installs jax-neuronx and runs this repository's engine, so "
            "it must fetch the source; nothing is baked into an image."
        )
    if _neuron_devices(instance_type) != 1:
        raise ValueError(
            "serving='jax' compiles a single-NeuronCore graph "
            "(--logical-nc-config=1); use inf2.xlarge or inf2.8xlarge"
        )
    template = (_jax_deploy_dir() / "user_data.sh").read_text()
    substitutions = {
        "__SOURCE_URI__": source_uri,
        "__MODEL_ID__": model,
        "__HF_SECRET_ID__": hf_secret_id,
        "__AWS_REGION__": AWS_REGION,
        "__MAX_MODEL_LEN__": str(max_model_len),
        "__SWAP_GIB__": str(swap_gib),
        "__NEURON_CC_FLAGS__": neuron_cc_flags,
    }
    for marker, value in substitutions.items():
        template = template.replace(marker, shlex.quote(value))
    if "__" in template:
        raise ValueError("Unresolved user-data template marker")
    return template


def _session():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed; run `python3 -m pip install -r requirements.txt`")
    kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        kwargs["profile_name"] = AWS_PROFILE
    return boto3.Session(**kwargs)


def _client(service: str):
    return _session().client(service)


def _is_inf2(instance_type: str) -> bool:
    return instance_type.startswith("inf2.")


def _neuron_devices(instance_type: str) -> int:
    return {
        "inf2.xlarge": 1,
        "inf2.8xlarge": 1,
        "inf2.24xlarge": 6,
        "inf2.48xlarge": 12,
    }.get(instance_type, 0)


def _neuron_cores(instance_type: str) -> int:
    return _neuron_devices(instance_type) * 2


def _validate_instance_type(instance_type: str) -> None:
    if not _is_inf2(instance_type) or not _neuron_devices(instance_type):
        raise ValueError(
            "instance_type must be one of inf2.xlarge, inf2.8xlarge, "
            "inf2.24xlarge, or inf2.48xlarge"
        )


def _device_flags(instance_type: str) -> str:
    return " ".join(
        f"--device=/dev/neuron{i}" for i in range(_neuron_devices(instance_type))
    )


def _user_data(
    model: str,
    instance_type: str,
    serving: str = "vllm",
    source_uri: Optional[str] = None,
) -> str:
    """Render idempotent cloud-init for the chosen serving stack.

    ``vllm`` runs the Neuron vLLM DLC and loads ``model`` from Hugging Face.
    ``optb`` runs the prebuilt Gemma-4 E2B Option-B container; the model is
    baked into the image, so ``model`` is ignored and no HF token is needed.
    ``jax`` runs this repository's pure-JAX engine under systemd — no docker at
    all — from a source bundle in S3; ``source_uri`` is required.
    """
    _validate_instance_type(instance_type)
    if serving == "jax":
        return _jax_user_data(model, instance_type, source_uri)
    if serving == "optb":
        if _neuron_devices(instance_type) != 1:
            raise ValueError(
                "serving='optb' is a single-device (2-core) build; use "
                "inf2.xlarge or inf2.8xlarge"
            )
        # The one-time neff load peaks ~14.5 GB of host RAM; without swap the
        # 16 GB inf2.xlarge OOM-kills the container and the SSM agent.
        return f"""#!/usr/bin/env bash
set -euxo pipefail
if ! swapon --show --noheadings | grep -q /swapfile; then
  fallocate -l 16G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
systemctl enable --now docker
mkdir -p /opt/vllm-inf2
cat >/opt/vllm-inf2/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
docker rm -f vllm-neuron 2>/dev/null || true
docker run -d --name vllm-neuron --restart unless-stopped --ipc=host \
  --device=/dev/neuron0 \
  -p {VLLM_PORT}:8080 \
  {OPTB_IMAGE}
SCRIPT
chmod 700 /opt/vllm-inf2/start.sh
/opt/vllm-inf2/start.sh
"""
    if serving != "vllm":
        raise ValueError("serving must be 'vllm', 'optb', or 'jax'")
    devices = _device_flags(instance_type)
    tp = _neuron_cores(instance_type)
    return f"""#!/usr/bin/env bash
set -euxo pipefail
systemctl enable --now docker
mkdir -p /opt/vllm-inf2
cat >/opt/vllm-inf2/start.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
HF_TOKEN=$(aws secretsmanager get-secret-value --region {AWS_REGION!r} --secret-id {HF_SECRET_ID!r} --query SecretString --output text 2>/dev/null || true)
docker rm -f vllm-neuron 2>/dev/null || true
docker run -d --name vllm-neuron --restart unless-stopped --ipc=host \
  {devices} --cap-add SYS_ADMIN --cap-add IPC_LOCK \
  -e HF_TOKEN="$HF_TOKEN" -p {VLLM_PORT}:8000 \
  {VLLM_IMAGE} \
  vllm serve {model} --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size {tp} --max-num-seqs 4 --max-model-len 4096
SCRIPT
chmod 700 /opt/vllm-inf2/start.sh
/opt/vllm-inf2/start.sh
"""


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


async def _resolve_ami(ec2=None) -> str:
    ec2 = ec2 or _client("ec2")
    result = await _call(
        ec2.describe_images,
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [NEURON_AMI_NAME]},
            {"Name": "architecture", "Values": ["x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(result.get("Images", []), key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"No Neuron DLAMI matching {NEURON_AMI_NAME!r} in {AWS_REGION}")
    return images[0]["ImageId"]


async def _instances(name: Optional[str] = None, states: Optional[list[str]] = None):
    filters = [
        {"Name": "tag:ManagedBy", "Values": ["inf2-devops"]},
        {"Name": "instance-state-name", "Values": states or ["pending", "running", "stopping", "stopped"]},
    ]
    if name:
        filters.append({"Name": "tag:Name", "Values": [name]})
    response = await _call(_client("ec2").describe_instances, Filters=filters)
    return [i for r in response.get("Reservations", []) for i in r.get("Instances", [])]


async def _ssm(instance_id: str, command: str, timeout: int = 300) -> str:
    ssm = _client("ssm")
    response = await _call(
        ssm.send_command,
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=timeout,
    )
    command_id = response["Command"]["CommandId"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = await _call(
                ssm.get_command_invocation,
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvocationDoesNotExist":
                await asyncio.sleep(2)
                continue
            raise
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            output = (result.get("StandardOutputContent", "") + result.get("StandardErrorContent", "")).strip()
            if result["Status"] != "Success":
                raise RuntimeError(f"SSM {result['Status']}: {output}")
            return output
        await asyncio.sleep(2)
    raise TimeoutError(f"SSM command did not finish in {timeout}s")


def _error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        detail = exc.response.get("Error", {})
        return f"❌ AWS {detail.get('Code', 'error')}: {detail.get('Message', exc)}"
    if isinstance(exc, BotoCoreError):
        return f"❌ AWS client error: {exc}"
    return f"❌ {exc}"


@mcp.tool(title="Save Hugging Face token", annotations=WRITE)
async def save_hf_token(token: str) -> str:
    """Create or update the configured AWS Secrets Manager secret."""
    secrets = _client("secretsmanager")
    try:
        try:
            await _call(secrets.put_secret_value, SecretId=HF_SECRET_ID, SecretString=token)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            await _call(secrets.create_secret, Name=HF_SECRET_ID, SecretString=token)
        return f"✅ Stored token in Secrets Manager secret `{HF_SECRET_ID}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Generate Inf2 deployment configuration", annotations=READ_ONLY)
async def get_deployment_config(
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    subnet_id: str = "<subnet-id>",
    security_group_id: str = "<security-group-id>",
    iam_instance_profile: str = "inf2-vllm-instance-profile",
    serving: Literal["vllm", "optb", "jax"] = "vllm",
    spot: bool = True,
    source_uri: Optional[str] = None,
) -> str:
    """Return cloud-init and an AWS CLI launch command without changing AWS.

    source_uri is required for serving='jax' (s3://bucket/tpu-jax-inf2.tar.gz)
    and ignored otherwise.
    """
    try:
        _validate_instance_type(instance_type)
        script = _user_data(model_name, instance_type, serving, source_uri)
        encoded = base64.b64encode(script.encode()).decode()
        market = (
            "--instance-market-options 'MarketType=spot,"
            "SpotOptions={SpotInstanceType=one-time}' "
            if spot
            else ""
        )
        return (
            f"### AWS Inferentia2 deployment\n\n```bash\n"
            f"AMI_ID=$(aws ec2 describe-images --region {AWS_REGION} --owners amazon "
            f"--filters 'Name=name,Values={NEURON_AMI_NAME}' 'Name=state,Values=available' "
            "--query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)\n"
            f"aws ec2 run-instances --region {AWS_REGION} --image-id \"$AMI_ID\" "
            f"--instance-type {instance_type} --subnet-id {subnet_id} "
            f"--security-group-ids {security_group_id} "
            f"--iam-instance-profile Name={iam_instance_profile} "
            f"{market}"
            f"--block-device-mappings 'DeviceName=/dev/sda1,Ebs={{VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}}' "
            f"--user-data '{encoded}' --tag-specifications "
            f"'ResourceType=instance,Tags=[{{Key=Name,Value={SERVICE_NAME}}},"
            "{Key=ManagedBy,Value=inf2-devops}]'\n```\n\n"
            f"User data is base64-encoded in the command.\n\n" + (
                f"The bootstrap installs jax-neuronx and runs this repository's "
                f"engine under systemd from `{source_uri}` — no container. The "
                f"compiled graph uses ONE of this instance's "
                f"{_neuron_cores(instance_type)} NeuronCore(s) "
                f"(`--logical-nc-config=1`); the API binds to 127.0.0.1:"
                f"{JAX_PORT}. For the plan/apply workflow with a persistent "
                f"compile-cache volume, use deployments/aws-inf2/deploy.py."
                if serving == "jax" else
                f"It exposes {_neuron_devices(instance_type)} Neuron device(s) / "
                f"{_neuron_cores(instance_type)} NeuronCore(s) to the container."
            )
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Create Inf2 instance", annotations=WRITE)
async def create_inf2_instance(
    subnet_id: str,
    security_group_id: str,
    iam_instance_profile: str,
    name: str = SERVICE_NAME,
    model_name: str = MODEL_NAME,
    instance_type: str = INSTANCE_TYPE,
    serving: Literal["vllm", "optb", "jax"] = "vllm",
    spot: bool = True,
    source_uri: Optional[str] = None,
) -> str:
    """Launch one tagged Inf2 instance using the latest regional Neuron DLAMI.

    serving='vllm' runs the Neuron vLLM DLC with model_name; serving='optb'
    runs the prebuilt Gemma-4 E2B container (model baked in, single device);
    serving='jax' runs this repository's pure-JAX engine under systemd from the
    S3 bundle at source_uri (required for that mode, ignored otherwise).
    Spot is the default; pass spot=False for on-demand.
    """
    try:
        _validate_instance_type(instance_type)
        # Render before the existence check and before any AWS call: for
        # serving='jax' this validates source_uri, the instance type, and that
        # the template is reachable. Failing here costs nothing; failing after
        # run_instances leaves a billing instance running the wrong bootstrap.
        user_data = _user_data(model_name, instance_type, serving, source_uri)
        if await _instances(name):
            return f"❌ A managed instance named `{name}` already exists."
        ec2 = _client("ec2")
        args = {
            "ImageId": await _resolve_ami(ec2),
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "SubnetId": subnet_id,
            "SecurityGroupIds": [security_group_id],
            "IamInstanceProfile": {"Name": iam_instance_profile},
            "UserData": user_data,
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": name}, {"Key": "ManagedBy", "Value": "inf2-devops"}, {"Key": "Serving", "Value": serving}]}],
        }
        if spot:
            args["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}
        response = await _call(ec2.run_instances, **args)
        instance_id = response["Instances"][0]["InstanceId"]
        market = "spot" if spot else "on-demand"
        if serving == "jax":
            return (
                f"✅ Launching `{instance_id}` ({instance_type}, {market}) in "
                f"`{AWS_REGION}` running the pure-JAX engine from `{source_uri}`.\n\n"
                f"No docker. The API binds to 127.0.0.1:{JAX_PORT}, so reach it "
                f"through SSM port forwarding. First start is slow: neuronx-cc "
                f"compiles the decode graph (~20 min at -O1, measured off-device) "
                f"before health passes. Follow with "
                f"`get_vllm_logs(instance_id, serving='jax')`."
            )
        image = OPTB_IMAGE if serving == "optb" else VLLM_IMAGE
        return (
            f"✅ Launching `{instance_id}` ({instance_type}, {market}) in "
            f"`{AWS_REGION}` serving `{image}`."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List managed Inf2 instances", annotations=READ_ONLY)
async def list_inf2_instances() -> str:
    """List instances tagged ManagedBy=inf2-devops."""
    try:
        instances = await _instances()
        if not instances:
            return "No managed Inf2 instances found."
        rows = ["instance_id\tname\ttype\tstate\tprivate_ip\tpublic_ip"]
        for item in instances:
            tags = {t["Key"]: t["Value"] for t in item.get("Tags", [])}
            rows.append("\t".join([
                item["InstanceId"], tags.get("Name", ""), item["InstanceType"],
                item["State"]["Name"], item.get("PrivateIpAddress", "-"),
                item.get("PublicIpAddress", "-"),
            ]))
        return "\n".join(rows)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Stop Inf2 instance", annotations=DESTRUCTIVE)
async def stop_inf2_instance(instance_id: str) -> str:
    """Stop an Inf2 instance, preserving its EBS volume."""
    try:
        await _call(_client("ec2").stop_instances, InstanceIds=[instance_id])
        return f"✅ Stopping `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Start Inf2 instance", annotations=WRITE)
async def start_inf2_instance(instance_id: str) -> str:
    """Start a stopped Inf2 instance."""
    try:
        await _call(_client("ec2").start_instances, InstanceIds=[instance_id])
        return f"✅ Starting `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Terminate Inf2 instance", annotations=DESTRUCTIVE)
async def terminate_inf2_instance(instance_id: str) -> str:
    """Permanently terminate an instance. Root EBS is deleted by default."""
    try:
        await _call(_client("ec2").terminate_instances, InstanceIds=[instance_id])
        return f"✅ Terminating `{instance_id}`."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check Inf2 and Neuron health", annotations=READ_ONLY)
async def verify_neuron_health(
    instance_id: str,
    serving: Literal["vllm", "optb", "jax"] = "vllm",
) -> str:
    """Use SSM to inspect Neuron devices, the serving process, and API health.

    serving='jax' has no container to inspect — the engine runs under systemd —
    so pass it, or the container probe reports a false failure on a healthy host.
    """
    # AWS-RunShellScript executes under sh (dash on Ubuntu), so no bashisms.
    # The SSM agent's PATH omits the Neuron tools directory.
    if serving == "jax":
        command = (
            "PATH=$PATH:/opt/aws/neuron/bin; neuron-ls; "
            f"systemctl is-active {JAX_SERVICE_NAME}.service; "
            f"curl -fsS http://127.0.0.1:{JAX_PORT}/health "
            f"|| curl -fsS http://127.0.0.1:{JAX_PORT}/v1/models"
        )
    else:
        command = (
            "PATH=$PATH:/opt/aws/neuron/bin; neuron-ls; "
            "docker inspect --format '{{.State.Status}}' vllm-neuron; "
            f"curl -fsS http://127.0.0.1:{VLLM_PORT}/health "
            f"|| curl -fsS http://127.0.0.1:{VLLM_PORT}/v1/models"
        )
    try:
        return f"### `{instance_id}` Neuron health ({serving})\n\n```\n{await _ssm(instance_id, command)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get serving logs", annotations=READ_ONLY)
async def get_vllm_logs(
    instance_id: str,
    tail: Annotated[int, Field(ge=1, le=5000)] = 200,
    serving: Literal["vllm", "optb", "jax"] = "vllm",
) -> str:
    """Read bounded serving logs through SSM.

    serving='jax' reads journalctl for the systemd unit; the container modes
    read docker logs. On a JAX host the first minutes are neuronx-cc compiling
    the decode graph, which is expected and not an error.
    """
    try:
        if serving == "jax":
            command = (
                f"journalctl -u {JAX_SERVICE_NAME}.service -n {tail} --no-pager 2>&1"
                f" || tail -n {tail} /var/log/{JAX_SERVICE_NAME}-bootstrap.log"
            )
        else:
            command = f"docker logs --tail {tail} vllm-neuron 2>&1"
        output = await _ssm(instance_id, command)
        return f"```\n{output}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get Inf2 endpoint", annotations=READ_ONLY)
async def get_endpoint(
    instance_id: str,
    serving: Literal["vllm", "optb", "jax"] = "vllm",
) -> str:
    """Resolve the instance addresses and probe the OpenAI-compatible API."""
    try:
        response = await _call(_client("ec2").describe_instances, InstanceIds=[instance_id])
        item = response["Reservations"][0]["Instances"][0]
        host = item.get("PublicIpAddress") or item.get("PrivateIpAddress")
        if serving == "jax":
            # Not a probe failure to report — user_data.sh binds the API to
            # loopback on purpose, so there is nothing here to reach. Probing
            # anyway and reporting "not reachable" would read as a fault.
            return (
                f"`{instance_id}` serves the JAX engine on 127.0.0.1:{JAX_PORT} "
                f"(loopback only, by design — there is no public listener).\n\n"
                f"Forward it:\n```bash\naws ssm start-session --target {instance_id} "
                f"--document-name AWS-StartPortForwardingSession "
                f"--parameters '{{\"portNumber\":[\"{JAX_PORT}\"],"
                f"\"localPortNumber\":[\"{JAX_PORT}\"]}}'\n```\n"
                f"Then the endpoint is `http://127.0.0.1:{JAX_PORT}/v1`. "
                f"Use verify_neuron_health(serving='jax') to check it in place."
            )
        url = f"http://{host}:{VLLM_PORT}"
        healthy = False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                healthy = (await client.get(f"{url}/health")).is_success
                if not healthy:
                    healthy = (await client.get(f"{url}/v1/models")).is_success
        except httpx.HTTPError:
            pass
        return f"Endpoint: `{url}/v1` — {'healthy' if healthy else 'not reachable from this host'}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(
    endpoint: str,
    prompt: str,
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = 256,
    model: str = MODEL_NAME,
) -> str:
    """Send a prompt to an Inf2-hosted OpenAI-compatible endpoint."""
    try:
        client = AsyncOpenAI(base_url=endpoint.rstrip("/") + "/v1", api_key="not-required")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check Inferentia quotas", annotations=READ_ONLY)
async def check_inf2_quotas() -> str:
    """List EC2 On-Demand and Spot Inferentia quota values in the active region."""
    try:
        quotas = await _call(_client("service-quotas").list_service_quotas, ServiceCode="ec2")
        matches = [
            q for q in quotas.get("Quotas", [])
            if "Inf" in q["QuotaName"] and ("On-Demand" in q["QuotaName"] or "Spot" in q["QuotaName"])
        ]
        return "\n".join(f"- {q['QuotaName']}: {q['Value']} vCPUs" for q in matches) or "No Inferentia quotas returned."
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show active AWS/Neuron configuration and operational prerequisites."""
    return (
        "### AWS Inferentia2 DevOps agent\n\n"
        f"- Region: `{AWS_REGION}`\n- Instance type: `{INSTANCE_TYPE}`\n"
        f"- Model: `{MODEL_NAME}`\n- DLC: `{VLLM_IMAGE}`\n- Port: `{VLLM_PORT}`\n"
        f"- Gemma-4 E2B image (serving='optb'): `{OPTB_IMAGE}`\n"
        f"- JAX service (serving='jax'): `{JAX_SERVICE_NAME}` on "
        f"127.0.0.1:{JAX_PORT}\n\n"
        "Launches default to spot capacity (pass spot=False for on-demand).\n\n"
        "**Three serving modes:**\n"
        "- `vllm` — Neuron vLLM DLC. Cannot serve Gemma-4 (`optimum-neuron` has "
        "no model class; it comes up healthy and emits gibberish).\n"
        "- `optb` — prebuilt Gemma-4 E2B `torch_neuronx` container, model baked "
        "in, single Neuron device, no HF token.\n"
        "- `jax` — this repository's pure-JAX engine under systemd. No docker, "
        "no vLLM, no torch_neuronx; the same model code that serves on TPU v6e, "
        "via the jax-neuronx PJRT plugin. Requires `source_uri` (an S3 bundle of "
        "the repo) and a single-device instance. The API binds to loopback; "
        "reach it with SSM port forwarding. Its graphs are known to compile for "
        "inf2 (benchmarks/runs/2026-07-30-neuron-compile-e2b), but have not been "
        "validated for numerics or throughput on a device — treat it as the "
        "porting target it is.\n\n"
        "`verify_neuron_health`, `get_vllm_logs`, and `get_endpoint` all take a "
        "`serving` argument; pass `serving='jax'` for JAX hosts or they will "
        "probe for a docker container that does not exist.\n\n"
        "The EC2 instance profile needs SSM core permissions, ECR Public read access, "
        f"and `secretsmanager:GetSecretValue` for `{HF_SECRET_ID}`. The caller needs "
        "EC2, SSM Run Command, Secrets Manager, and Service Quotas permissions."
    )


if __name__ == "__main__":
    mcp.run()
