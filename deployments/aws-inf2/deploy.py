#!/usr/bin/env python3
"""Safe EC2 planner/launcher for the pure-JAX Gemma 4 Inf2 scaffold."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Any


ACTIVE_STATES = ("pending", "running", "stopping", "stopped")
ROOT_DEVICE = "/dev/sda1"
# Attachment point for the persistent compile-cache volume. Nitro remaps this to
# an NVMe name on the guest; user_data.sh locates it as the non-root disk.
CACHE_DEVICE = "/dev/sdf"
LAUNCH_REQUIRED = (
    "source_uri",
    "subnet_id",
    "security_group_id",
    "instance_profile_name",
)


@dataclass(frozen=True)
class Config:
    region: str
    project: str
    source_uri: str | None
    subnet_id: str | None
    security_group_id: str | None
    instance_profile_name: str | None
    instance_type: str
    market_type: str
    model_id: str
    hf_secret_id: str
    max_model_len: int
    volume_gib: int
    cache_volume_gib: int
    swap_gib: int
    neuron_cc_flags: str
    ami_id: str | None
    # Optional and last so existing callers constructing Config positionally
    # keep working.
    cache_volume_id: str | None = None


def _boto3():
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("boto3 is required: python3 -m pip install boto3") from exc
    return boto3


def render_user_data(config: Config) -> str:
    template = (Path(__file__).with_name("user_data.sh")).read_text()
    substitutions = {
        "__SOURCE_URI__": config.source_uri,
        "__MODEL_ID__": config.model_id,
        "__HF_SECRET_ID__": config.hf_secret_id,
        "__AWS_REGION__": config.region,
        "__MAX_MODEL_LEN__": str(config.max_model_len),
        "__SWAP_GIB__": str(config.swap_gib),
        "__NEURON_CC_FLAGS__": config.neuron_cc_flags,
    }
    for marker, value in substitutions.items():
        template = template.replace(marker, shlex.quote(value))
    if "__" in template:
        raise ValueError("Unresolved user-data template marker")
    return template


def resolve_ami(ec2: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {
                "Name": "name",
                "Values": [
                    "*Deep Learning Base Neuron AMI*Ubuntu 22.04*",
                    "*Deep Learning AMI Neuron*Ubuntu 22.04*",
                ],
            },
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(
        response.get("Images", []),
        key=lambda image: image.get("CreationDate", ""),
        reverse=True,
    )
    if not images:
        raise RuntimeError("No Neuron DLAMI found; pass --ami-id explicitly")
    return images[0]["ImageId"]


def existing_hosts(ec2: Any, project: str) -> list[dict[str, Any]]:
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [project]},
            {"Name": "instance-state-name", "Values": list(ACTIVE_STATES)},
        ]
    )
    return [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]


def cache_volume_id(instance: dict[str, Any]) -> str | None:
    for mapping in instance.get("BlockDeviceMappings", []):
        if mapping.get("DeviceName") == CACHE_DEVICE:
            return mapping.get("Ebs", {}).get("VolumeId")
    return None


def launch(config: Config, apply: bool) -> dict[str, Any]:
    missing = [name for name in LAUNCH_REQUIRED if getattr(config, name) is None]
    if missing:
        raise ValueError(
            "launch requires " + ", ".join("--" + n.replace("_", "-") for n in missing)
        )

    boto3 = _boto3()
    ec2 = boto3.client("ec2", region_name=config.region)
    hosts = existing_hosts(ec2, config.project)
    if hosts:
        ids = ", ".join(host["InstanceId"] for host in hosts)
        raise RuntimeError(
            f"Project {config.project!r} already has an EC2 host in "
            f"{config.region}: {ids}"
        )

    ami_id = resolve_ami(ec2, config.ami_id)

    # A retained cache volume can be reattached instead of provisioning a blank
    # one. It holds the weight download and the Neuron/XLA compile caches, so
    # reuse turns a ~20 minute cold start into a ~2 minute one. EBS cannot cross
    # an Availability Zone, so the volume pins the subnet -- check that here
    # rather than letting run_instances fail after the request is built.
    reuse_cache = config.cache_volume_id is not None
    if reuse_cache:
        vol = ec2.describe_volumes(VolumeIds=[config.cache_volume_id])["Volumes"][0]
        if vol["State"] != "available":
            raise RuntimeError(
                f"cache volume {config.cache_volume_id} is {vol['State']}, not "
                "available; detach it from its current host first"
            )
        subnet_az = ec2.describe_subnets(
            SubnetIds=[config.subnet_id])["Subnets"][0]["AvailabilityZone"]
        if vol["AvailabilityZone"] != subnet_az:
            raise RuntimeError(
                f"cache volume {config.cache_volume_id} is in "
                f"{vol['AvailabilityZone']} but --subnet-id is in {subnet_az}; "
                "EBS volumes cannot cross an Availability Zone"
            )

    request: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": config.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": config.subnet_id,
        "SecurityGroupIds": [config.security_group_id],
        "IamInstanceProfile": {"Name": config.instance_profile_name},
        "UserData": render_user_data(config),
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
        },
        "BlockDeviceMappings": [
            # The root volume is rebuilt from the AMI on every launch, so
            # retaining it would only orphan a paid volume nothing reattaches.
            {
                "DeviceName": ROOT_DEVICE,
                "Ebs": {
                    "VolumeSize": config.volume_gib,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                },
            },
            # The Neuron/XLA compile caches do survive: recompiling the Gemma
            # graph from cold costs far more than the idle volume does.
            {
                "DeviceName": CACHE_DEVICE,
                "Ebs": {
                    "VolumeSize": config.cache_volume_gib,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": False,
                    "Encrypted": True,
                },
            },
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": config.project},
                    {"Key": "Project", "Value": config.project},
                    {"Key": "Workload", "Value": "gemma4-jax-inf2"},
                ],
            },
            {
                "ResourceType": "volume",
                # run_instances cannot tag per-device, so this covers both.
                "Tags": [
                    {"Key": "Name", "Value": f"{config.project}-vol"},
                    {"Key": "Project", "Value": config.project},
                ],
            },
        ],
    }
    if reuse_cache:
        # Drop the blank cache volume from the mapping; the retained one is
        # attached after the instance reaches `running` (attach_volume cannot
        # run against a pending instance).
        #
        # NoDevice, not merely absent: an AMI captured from a host that had a
        # cache volume carries its own CACHE_DEVICE mapping, so leaving it out
        # of the request silently restores a volume from the AMI's snapshot and
        # AttachVolume then fails with "attachment point already in use".
        # Suppressing it also avoids the restore itself, which is worth avoiding
        # on its own: a snapshot-restored volume lazy-loads from S3 on first
        # touch -- measured at 6 MB/s here, so reading back a 7.8 GB weight
        # cache takes ~22 minutes, slower than downloading the weights fresh.
        # The retained volume has no such penalty.
        request["BlockDeviceMappings"] = [
            m for m in request["BlockDeviceMappings"] if m["DeviceName"] != CACHE_DEVICE
        ] + [{"DeviceName": CACHE_DEVICE, "NoDevice": ""}]

    if config.market_type == "spot":
        request["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    plan = {
        "action": "launch" if apply else "plan",
        "region": config.region,
        "ami_id": ami_id,
        "ami_source": "explicit" if config.ami_id else "auto-discovered (SDK line NOT pinned)",
        "instance_type": config.instance_type,
        "market_type": config.market_type,
        "project": config.project,
        "source_uri": config.source_uri,
        "root_volume_gib": config.volume_gib,
        "cache_volume_gib": config.cache_volume_gib,
        "cache_volume_retained_on_terminate": True,
        "cache_volume_reused": config.cache_volume_id or "no (fresh blank volume)",
        "swap_gib": config.swap_gib,
        "neuron_cc_flags": config.neuron_cc_flags,
        "api_exposure": "127.0.0.1:8000 (SSM/private proxy required)",
    }
    if not apply:
        return plan

    response = ec2.run_instances(**request)
    instance = response["Instances"][0]
    instance_id = instance["InstanceId"]

    if reuse_cache:
        # user_data mounts any already-formatted non-root disk it finds and
        # skips mkfs, so the attach only has to win the race against the
        # bootstrap reaching that step -- which it does, since the bootstrap
        # spends its first stretch on apt.
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        ec2.attach_volume(VolumeId=config.cache_volume_id,
                          InstanceId=instance_id, Device=CACHE_DEVICE)

    return {**plan, "instance_id": instance_id, "state": "pending"}


def terminate(config: Config, apply: bool) -> dict[str, Any]:
    boto3 = _boto3()
    ec2 = boto3.client("ec2", region_name=config.region)
    hosts = existing_hosts(ec2, config.project)
    if not hosts:
        raise RuntimeError(
            f"No active EC2 host tagged Project={config.project!r} in {config.region}"
        )

    targets = [
        {"instance_id": host["InstanceId"], "cache_volume_id": cache_volume_id(host)}
        for host in hosts
    ]
    plan = {
        "action": "terminate" if apply else "plan",
        "region": config.region,
        "project": config.project,
        "targets": targets,
        # Retained volumes bill until deleted; the launcher never reattaches
        # them, so reuse is a manual attach and deletion is a manual choice.
        "note": "cache volumes are retained and keep billing; delete or reattach them yourself",
    }
    if not apply:
        return plan

    ec2.terminate_instances(
        InstanceIds=[target["instance_id"] for target in targets]
    )
    return {**plan, "state": "shutting-down"}


def parse_args() -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "launch", "terminate"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--region", required=True)
    parser.add_argument("--project", default="gemma4-jax-inf2")
    # Required for plan/launch, irrelevant to terminate; enforced in launch().
    parser.add_argument("--source-uri")
    parser.add_argument("--subnet-id")
    parser.add_argument("--security-group-id")
    parser.add_argument("--instance-profile-name")
    parser.add_argument("--instance-type", default="inf2.xlarge")
    parser.add_argument("--market-type", choices=("on-demand", "spot"), default="on-demand")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it-qat-w4a16-ct")
    parser.add_argument("--hf-secret-id", default="hf-token")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--volume-gib", type=int, default=200)
    parser.add_argument("--cache-volume-gib", type=int, default=200)
    # Reattach a retained cache volume (weights + Neuron/XLA compile caches)
    # instead of provisioning a blank one. Must be `available` and in the same
    # AZ as --subnet-id. `terminate` prints the id of the volume it kept.
    parser.add_argument("--cache-volume-id")
    # A 16 GiB inf2.xlarge host OOM-kills the Neuron graph load without swap.
    parser.add_argument("--swap-gib", type=int, default=32)
    parser.add_argument("--neuron-cc-flags", default="--model-type=transformer")
    parser.add_argument("--ami-id")
    args = parser.parse_args()
    if args.apply and args.command not in ("launch", "terminate"):
        parser.error("--apply is only valid with launch or terminate")
    config = Config(**{
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "apply"}
    })
    return args, config


def main() -> None:
    args, config = parse_args()
    # Every command stays a dry-run unless --apply is deliberately supplied.
    try:
        if args.command == "terminate":
            result = terminate(config, apply=args.apply)
        else:
            result = launch(config, apply=args.command == "launch" and args.apply)
    except (ValueError, RuntimeError) as exc:
        # These are operator errors, not crashes; a traceback only hides them.
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
