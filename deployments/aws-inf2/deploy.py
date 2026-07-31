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
    # Defaulted so existing callers (and the MCP server, which builds a Config
    # to render user-data) keep working unchanged. The safe default is the old
    # behaviour: cold start, retain the volume.
    cache_volume_id: str | None = None
    reuse_cache: bool = False
    delete_cache: bool = False


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


def subnet_az(ec2: Any, subnet_id: str) -> str:
    subnets = ec2.describe_subnets(SubnetIds=[subnet_id]).get("Subnets", [])
    if not subnets:
        raise RuntimeError(f"Subnet {subnet_id} not found")
    return subnets[0]["AvailabilityZone"]


def find_reusable_cache_volume(ec2: Any, project: str, az: str) -> str | None:
    """An unattached cache volume for this project in the launch AZ, if exactly one.

    EBS volumes are AZ-locked, so the AZ filter is a correctness constraint and
    not a preference. Returns None when there is nothing to reuse; raises when
    the choice is ambiguous rather than picking arbitrarily — attaching the
    wrong cache is worse than a cold start, because it looks like it worked.
    """
    response = ec2.describe_volumes(
        Filters=[
            {"Name": "tag:Project", "Values": [project]},
            {"Name": "status", "Values": ["available"]},
            {"Name": "availability-zone", "Values": [az]},
        ]
    )
    volumes = sorted(
        response.get("Volumes", []),
        key=lambda v: v.get("CreateTime"),
        reverse=True,
    )
    if not volumes:
        return None
    if len(volumes) > 1:
        ids = ", ".join(v["VolumeId"] for v in volumes)
        raise RuntimeError(
            f"{len(volumes)} reusable cache volumes for project {project!r} in {az}: "
            f"{ids}. Pass --cache-volume-id to choose, or delete the extras."
        )
    return volumes[0]["VolumeId"]


def resolve_cache_volume(ec2: Any, config: Config, az: str) -> str | None:
    """The existing volume to attach after launch, or None to create a fresh one."""
    if config.cache_volume_id:
        volumes = ec2.describe_volumes(VolumeIds=[config.cache_volume_id]).get("Volumes", [])
        if not volumes:
            raise RuntimeError(f"Cache volume {config.cache_volume_id} not found")
        volume = volumes[0]
        if volume["AvailabilityZone"] != az:
            raise RuntimeError(
                f"Cache volume {config.cache_volume_id} is in "
                f"{volume['AvailabilityZone']} but the subnet is in {az}. EBS volumes "
                "cannot cross an Availability Zone — launch into that AZ, or omit "
                "--cache-volume-id to start cold."
            )
        if volume["State"] != "available":
            raise RuntimeError(
                f"Cache volume {config.cache_volume_id} is {volume['State']}, not "
                "available; it is probably still attached to another host."
            )
        return config.cache_volume_id
    if config.reuse_cache:
        return find_reusable_cache_volume(ec2, config.project, az)
    return None


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

    az = subnet_az(ec2, config.subnet_id)
    reuse_volume = resolve_cache_volume(ec2, config, az)

    ami_id = resolve_ami(ec2, config.ami_id)
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
        # Only the root volume is described here. RunInstances cannot attach an
        # EXISTING volume — BlockDeviceMappings only ever creates new ones — so
        # the cache volume is attached separately below, whether reused or fresh.
        # Creating it here is what silently minted a new 150 GB volume on every
        # launch while the previous one sat unattached and billing.
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
    if config.market_type == "spot":
        request["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    plan = {
        "action": "launch" if apply else "plan",
        "region": config.region,
        "availability_zone": az,
        "ami_id": ami_id,
        "ami_source": "explicit" if config.ami_id else "auto-discovered (SDK line NOT pinned)",
        "instance_type": config.instance_type,
        "market_type": config.market_type,
        "project": config.project,
        "source_uri": config.source_uri,
        "root_volume_gib": config.volume_gib,
        "cache_volume": (
            {"action": "reuse", "volume_id": reuse_volume,
             "effect": "skips the checkpoint download and the NEFF compile"}
            if reuse_volume else
            {"action": "create", "size_gib": config.cache_volume_gib,
             "effect": "cold start: full download and compile"}
        ),
        "cache_volume_retained_on_terminate": True,
        "swap_gib": config.swap_gib,
        "neuron_cc_flags": config.neuron_cc_flags,
        "api_exposure": "127.0.0.1:8000 (SSM/private proxy required)",
    }
    if not apply:
        return plan

    response = ec2.run_instances(**request)
    instance = response["Instances"][0]
    instance_id = instance["InstanceId"]

    volume_id = reuse_volume
    if volume_id is None:
        volume_id = ec2.create_volume(
            AvailabilityZone=az,
            Size=config.cache_volume_gib,
            VolumeType="gp3",
            Encrypted=True,
            TagSpecifications=[{
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "Name", "Value": f"{config.project}-cache"},
                    {"Key": "Project", "Value": config.project},
                ],
            }],
        )["VolumeId"]
        ec2.get_waiter("volume_available").wait(VolumeIds=[volume_id])

    # Attach cannot proceed until the instance leaves `pending`. user_data.sh
    # waits up to CACHE_WAIT_SECS for the device, so a slow attach costs a pause
    # rather than a silent fall back onto the ephemeral root volume.
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    ec2.attach_volume(Device=CACHE_DEVICE, InstanceId=instance_id, VolumeId=volume_id)

    return {
        **plan,
        "instance_id": instance_id,
        "state": "running",
        "cache_volume_attached": volume_id,
    }


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
        "cache_volumes": "delete" if config.delete_cache else "retain",
        "note": (
            "cache volumes will be DELETED; the next launch downloads the "
            "checkpoint and recompiles from cold"
            if config.delete_cache else
            "cache volumes are retained and keep billing (~$0.08/GiB-month). "
            "Relaunch with --reuse-cache to pick them up, or terminate with "
            "--delete-cache to stop the charge."
        ),
    }
    if not apply:
        return plan

    ec2.terminate_instances(
        InstanceIds=[target["instance_id"] for target in targets]
    )

    deleted = []
    if config.delete_cache:
        volume_ids = [t["cache_volume_id"] for t in targets if t["cache_volume_id"]]
        if volume_ids:
            # A volume cannot be deleted while it is still attached, and
            # terminate_instances returns before the detach completes.
            ec2.get_waiter("instance_terminated").wait(
                InstanceIds=[t["instance_id"] for t in targets]
            )
        for volume_id in volume_ids:
            ec2.delete_volume(VolumeId=volume_id)
            deleted.append(volume_id)

    return {**plan, "state": "shutting-down", "deleted_volumes": deleted}


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
    # A 16 GiB inf2.xlarge host OOM-kills the Neuron graph load without swap.
    parser.add_argument("--swap-gib", type=int, default=32)
    parser.add_argument("--neuron-cc-flags", default="--model-type=transformer")
    parser.add_argument("--ami-id")
    # Reusing the cache volume is the single biggest startup win: it carries the
    # ~9.6 GB checkpoint and the Neuron compile cache, which together dominate a
    # cold start. EBS is AZ-locked, so the volume's AZ constrains --subnet-id.
    parser.add_argument("--cache-volume-id",
                        help="Attach this existing volume instead of creating one.")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Attach an available volume tagged Project=<project> "
                             "in the launch AZ, if exactly one exists.")
    parser.add_argument("--delete-cache", action="store_true",
                        help="terminate only: also delete the cache volumes, "
                             "which otherwise keep billing.")
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
