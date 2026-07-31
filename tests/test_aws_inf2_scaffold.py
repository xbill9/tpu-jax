"""Unit tests for the AWS Inf2 deployment scaffold (no AWS calls)."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deployments" / "aws-inf2"


def load(name, filename):
    path = DEPLOY_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy = load("aws_inf2_deploy", "deploy.py")
# Safe to import without JAX installed: the jax import lives inside
# verify_neuron(), not at module scope.
entrypoint = load("aws_inf2_entrypoint", "neuron_entrypoint.py")


def config(**overrides):
    values = dict(
        region="us-east-1",
        project="test-gemma",
        source_uri="s3://bucket/source.tar.gz",
        subnet_id="subnet-1",
        security_group_id="sg-1",
        instance_profile_name="profile",
        instance_type="inf2.xlarge",
        market_type="on-demand",
        model_id="google/gemma-4-E2B-it-qat-w4a16-ct",
        hf_secret_id="hf-token",
        max_model_len=4096,
        volume_gib=200,
        cache_volume_gib=200,
        swap_gib=32,
        neuron_cc_flags="--model-type=transformer",
        ami_id="ami-test",
    )
    values.update(overrides)
    return deploy.Config(**values)


class FakeEc2:
    def __init__(self, instances=None):
        self.instances = instances or []
        self.run_request = None
        self.terminated = None

    def describe_instances(self, **_kwargs):
        return {"Reservations": [{"Instances": self.instances}]}

    def run_instances(self, **kwargs):
        self.run_request = kwargs
        return {"Instances": [{"InstanceId": "i-new"}]}

    def terminate_instances(self, **kwargs):
        self.terminated = kwargs
        return {"TerminatingInstances": []}


def patched(ec2):
    fake_boto3 = mock.Mock()
    fake_boto3.client.return_value = ec2
    return mock.patch.object(deploy, "_boto3", return_value=fake_boto3)


def devices(request):
    return {
        mapping["DeviceName"]: mapping["Ebs"]
        for mapping in request["BlockDeviceMappings"]
    }


class Inf2ScaffoldTests(unittest.TestCase):
    def test_user_data_quotes_values_and_has_no_token(self):
        rendered = deploy.render_user_data(
            config(model_id="model with spaces; unsafe", hf_secret_id="secret/name")
        )
        self.assertIn("'model with spaces; unsafe'", rendered)
        self.assertIn("aws secretsmanager get-secret-value", rendered)
        self.assertNotIn("hf_actual_secret_value", rendered)
        self.assertIn('HF_TOKEN="$(cat /run/gemma4-hf-token)"', rendered)
        self.assertNotIn("__MODEL_ID__", rendered)

    def test_refuses_second_tagged_host(self):
        ec2 = FakeEc2([{"InstanceId": "i-existing"}])
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "i-existing"):
                deploy.launch(config(), apply=True)
        self.assertIsNone(ec2.run_request)

    def test_plan_does_not_launch(self):
        ec2 = FakeEc2()
        with patched(ec2):
            result = deploy.launch(config(), apply=False)
        self.assertEqual(result["action"], "plan")
        self.assertIsNone(ec2.run_request)

    def test_launch_requires_networking_arguments(self):
        ec2 = FakeEc2()
        with patched(ec2):
            with self.assertRaisesRegex(ValueError, "--subnet-id"):
                deploy.launch(config(subnet_id=None), apply=False)

    def test_apply_enforces_imdsv2_and_spot_choice(self):
        ec2 = FakeEc2()
        with patched(ec2):
            result = deploy.launch(config(market_type="spot"), apply=True)
        self.assertEqual(result["instance_id"], "i-new")
        self.assertEqual(ec2.run_request["MetadataOptions"]["HttpTokens"], "required")
        self.assertEqual(
            ec2.run_request["InstanceMarketOptions"]["MarketType"], "spot"
        )

    def test_root_volume_is_disposable_and_cache_volume_persists(self):
        ec2 = FakeEc2()
        with patched(ec2):
            deploy.launch(config(), apply=True)
        mappings = devices(ec2.run_request)
        # A retained root volume is a pure cost leak: nothing ever reattaches it.
        self.assertTrue(mappings[deploy.ROOT_DEVICE]["DeleteOnTermination"])
        self.assertFalse(mappings[deploy.CACHE_DEVICE]["DeleteOnTermination"])
        self.assertTrue(mappings[deploy.CACHE_DEVICE]["Encrypted"])

    def test_auto_discovered_ami_is_flagged_as_unpinned(self):
        ec2 = FakeEc2()
        with patched(ec2):
            explicit = deploy.launch(config(), apply=False)
        self.assertEqual(explicit["ami_source"], "explicit")

    def test_terminate_plan_reports_retained_cache_volume(self):
        host = {
            "InstanceId": "i-existing",
            "BlockDeviceMappings": [
                {"DeviceName": deploy.ROOT_DEVICE, "Ebs": {"VolumeId": "vol-root"}},
                {"DeviceName": deploy.CACHE_DEVICE, "Ebs": {"VolumeId": "vol-cache"}},
            ],
        }
        ec2 = FakeEc2([host])
        with patched(ec2):
            result = deploy.terminate(config(), apply=False)
        self.assertEqual(result["action"], "plan")
        self.assertEqual(result["targets"][0]["cache_volume_id"], "vol-cache")
        self.assertIsNone(ec2.terminated)

    def test_terminate_apply_terminates_only_tagged_hosts(self):
        ec2 = FakeEc2([{"InstanceId": "i-existing"}])
        with patched(ec2):
            result = deploy.terminate(config(), apply=True)
        self.assertEqual(result["state"], "shutting-down")
        self.assertEqual(ec2.terminated["InstanceIds"], ["i-existing"])

    def test_terminate_without_a_host_is_an_error(self):
        ec2 = FakeEc2()
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "No active EC2 host"):
                deploy.terminate(config(), apply=True)
        self.assertIsNone(ec2.terminated)


class NeuronEntrypointTests(unittest.TestCase):
    def test_pallas_is_refused_by_the_engine_not_routed_to_the_interpreter(self):
        """The entrypoint must NOT set JAX_E_PALLAS_INTERPRET.

        It used to, on the reasoning that the interpreter is better than a hard
        failure. It is not: the Pallas interpreter traces the fused W4A16 kernel
        body into the enclosing graph, unrolling its K loop per tile, so the
        "fallback" silently produces a far worse graph than the reference path.
        `ports/gemma4/backend.py` now reports that Neuron has no Pallas backend
        and `set_w4a16_impl` raises on "fused", which fails loudly at
        configuration time instead of quietly at serving time.
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            entrypoint.configure_neuron()
            self.assertNotIn("JAX_E_PALLAS_INTERPRET", os.environ)
            # "neuron,cpu": quantize_ple_table needs a host device for the
            # 4.70 GB per-layer embedding table.
            self.assertEqual(os.environ["JAX_PLATFORMS"], "neuron,cpu")
            self.assertEqual(os.environ["JAX_DEFAULT_PRNG_IMPL"], "rbg")

    def test_entrypoint_does_not_pin_the_engine_platform(self):
        """JAX_E_PLATFORM is a testing override; on a real host it must be unset
        so detection reads the actual PJRT device."""
        with mock.patch.dict(os.environ, {}, clear=True):
            entrypoint.configure_neuron()
            self.assertNotIn("JAX_E_PLATFORM", os.environ)

    def test_configure_neuron_respects_a_preset_platform(self):
        with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=True):
            entrypoint.configure_neuron()
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")


@unittest.skipIf(shutil.which("bash") is None, "bash not available")
class UserDataShellTests(unittest.TestCase):
    def rendered(self):
        return deploy.render_user_data(config())

    def test_rendered_user_data_is_valid_bash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
            handle.write(self.rendered())
            path = handle.name
        try:
            result = subprocess.run(
                ["bash", "-n", path], capture_output=True, text=True
            )
        finally:
            os.unlink(path)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_token_is_handed_to_the_service_user(self):
        text = self.rendered()
        # ExecStartPre=+ runs as root; ExecStart runs as ubuntu. Without the
        # chown the unit crash-loops on a permission error it never logs well.
        self.assertIn("ExecStartPre=+/usr/local/bin/gemma4-fetch-hf-token", text)
        self.assertIn('chown ubuntu:ubuntu "$tmp"', text)
        self.assertIn("User=ubuntu", text)

    def test_swap_is_provisioned_before_the_memory_hungry_steps(self):
        text = self.rendered()
        # The 16 GiB inf2.xlarge host OOM-kills the Neuron graph load without
        # swap, taking the SSM agent with it -- and there is no inbound SSH.
        swap = text.index("mkswap /swapfile")
        self.assertLess(swap, text.index("pip install"))
        self.assertLess(swap, text.index("systemctl enable"))
        self.assertIn("'/swapfile none swap sw 0 0'", text)

    def test_swap_can_be_disabled_on_a_large_host(self):
        self.assertIn('SWAP_GIB=0\n', deploy.render_user_data(config(swap_gib=0)))

    def test_compiler_flags_reach_the_service_environment(self):
        rendered = deploy.render_user_data(
            config(neuron_cc_flags="--model-type=transformer --target=inf2")
        )
        self.assertIn(
            "NEURON_CC_FLAGS_VALUE='--model-type=transformer --target=inf2'", rendered
        )
        self.assertIn("NEURON_CC_FLAGS=$NEURON_CC_FLAGS_VALUE", rendered)

    def test_entrypoint_defers_to_the_deployed_compiler_flags(self):
        # user_data puts NEURON_CC_FLAGS in the unit's EnvironmentFile, so the
        # entrypoint must not clobber it -- setdefault, never assignment.
        with mock.patch.dict(os.environ, {"NEURON_CC_FLAGS": "--target=inf2"}, clear=True):
            entrypoint.configure_neuron()
            self.assertEqual(os.environ["NEURON_CC_FLAGS"], "--target=inf2")

    def test_cache_volume_is_mounted_before_cache_dirs_are_created(self):
        text = self.rendered()
        self.assertLess(text.index('mount "$CACHE_ROOT"'), text.index("/huggingface"))
        # A reattached volume already holds the compile caches.
        self.assertIn('blkid "$cache_dev"', text)


if __name__ == "__main__":
    unittest.main()
