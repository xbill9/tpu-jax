"""The inf2-devops MCP server's serving='jax' mode (no AWS calls).

The JAX mode is the one serving path in that server with no container behind
it: the host installs jax-neuronx and runs this repository's engine under
systemd. That makes two things easy to get wrong and expensive to discover on a
running instance, so both are pinned here:

  1. The cloud-init must be the SAME file `deployments/aws-inf2/deploy.py`
     renders. Two bootstraps that drift produce a host that comes up healthy
     serving something other than what was reviewed.
  2. The observability tools must not probe for a docker container that a JAX
     host does not have, or a healthy host reads as broken.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / ".claude" / "skills" / "inf2-management" / "mcp" / "server.py"
DEPLOY_DIR = ROOT / "deployments" / "aws-inf2"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server = _load("inf2_mcp_server", SERVER)
deploy = _load("inf2_deploy_for_mcp", DEPLOY_DIR / "deploy.py")

SOURCE_URI = "s3://bucket/tpu-jax-inf2.tar.gz"
MODEL = "google/gemma-4-E2B-it-qat-q4_0-unquantized"


class DeployTemplateIsSharedTests(unittest.TestCase):
    def test_mcp_renders_byte_identical_user_data_to_deploy_py(self):
        """One template, two callers. This is the whole point of reading the
        file instead of embedding a copy in the server."""
        from_mcp = server._user_data(MODEL, "inf2.xlarge", "jax", SOURCE_URI)
        from_deploy = deploy.render_user_data(deploy.Config(
            region=server.AWS_REGION,
            project="p",
            source_uri=SOURCE_URI,
            subnet_id="subnet-1",
            security_group_id="sg-1",
            instance_profile_name="prof",
            instance_type="inf2.xlarge",
            market_type="spot",
            model_id=MODEL,
            hf_secret_id=server.HF_SECRET_ID,
            max_model_len=server.JAX_MAX_MODEL_LEN,
            volume_gib=200,
            cache_volume_gib=200,
            swap_gib=server.JAX_SWAP_GIB,
            neuron_cc_flags=server.JAX_NEURON_CC_FLAGS,
            ami_id=None,
        ))
        self.assertEqual(from_mcp, from_deploy)

    def test_deploy_dir_is_found_by_walking_up(self):
        self.assertEqual(server._jax_deploy_dir().resolve(), DEPLOY_DIR.resolve())

    def test_no_markers_survive_rendering(self):
        self.assertNotIn("__", server._user_data(MODEL, "inf2.xlarge", "jax", SOURCE_URI))


class JaxModeGuardrailTests(unittest.TestCase):
    def test_source_uri_is_required(self):
        """Nothing is baked into an image; without the bundle the host has no
        engine to run, and the instance would bill while failing silently."""
        with self.assertRaises(ValueError) as ctx:
            server._user_data(MODEL, "inf2.xlarge", "jax", None)
        self.assertIn("source_uri", str(ctx.exception))

    def test_multi_device_instances_are_refused(self):
        """The compiled graph is --logical-nc-config=1; a 24xlarge would leave
        11 of 12 devices idle while looking like a bigger deployment."""
        with self.assertRaises(ValueError) as ctx:
            server._user_data(MODEL, "inf2.24xlarge", "jax", SOURCE_URI)
        self.assertIn("single-NeuronCore", str(ctx.exception))

    def test_jax_bootstrap_uses_systemd_and_never_docker(self):
        script = server._user_data(MODEL, "inf2.xlarge", "jax", SOURCE_URI)
        self.assertIn(f"{server.JAX_SERVICE_NAME}.service", script)
        self.assertIn("neuron_entrypoint.py", script)
        self.assertNotIn("docker", script.lower())

    def test_unknown_serving_mode_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            server._user_data(MODEL, "inf2.xlarge", "nxd")
        self.assertIn("jax", str(ctx.exception))


class ContainerModesUnchangedTests(unittest.TestCase):
    """Adding a third mode must not disturb the two that already worked."""

    def test_vllm_and_optb_still_render_docker_bootstraps(self):
        for mode in ("vllm", "optb"):
            script = server._user_data("meta-llama/Llama-3.1-8B-Instruct",
                                       "inf2.xlarge", mode)
            self.assertIn("docker run", script, f"{mode} lost its container")

    def test_optb_still_refuses_multi_device(self):
        with self.assertRaises(ValueError):
            server._user_data("m", "inf2.24xlarge", "optb")


class ServingDependencyTests(unittest.TestCase):
    """The bootstrap must install what the server actually imports.

    It originally installed the repo-root requirements.txt — the MCP server's
    dependencies — which contains none of the serving stack. That host boots,
    installs cleanly, reports success, and dies on `import fastapi` before ever
    touching the model: a failure that costs an instance-hour to discover and
    looks like a Neuron problem.
    """

    REQUIREMENTS = DEPLOY_DIR / "requirements-serving.txt"

    def _pinned(self):
        names = set()
        for line in self.REQUIREMENTS.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(line.split("[")[0].split("==")[0].split(">")[0].strip())
        return names

    def test_user_data_installs_the_serving_requirements(self):
        script = (DEPLOY_DIR / "user_data.sh").read_text()
        self.assertIn("requirements-serving.txt", script)
        self.assertNotIn("-r /opt/gemma4/app/requirements.txt", script)

    def test_every_third_party_import_of_the_server_is_covered(self):
        """Scanned from the source rather than listed by hand, so a new import
        in the server fails here instead of on the instance."""
        import ast

        stdlib = set(sys.stdlib_module_names)
        local = {"ports", "jax_engine", "jax_openai_server"}
        # jax/jaxlib come from the Neuron index via jax-neuronx, not from this file.
        from_neuron_index = {"jax", "jaxlib"}
        provided = self._pinned() | from_neuron_index

        missing = {}
        for rel in ("jax_openai_server.py", "jax_engine.py",
                    "ports/gemma4/jax_e_loader.py", "ports/gemma4/jax_e_model.py"):
            tree = ast.parse((ROOT / rel).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                else:
                    continue
                for mod in mods:
                    top = mod.split(".")[0]
                    if not top or top in stdlib or top in local or top in provided:
                        continue
                    missing.setdefault(top, rel)
        self.assertEqual(missing, {},
                         f"imports not covered by requirements-serving.txt: {missing}")

    def test_jax_is_not_pinned_from_pypi(self):
        """A PyPI jaxlib has no Neuron PJRT plugin. It would install fine and
        then report 'no Neuron device found' at runtime."""
        self.assertNotIn("jax", self._pinned())
        self.assertNotIn("jaxlib", self._pinned())


class ServingAwareToolTests(unittest.TestCase):
    """The health/log/endpoint tools branch on `serving`. Checked through the
    FastMCP tool objects so the exposed signature is what gets tested."""

    @staticmethod
    def _fn(name):
        # FastMCP's @mcp.tool registers and returns the original function here;
        # older/newer versions hand back a wrapper carrying `.fn`. Accept both
        # so the test pins the server's behaviour, not the decorator's.
        tool = getattr(server, name)
        return getattr(tool, "fn", tool)

    def test_health_probes_systemd_for_jax_and_docker_otherwise(self):
        import inspect

        source = inspect.getsource(self._fn("verify_neuron_health"))
        self.assertIn("systemctl is-active", source)
        self.assertIn("docker inspect", source)

    def test_serving_argument_exists_on_every_host_facing_tool(self):
        import inspect

        for name in ("verify_neuron_health", "get_vllm_logs", "get_endpoint",
                     "create_inf2_instance", "get_deployment_config"):
            params = inspect.signature(self._fn(name)).parameters
            self.assertIn("serving", params, f"{name} cannot target a JAX host")

    def test_jax_is_a_valid_choice_everywhere_serving_is_accepted(self):
        import inspect
        import typing

        for name in ("verify_neuron_health", "get_vllm_logs", "get_endpoint",
                     "create_inf2_instance", "get_deployment_config"):
            annotation = inspect.signature(self._fn(name)).parameters["serving"].annotation
            self.assertIn("jax", typing.get_args(annotation), f"{name} omits jax")

    def test_source_uri_is_plumbed_through_the_launch_tools(self):
        import inspect

        for name in ("create_inf2_instance", "get_deployment_config"):
            params = inspect.signature(self._fn(name)).parameters
            self.assertIn("source_uri", params, f"{name} cannot deploy the JAX engine")


class HelpTextTests(unittest.TestCase):
    def test_help_documents_all_three_modes_and_the_loopback_caveat(self):
        import asyncio

        get_help = getattr(server.get_help, "fn", server.get_help)
        text = asyncio.run(get_help())
        for token in ("vllm", "optb", "jax", "source_uri", "loopback"):
            self.assertIn(token, text, f"get_help omits {token!r}")


if __name__ == "__main__":
    unittest.main()
