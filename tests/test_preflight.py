from __future__ import annotations

import collections
import json
import resource
import shutil
import subprocess
import sys

import pytest

from tests.conftest import FAKE_BIN, fake_env
from tpprof import constants, engine, envcapture, preflight

QUICK_NAMES = ["gpu_count", "gpu_name", "power_limit", "driver", "topology", "fabric", "gpu_idle",
               "shm", "disk", "nofile", "cpus", "ram"]
Usage = collections.namedtuple("Usage", "total used free")
BIG = Usage(10**13, 0, 10**13)


@pytest.fixture
def roomy(monkeypatch):
    """A host with plenty of disk, /dev/shm and file descriptors, whatever machine runs the tests."""
    monkeypatch.setattr(shutil, "disk_usage", lambda path: BIG)
    monkeypatch.setattr(resource, "getrlimit", lambda which: (1024, 1048576))


def _ctx(tmp_path, scenario: str = "ok", **kw) -> preflight.PreflightContext:
    env = fake_env(tmp_path, FAKE_NVSMI_SCENARIO=scenario)
    args = dict(nvidia_smi=str(FAKE_BIN / "nvidia-smi"), vllm_bin=str(FAKE_BIN / "vllm"),
                results_dir=str(tmp_path / "results"), shm_path=str(tmp_path), on_box=False, env=env)
    args.update(kw)
    return preflight.PreflightContext(**args)


def _by_name(checks) -> dict[str, preflight.Check]:
    names = [c.name for c in checks]
    assert len(names) == len(set(names)), names
    return {c.name: c for c in checks}


def _model_dir(tmp_path, sizes=None, total_size=constants.MODEL_TENSOR_BYTES_TOTAL):
    d = tmp_path / "model"
    d.mkdir(parents=True)
    for name, size in (sizes or constants.MODEL_SHARD_SIZES).items():
        with open(d / name, "wb") as f:
            f.truncate(size)  # sparse: the exact size without the bytes
    for name in constants.MODEL_REQUIRED_FILES:
        (d / name).write_text("{}")
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size},
         "weight_map": {"lm_head.weight": "model-00004-of-00004.safetensors"}}))
    return d


def test_quick_ok_with_fake_ok_scenario(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path))
    assert [c.name for c in checks] == QUICK_NAMES
    by = _by_name(checks)
    assert all(c.ok for c in checks if c.hard), [(c.name, c.detail) for c in checks if not c.ok]
    assert not by["cpus"].hard and not by["ram"].hard
    assert by["gpu_name"].detail.count(constants.H100_SXM.name) == 2
    assert "NV18" in by["topology"].detail
    ok, text = preflight.verdict(checks)
    assert ok
    assert "PASS" in text and "gpu_count" in text


EXPECTED_FAILS = {
    "one_gpu": {"gpu_count", "topology"},
    "pcie": {"gpu_name", "power_limit", "topology"},
    "old_driver": {"driver"},
    "nv12": {"topology"},
    "busy": {"gpu_idle"},
    "no_fabric": {"fabric"},
    "power_capped": {"power_limit"},
}


@pytest.mark.parametrize("scenario", sorted(EXPECTED_FAILS))
def test_preflight_hard_fails(tmp_path, roomy, scenario):
    checks = preflight.quick_checks(_ctx(tmp_path, scenario))
    failed = {c.name for c in checks if c.hard and not c.ok}
    assert failed == EXPECTED_FAILS[scenario]
    for c in checks:
        if c.name in failed:
            assert c.fix, f"{c.name} has no actionable fix"
    ok, text = preflight.verdict(checks)
    assert not ok
    for name in failed:
        assert f"FAIL  {name}" in text


def test_preflight_hard_fail_details_name_the_problem(tmp_path, roomy):
    by = _by_name(preflight.quick_checks(_ctx(tmp_path, "old_driver")))
    assert "575.57.08" in by["driver"].detail
    by = _by_name(preflight.quick_checks(_ctx(tmp_path, "busy")))
    assert "424242" in by["gpu_idle"].detail and "30000" in by["gpu_idle"].detail
    by = _by_name(preflight.quick_checks(_ctx(tmp_path, "no_fabric")))
    assert "In Progress" in by["fabric"].detail
    by = _by_name(preflight.quick_checks(_ctx(tmp_path, "power_capped")))
    assert "500" in by["power_limit"].detail


def test_quick_checks_import_only_the_standard_library(tmp_path):
    """AM27: the quick preflight runs before any install, so it must not pull numpy/psutil/torch/vllm."""
    code = ("import sys; from tpprof import preflight; "
            "preflight.quick_checks(preflight.PreflightContext(nvidia_smi=sys.argv[1], results_dir=sys.argv[2], "
            "shm_path=sys.argv[2])); "
            "bad = sorted({'numpy', 'psutil', 'torch', 'vllm', 'transformers'} & set(sys.modules)); "
            "print(bad); sys.exit(1 if bad else 0)")
    env = fake_env(tmp_path)
    r = subprocess.run([sys.executable, "-c", code, str(FAKE_BIN / "nvidia-smi"), str(tmp_path)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_accept_topology(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path, "nv12", accept_topology=True))
    topo = _by_name(checks)["topology"]
    assert not topo.ok and not topo.hard          # downgraded to a recorded warning
    assert "NV12" in topo.detail and "--accept-topology" in topo.detail
    ok, text = preflight.verdict(checks)
    assert ok
    assert "WARN  topology" in text


def test_accept_topology_does_not_hide_a_missing_gpu(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path, "one_gpu", accept_topology=True))
    assert not preflight.verdict(checks)[0]


def test_shm_too_small(tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    tiny = Usage(64 << 20, 0, 64 << 20)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: tiny if str(path) == str(shm) else BIG)
    monkeypatch.setattr(resource, "getrlimit", lambda which: (1024, 1048576))
    checks = preflight.quick_checks(_ctx(tmp_path, shm_path=str(shm)))
    by = _by_name(checks)
    assert not by["shm"].ok and by["shm"].hard
    assert "64" in by["shm"].detail and "--ipc=host" in by["shm"].fix
    assert by["disk"].ok
    assert not preflight.verdict(checks)[0]


def test_missing_shm_and_low_disk_and_nofile_fail(tmp_path, monkeypatch):
    small = Usage(10**11, 0, 10 * 10**9)
    real_usage = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: small if "results" in str(path) or
                        str(path) == str(tmp_path) else real_usage(path))
    monkeypatch.setattr(resource, "getrlimit", lambda which: (1024, 4096))
    checks = preflight.quick_checks(_ctx(tmp_path, shm_path=str(tmp_path / "no-such-shm")))
    by = _by_name(checks)
    assert not by["shm"].ok and "no-such-shm" in by["shm"].detail
    assert not by["disk"].ok and "10.0 GB" in by["disk"].detail
    assert not by["nofile"].ok and "4096" in by["nofile"].detail


def test_missing_nvidia_smi_fails_gpu_gates_without_raising(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path, nvidia_smi=str(tmp_path / "no-nvidia-smi")))
    by = _by_name(checks)
    for name in ("gpu_count", "gpu_name", "power_limit", "driver", "topology", "fabric", "gpu_idle"):
        assert not by[name].ok, name
    assert not preflight.verdict(checks)[0]


def test_model_files_exact_sizes(tmp_path):
    good = preflight.check_model_files(str(_model_dir(tmp_path / "a")))
    assert good.ok and good.hard, good.detail

    sizes = dict(constants.MODEL_SHARD_SIZES)
    sizes["model-00003-of-00004.safetensors"] -= 1
    bad = preflight.check_model_files(str(_model_dir(tmp_path / "b", sizes=sizes)))
    assert not bad.ok
    assert "model-00003-of-00004.safetensors" in bad.detail and "4915916175" in bad.detail

    wrong_total = preflight.check_model_files(str(_model_dir(tmp_path / "c", total_size=16060556376)))
    assert not wrong_total.ok and "total_size" in wrong_total.detail

    d = _model_dir(tmp_path / "d")
    (d / "tokenizer.json").unlink()
    missing = preflight.check_model_files(str(d))
    assert not missing.ok and "tokenizer.json" in missing.detail

    assert not preflight.check_model_files("").ok


def test_help_flags_all_known(tmp_path):
    ctx = _ctx(tmp_path, model_dir="/models/llama")
    problems = preflight.help_flag_problems(ctx)
    labels = {f"serve {c.name}/{c.arm}" for c in engine.all_configs()}
    labels |= {f"bench latency {c.name}/{c.arm}" for c in engine.all_configs() if c.dp == 1}
    assert labels <= set(problems)
    assert {k: v for k, v in problems.items() if k != "bench serve client"} == {k: [] for k in labels}
    pytest.importorskip("tpprof.client")
    assert problems["bench serve client"] == []
    check = _by_name(preflight.full_checks(ctx))["help_flags"]
    assert check.ok and check.hard, check.detail


def test_help_flags_names_an_unknown_flag(tmp_path, monkeypatch):
    bogus = engine.base_config("TP1").with_arm("base", set_flags={"--made-up-flag": None})
    monkeypatch.setattr(engine, "all_configs", lambda: [bogus])
    problems = preflight.help_flag_problems(_ctx(tmp_path, model_dir="/models/llama"))
    assert problems["serve TP1/base"] == ["--made-up-flag"]
    assert problems["bench latency TP1/base"] == ["--made-up-flag"]


def test_full_checks_off_box_with_fakes(tmp_path, roomy, monkeypatch):
    ctx = _ctx(tmp_path, model_dir=str(_model_dir(tmp_path)))
    checks = preflight.full_checks(ctx)
    by = _by_name(checks)
    assert [c.name for c in checks][:len(QUICK_NAMES)] == QUICK_NAMES
    for name in ("vllm_version", "model_files", "nsys", "nsys_status", "monitor_field", "imports"):
        assert by[name].ok, (name, by[name].detail)
    assert "Root privilege" in by["nsys_status"].detail
    assert by["monitor_field"].detail == "clocks_event_reasons.active"
    for name in ("nvcc", "flashinfer_jit_cache"):
        assert name in by and by[name].hard
    # the on-box checks import torch/vllm/transformers and are not run off the box
    for name in ("torch", "vllm_env_names", "json_configs", "tokenizer_offline", "multicast"):
        assert name not in by


def test_pinned_thresholds_come_from_constants(tmp_path, roomy, monkeypatch):
    for name in ("EXPECTED_LINK", "MIN_CPUS", "MIN_RAM_BYTES", "FLASHINFER_JIT_CACHE"):
        assert not hasattr(preflight, name), f"{name} must live only in tpprof/constants.py"
    dist, want = constants.FLASHINFER_JIT_CACHE
    installed = {dist: want}
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda d: installed[d])
    by = _by_name(preflight.full_checks(_ctx(tmp_path)))
    assert by["flashinfer_jit_cache"].ok and want in by["flashinfer_jit_cache"].detail
    installed[dist] = "0.6.17"
    by = _by_name(preflight.full_checks(_ctx(tmp_path)))
    assert not by["flashinfer_jit_cache"].ok and "0.6.17" in by["flashinfer_jit_cache"].detail
    assert f"{dist}=={want}" in by["flashinfer_jit_cache"].fix
    monkeypatch.setattr(constants, "MIN_CPUS", 10**6)
    by = _by_name(preflight.quick_checks(_ctx(tmp_path)))
    assert not by["cpus"].ok and str(10**6) in by["cpus"].detail


def test_vllm_version_mismatch_fails(tmp_path, roomy):
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\necho 0.29.1\n")
    fake.chmod(0o755)
    by = _by_name(preflight.full_checks(_ctx(tmp_path, vllm_bin=str(fake))))
    assert not by["vllm_version"].ok and "0.29.1" in by["vllm_version"].detail


def test_nsys_gate(tmp_path):
    ctx = _ctx(tmp_path)
    assert preflight.check_nsys(ctx).ok
    on_box = _ctx(tmp_path, on_box=True)
    check = preflight.check_nsys(on_box)
    assert not check.ok and "TPPROF_NSYS" in check.detail
    env = dict(ctx.env)
    env["TPPROF_NSYS"] = str(tmp_path / "missing-nsys")
    missing = preflight.check_nsys(_ctx(tmp_path, env=env))
    assert not missing.ok and constants.NSYS_APT_PACKAGE in missing.fix


def test_monitor_field_falls_back_to_legacy_name(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\ncase \"$1\" in *clocks_event_reasons*) echo 'Field is not valid' >&2; exit 2;; "
                    "esac\necho 0x0000000000000001\n")
    fake.chmod(0o755)
    ctx = _ctx(tmp_path, nvidia_smi=str(fake))
    assert preflight.monitor_field(ctx) == "clocks_throttle_reasons.active"
    assert preflight.monitor_field(_ctx(tmp_path)) == "clocks_event_reasons.active"


def test_skip_gate_recorded(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path, "busy"))
    assert not preflight.verdict(checks)[0]
    ok, text = preflight.verdict(checks, skip=["gpu_idle", "no_such_gate"])
    assert ok
    assert "SKIP  gpu_idle" in text and "--skip-gate" in text
    assert "no_such_gate" in text
    out = tmp_path / "out" / "preflight.json"
    preflight.write_report(checks, str(out), skip=["gpu_idle"])
    report = json.loads(out.read_text())
    assert report["ok"] is True and report["skipped_gates"] == ["gpu_idle"]
    assert {"t_wall", "t_mono"} <= set(report)
    gpu_idle = next(c for c in report["checks"] if c["name"] == "gpu_idle")
    assert gpu_idle["ok"] is False and gpu_idle["skipped"] is True
    assert set(gpu_idle) == {"name", "ok", "hard", "detail", "fix", "skipped"}


def test_skip_gate_from_context_reaches_verdict_and_report(tmp_path, roomy):
    """A caller that sets ctx.skip_gates and uses the brief's signatures gets one consistent answer."""
    checks = preflight.quick_checks(_ctx(tmp_path, "busy", skip_gates=("gpu_idle", "shm")))
    by = _by_name(checks)
    assert by["gpu_idle"].skipped and not by["gpu_idle"].ok   # recorded, not hidden
    assert not by["shm"].skipped                              # it passed: nothing to override
    ok, text = preflight.verdict(checks)
    assert ok and "SKIP  gpu_idle" in text and "skipped gates: gpu_idle" in text
    out = tmp_path / "preflight.json"
    preflight.write_report(checks, str(out))
    report = json.loads(out.read_text())
    assert report["ok"] is True and report["skipped_gates"] == ["gpu_idle"]
    gpu_idle = next(c for c in report["checks"] if c["name"] == "gpu_idle")
    assert gpu_idle["ok"] is False and gpu_idle["skipped"] is True
    # passing the same names again changes nothing
    assert preflight.verdict(checks, ctx_skip := ("gpu_idle",)) == (ok, text)
    preflight.write_report(checks, str(out), skip=ctx_skip)
    assert json.loads(out.read_text())["skipped_gates"] == ["gpu_idle"]


def test_skip_gate_from_context_in_full_checks(tmp_path, roomy, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path=None: None)   # no nvcc, whatever the host
    ctx = _ctx(tmp_path, model_dir=str(_model_dir(tmp_path)), skip_gates=("nvcc",))
    by = _by_name(preflight.full_checks(ctx))
    assert not by["nvcc"].ok and by["nvcc"].skipped
    assert not by["gpu_idle"].skipped
    _, text = preflight.verdict(list(by.values()))
    summary = text.splitlines()[-1]
    assert "SKIP  nvcc" in text and "skipped gates: nvcc" in summary
    assert "nvcc" not in summary.split(", skipped gates:")[0]


def test_write_report_without_skips_fails_on_hard_gate(tmp_path, roomy):
    checks = preflight.quick_checks(_ctx(tmp_path, "nv12"))
    out = tmp_path / "preflight.json"
    preflight.write_report(checks, str(out))
    report = json.loads(out.read_text())
    assert report["ok"] is False and report["skipped_gates"] == []


def test_capture_env_has_keys(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.env["VLLM_BUILD_COMMIT"] = "ced6857"
    ctx.env["HF_TOKEN"] = "hf_secret_value"
    ctx.env["NCCL_DEBUG"] = "INFO"
    ctx.env["UNRELATED_VAR"] = "x"
    env = envcapture.capture_env(ctx)
    for key in ("nvidia_smi_q", "topo_m", "nvlink_s", "driver", "gpu_names", "pip_freeze", "uname",
                "os_release", "lscpu", "free_g", "environ", "vllm_build_commit", "nsys_version",
                "monitor_field", "t_wall", "t_mono", "errors"):
        assert key in env, key
    assert "Fabric" in env["nvidia_smi_q"] and "NV18" in env["topo_m"] and "Link 17" in env["nvlink_s"]
    assert env["driver"] == "580.95.05"
    assert env["gpu_names"] == [constants.H100_SXM.name] * 2
    assert "pytest==" in env["pip_freeze"].lower()
    assert env["vllm_build_commit"] == "ced6857"
    assert constants.NSYS_VERSION in env["nsys_version"]
    assert env["monitor_field"] == "clocks_event_reasons.active"
    environ = env["environ"]
    assert environ["NCCL_DEBUG"] == "INFO" and "PATH" in environ and "TPPROF_NSYS" in environ
    assert "UNRELATED_VAR" not in environ
    assert environ["HF_TOKEN"] == "<redacted>"
    assert "hf_secret_value" not in json.dumps(env)
    json.dumps(env)  # env.json content is JSON-serializable


def test_capture_env_tolerates_missing_tools(tmp_path):
    ctx = _ctx(tmp_path, nvidia_smi=str(tmp_path / "no-nvidia-smi"))
    ctx.env["PATH"] = str(tmp_path / "empty-bin")
    ctx.env["TPPROF_NSYS"] = str(tmp_path / "no-nsys")
    env = envcapture.capture_env(ctx)
    assert env["nvidia_smi_q"] is None and env["driver"] is None and env["gpu_names"] == []
    assert "nvidia_smi_q" in env["errors"] and "nsys_version" in env["errors"]
    assert env["lscpu"] is None and "lscpu" in env["errors"]
    assert env["vllm_build_commit"] is None


def test_parsers_tolerate_real_layout_variations():
    topo = ("\x1b[4m\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
            "GPU0\t X \tNV18\tSYS\t0-51\t0\t\tN/A\n"
            "GPU1\tNV18\t X \tSYS\t0-51\t0\t\tN/A\n")
    assert preflight.topology_link(topo) == "NV18"
    assert preflight.topology_link(topo, "GPU1", "GPU0") == "NV18"
    assert preflight.topology_link("\tGPU0\tCPU Affinity\nGPU0\t X \t0-51\n") is None
    q = ("GPU 00000000:18:00.0\n"
         "    Product Name                          : NVIDIA H100 80GB HBM3\n"
         "    Fabric\n"
         "        State                             : Completed\n"
         "        Status                            : Success\n"
         "        Health\n"
         "            Bandwidth                     : Full\n"
         "            Status                        : Healthy\n"
         "    Processes                             : None\n"
         "    Reset Status\n"
         "        State                             : Not Needed\n")
    assert preflight.fabric_states(q) == [("00000000:18:00.0", "Completed", "Success")]
