from __future__ import annotations

import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
FAKE_BIN = ROOT / "tests" / "fake_bin"


def fake_env(tmp_path: pathlib.Path, **overrides: str) -> dict[str, str]:
    """Environment for running the fake tools: fake_bin first on PATH, fast time scale."""
    env = dict(os.environ)
    env["PATH"] = f"{FAKE_BIN}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_TIME_SCALE"] = "0.01"
    env["FAKE_GPU_STATE_DIR"] = str(tmp_path / "gpustate")
    env["TPPROF_NSYS"] = str(FAKE_BIN / "nsys")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    (tmp_path / "gpustate").mkdir(exist_ok=True)
    env.update(overrides)
    return env


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    return FIXTURES
