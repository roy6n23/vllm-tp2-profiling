from __future__ import annotations

from tpprof import constants as c


def test_shard_sizes_sum_to_file_total():
    assert sum(c.MODEL_SHARD_SIZES.values()) == c.MODEL_SHARD_FILES_TOTAL == 16060556376
    assert c.MODEL_TENSOR_BYTES_TOTAL == 16060522496


def test_pins():
    assert c.VLLM_VERSION == "0.30.0"
    assert c.MODEL.repo == "NousResearch/Meta-Llama-3.1-8B-Instruct"
    assert c.H100_SXM.name == "NVIDIA H100 80GB HBM3" and c.H100_SXM.sm_count == 132


def test_nsys_path_env_override(monkeypatch):
    monkeypatch.delenv("TPPROF_NSYS", raising=False)
    assert c.nsys_path() == c.NSYS_DEFAULT_PATH
    monkeypatch.setenv("TPPROF_NSYS", "/tmp/fake-nsys")
    assert c.nsys_path() == "/tmp/fake-nsys"


def test_latin_square_is_a_latin_square():
    rows = c.LATIN_SQUARE
    assert all(sorted(r) == ["DP2", "TP1", "TP2"] for r in rows)
    assert all(sorted(col) == ["DP2", "TP1", "TP2"] for col in zip(*rows))
