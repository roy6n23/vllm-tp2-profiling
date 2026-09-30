from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import fake_env
from tpprof import tokbench


class ToyTokenizer:
    """Word-level stand-in for a HF tokenizer: id i <-> "w<i>", ids >= 90 are special."""

    vocab_size = 100
    all_special_ids = [90, 91, 99]
    bos_token_id = 99

    def __init__(self):
        self.decode_calls = 0

    def encode(self, text, add_special_tokens=True):
        ids = [int(w[1:]) for w in text.split()]
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, ids, skip_special_tokens=False):
        self.decode_calls += 1
        special = set(self.all_special_ids)
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in special))


class ToyStream:
    """Mimics tokenizers.decoders.DecodeStream(ids=..., skip_special_tokens=...).step(tokenizer, id)."""

    instances: list = []

    def __init__(self, ids=None, skip_special_tokens=False):
        self.ids = list(ids or [])
        self.steps = 0
        ToyStream.instances.append(self)

    def step(self, tokenizer, token_id):
        self.steps += 1
        return f" w{token_id}"


def test_synthetic_writes_the_keys(tmp_path):
    out = tmp_path / "tokbench.json"
    assert tokbench.main(["--model-dir", "/nonexistent", "--n", "200", "--input-len", "1024",
                          "--output-len", "256", "--out", str(out), "--synthetic"]) == 0
    d = json.loads(out.read_text())
    assert set(tokbench.KEYS) <= set(d)
    assert d["synthetic"] is True and d["detok_method"] == "synthetic"
    assert (d["n"], d["input_len"], d["output_len"], d["prompt_tokens"]) == (200, 1024, 256, 1023)
    for m in ("encode_ms", "decode_ms", "detok_us_per_token"):
        assert 0 < d[f"{m}_p50"] <= d[f"{m}_p90"]
    assert d["t_wall_end"] >= d["t_wall_start"] and d["t_mono_end"] >= d["t_mono_start"]


def test_synthetic_cli_does_not_need_transformers(tmp_path):
    out = tmp_path / "tb.json"
    code = ("import sys; sys.modules['transformers'] = None; sys.modules['tokenizers'] = None\n"
            "from tpprof import tokbench\n"
            f"raise SystemExit(tokbench.main(['--model-dir', 'x', '--out', {str(out)!r}, '--synthetic']))")
    r = subprocess.run([sys.executable, "-c", code], env=fake_env(tmp_path), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert (d["n"], d["input_len"], d["output_len"]) == (200, 1024, 256)       # the CLI defaults


def test_module_runs_as_script(tmp_path):
    out = tmp_path / "tb.json"
    r = subprocess.run([sys.executable, "-m", "tpprof.tokbench", "--model-dir", "x", "--out", str(out),
                        "--synthetic", "--n", "5"], env=fake_env(tmp_path), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["n"] == 5


def test_prompts_have_exactly_input_len_minus_one_tokens_without_special_ids():
    tok = ToyTokenizer()
    prompts, mismatched = tokbench.build_prompts(tok, n=7, input_len=33, rng=np.random.default_rng(0))
    assert len(prompts) == 7 and mismatched == 0
    for text, ids in prompts:
        assert ids == tok.encode(text, add_special_tokens=False)
        assert len(ids) == 32
        assert not set(ids) & set(tok.all_special_ids)
    assert len({text for text, _ in prompts}) == 7


def test_measure_with_decode_stream_primes_it_with_the_prompt():
    ToyStream.instances = []
    tok = ToyTokenizer()
    r = tokbench.measure(tok, n=4, input_len=17, output_len=9, seed=0, decode_stream=ToyStream)
    assert r["detok_method"] == "DecodeStream(ids=prompt)"
    assert r["prompt_tokens"] == 16 and r["token_mismatch_requests"] == 0
    measured = ToyStream.instances[-4:]
    assert all(len(s.ids) == 17 and s.ids[0] == tok.bos_token_id and s.steps == 9 for s in measured)
    for m in ("encode_ms", "decode_ms", "detok_us_per_token"):
        assert 0 <= r[f"{m}_p50"] <= r[f"{m}_p90"]


def test_measure_without_decode_stream_decodes_growing_prefixes():
    tok = ToyTokenizer()
    r = tokbench.measure(tok, n=3, input_len=9, output_len=5, seed=0, decode_stream=None)
    assert r["detok_method"] == "prefix_decode"
    assert r["prompt_tokens"] == 8
    assert r["detok_us_per_token_p90"] >= r["detok_us_per_token_p50"] >= 0


def test_decode_stream_without_prefill_support_is_recorded():
    class OldStream:
        def __init__(self, skip_special_tokens=False):
            pass

        def step(self, tokenizer, token_id):
            return None

    r = tokbench.measure(ToyTokenizer(), n=2, input_len=9, output_len=4, seed=0, decode_stream=OldStream)
    assert r["detok_method"] == "DecodeStream"


def test_real_mode_names_transformers_when_missing(tmp_path):
    code = ("import sys; sys.modules['transformers'] = None\n"
            "from tpprof import tokbench\n"
            f"raise SystemExit(tokbench.main(['--model-dir', 'x', '--out', {str(tmp_path / 'o.json')!r}]))")
    r = subprocess.run([sys.executable, "-c", code], env=fake_env(tmp_path), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode != 0 and "transformers" in r.stderr


@pytest.mark.parametrize("args", [["--n", "0"], ["--input-len", "1"], ["--output-len", "0"]])
def test_bad_sizes_are_rejected(tmp_path, args):
    with pytest.raises(SystemExit):
        tokbench.main(["--model-dir", "x", "--out", str(tmp_path / "o.json"), "--synthetic", *args])
