from __future__ import annotations

from tpprof import model, predict

def test_render_is_deterministic_and_complete():
    a = predict.render_predictions_md(model.predictions())
    b = predict.render_predictions_md(model.predictions())
    assert a == b
    for token in ("# Predictions", "H1", "H2", "H3", "H4", "H5", "H8", "central", "optimistic", "pessimistic",
                  "decode step", "prefill", "KV cache", "saturation"):
        assert token in a
    assert "d8" not in a.split("## Constants")[0]   # the d8 set only appears in the constants appendix

def test_main_writes_file(tmp_path):
    out = tmp_path / "p.md"
    assert predict.main(["--out", str(out)]) == 0
    assert out.read_text() == predict.render_predictions_md(model.predictions())
