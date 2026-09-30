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


def test_ms_format_never_shows_more_than_4_significant_digits():
    assert predict._ms(5.85) == "5.85"
    assert predict._ms(48.26) == "48.26"
    assert predict._ms(223.6) == "223.6"
    assert predict._ms(1234.0) == "1234"
    assert predict._ms(0.5) == "0.50"


def test_hypotheses_table_has_the_h2_prefill_leg():
    md = predict.render_predictions_md(model.predictions())
    assert "| H2 (prefill) |" in md


def _hypothesis_rows(md: str) -> dict[str, list[str]]:
    section = md.split("## Hypotheses")[1].split("\n## ")[0]
    rows = [line.strip("|").split(" | ") for line in section.splitlines() if line.startswith("| H")]
    return {r[0].strip(): [c.strip() for c in r] for r in rows}


def test_hypotheses_central_column_comes_from_the_central_constants():
    pred = model.predictions()
    assert "hypotheses_central" not in pred
    rows = _hypothesis_rows(predict.render_predictions_md(pred))
    central = model.hypothesis_values(model.CONSTANTS["central"])
    for label, key, _, kind in predict.HYPOTHESES:
        assert rows[label][3] == predict._hyp_value(central[key], kind), label
    assert rows["H8"][3] == "1.060"


def test_hypotheses_central_column_follows_the_constants_in_pred():
    # The renderer reads the central constants from pred itself, not from module state.
    pred = model.predictions()
    pred["constants"]["central"] = dict(pred["constants"]["central"], unfused_norm_s=0.0)
    rows = _hypothesis_rows(predict.render_predictions_md(pred))
    c = model.Constants(**pred["constants"]["central"])
    assert rows["H8"][3] == predict._hyp_value(model.hypothesis_values(c)["H8"], "ratio")
    assert rows["H8"][3] != "1.060"
