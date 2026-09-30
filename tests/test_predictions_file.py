from __future__ import annotations

from tests.conftest import ROOT
from tpprof import model, predict


def _hypothesis_rows(md: str, section: str) -> dict[str, list[list[str]]]:
    """Rows of the markdown table under `section`, keyed by the first cell's H-id ("H2 (prefill)" -> "H2")."""
    body = md.split(section, 1)[1].split("\n## ", 1)[0]
    rows: dict[str, list[list[str]]] = {}
    for line in body.splitlines():
        if line.startswith("| H"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.setdefault(cells[0].split()[0], []).append(cells)
    return rows


def test_committed_predictions_md_matches_the_model():
    committed = (ROOT / "predictions.md").read_text(encoding="utf-8")
    assert committed == predict.render_predictions_md(model.predictions())


def test_readme_quotes_the_committed_bands():
    predicted = _hypothesis_rows((ROOT / "predictions.md").read_text(encoding="utf-8"), "## Hypotheses")
    readme = _hypothesis_rows((ROOT / "README.md").read_text(encoding="utf-8"), "## Hypotheses")
    assert set(predicted) <= set(readme)
    for hid, rows in predicted.items():
        (readme_row,) = readme[hid]
        quoted = " | ".join(readme_row)
        for _, _, band, central in rows:
            assert band in quoted and f"({central})" in quoted, (hid, band, central, quoted)
