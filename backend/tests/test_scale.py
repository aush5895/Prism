"""Guards on the scale proof (evaluation/scale.py).

A synthetic corpus is only useful if it is reproducible and if nothing it generates can
ever be mistaken for Samsung's data. Both are asserted here, along with the rule that the
scale run adds a section to metrics.md without disturbing any existing figure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import scale  # noqa: E402
from evaluation.run_eval import render_metrics  # noqa: E402

REPORT_PATH = ROOT / "evaluation" / "report.json"


# ----------------------------------------------------------------- reproducibility
def test_the_corpus_is_deterministic_for_a_seed():
    """Two runs must produce the same corpus, or the published numbers are unrepeatable."""
    a = scale.build_corpus(120, seed=4242)
    b = scale.build_corpus(120, seed=4242)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [s.query for s in a] == [s.query for s in b]
    assert [s.evidence for s in a] == [s.evidence for s in b]


def test_a_different_seed_gives_a_different_corpus():
    a = scale.build_corpus(120, seed=1)
    b = scale.build_corpus(120, seed=2)
    assert [s.query for s in a] != [s.query for s in b]


def test_a_prefix_of_the_corpus_is_stable():
    """Each size measures corpus[:n], so the first n scenarios must not shift when a
    larger corpus is generated -- otherwise the sizes are not comparable."""
    big = scale.build_corpus(200, seed=99)
    assert [s.scenario_id for s in big[:50]] == [f"SYN-{i:06d}" for i in range(50)]


# ----------------------------------------------------------------- labelling
def test_every_generated_artefact_is_labelled_synthetic():
    """Nothing here may ever be presented as Samsung data."""
    for s in scale.build_corpus(200, seed=7):
        assert s.scenario_id.startswith("SYN-")
        assert s.article_id.startswith("SYN-ART-")
        assert "SYNTHETIC ARTICLE" in s.evidence.splitlines()[0]
        assert s.plan["contexts"][0]["title"] == "Synthetic scenario"


def test_article_variants_are_built_from_real_sections():
    """The prose is Samsung's; only the arrangement is ours. If the generator started
    inventing article text, the measurement would be about the generator."""
    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text(encoding="utf-8"))
    real = "\n".join(
        f"{r['siis_response']['title']}\n{r['siis_response']['content']}"
        for r in rows["responses"])
    real_words = set(re.findall(r"[a-z]{6,}", real.lower()))

    for s in scale.build_corpus(40, seed=11):
        body = "\n".join(s.evidence.splitlines()[1:])          # drop the SYNTHETIC header
        words = set(re.findall(r"[a-z]{6,}", body.lower()))
        assert words, "an article variant must carry text"
        invented = words - real_words
        assert not invented, f"article variant contains invented words: {sorted(invented)[:5]}"


def test_scenarios_cluster_onto_shared_articles():
    """Real corpora cluster -- many complaints map to one knowledge article -- and that
    clustering is where within-article collisions live."""
    corpus = scale.build_corpus(400, seed=5)
    articles = {s.article_id for s in corpus}
    assert 1 < len(articles) < len(corpus)
    assert len(corpus) / len(articles) >= 2


# ----------------------------------------------------------------- report + rendering
@pytest.mark.skipif(not REPORT_PATH.exists(), reason="run the harness first")
def test_section_8_renders_only_from_measured_numbers():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if "scale" not in report:
        pytest.skip("no scale run recorded yet")

    markdown = render_metrics(report)
    assert "## 8. Scale" in markdown
    for row in report["scale"]["by_size"]:
        assert f"{row['scenarios']:,}" in markdown
    # the honesty statement is not optional
    assert report["scale"]["does_not_measure"] in markdown
    assert "synthetic" in markdown.lower()


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="run the harness first")
def test_a_scale_run_does_not_disturb_the_other_sections():
    """Section 8 is additive. Rendering with and without the scale block must leave every
    earlier section byte-identical."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if "scale" not in report:
        pytest.skip("no scale run recorded yet")

    without = dict(report)
    without.pop("scale")
    before, after = render_metrics(without), render_metrics(report)

    marker = "## 7. Limitations"
    assert before[:before.index(marker)] == after[:after.index(marker)]
    assert "## 8. Scale" not in before
