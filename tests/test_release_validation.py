from __future__ import annotations

import shutil
from pathlib import Path

from scripts import validate_final_outputs as validator


def _copy_final_outputs(tmp_path: Path, monkeypatch) -> Path:
    final = tmp_path / "final"
    shutil.copytree(validator.FINAL, final)
    monkeypatch.setattr(validator, "FINAL", final)
    monkeypatch.setattr(validator, "RECALCULATION_STARTED", None)
    return final


def test_release_mode_passes_in_public_checkout() -> None:
    assert validator.main(full=False) == 0


def test_full_mode_reports_missing_generated_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validator, "DIAGNOSTICS", tmp_path / "diagnostics")
    monkeypatch.setattr(validator, "FIGURES", tmp_path / "figures")

    assert validator.main(full=True) == 1
    output = capsys.readouterr().out
    assert "missing treatment diagnostic" in output
    assert "thesis figure is missing" in output
    assert "run the full analysis pipeline" in output


def test_malformed_canonical_csv_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    final = _copy_final_outputs(tmp_path, monkeypatch)
    target = final / "main_event_study_post_summary.csv"
    text = target.read_text(encoding="utf-8")
    target.write_text("Unnamed: 0," + text, encoding="utf-8")

    assert validator.main(full=False) == 1
    assert "random index column" in capsys.readouterr().out


def test_missing_required_tracked_csv_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    final = _copy_final_outputs(tmp_path, monkeypatch)
    (final / "main_event_study_post_summary.csv").unlink()

    assert validator.main(full=False) == 1
    assert "missing file" in capsys.readouterr().out


def _prepare_publication_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, set[str]]:
    thesis = tmp_path / "thesis"
    figures = tmp_path / "figures" / "empirical"
    thesis.mkdir()
    figures.mkdir(parents=True)
    includes = "\n".join(
        f"\\includegraphics{{{name}}}" for name in validator.PUBLICATION_FIGURES
    )
    (thesis / "main.tex").write_text(
        "\n".join([r"13\,255 9\,327 302\,398", includes]),
        encoding="utf-8",
    )
    (thesis / "references.bib").write_text("% references\n", encoding="utf-8")
    (thesis / "main.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    for name in validator.PUBLICATION_FIGURES:
        (figures / name).write_bytes(b"%PDF-1.4\n%%EOF\n")

    tracked = {
        "thesis/main.pdf",
        *(f"figures/empirical/{name}" for name in validator.PUBLICATION_FIGURES),
    }
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "FIGURES", figures)
    monkeypatch.setattr(validator, "_tracked_files", lambda: tracked)
    return thesis, tracked


def test_publication_artifacts_accept_canonical_release(
    tmp_path: Path, monkeypatch
) -> None:
    _prepare_publication_fixture(tmp_path, monkeypatch)
    result = validator.Validator()

    validator.validate_publication_artifacts(result)

    assert result.errors == []


def test_publication_artifacts_reject_regressions(tmp_path: Path, monkeypatch) -> None:
    thesis, tracked = _prepare_publication_fixture(tmp_path, monkeypatch)
    tex_path = thesis / "main.tex"
    tex_path.write_text(
        tex_path.read_text(encoding="utf-8")
        + "\n13\\,256 9\\,328 author@example.com\n"
        + "причинная альтернативная гипотеза не принимается\n",
        encoding="utf-8",
    )
    (thesis / "main.pdf").write_bytes(b"not a PDF")
    tracked.add("figures/empirical/unexpected.png")
    result = validator.Validator()

    validator.validate_publication_artifacts(result)

    combined = "\n".join(result.errors)
    assert "obsolete panel denominator" in combined
    assert "obsolete treated-hex denominator" in combined
    assert "personal email" in combined
    assert "obsolete causal-hypothesis wording" in combined
    assert "PDF magic header" in combined
    assert "contain PNG files" in combined


def test_assoc_rejects_legacy_total_request_label(tmp_path: Path, monkeypatch) -> None:
    final = _copy_final_outputs(tmp_path, monkeypatch)
    target = final / "speed_conversion_association_summary.csv"
    text = target.read_text(encoding="utf-8")
    text = text.replace(
        "n_analysis_requests_before_t_available_filter", "n_total_requests", 1
    ).replace(
        "combined treated/control sample from build_did_samples; not the full raw dataset",
        "all loaded requests",
        1,
    )
    target.write_text(text, encoding="utf-8")
    result = validator.Validator()

    validator.validate_assoc(result, target)

    combined = "\n".join(result.errors)
    assert "forbidden legacy total-request label" in combined
    assert "must contain exactly one n_analysis_requests_before_t_available_filter row" in combined
