from __future__ import annotations

import json

import pytest

from scripts import run_full_recalculation as runner


def test_notebook_selection_is_explicit_and_ordered() -> None:
    assert runner.select_notebooks([]) == runner.NOTEBOOKS
    assert runner.select_notebooks(["1"]) == runner.NOTEBOOKS[:1]
    assert runner.select_notebooks(["from:10"]) == runner.NOTEBOOKS[9:]
    with pytest.raises(ValueError):
        runner.select_notebooks(["from:0"])
    with pytest.raises(ValueError):
        runner.select_notebooks(["missing-notebook"])


def test_full_selection_enables_strict_validation() -> None:
    full_command = runner.build_validation_command(list(runner.NOTEBOOKS))
    partial_command = runner.build_validation_command(runner.NOTEBOOKS[:1])

    assert full_command[-1] == "--full"
    assert "--full" not in partial_command


def test_summary_contains_only_the_current_run(tmp_path) -> None:
    path = tmp_path / "recalculation_summary.json"
    runner.write_summary(
        path,
        run_id="run-current",
        overall_start="2026-08-04T00:00:00+00:00",
        selected_notebooks=["notebooks/current.ipynb"],
        notebook_results=[{"notebook": "notebooks/current.ipynb", "exit_code": 0}],
        stages=[{"stage": "validation", "exit_code": 0}],
        overall_end="2026-08-04T00:01:00+00:00",
        exit_code=0,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-current"
    assert payload["selected_notebooks"] == ["notebooks/current.ipynb"]
    assert payload["notebook_results"] == [
        {"notebook": "notebooks/current.ipynb", "exit_code": 0}
    ]
    assert payload["stages"] == [{"stage": "validation", "exit_code": 0}]
    assert payload["ok"] is True
