# -*- coding: utf-8 -*-
"""Run the complete research recalculation and record one isolated run summary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

NOTEBOOKS = [
    "notebooks/conversionfunnel_revised.ipynb",
    "notebooks/hexagonsclosingcheck.ipynb",
    "notebooks/cohort_diagnostics.ipynb",
    "notebooks/preestimationDID.ipynb",
    "notebooks/utilization_censoring_threshold_diagnostics.ipynb",
    "notebooks/final_empirical_recalculation.ipynb",
    "notebooks/conditional_funnel_did.ipynb",
    "notebooks/speed_conversion_association.ipynb",
    "notebooks/honest_did_sensitivity.ipynb",
    "notebooks/harmonized_empirical_figures.ipynb",
]

MANIFEST_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "h3",
    "jupyterlab",
    "nbconvert",
    "nbclient",
    "nbformat",
    "linearmodels",
    "differences",
    "scipy",
    "statsmodels",
    "tqdm",
    "pytest",
    "ruff",
    "dvc",
]

BASE_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _relative_posix(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _public_command(command: list[str]) -> list[str]:
    """Return a portable command representation for the run summary."""
    public: list[str] = []
    for item in command:
        if item == sys.executable:
            public.append("python")
            continue
        candidate = Path(item)
        if candidate.is_absolute():
            try:
                public.append(candidate.resolve().relative_to(ROOT.resolve()).as_posix())
                continue
            except ValueError:
                public.append(candidate.name)
                continue
        public.append(item.replace("\\", "/"))
    return public


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reproducibility_manifest(env: dict[str, str]) -> Path:
    """Record code, runtime and input-data identities for this recalculation."""
    versions: dict[str, str | None] = {}
    for package in MANIFEST_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    dvc = subprocess.run(
        ["dvc", "status", "--json"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    data_files = []
    for directory in (ROOT / "data", ROOT / "datasets"):
        if directory.exists():
            for path in sorted(item for item in directory.rglob("*") if item.is_file()):
                data_files.append(
                    {
                        "path": _relative_posix(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    dvc_files = [
        {"path": _relative_posix(path), "sha256": _sha256(path)}
        for path in sorted(ROOT.rglob("*.dvc"))
        if path.is_file() and ".git" not in path.parts and path.name != ".dvc"
    ]
    manifest = {
        "generated_at": _iso(_now()),
        "python": sys.version,
        "platform": sys.platform,
        "commit_hash": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_error": commit.stderr.strip() or None,
        "packages": versions,
        "dvc_status": (
            json.loads(dvc.stdout) if dvc.returncode == 0 and dvc.stdout.strip() else None
        ),
        "dvc_error": dvc.stderr.strip() or None,
        "dvc_files": dvc_files,
        "data_files": data_files,
    }
    path = ROOT / "outputs" / "diagnostics" / "reproducibility_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prepare_build_input(source: Path, build_input_dir: Path) -> Path:
    """Create a validated build-only notebook copy with no stale outputs."""
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            source=f"import os\nos.chdir({str(source.parent)!r})\n",
            metadata={"tags": ["build-bootstrap"]},
        ),
    )
    nbformat.validate(notebook)
    build_input = build_input_dir / source.name
    nbformat.write(notebook, build_input)
    return build_input


def run_one(
    rel: str,
    *,
    run_id: str,
    log_dir: Path,
    build_dir: Path,
    build_input_dir: Path,
    env: dict[str, str],
) -> dict:
    notebook_path = ROOT / rel
    build_input = prepare_build_input(notebook_path, build_input_dir)
    name = notebook_path.stem
    log_path = log_dir / f"{name}.log"
    started = _now()
    timer = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--output-dir",
        str(build_dir),
        "--output",
        notebook_path.name,
        str(build_input),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"START {_iso(started)}\n")
        log.write("CMD " + " ".join(_public_command(command)) + "\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - timer
        ended = _now()
        log.write(f"\nEND {_iso(ended)}\n")
        log.write(f"ELAPSED_SEC {elapsed:.3f}\n")
        log.write(f"EXIT_CODE {process.returncode}\n")
    return {
        "run_id": run_id,
        "notebook": rel.replace("\\", "/"),
        "name": name,
        "log": _relative_posix(log_path),
        "started": _iso(started),
        "ended": _iso(ended),
        "elapsed_sec": elapsed,
        "exit_code": process.returncode,
        "ok": process.returncode == 0,
    }


def run_stage(name: str, command: list[str], *, env: dict[str, str]) -> dict:
    """Run a non-notebook stage and preserve its real exit code."""
    started = _now()
    timer = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, env=env, check=False)
    ended = _now()
    return {
        "stage": name,
        "command": _public_command(command),
        "started": _iso(started),
        "ended": _iso(ended),
        "elapsed_sec": time.perf_counter() - timer,
        "exit_code": process.returncode,
        "ok": process.returncode == 0,
    }


def write_summary(
    path: Path,
    *,
    run_id: str,
    overall_start: str,
    selected_notebooks: list[str],
    notebook_results: list[dict],
    stages: list[dict],
    overall_end: str | None = None,
    exit_code: int | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "overall_start": overall_start,
        "updated": _iso(_now()),
        "overall_end": overall_end,
        "exit_code": exit_code,
        "ok": exit_code == 0 if exit_code is not None else None,
        "selected_notebooks": [item.replace("\\", "/") for item in selected_notebooks],
        "notebook_results": notebook_results,
        "stages": stages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_notebooks(arguments: list[str]) -> list[str]:
    if not arguments:
        return list(NOTEBOOKS)
    selection = arguments[0]
    if selection.startswith("from:") and selection.split(":", 1)[1].isdigit():
        start = int(selection.split(":", 1)[1]) - 1
        if start < 0 or start >= len(NOTEBOOKS):
            raise ValueError(f"notebook index out of range: {selection}")
        return NOTEBOOKS[start:]
    if selection.isdigit():
        index = int(selection) - 1
        if index < 0 or index >= len(NOTEBOOKS):
            raise ValueError(f"notebook index out of range: {selection}")
        return [NOTEBOOKS[index]]
    matched = [notebook for notebook in NOTEBOOKS if selection in notebook]
    if not matched:
        raise ValueError(f"no notebook matches: {selection}")
    return matched


def build_validation_command(selected_notebooks: list[str]) -> list[str]:
    """Use strict validation only after the complete canonical notebook sequence."""
    command = [sys.executable, "scripts/validate_final_outputs.py"]
    if selected_notebooks == NOTEBOOKS:
        command.append("--full")
    return command


def main() -> int:
    selected_notebooks = select_notebooks(sys.argv[1:])
    started = _now()
    overall_start = _iso(started)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")

    log_dir = ARTIFACTS_DIR / "execution_logs" / run_id
    build_dir = ARTIFACTS_DIR / "notebook_build" / run_id
    build_input_dir = build_dir / "inputs"
    log_dir.mkdir(parents=True, exist_ok=False)
    build_input_dir.mkdir(parents=True, exist_ok=False)

    summary_path = log_dir / "recalculation_summary.json"
    (log_dir / "recalculation_started.txt").write_text(overall_start, encoding="utf-8")
    notebook_results: list[dict] = []
    stages: list[dict] = []
    env = {
        **BASE_SUBPROCESS_ENV,
        "LAST_MILE_RECALCULATION_STARTED": overall_start,
        "LAST_MILE_RECALCULATION_RUN_ID": run_id,
    }

    def persist(*, exit_code: int | None = None) -> None:
        write_summary(
            summary_path,
            run_id=run_id,
            overall_start=overall_start,
            selected_notebooks=selected_notebooks,
            notebook_results=notebook_results,
            stages=stages,
            overall_end=_iso(_now()) if exit_code is not None else None,
            exit_code=exit_code,
        )

    persist()
    try:
        for rel in selected_notebooks:
            print(f"=== RUNNING {rel} ===", flush=True)
            result = run_one(
                rel,
                run_id=run_id,
                log_dir=log_dir,
                build_dir=build_dir,
                build_input_dir=build_input_dir,
                env=env,
            )
            notebook_results.append(result)
            persist()
            print(
                f"=== DONE {rel} exit={result['exit_code']} "
                f"elapsed={result['elapsed_sec']:.1f}s ===",
                flush=True,
            )
            if not result["ok"]:
                print(f"FAILED: see {result['log']}", flush=True)
                persist(exit_code=int(result["exit_code"]))
                return int(result["exit_code"])

            if rel == "notebooks/hexagonsclosingcheck.ipynb":
                for stage_name, script in (
                    ("hex_closure_diagnostics", "scripts/run_hex_closure_diagnostics.py"),
                    ("closure_did_sensitivity", "scripts/run_closure_did_sensitivity.py"),
                ):
                    stage = run_stage(
                        stage_name,
                        [sys.executable, script],
                        env=env,
                    )
                    stages.append(stage)
                    persist()
                    if not stage["ok"]:
                        persist(exit_code=int(stage["exit_code"]))
                        return int(stage["exit_code"])

        manifest_started = _now()
        manifest_path = write_reproducibility_manifest(env)
        stages.append(
            {
                "stage": "reproducibility_manifest",
                "command": ["internal", "write_reproducibility_manifest"],
                "output": _relative_posix(manifest_path),
                "started": _iso(manifest_started),
                "ended": _iso(_now()),
                "exit_code": 0,
                "ok": True,
            }
        )
        persist()
        print(f"=== MANIFEST {_relative_posix(manifest_path)} ===", flush=True)

        validation_is_full = selected_notebooks == NOTEBOOKS
        validation_command = build_validation_command(selected_notebooks)
        for stage_name, command in (
            (
                "generate_figures",
                [sys.executable, "scripts/generate_figures_from_final_csvs.py"],
            ),
            (
                "validate_final_outputs_full"
                if validation_is_full
                else "validate_final_outputs_release",
                validation_command,
            ),
        ):
            stage = run_stage(stage_name, command, env=env)
            stages.append(stage)
            persist()
            if not stage["ok"]:
                persist(exit_code=int(stage["exit_code"]))
                return int(stage["exit_code"])
    except Exception as exc:  # noqa: BLE001
        stages.append(
            {
                "stage": "orchestration_error",
                "started": _iso(_now()),
                "ended": _iso(_now()),
                "exit_code": 1,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        persist(exit_code=1)
        return 1

    persist(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
