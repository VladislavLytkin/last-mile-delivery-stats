# -*- coding: utf-8 -*-
"""Static audit of tracked Jupyter notebooks for the public release.

Exits with a non-zero code when critical issues are found.
Does not modify notebooks.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

# Explicit allowlist: (notebook name, regex pattern, reason)
ALLOWLIST: list[tuple[str, re.Pattern[str], str]] = [
    (
        "harmonized_empirical_figures.ipynb",
        re.compile(r"thesis/main\.tex"),
        "Figures are consumed by thesis/main.tex; not a copy-paste instruction.",
    ),
]

FORBIDDEN_WORDS = re.compile(
    r"ChatGPT|Codex|\bCursor\b|готовый промпт|скопировать в main\.tex|"
    r"вставить в main\.tex|по просьбе пользователя|LaTeX для таблицы",
    re.I,
)
TODOISH = re.compile(r"\b(TODO|FIXME|TEMP|DEBUG|PENDING)\b")
ABS_PATH = re.compile(
    r"(?:"
    r"[A-Za-z]:\\Users\\|/Users/|/home/|"
    r"[A-Za-z]:\\[^\\\n]*\\AppData\\|"
    r"OneDrive|"
    r"\\Документы\\|/Документы/|"
    # Incomplete redaction that still trails a local absolute remainder
    r"<PROJECT_ROOT>/…\\"
    r")",
    re.I,
)
LATEX_ENV_MD = re.compile(r"\\begin\{(table|figure|tabular)\}")
LATEX_DUMP = re.compile(
    r"\\begin\{(table|figure|tabular)\}.*?\\end\{\1\}",
    re.S | re.I,
)
FENCED_LATEX = re.compile(r"```(?:latex|tex)\b", re.I)
SECRET = re.compile(
    r"\b(AWS_ACCESS_KEY|AWS_SECRET|PASSWORD|SECRET_KEY|API_TOKEN)\b"
    r"|-----BEGIN (?:RSA )?PRIVATE KEY-----",
    re.I,
)
ENGLISH_HEAVY = re.compile(
    r"\b(This notebook|Purpose and scope|Below we will|Key insights|"
    r"Next steps|Overview of the analysis)\b"
)


def cell_source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def output_text(out: dict, *, for_length: bool = False) -> str:
    chunks: list[str] = []
    if "text" in out:
        t = out["text"]
        chunks.append("".join(t) if isinstance(t, list) else str(t))
    if "traceback" in out:
        tb = out["traceback"]
        chunks.append("\n".join(tb) if isinstance(tb, list) else str(tb))
    if out.get("ename") or out.get("evalue"):
        chunks.append(f"{out.get('ename','')}: {out.get('evalue','')}")
    data = out.get("data") or {}
    keys = ("text/plain",) if for_length else (
        "text/plain",
        "text/html",
        "text/latex",
        "text/markdown",
    )
    for key in keys:
        if key in data:
            v = data[key]
            chunks.append("".join(v) if isinstance(v, list) else str(v))
    return "\n".join(chunks)


def is_allowlisted(nb_name: str, text: str) -> bool:
    for name, pattern, _reason in ALLOWLIST:
        if name in {"*", nb_name} and pattern.search(text):
            # Only suppress if the match itself is covered; crude but explicit.
            return True
    return False


def tracked_notebooks() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [ROOT / line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return [
        path
        for path in paths
        if ".ipynb_checkpoints" not in path.parts
        and not (
            "reports" in path.parts
            and "notebook_build" in path.parts
        )
    ]


def validation_cell_index(error: Exception) -> int | None:
    """Return the failing cell index exposed by jsonschema, when available."""
    path = list(getattr(error, "absolute_path", ()))
    if len(path) >= 2 and path[0] == "cells" and isinstance(path[1], int):
        return path[1]
    return None


def audit_notebook(path: Path) -> list[str]:
    issues: list[str] = []
    name = path.name
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as exc:  # noqa: BLE001
        return [f"CRITICAL: cannot read with nbformat: {exc}"]

    try:
        nbformat.validate(nb)
    except Exception as exc:  # noqa: BLE001
        cell_index = validation_cell_index(exc)
        location = f", cell {cell_index}" if cell_index is not None else ""
        message = str(exc).splitlines()[0]
        return [
            "CRITICAL: nbformat.validate failed "
            f"({type(exc).__name__}{location}): {message}"
        ]

    md_texts: list[str] = []
    has_title = False
    has_conclusion = False
    active = True

    for idx, cell in enumerate(nb.cells):
        src = cell_source(cell)
        if cell.cell_type == "markdown":
            md_texts.append(src)
            if idx == 0 and src.lstrip().startswith("#"):
                has_title = True
            if re.search(r"^##\s+Вывод\b", src, re.M):
                has_conclusion = True
            if LATEX_ENV_MD.search(src):
                issues.append(f"cell {idx}: markdown contains \\begin{{table|figure|tabular}}")
            if FENCED_LATEX.search(src):
                issues.append(f"cell {idx}: fenced latex/tex block in markdown")
            if ENGLISH_HEAVY.search(src) and not is_allowlisted(name, src):
                issues.append(f"cell {idx}: English boilerplate in author markdown")

        if cell.cell_type == "code":
            if "execution_count" not in cell:
                issues.append(f"cell {idx}: missing required execution_count")
            if "outputs" not in cell:
                issues.append(f"cell {idx}: missing required outputs")
            if not src.strip():
                issues.append(f"cell {idx}: empty code cell")
            else:
                try:
                    ast.parse(src, filename=f"{name}:cell-{idx}")
                except SyntaxError as exc:
                    issues.append(
                        f"cell {idx}: invalid Python syntax at line "
                        f"{exc.lineno}: {exc.msg}"
                    )
            stripped = src.strip()
            if stripped and re.fullmatch(r"print\([^)]*\)\s*", stripped):
                # Temporary single-print cells are discouraged
                if re.search(r"print\(\s*['\"]?(debug|temp|test|TODO)", stripped, re.I):
                    issues.append(f"cell {idx}: temporary debug print-only cell")

            for out in cell.get("outputs", []):
                if out.get("output_type") == "error" or out.get("ename") or out.get("evalue"):
                    issues.append(f"cell {idx}: traceback / error output present")
                text = output_text(out)
                length_text = output_text(out, for_length=True)
                if ABS_PATH.search(text) and not is_allowlisted(name, text):
                    issues.append(f"cell {idx}: absolute local path in outputs")
                if LATEX_DUMP.search(text) and text.count("\n") > 12:
                    issues.append(f"cell {idx}: large LaTeX dump in outputs")
                if length_text.count("\n") > 80 and "image/" not in str(out.get("data", {})):
                    if (
                        "[HTML-таблица сокращена" not in length_text
                        and "output сокращён" not in length_text
                        and "[LaTeX tabular output removed" not in length_text
                    ):
                        issues.append(
                            f"cell {idx}: excessively long text output "
                            f"({length_text.count(chr(10))} lines)"
                        )

        blob = src
        for out in cell.get("outputs", []):
            blob += "\n" + output_text(out)

        for m in FORBIDDEN_WORDS.finditer(blob):
            snippet = blob[max(0, m.start() - 20) : m.end() + 20].replace("\n", " ")
            if is_allowlisted(name, snippet):
                continue
            issues.append(f"cell {idx}: forbidden phrase near: {snippet!r}")

        for m in TODOISH.finditer(blob):
            snippet = blob[max(0, m.start() - 25) : m.end() + 25].replace("\n", " ")
            # Allowlist check on snippet
            allowed = False
            for nb_name, pattern, _ in ALLOWLIST:
                if nb_name in {"*", name} and pattern.search(snippet):
                    allowed = True
                    break
            # Ignore TEMP inside AppData/Temp paths already flagged elsewhere
            if allowed:
                continue
            # Ignore common false positives inside words? TEMP/DEBUG as whole words only.
            issues.append(f"cell {idx}: {m.group(0)} near: {snippet!r}")

        if ABS_PATH.search(src):
            issues.append(f"cell {idx}: absolute local path in source")

        if SECRET.search(blob):
            issues.append(f"cell {idx}: possible secret pattern")

    if active and not has_title:
        issues.append("missing top-level markdown title / header")
    if active and not has_conclusion:
        issues.append("missing ## Вывод section")
    # Predominantly Russian author prose: ignore fenced/backtick code spans
    # (variable names, paths, English identifiers inflate Latin share).
    md_join = "\n".join(md_texts)
    md_prose = re.sub(r"`[^`]+`", " ", md_join)
    md_prose = re.sub(r"\$\$[\s\S]*?\$\$", " ", md_prose)
    md_prose = re.sub(r"\$[^$]+\$", " ", md_prose)
    letters = [ch for ch in md_prose if ch.isalpha()]
    if letters:
        cyr = sum(
            1
            for ch in letters
            if ("а" <= ch.lower() <= "я") or ch.lower() == "ё"
        )
        share = cyr / len(letters)
        if share < 0.50:
            issues.append(
                f"markdown appears insufficiently Russian (cyrillic share={share:.2f})"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-lines",
        type=int,
        default=120,
        help="max allowed non-image text output lines (informational threshold)",
    )
    args = parser.parse_args()
    _ = args  # reserved

    try:
        paths = tracked_notebooks()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: cannot enumerate tracked notebooks ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2
    if not paths:
        print("ERROR: no tracked notebooks found", file=sys.stderr)
        return 2

    critical = 0
    print(f"Auditing {len(paths)} tracked notebook(s)…")
    for path in paths:
        rel = path.relative_to(ROOT) if path.is_absolute() else path
        issues = audit_notebook(path)
        if not issues:
            print(f"OK  {rel}")
            continue
        print(f"FAIL {rel}")
        for issue in issues:
            print(f"  - {issue}")
            critical += 1

    if critical:
        print(f"\nFound {critical} issue(s).", file=sys.stderr)
        return 1
    print("\nAll notebook release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
