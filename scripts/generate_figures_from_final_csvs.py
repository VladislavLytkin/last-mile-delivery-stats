# -*- coding: utf-8 -*-
"""Generate thesis publication figures from confirmed CSVs without re-estimation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from last_mile.publication_figures import generate_all  # noqa: E402


def main() -> int:
    for path in generate_all(ROOT):
        print("generated:", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
