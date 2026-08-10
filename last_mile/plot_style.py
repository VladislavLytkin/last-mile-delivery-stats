"""Shared, isolated Matplotlib style for publication figures."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PALETTE = {
    "region_and_workmode": "#234E70",
    "region_only": "#4C78A8",
    "workmode_only": "#E07A3F",
    "treated": "#245A92",
    "control": "#7B8794",
    "text": "#202A35",
    "text_muted": "#667383",
    "grid": "#DCE3EA",
    "post": "#F2F5F8",
    "zero": "#303A44",
    "paper": "#FFFFFF",
}

TYPE_ORDER = ["region_and_workmode", "region_only", "workmode_only"]
TYPE_LABELS = {
    "region_and_workmode": "Смена региона и режима работы",
    "region_only": "Только смена региона",
    "workmode_only": "Только смена режима работы",
}
TYPE_MARKERS = {
    "region_and_workmode": "o",
    "region_only": "s",
    "workmode_only": "^",
}
TYPE_LINESTYLES = {
    "region_and_workmode": "-",
    "region_only": "--",
    "workmode_only": "-.",
}

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "font.weight": "normal",
    "axes.titlesize": 9.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "axes.labelcolor": PALETTE["text"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.color": PALETTE["text_muted"],
    "ytick.color": PALETTE["text_muted"],
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.65,
    "grid.alpha": 0.8,
    "figure.facecolor": PALETTE["paper"],
    "axes.facecolor": PALETTE["paper"],
    "savefig.facecolor": PALETTE["paper"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
}

F = TypeVar("F", bound=Callable)


@contextmanager
def plot_style() -> Iterator[None]:
    """Apply the publication style without leaking global Matplotlib state."""
    with mpl.rc_context(RC_PARAMS):
        yield


def with_plot_style(func: F) -> F:
    """Decorator variant of :func:`plot_style` for notebook helpers."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        with plot_style():
            return func(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def style_axes(ax: mpl.axes.Axes, *, horizontal_grid: bool = True) -> None:
    """Apply quiet axes scaffolding consistently."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.grid(False)
    if horizontal_grid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.65, alpha=0.8)


def percent_formatter(xmax: float = 1.0, decimals: int = 0) -> PercentFormatter:
    return PercentFormatter(xmax=xmax, decimals=decimals)


def save_figure(
    fig: mpl.figure.Figure,
    path: str | Path,
    *,
    preview_png: bool = True,
    preview_dpi: int = 240,
    close: bool = True,
) -> Path:
    """Save a vector PDF and an optional QA PNG with stable export settings."""
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        pdf_path = pdf_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "last_mile.plot_style",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
        metadata=metadata,
    )
    if preview_png:
        fig.savefig(
            pdf_path.with_suffix(".png"),
            format="png",
            dpi=preview_dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    if close:
        plt.close(fig)
    return pdf_path


def contrasting_text_color(hex_color: str) -> str:
    """Return readable white/dark ink for a solid segment fill."""
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "#FFFFFF" if luminance < 0.56 else PALETTE["text"]
