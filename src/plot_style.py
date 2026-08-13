"""Shared figure styling: palette, matplotlib defaults and save helpers.

Every figure is a static PNG destined for the report, the notebook and the
printable study guide, so the design deliberately commits to a single light
theme rather than shipping a theme-aware pair.

Colour follows the reference categorical palette **in its documented order**,
which is the ordering that clears the colour-blindness gates on adjacent
pairs. Two consequences are honoured throughout:

* Scatter plots -- where every pair of series is visually adjacent -- use at
  most the first three slots, the subset validated across all pairs.
* Identity is never carried by colour alone: every multi-series figure has a
  legend, and small series counts are additionally direct-labelled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
import numpy as np  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402

logger = logging.getLogger(__name__)

#: Categorical palette, light mode, in validated order.
PALETTE: Final[tuple[str, ...]] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
)

#: Slots safe for forms where every series pair is adjacent (scatter).
ALL_PAIRS_PALETTE: Final[tuple[str, ...]] = PALETTE[:3]

SURFACE: Final[str] = "#fcfcfb"
TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
GRID_COLOR: Final[str] = "#e5e4e0"
REFERENCE_COLOR: Final[str] = "#9a998f"

#: Ordinal ramp for the three confidence tiers, light→dark.
TIER_RAMP: Final[tuple[str, str, str]] = ("#86b6ef", "#2a78d6", "#184f95")

#: Dash patterns used as a secondary encoding alongside hue, so overlapping
#: curves stay distinguishable and identity never rests on colour alone.
LINE_STYLES: Final[tuple[Any, ...]] = (
    "solid",
    (0, (5, 2)),
    (0, (1, 1.6)),
    (0, (7, 2, 1.5, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
)

#: Line and marker geometry, per the mark specification.
LINE_WIDTH: Final[float] = 2.0
MARKER_SIZE: Final[float] = 8.0
BAR_GAP: Final[float] = 0.02

#: Default figure sizes in inches.
WIDE_FIGURE: Final[tuple[float, float]] = (10.0, 5.5)
SQUARE_FIGURE: Final[tuple[float, float]] = (6.0, 6.0)

#: Resolution for saved figures, high enough for print.
FIGURE_DPI: Final[int] = 160


def apply_style() -> None:
    """Install the shared matplotlib style: recessive axes, quiet grid."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def finish(figure: plt.Figure, path: Path) -> Path:
    """Tighten, save and close a figure.

    Args:
        figure: The figure to write.
        path: Destination path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path


def hide_spines(axis: plt.Axes, keep: Sequence[str] = ("left", "bottom")) -> None:
    """Remove chart junk by hiding all but the named spines.

    Args:
        axis: Axes to modify.
        keep: Spine names to retain.
    """
    for name, spine in axis.spines.items():
        spine.set_visible(name in keep)


