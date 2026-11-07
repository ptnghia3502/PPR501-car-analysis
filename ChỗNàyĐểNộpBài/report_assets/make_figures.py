"""Regenerate the figures used by Assignment3_Report.docx.

All numbers are recomputed from the same sources the notebook uses:

* ``sources/Dataset/car-price.csv`` for the listing statistics and the two OLS
  models (production year and odometer mileage).
* ``sources/CNN/outputs/car_cnn/metrics.json`` for the MobileNetV2 confusion
  matrix and the per-class precision / recall / F1 scores.
* ``sources/Dataset/Train`` and ``sources/Dataset/Test`` for the image counts.

The figures are written to ``report_assets/figures`` and consumed by
``build_report.py``.

Run from the repository root::

    python "ChỗNàyĐểNộpBài/report_assets/make_figures.py"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "sources" / "Dataset" / "car-price.csv"
CNN_DIR = ROOT / "sources" / "CNN" / "outputs" / "car_cnn"
FIG_DIR = Path(__file__).resolve().parent / "figures"

NAVY = "#2F4B7C"
ORANGE = "#F4A261"
RED = "#E5484D"
GREY = "#8A94A6"
LIGHT = "#E4EAF2"

CLASS_ORDER = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Maruti Suzuki Swift",
    "Tata Safari",
    "Toyota Innova",
]
# Two-line short forms for the rotated axis labels of the evaluation figure.
CLASS_LABELS_WRAPPED = [
    "Audi",
    "Hyundai\nCreta",
    "Mahindra\nScorpio",
    "Rolls\nRoyce",
    "Maruti\nSuzuki\nSwift",
    "Tata\nSafari",
    "Toyota\nInnova",
]
FOLDER_ORDER = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova",
]


def apply_style() -> None:
    """One visual language for every figure in the report."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4A5568",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def convert_price(value: object) -> float:
    """Convert a Vietnamese price string into millions of VND."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("₫", "")
    ty = re.search(r"([\d.,]+)\s*tỷ", text, re.IGNORECASE)
    trieu = re.search(r"([\d.,]+)\s*triệu", text, re.IGNORECASE)
    if not ty and not trieu:
        try:
            return float(text.replace(",", ""))
        except ValueError:
            return np.nan
    ty_value = float(ty.group(1).replace(".", "").replace(",", ".")) if ty else 0.0
    trieu_value = (
        float(trieu.group(1).replace(".", "").replace(",", ".")) if trieu else 0.0
    )
    return ty_value * 1000.0 + trieu_value


def load_listings() -> pd.DataFrame:
    frame = pd.read_csv(DATASET)
    frame["Price_Million"] = frame["Price"].apply(convert_price)
    frame = frame[frame["Price_Million"].gt(0)].copy()
    frame["Year"] = pd.to_numeric(frame["Năm sản xuất"], errors="coerce")
    frame["Km"] = pd.to_numeric(
        frame["Số Km đã đi"].astype(str).str.replace(r"[^0-9]", "", regex=True),
        errors="coerce",
    )
    return frame


def figure_year_distribution(frame: pd.DataFrame) -> Path:
    """Fig. 1 - how many valid listings each production year contributes."""
    valid = frame.dropna(subset=["Year", "Price_Million"])
    valid = valid[valid["Year"].between(1900, 2035)]
    counts = valid["Year"].astype(int).value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(5.6, 2.9))
    axis.bar(counts.index, counts.values, color=NAVY, width=0.82, linewidth=0)
    axis.set_xlabel("Production year")
    axis.set_ylabel("Number of listings")
    axis.set_title(
        f"Valid listings by production year (n = {len(valid):,})",
        pad=7,
    )
    axis.set_xlim(counts.index.min() - 1.2, counts.index.max() + 1.2)
    axis.set_xticks(np.arange(counts.index.min(), counts.index.max() + 1, 5))
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    figure.tight_layout()
    path = FIG_DIR / "fig1_year_distribution.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_image_counts() -> Path:
    """Fig. 2 - train / test split for every vehicle class."""
    train_dir = ROOT / "sources" / "Dataset" / "Train"
    test_dir = ROOT / "sources" / "Dataset" / "Test"
    rows = []
    for index, folder in enumerate(FOLDER_ORDER):
        train = len(list((train_dir / folder).glob("*.jpg")))
        test = len(list((test_dir / folder).glob("*.jpg")))
        rows.append((CLASS_ORDER[index], train, test))
    counts = pd.DataFrame(rows, columns=["Class", "Train", "Test"])
    counts["Total"] = counts["Train"] + counts["Test"]
    counts = counts.sort_values("Total", ascending=False).reset_index(drop=True)

    positions = np.arange(len(counts))
    figure, axis = plt.subplots(figsize=(5.6, 3.0))
    axis.bar(
        positions - 0.19,
        counts["Train"],
        width=0.36,
        color=NAVY,
        label="Training",
        linewidth=0,
    )
    axis.bar(
        positions + 0.19,
        counts["Test"],
        width=0.36,
        color=ORANGE,
        label="Test",
        linewidth=0,
    )
    for index, row in counts.iterrows():
        axis.text(
            index - 0.19,
            row["Train"] + 12,
            f"{row['Train']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=NAVY,
        )
        axis.text(
            index + 0.19,
            row["Test"] + 12,
            f"{row['Test']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#B9722E",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(counts["Class"], rotation=18, ha="right")
    axis.set_ylabel("Number of images")
    axis.set_title("Vehicle image dataset: training and test splits", pad=7)
    axis.set_ylim(0, int(counts["Total"].max()) * 1.34)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", ncols=2)
    figure.tight_layout()
    path = FIG_DIR / "fig2_image_counts.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _architecture_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 7.4,
    weight: str = "normal",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            facecolor=face,
            edgecolor=edge,
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1F2937",
        weight=weight,
        zorder=3,
        linespacing=1.35,
    )


def _architecture_arrow(axis, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#7A869A",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def figure_architecture() -> Path:
    """Fig. 3 - the three decoupled processing tracks of the application."""
    figure, axis = plt.subplots(figsize=(6.4, 3.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.text(
        0.5,
        0.965,
        "Modular architecture of the Streamlit application",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    _architecture_box(
        axis, 0.015, 0.60, 0.17, 0.20, "User text query\n(e.g. “Toyota\nInnova 2020 7 seats”)", "#F4F6FA", "#C6CFDC"
    )
    _architecture_box(
        axis, 0.015, 0.16, 0.17, 0.20, "Uploaded photo\n(vehicle image)", "#F4F6FA", "#C6CFDC"
    )
    _architecture_box(
        axis, 0.015, 0.36, 0.17, 0.17, "Listing table\ncar-price.csv", "#F4F6FA", "#C6CFDC"
    )

    _architecture_box(
        axis, 0.245, 0.60, 0.21, 0.20, "Query parsing\nregular expressions\n(attribute filters)", "#EFF4FB", NAVY
    )
    _architecture_box(
        axis, 0.245, 0.16, 0.21, 0.20, "Pre-processing\n224 × 224 resize,\nnormalisation", "#EFF4FB", NAVY
    )
    _architecture_box(
        axis, 0.245, 0.36, 0.21, 0.17, "Price parsing\nand cleaning", "#EFF4FB", NAVY
    )

    _architecture_box(
        axis,
        0.505,
        0.60,
        0.21,
        0.20,
        "Hybrid ranking\nMiniLM bi-encoder\n+ TF-IDF fusion",
        "#FFF4E8",
        ORANGE,
    )
    _architecture_box(
        axis, 0.505, 0.16, 0.21, 0.20, "MobileNetV2\ntransfer learning\n(7 classes)", "#FFF4E8", ORANGE
    )
    _architecture_box(
        axis, 0.505, 0.36, 0.21, 0.17, "Ordinary least\nsquares model", "#FFF4E8", ORANGE
    )

    _architecture_box(
        axis, 0.775, 0.60, 0.21, 0.20, "Ranked listing cards\nand price trend charts", "#EAF6F4", "#3E9B90"
    )
    _architecture_box(
        axis, 0.775, 0.16, 0.21, 0.20, "Class label, probability\nand confidence gate", "#EAF6F4", "#3E9B90"
    )
    _architecture_box(
        axis, 0.775, 0.36, 0.21, 0.17, "Trend line, R² and\nRMSE summary", "#EAF6F4", "#3E9B90"
    )

    _architecture_arrow(axis, (0.19, 0.70), (0.24, 0.70))
    _architecture_arrow(axis, (0.19, 0.26), (0.24, 0.26))
    _architecture_arrow(axis, (0.19, 0.445), (0.24, 0.445))

    _architecture_arrow(axis, (0.46, 0.70), (0.50, 0.70))
    _architecture_arrow(axis, (0.46, 0.26), (0.50, 0.26))
    _architecture_arrow(axis, (0.46, 0.445), (0.50, 0.445))

    _architecture_arrow(axis, (0.72, 0.70), (0.77, 0.70))
    _architecture_arrow(axis, (0.72, 0.26), (0.77, 0.26))
    _architecture_arrow(axis, (0.72, 0.445), (0.77, 0.445))

    axis.text(
        0.5,
        0.045,
        "Shared session state · SHA-256 keyed caches for embeddings and TF-IDF · CPU-only inference",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#5B6577",
        style="italic",
    )
    figure.tight_layout()
    path = FIG_DIR / "fig3_architecture.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _regression_panel(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    title: str,
    xlim: tuple[float, float] | None,
    r2: float,
    n: int,
    y_cap: float = 5000.0,
) -> None:
    """Scatter + OLS line + binned means with an inter-quartile band.

    The panel title is kept short so that it cannot run into the title of the
    neighbouring panel; the sample size and the goodness of fit are printed
    inside the axes instead.
    """
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    grid = np.linspace(x.min(), x.max(), 200)
    axis.scatter(
        x,
        y,
        s=6,
        alpha=0.16,
        color=NAVY,
        linewidths=0,
        label="Listings",
        zorder=1,
    )
    axis.plot(
        grid,
        model.predict(grid.reshape(-1, 1)),
        color=RED,
        linewidth=1.9,
        label=f"OLS fit (R² = {r2:.3f})",
        zorder=3,
    )

    bins = np.linspace(x.min(), x.max(), 41)
    index = np.digitize(x, bins)
    centres, means, low, high = [], [], [], []
    for bucket in range(1, len(bins)):
        mask = index == bucket
        if mask.sum() < 15:
            continue
        values = y[mask]
        centres.append(0.5 * (bins[bucket - 1] + bins[bucket]))
        means.append(np.median(values))
        low.append(np.percentile(values, 25))
        high.append(np.percentile(values, 75))
    axis.fill_between(
        centres,
        low,
        high,
        color=LIGHT,
        alpha=0.95,
        linewidth=0,
        label="Inter-quartile range",
        zorder=2,
    )
    axis.plot(
        centres,
        means,
        color=NAVY,
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=2.6,
        label="Binned median",
        zorder=4,
    )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Listing price (million VND)")
    axis.set_ylim(0, y_cap)
    axis.set_title(f"{title}\nn = {n:,};  R² = {r2:.3f}", pad=6, fontsize=9.5, linespacing=1.4)
    if xlim is not None:
        axis.set_xlim(*xlim)
    else:
        axis.set_xlim(x.min() - 1.0, x.max() + 1.0)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        ncols=2,
        columnspacing=1.0,
        handlelength=1.6,
        fontsize=7.5,
    )


def figure_year_regression(frame: pd.DataFrame) -> Path:
    """Fig. 4 - price against production year."""
    valid = frame.dropna(subset=["Year", "Price_Million"])
    valid = valid[valid["Year"].between(1900, 2035)]
    x = valid["Year"].to_numpy(dtype=float)
    y = valid["Price_Million"].to_numpy(dtype=float)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = model.score(x.reshape(-1, 1), y)

    figure, axis = plt.subplots(figsize=(5.6, 3.1))
    _regression_panel(
        axis,
        x,
        y,
        "Production year",
        "Listing price against production year",
        None,
        r2,
        len(valid),
    )
    figure.tight_layout()
    path = FIG_DIR / "fig4_price_vs_year.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_mileage_regression(frame: pd.DataFrame) -> Path:
    """Fig. 5 - price against odometer reading."""
    known = frame.dropna(subset=["Km", "Price_Million"])
    known = known[known["Km"].ge(0)]
    capped = known[known["Km"].between(0, 100_000)]
    x = capped["Km"].to_numpy(dtype=float)
    y = capped["Price_Million"].to_numpy(dtype=float)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = model.score(x.reshape(-1, 1), y)

    figure, axis = plt.subplots(figsize=(5.6, 3.1))
    _regression_panel(
        axis,
        x,
        y,
        "Odometer reading (km)",
        "Listing price against odometer reading",
        (0, 100_000),
        r2,
        len(capped),
    )
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k")
    )
    figure.tight_layout()
    path = FIG_DIR / "fig5_price_vs_mileage.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_body_style(frame: pd.DataFrame) -> Path:
    """Fig. 6 - price distribution per body style."""
    styles = frame["Kiểu dáng"].value_counts()
    keep = [style for style in styles.index if styles[style] >= 60]
    subset = frame[frame["Kiểu dáng"].isin(keep)]
    groups = [
        subset.loc[subset["Kiểu dáng"] == style, "Price_Million"].to_numpy()
        for style in keep
    ]
    groups = [np.clip(group, 0.2, None) for group in groups]
    order = np.argsort([np.median(group) for group in groups])

    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    box = axis.boxplot(
        [groups[index] for index in order],
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": RED, "linewidth": 1.5},
        whiskerprops={"color": GREY, "linewidth": 0.9},
        capprops={"color": GREY, "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(LIGHT)
        patch.set_edgecolor(NAVY)
        patch.set_linewidth(0.9)
    axis.set_yticks(range(1, len(order) + 1))
    axis.set_yticklabels([keep[index] for index in order])
    axis.set_xscale("log")
    axis.set_xlabel("Listing price (million VND, logarithmic scale)")
    axis.set_title("Listing price distribution by body style", pad=7)
    axis.set_xlim(right=axis.get_xlim()[1] * 1.35)
    axis.set_xticks([10, 30, 100, 300, 1000, 3000, 10000, 30000])
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda value, _pos: f"{value:,.0f}".replace(",", " ")
        )
    )
    axis.set_axisbelow(True)
    figure.tight_layout()
    path = FIG_DIR / "fig6_price_by_body_style.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def figure_confusion_matrix() -> tuple[Path, pd.DataFrame]:
    """Fig. 7 - row-normalised confusion matrix."""
    metrics = json.loads((CNN_DIR / "metrics.json").read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = _safe_divide(np.diag(matrix), predicted)
    recall = _safe_divide(np.diag(matrix), support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    per_class = pd.DataFrame(
        {
            "Class": CLASS_ORDER,
            "Support": support.astype(int),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )

    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    image = axis.imshow(
        _safe_divide(matrix, support[:, None]),
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column] / support[row]
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=6.6,
                color="white" if value > 0.5 else "#1F2937",
            )
    axis.set_xticks(range(len(CLASS_ORDER)))
    axis.set_xticklabels(CLASS_LABELS_WRAPPED, rotation=40, ha="right")
    axis.set_yticks(range(len(CLASS_ORDER)))
    axis.set_yticklabels(CLASS_LABELS_WRAPPED)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"MobileNetV2 confusion matrix (n = {int(support.sum())} test images)",
        pad=7,
    )
    axis.grid(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    bar.set_label("Recall per row", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    figure.tight_layout()
    path = FIG_DIR / "fig7_confusion_matrix.png"
    figure.savefig(path)
    plt.close(figure)
    return path, per_class


def figure_per_class_scores(per_class: pd.DataFrame) -> Path:
    """Fig. 8 - per-class precision / recall / F1 against the overall metrics."""
    accuracy = per_class["Support"][0] / per_class["Support"].sum()
    accuracy = (
        float(
            pd.read_json(CNN_DIR / "metrics.json", typ="series")["test_accuracy"]
        )
    )
    balanced = float(per_class["Recall"].mean())

    positions = np.arange(len(per_class))
    width = 0.26
    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    axis.bar(
        positions - width,
        per_class["Precision"],
        width=width,
        color=NAVY,
        label="Precision",
        linewidth=0,
    )
    axis.bar(
        positions,
        per_class["Recall"],
        width=width,
        color=ORANGE,
        label="Recall",
        linewidth=0,
    )
    axis.bar(
        positions + width,
        per_class["F1"],
        width=width,
        color="#8FB6E3",
        label="F1",
        linewidth=0,
    )
    axis.axhline(
        accuracy, color=RED, linewidth=1.2, linestyle="--", label=f"Accuracy {accuracy:.2%}"
    )
    axis.axhline(
        balanced,
        color="#3E9B90",
        linewidth=1.2,
        linestyle=":",
        label=f"Balanced accuracy {balanced:.2%}",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASS_LABELS_WRAPPED, rotation=18, ha="right")
    axis.set_ylim(0.78, 1.03)
    axis.set_ylabel("Score")
    axis.set_title("Per-class classification scores on the test split", pad=6)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncols=3,
        columnspacing=1.2,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    figure.tight_layout()
    path = FIG_DIR / "fig8_per_class_scores.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def build_price_panels_figure(frame: pd.DataFrame):
    """Build the two-panel price figure and return ``(figure, axes)``."""
    valid = frame.dropna(subset=["Year", "Price_Million"])
    valid = valid[valid["Year"].between(1900, 2035)]
    year_x = valid["Year"].to_numpy(dtype=float)
    year_y = valid["Price_Million"].to_numpy(dtype=float)
    year_model = LinearRegression().fit(year_x.reshape(-1, 1), year_y)
    year_r2 = year_model.score(year_x.reshape(-1, 1), year_y)

    known = frame.dropna(subset=["Km", "Price_Million"])
    known = known[known["Km"].ge(0)]
    capped = known[known["Km"].between(0, 100_000)]
    km_x = capped["Km"].to_numpy(dtype=float)
    km_y = capped["Price_Million"].to_numpy(dtype=float)
    km_model = LinearRegression().fit(km_x.reshape(-1, 1), km_y)
    km_r2 = km_model.score(km_x.reshape(-1, 1), km_y)

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    _regression_panel(
        axes[0],
        year_x,
        year_y,
        "Production year",
        "Price against production year",
        None,
        year_r2,
        len(valid),
    )
    _regression_panel(
        axes[1],
        km_x,
        km_y,
        "Odometer reading (km)",
        "Price against odometer reading",
        (0, 100_000),
        km_r2,
        len(capped),
    )
    axes[1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k")
    )
    figure.tight_layout()
    return figure, axes


def figure_price_panels(frame: pd.DataFrame) -> Path:
    """Fig. 4 - the two regressions side by side in a single figure."""
    figure, _axes = build_price_panels_figure(frame)
    path = FIG_DIR / "fig4_price_panels.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def build_cnn_evaluation_figure(matrix: np.ndarray, per_class: pd.DataFrame, accuracy: float):
    """Build the two-panel CNN evaluation figure and return ``(figure, axes)``."""
    support = matrix.sum(axis=1)
    balanced = float(per_class["Recall"].mean())

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), width_ratios=[1.15, 1.15])

    image = axes[0].imshow(
        _safe_divide(matrix, support[:, None]),
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column] / support[row]
            axes[0].text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=6.6,
                color="white" if value > 0.5 else "#1F2937",
            )
    axes[0].set_xticks(range(len(CLASS_ORDER)))
    axes[0].set_xticklabels(CLASS_LABELS_WRAPPED, rotation=0, ha="center", fontsize=7)
    axes[0].set_yticks(range(len(CLASS_ORDER)))
    axes[0].set_yticklabels(CLASS_ORDER, fontsize=6.4)
    axes[0].set_xlabel("Predicted class", fontsize=8)
    axes[0].set_ylabel("True class", fontsize=8)
    axes[0].set_title(
        f"Confusion matrix (n = {int(support.sum())} test images)", pad=6, fontsize=9
    )
    axes[0].grid(False)
    bar = figure.colorbar(image, ax=axes[0], fraction=0.04, pad=0.02)
    bar.set_label("Recall per row", fontsize=7)
    bar.ax.tick_params(labelsize=6)

    positions = np.arange(len(per_class))
    width = 0.26
    axes[1].bar(
        positions - width,
        per_class["Precision"],
        width=width,
        color=NAVY,
        label="Precision",
        linewidth=0,
    )
    axes[1].bar(
        positions,
        per_class["Recall"],
        width=width,
        color=ORANGE,
        label="Recall",
        linewidth=0,
    )
    axes[1].bar(
        positions + width,
        per_class["F1"],
        width=width,
        color="#8FB6E3",
        label="F1",
        linewidth=0,
    )
    axes[1].axhline(
        accuracy, color=RED, linewidth=1.1, linestyle="--", label=f"Accuracy {accuracy:.2%}"
    )
    axes[1].axhline(
        balanced,
        color="#3E9B90",
        linewidth=1.1,
        linestyle=":",
        label=f"Balanced accuracy {balanced:.2%}",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(CLASS_LABELS_WRAPPED, rotation=0, ha="center", fontsize=7)
    axes[1].set_ylim(0.78, 1.03)
    axes[1].set_ylabel("Score", fontsize=8)
    axes[1].set_title("Per-class scores on the test split", pad=6, fontsize=9)
    axes[1].set_axisbelow(True)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        ncols=3,
        columnspacing=1.0,
        handlelength=1.4,
        fontsize=7,
        borderaxespad=0.0,
    )
    figure.tight_layout()
    return figure, axes


def figure_confusion_and_scores(
    matrix: np.ndarray, per_class: pd.DataFrame, accuracy: float
) -> Path:
    """Fig. 5 - confusion matrix and per-class scores in a single figure."""
    figure, _axes = build_cnn_evaluation_figure(matrix, per_class, accuracy)
    path = FIG_DIR / "fig5_cnn_evaluation.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_listings()
    figure_year_distribution(frame)
    figure_architecture()
    figure_price_panels(frame)
    figure_body_style(frame)
    metrics = json.loads((CNN_DIR / "metrics.json").read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = _safe_divide(np.diag(matrix), predicted)
    recall = _safe_divide(np.diag(matrix), support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    per_class = pd.DataFrame(
        {
            "Class": CLASS_ORDER,
            "Support": support.astype(int),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )
    figure_confusion_and_scores(
        matrix, per_class, float(metrics["test_accuracy"])
    )
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
