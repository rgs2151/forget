"""Secondary diagnostics for the completed direction-transfer experiment."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


UNIT_DIR = Path(__file__).resolve().parent
RUN_DIR = UNIT_DIR / "cache" / "main"
FIGURE_DIR = UNIT_DIR / "plots"
DIAGNOSTIC_DIR = RUN_DIR / "diagnostics"

PRIMARY = "#980000"
DARK_GRAY = "#444444"
REFUSAL_CMAP = LinearSegmentedColormap.from_list(
    "refusal_rate",
    ["#ffffff", PRIMARY],
)
CATEGORIES = ("space", "places", "engineering")
CONDITIONS = ("baseline", "native", "transfer")
CONCEPTS = (
    "bacteria",
    "cats",
    "chess",
    "dogs",
    "lasers",
    "obama",
    "paris",
    "people",
    "the_moon",
    "united_states",
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "savefig.facecolor": "white",
    })


def load_config() -> dict:
    return json.loads((UNIT_DIR / "frozen_config.json").read_text())


def save_csv(frame: pd.DataFrame, name: str) -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DIAGNOSTIC_DIR / name, index=False)


def concept_rates(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for category in CATEGORIES:
        targets = set(config["categories"][category]["target_concepts"])
        for condition in CONDITIONS:
            path = (
                RUN_DIR
                / "full"
                / "evaluation"
                / f"{category}__{condition}_judged.csv"
            )
            judged = pd.read_csv(path)
            grouped = judged.groupby("concept", as_index=False).agg(
                refusal_rate=("judge_refusal", "mean"),
                n=("judge_refusal", "size"),
            )
            for row in grouped.itertuples(index=False):
                rows.append({
                    "category": category,
                    "condition": condition,
                    "concept": row.concept,
                    "is_target": row.concept in targets,
                    "refusal_rate": float(row.refusal_rate),
                    "n": int(row.n),
                })
    result = pd.DataFrame(rows)
    save_csv(result, "concept_rates.csv")
    return result


def display_concept(concept: str) -> str:
    if concept == "united_states":
        return "USA"
    return concept.replace("_", " ")


def plot_concept_footprint(config: dict, rates: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.2, 3.2))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.035],
        wspace=0.13,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_ax = fig.add_subplot(grid[0, 3])
    image = None
    for index, (ax, category) in enumerate(zip(axes, CATEGORIES)):
        data = rates[rates["category"].eq(category)]
        matrix = np.full((len(CONDITIONS), len(CONCEPTS)), np.nan)
        for row_index, condition in enumerate(CONDITIONS):
            lookup = (
                data[data["condition"].eq(condition)]
                .set_index("concept")["refusal_rate"]
                .to_dict()
            )
            matrix[row_index] = [lookup.get(concept, np.nan) for concept in CONCEPTS]

        image = ax.imshow(
            matrix,
            cmap=REFUSAL_CMAP,
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="none",
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    continue
                color = "white" if value >= 0.55 else "black"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6.5,
                )

        targets = set(config["categories"][category]["target_concepts"])
        for column_index, concept in enumerate(CONCEPTS):
            if concept in targets:
                ax.add_patch(Rectangle(
                    (column_index - 0.5, -0.5),
                    1,
                    len(CONDITIONS),
                    fill=False,
                    edgecolor="black",
                    linewidth=1.8,
                ))

        ax.set_title(category.capitalize(), fontweight="bold")
        ax.set_xticks(
            np.arange(len(CONCEPTS)),
            [display_concept(concept) for concept in CONCEPTS],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(len(CONDITIONS)), [value.capitalize() for value in CONDITIONS])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        if index > 0:
            ax.tick_params(axis="y", labelleft=False)

    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Refusal rate")
    colorbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.32, top=0.87)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "concept_footprint.png", dpi=300)
    plt.close(fig)


def parse_layer(value: object) -> int:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return int(parsed[0])
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(value[0])
    return int(value)


def calibration_rates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for category in CATEGORIES:
        for condition in ("native", "transfer"):
            path = (
                RUN_DIR
                / "full"
                / "calibration"
                / f"{category}__{condition}_judged.csv"
            )
            judged = pd.read_csv(path)
            judged["layer"] = judged["source_layer"].map(parse_layer)
            grouped = judged.groupby(["layer", "scale"], as_index=False).agg(
                refusal_rate=("judge_refusal", "mean"),
                fluency_rate=("judge_fluency", "mean"),
                n=("judge_refusal", "size"),
            )
            grouped.insert(0, "condition", condition)
            grouped.insert(0, "category", category)
            rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    save_csv(result, "calibration_rates.csv")
    return result


def plot_calibration_landscape(rates: pd.DataFrame) -> None:
    points = pd.read_csv(RUN_DIR / "full" / "operating_points.csv")
    fig, axes = plt.subplots(3, 2, figsize=(5.2, 6.2), sharex=True, sharey=True)
    image = None
    for row_index, category in enumerate(CATEGORIES):
        for column_index, condition in enumerate(("native", "transfer")):
            ax = axes[row_index, column_index]
            data = rates[
                rates["category"].eq(category)
                & rates["condition"].eq(condition)
            ]
            matrix = (
                data.pivot(index="layer", columns="scale", values="refusal_rate")
                .reindex(index=range(16), columns=range(1, 11))
                .to_numpy()
            )
            image = ax.imshow(
                matrix,
                origin="lower",
                cmap=REFUSAL_CMAP,
                vmin=0,
                vmax=1,
                aspect="auto",
                interpolation="none",
            )
            point = points[
                points["category"].eq(category)
                & points["condition"].eq(condition)
            ].iloc[0]
            ax.scatter(
                float(point["scale"]) - 1,
                int(point["layer"]),
                marker="*",
                s=95,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=3,
            )
            ax.set_xticks(np.arange(10), np.arange(1, 11))
            ax.set_yticks([0, 5, 10, 15])
            ax.tick_params(length=0)
            if row_index == 0:
                title = "Native Inhouse" if condition == "native" else "Transferred MMLU"
                ax.set_title(title, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{category.capitalize()}\nLayer", fontweight="bold")
            if row_index == len(CATEGORIES) - 1:
                ax.set_xlabel("Scale")
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Refusal rate")
    colorbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.08, top=0.94, hspace=0.18, wspace=0.10)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "calibration_landscape.png", dpi=300)
    plt.close(fig)


def vector_stability() -> pd.DataFrame:
    points = pd.read_csv(RUN_DIR / "samples" / "operating_points.csv")
    rows: list[dict] = []
    for category in CATEGORIES:
        reference_path = (
            RUN_DIR
            / "vectors"
            / "samples"
            / f"{category}__n093__full.pt"
        )
        reference = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=True,
        )["v_detect"]["__direction__"]
        category_points = points[points["category"].eq(category)]
        for row in category_points.itertuples(index=False):
            bundle = torch.load(
                row.vector_path,
                map_location="cpu",
                weights_only=True,
            )
            direction = bundle["v_detect"]["__direction__"]
            layer = int(row.layer)
            cosine = float(F.cosine_similarity(
                direction[layer, 0].reshape(1, -1),
                reference[layer, 0].reshape(1, -1),
            ).item())
            rows.append({
                "variant_id": row.variant_id,
                "category": category,
                "sample_n": int(row.sample_n),
                "seed": row.seed,
                "layer": layer,
                "selected_scale": float(row.scale),
                "cosine_to_full_direction": cosine,
            })
    result = pd.DataFrame(rows)
    save_csv(result, "vector_stability.csv")
    return result


def plot_sample_diagnostics(stability: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.4), sharex=True)
    sample_sizes = [2, 4, 8, 16, 32, 64, 93]
    for column_index, category in enumerate(CATEGORIES):
        data = stability[stability["category"].eq(category)].copy()
        nonfull = data[data["seed"].astype(str).ne("full")]
        for seed, trace in nonfull.groupby("seed"):
            trace = trace.sort_values("sample_n")
            axes[0, column_index].plot(
                trace["sample_n"],
                trace["cosine_to_full_direction"],
                color="#999999",
                linewidth=0.9,
                alpha=0.8,
                label="draw" if column_index == 0 and str(seed) == "42" else None,
            )
            axes[1, column_index].plot(
                trace["sample_n"],
                trace["selected_scale"],
                color="#999999",
                linewidth=0.9,
                alpha=0.8,
            )

        means = data.groupby("sample_n", as_index=False).agg(
            cosine=("cosine_to_full_direction", "mean"),
            scale=("selected_scale", "mean"),
        )
        axes[0, column_index].plot(
            means["sample_n"],
            means["cosine"],
            color=PRIMARY,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="mean" if column_index == 0 else None,
        )
        axes[1, column_index].plot(
            means["sample_n"],
            means["scale"],
            color=PRIMARY,
            marker="o",
            linewidth=1.8,
            markersize=4,
        )

        axes[0, column_index].set_title(category.capitalize(), fontweight="bold")
        for row_index in range(2):
            ax = axes[row_index, column_index]
            ax.set_xscale("log", base=2)
            ax.set_xticks(sample_sizes)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(width=1.2)
        axes[0, column_index].set_ylim(0, 1.05)
        axes[1, column_index].set_ylim(0.5, 10.5)
        axes[1, column_index].set_yticks([1, 4, 7, 10])
        axes[1, column_index].set_xlabel("Source examples")

    axes[0, 0].set_ylabel("Cosine to full direction")
    axes[1, 0].set_ylabel("Selected scale")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "sample_vector_diagnostics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    configure_style()
    config = load_config()
    rates = concept_rates(config)
    plot_concept_footprint(config, rates)
    calibration = calibration_rates()
    plot_calibration_landscape(calibration)
    stability = vector_stability()
    plot_sample_diagnostics(stability)
    for name in (
        "concept_footprint.png",
        "calibration_landscape.png",
        "sample_vector_diagnostics.png",
    ):
        path = FIGURE_DIR / name
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
