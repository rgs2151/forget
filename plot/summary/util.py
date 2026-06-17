import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from refuse.calibration import select_optimal_config

from ..plot import (
    AXES,
    AXIS_COLOR,
    HARMONIC_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    custom_cmap,
    harmonic_refusal_fluency,
    setup_style,
)


ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "store"
OUT = ROOT / "plot" / "figures"
RESULTS = ("prefill_logit", "main")
CALIB_COLS = ("judge_refusal", "judge_retention", "judge_fluency")
SHORT_AXIS_LABEL = {
    "refusal": "Refusal",
    "retention": "Retain",
    "fluency": "Fluency",
}

DATA_MODELS = [
    ("Llama-3.1-8B-Instruct", "llama8b"),
    ("Mistral-7B-Instruct-v0.3", "mistral7b"),
    ("Qwen2.5-7B-Instruct", "qwen7b"),
]

CALIB_MODELS = [
    {"family": "Llama", "label": "3.2\n1B", "key": "llama32_1b"},
    {"family": "Llama", "label": "3.2\n3B", "key": "llama32_3b"},
    {"family": "Llama", "label": "3.1\n8B", "key": "llama8b"},
    {"family": "Mistral", "label": "7B", "key": "mistral7b"},
    {"family": "Mistral", "label": "Small\n24B", "key": "mistral_small24b"},
    {"family": "Qwen", "label": "0.5B", "key": "qwen05b"},
    {"family": "Qwen", "label": "3B", "key": "qwen3b"},
    {"family": "Qwen", "label": "7B", "key": "qwen7b"},
    {"family": "Qwen", "label": "14B", "key": "qwen14b"},
    {"family": "Phi", "label": "mini", "key": "phi4mini"},
    {"family": "Phi", "label": "14B", "key": "phi4"},
]

DATASETS = [
    ("inhouse", "inhouse"),
    ("MMLU", "mmlu"),
    ("RWKU", "rwku"),
    ("ConceptVectors", "conceptvectors"),
]

FULL_METRICS = [
    ("refuse", "judge_refusal", "Refusal rate"),
    ("retain", "judge_retention", "Retain rate"),
    ("fluency", "judge_fluency", "Fluency rate"),
]


def setup_summary_style():
    setup_style()
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", transparent=False)


def result_file(store_dir, filename):
    for result in RESULTS:
        path = store_dir / "results" / result / filename
        if path.exists():
            return path
    return store_dir / "results" / RESULTS[-1] / filename


def has_cols(file_path, required_cols):
    if not file_path.exists():
        return False
    cols = pd.read_csv(file_path, nrows=0).columns
    return all(col in cols for col in required_cols)


def _layer_key(source_layer):
    layers = ast.literal_eval(source_layer) if isinstance(source_layer, str) else list(source_layer)
    return str(layers)


def _layer_value(source_layer):
    layers = ast.literal_eval(source_layer) if isinstance(source_layer, str) else list(source_layer)
    return float(np.mean(layers))


def _first_intervention_layers(df):
    return _layer_key(df["source_layer"].iloc[0])


def _first_nonzero_scale(df):
    return float(sorted(s for s in df.loc[df["label"] == "intervention", "scale"].unique() if s != 0)[0])


def heatmap_pivot(judged_csv, metric):
    df = pd.read_csv(judged_csv)
    layers = _first_intervention_layers(df)
    scale = _first_nonzero_scale(df)
    plot_df = df[
        (df["label"] == "intervention")
        & (df["scale"] == scale)
        & (df["source_layer"].astype(str) == layers)
        & (df["target_layer"].astype(str) == layers)
    ]
    concepts = list(df["concept"].unique())
    scores = (
        plot_df.pivot_table(index="concept", columns="target", values=metric, aggfunc="mean")
        .reindex(index=concepts, columns=concepts)
    )
    return scores, concepts


def draw_heatmap(ax, judged_csv, metric, cbar_label):
    scores, concepts = heatmap_pivot(judged_csv, metric)
    n = scores.shape[0]
    label_each = n <= 12 and max(len(c) for c in concepts) <= 20
    yticklabels = [f"{c} $c_{{{i + 1}}}$" for i, c in enumerate(concepts)] if label_each else False
    sns.heatmap(
        scores.fillna(0), ax=ax, cmap="Greys", square=True, vmin=0, vmax=1,
        xticklabels=False, yticklabels=yticklabels,
        cbar_kws={"shrink": 0.7, "ticks": [0, 1]},
    )
    ax.set_xticks([0.5, n - 0.5], labels=["$c_1$", f"$c_{{{n}}}$"])
    if not label_each:
        ax.set_yticks([0.5, n - 0.5], labels=["$c_1$", f"$c_{{{n}}}$"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6, length=0)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=7)


def draw_bars(ax, judged_csv):
    df = pd.read_csv(judged_csv)
    if "label" in df:
        df = df[df["label"] == "intervention"]
    df = df.assign(kind=np.where(df["concept"] == df["target"], "Target", "Untargeted"))
    score_cols = [f"judge_{axis}" for axis in AXES if f"judge_{axis}" in df.columns]
    long_df = df[["kind"] + score_cols].melt(id_vars="kind", var_name="axis", value_name="score")
    long_df["axis"] = long_df["axis"].str.replace("judge_", "").map(lambda x: SHORT_AXIS_LABEL[x])
    sns.barplot(
        data=long_df, x="axis", y="score", hue="kind", ax=ax,
        order=[SHORT_AXIS_LABEL[axis] for axis in AXES if f"judge_{axis}" in df.columns],
        hue_order=["Target", "Untargeted"],
        palette={"Target": PRIMARY_COLOR, "Untargeted": SECONDARY_COLOR},
        errorbar=("ci", 95),
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    sns.despine(ax=ax, trim=True, offset=5)


def draw_cross(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.78")
        spine.set_linewidth(1)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="0.45", linewidth=2.4)
    ax.plot([0, 1], [1, 0], transform=ax.transAxes, color="0.45", linewidth=2.4)


def draw_calibration(ax, calibration_csv, legend=False):
    df = pd.read_csv(calibration_csv)
    df = df[df["label"] == "intervention"]
    layers, scale = select_optimal_config(df)
    d = df[df["source_layer"].astype(str) == str(layers)].copy()
    d = d.assign(judge_harmonic=harmonic_refusal_fluency(d))

    for axis in AXES:
        col = f"judge_{axis}"
        sns.lineplot(
            data=d, x="scale", y=col, ax=ax,
            color=AXIS_COLOR[axis], label=SHORT_AXIS_LABEL[axis],
            estimator="mean", errorbar=None,
        )

    sns.lineplot(
        data=d, x="scale", y="judge_harmonic", ax=ax,
        color=HARMONIC_COLOR, label="Harmonic",
        estimator="mean", errorbar=None, linestyle="--",
    )

    means = d.groupby("scale", as_index=False)["judge_harmonic"].mean()
    peak = means[means["scale"] == scale].iloc[0]
    ax.plot(
        peak["scale"], peak["judge_harmonic"],
        marker="*", color=SECONDARY_COLOR, markersize=11,
        fillstyle="none", linestyle="None", label="Optimal",
    )

    mx = float(d["scale"].max())
    ax.set_xlabel("Scale", fontsize=9)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_xlim(0, mx)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(loc="lower right", fontsize=7, frameon=True, facecolor="white", edgecolor="white")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax, trim=True, offset=5)


def draw_full_calibration(ax, calibration_csv, metric, cmap):
    df = pd.read_csv(calibration_csv)
    df = df[df["label"] == "intervention"].copy()
    df["_layer_value"] = df["source_layer"].map(_layer_value)
    mx = float(df["scale"].max())
    hi = max(float(df["_layer_value"].max()), 1.0)

    for layer, layer_df in df.groupby("source_layer"):
        value = _layer_value(layer)
        trace = layer_df.groupby("scale", as_index=False)[metric].mean()
        ax.plot(trace["scale"], trace[metric], color=cmap(value / hi), linewidth=0.8)

    ax.set_xlabel("Scale", fontsize=8)
    ax.set_ylabel("Rate", fontsize=8)
    ax.set_xlim(0, mx)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=7)
    sns.despine(ax=ax, trim=True, offset=4)


def tidy_grid_axis(ax, row_idx, col_idx, n_rows):
    if row_idx != n_rows - 1:
        ax.set_xlabel("")
    if col_idx != 0:
        ax.set_ylabel("")


def add_family_headers(fig, axes, models, family_y=0.965, model_y=0.925):
    fig.canvas.draw()
    for i, model in enumerate(models):
        bbox = axes[0, i].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, model_y, model["label"],
                 ha="center", va="bottom", fontsize=9, weight="bold")

    families = []
    start = 0
    for i, model in enumerate(models + [{"family": None}]):
        if i == len(models) or model["family"] != models[start]["family"]:
            families.append((models[start]["family"], start, i - 1))
            start = i

    for family, start, end in families:
        left = axes[0, start].get_position()
        right = axes[0, end].get_position()
        x0, x1 = left.x0, right.x1
        fig.text((x0 + x1) / 2, family_y, family,
                 ha="center", va="bottom", fontsize=14, weight="bold")
        fig.add_artist(plt.Line2D([x0, x1], [family_y - 0.004, family_y - 0.004],
                                  transform=fig.transFigure, color="0.3", linewidth=0.8))


def add_row_headers(fig, axes, rows, x=0.02):
    fig.canvas.draw()
    for r, (row_label, _) in enumerate(rows):
        bbox = axes[r, 0].get_position()
        fig.text(x, (bbox.y0 + bbox.y1) / 2, row_label,
                 ha="center", va="center", rotation=90, fontsize=14, weight="bold")


def layer_colorbar(fig, cmap):
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.965, 0.30, 0.012, 0.38])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["early", "late"])
    cbar.set_label("Layer depth", rotation=270, labelpad=14)


def full_cmap():
    return custom_cmap(32)

