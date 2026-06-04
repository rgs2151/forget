import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from refuse.calibration import select_optimal_config

from .plot import (
    AXES,
    AXIS_COLOR,
    HARMONIC_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    custom_cmap,
    harmonic_refusal_fluency,
    setup_style,
)


STORE = Path("store")
OUT = Path("plot/figures")
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

ERRORED = {
    ("mistral7b", "conceptvectors"),
    ("qwen7b", "mmlu"),
    ("qwen7b", "conceptvectors"),
}

SCORE_RUNS = [
    {"run": "qwen05b_inhouse", "model": "Qwen 0.5B", "label": "Qwen .5B", "size_b": 0.5},
    {"run": "llama32_1b_inhouse", "model": "Llama 3.2 1B", "label": "L3.2 1B", "size_b": 1.0},
    {"run": "llama32_3b_inhouse", "model": "Llama 3.2 3B", "label": "L3.2 3B", "size_b": 3.0},
    {"run": "qwen3b_inhouse", "model": "Qwen 3B", "label": "Qwen 3B", "size_b": 3.0},
    {"run": "phi4mini_inhouse", "model": "Phi-4 mini", "label": "Phi mini", "size_b": 3.8},
    {"run": "mistral7b_inhouse", "model": "Mistral 7B", "label": "Mistral 7B", "size_b": 7.0},
    {"run": "qwen7b_inhouse", "model": "Qwen 7B", "label": "Qwen 7B", "size_b": 7.0},
    {"run": "llama8b_inhouse", "model": "Llama 3.1 8B", "label": "L3.1 8B", "size_b": 8.0},
    {"run": "phi4_inhouse", "model": "Phi-4 14B", "label": "Phi-4 14B", "size_b": 14.0},
    {"run": "qwen14b_inhouse", "model": "Qwen 14B", "label": "Qwen 14B", "size_b": 14.0},
]

SCORE_PANELS = [
    ("judge_refusal", "Refusal rate", AXIS_COLOR["refusal"]),
    ("judge_retention", "Retain rate", AXIS_COLOR["retention"]),
    ("judge_fluency", "Fluency rate", AXIS_COLOR["fluency"]),
]

SCORE_LABEL_OFFSETS = {
    "judge_refusal": {
        "qwen7b_inhouse": (4, 4),
        "qwen14b_inhouse": (4, 2),
        "phi4mini_inhouse": (4, 8),
    },
    "judge_retention": {
        "llama32_1b_inhouse": (-10, -12),
        "llama32_3b_inhouse": (4, -12),
        "phi4mini_inhouse": (4, 12),
        "qwen05b_inhouse": (4, 5),
        "mistral7b_inhouse": (4, -12),
        "llama8b_inhouse": (4, 5),
        "phi4_inhouse": (4, -12),
        "qwen14b_inhouse": (4, 5),
    },
    "judge_fluency": {
        "phi4mini_inhouse": (4, 5),
        "mistral7b_inhouse": (4, 6),
        "qwen7b_inhouse": (4, -10),
        "qwen14b_inhouse": (4, -12),
    },
}

FULL_METRICS = [
    ("refuse", "judge_refusal", "Refusal rate"),
    ("retain", "judge_retention", "Retain rate"),
    ("fluency", "judge_fluency", "Fluency rate"),
]


def status_for(file_path, model_key, row_key, required_cols=()):
    if not file_path.exists():
        return "error" if (model_key, row_key) in ERRORED else "calculating"
    if required_cols:
        cols = pd.read_csv(file_path, nrows=0).columns
        if not all(c in cols for c in required_cols):
            return "calculating"
    return None


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


def draw_status_box(ax, status):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.75")
        spine.set_linewidth(1)
    ax.text(0.5, 0.5, status, transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color="0.45", style="italic")


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


def write_model_data(store, out):
    setup_style()
    fig, axes = plt.subplots(4, 6, figsize=(20, 13))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.55, hspace=0.55)

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, (_, model_key) in enumerate(DATA_MODELS):
            store_dir = store / f"{model_key}_{row_key}"
            conf_csv = store_dir / "confusion_judged.csv"
            bars_csv = store_dir / "bars_judged.csv"
            ax_left = axes[r, mi * 2]
            ax_right = axes[r, mi * 2 + 1]

            status = status_for(conf_csv, model_key, row_key, ("judge_refusal",))
            if status is None:
                draw_heatmap(ax_left, conf_csv, "judge_refusal", "Refusal")
            else:
                draw_status_box(ax_left, status)

            if row_key == "inhouse":
                status = status_for(conf_csv, model_key, row_key, ("judge_retention",))
                if status is None:
                    draw_heatmap(ax_right, conf_csv, "judge_retention", "Retain")
                else:
                    draw_status_box(ax_right, status)
            else:
                status = status_for(bars_csv, model_key, row_key, CALIB_COLS)
                if status is None:
                    draw_bars(ax_right, bars_csv)
                else:
                    draw_status_box(ax_right, status)

    fig.canvas.draw()
    for mi, (model_label, _) in enumerate(DATA_MODELS):
        left = axes[0, mi * 2].get_position()
        right = axes[0, mi * 2 + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, 0.95, model_label,
                 ha="center", va="bottom", fontsize=15, weight="bold")

    for r, (row_label, _) in enumerate(DATASETS):
        bbox = axes[r, 0].get_position()
        fig.text(0.025, (bbox.y0 + bbox.y1) / 2, row_label,
                 ha="center", va="center", rotation=90, fontsize=14, weight="bold")

    save_path = out / "model_data.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_calib_optimal(store, out):
    setup_style()
    n_rows, n_cols = len(DATASETS), len(CALIB_MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10.5))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.84, bottom=0.07,
                        wspace=0.40, hspace=0.55)
    legend_drawn = False

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, model in enumerate(CALIB_MODELS):
            model_key = model["key"]
            cal_csv = store / f"{model_key}_{row_key}" / "calibration_judged.csv"
            ax = axes[r, mi]
            if has_cols(cal_csv, CALIB_COLS):
                draw_calibration(ax, cal_csv, legend=not legend_drawn)
                legend_drawn = True
                tidy_grid_axis(ax, r, mi, n_rows)
            else:
                draw_cross(ax)

    add_family_headers(fig, axes, CALIB_MODELS, family_y=0.965, model_y=0.91)
    add_row_headers(fig, axes, DATASETS, x=0.018)

    save_path = out / "calib_optimal.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_calib_full_metric(store, out, title, metric, ylabel):
    setup_style()
    n_rows, n_cols = len(DATASETS), len(CALIB_MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10.8))
    fig.subplots_adjust(left=0.05, right=0.955, top=0.82, bottom=0.07,
                        wspace=0.40, hspace=0.55)
    cmap = custom_cmap(32)

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, model in enumerate(CALIB_MODELS):
            model_key = model["key"]
            cal_csv = store / f"{model_key}_{row_key}" / "calibration_judged.csv"
            ax = axes[r, mi]
            if has_cols(cal_csv, (metric,)):
                draw_full_calibration(ax, cal_csv, metric, cmap)
                if mi == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                tidy_grid_axis(ax, r, mi, n_rows)
            else:
                draw_cross(ax)

    add_family_headers(fig, axes, CALIB_MODELS, family_y=0.94, model_y=0.885)
    add_row_headers(fig, axes, DATASETS, x=0.018)
    fig.suptitle(title, fontsize=22, weight="bold", y=0.992)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.965, 0.30, 0.012, 0.38])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["early", "late"])
    cbar.set_label("Layer depth", rotation=270, labelpad=14)

    save_path = out / f"calib_full_{title}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_calib_full(store, out):
    return [write_calib_full_metric(store, out, title, metric, ylabel)
            for title, metric, ylabel in FULL_METRICS]


def optimal_cell(df):
    layers, scale = select_optimal_config(df)
    layer_key = str(layers)
    d = df[df["label"] == "intervention"]
    d = d[d["source_layer"].astype(str) == layer_key]
    d = d[np.isclose(d["scale"].astype(float), scale)]
    return d, layers, scale


def collect_score_points(store):
    rows = []
    for spec in SCORE_RUNS:
        csv = store / spec["run"] / "calibration_judged.csv"
        if status_for(csv, spec["run"], "inhouse", CALIB_COLS) is not None:
            continue
        df = pd.read_csv(csv)
        cell, layers, scale = optimal_cell(df)
        row = dict(spec)
        row.update({
            "optimal_layer": str(layers),
            "optimal_scale": scale,
            "n": len(cell),
            "judge_refusal": cell["judge_refusal"].mean(),
            "judge_retention": cell["judge_retention"].mean(),
            "judge_fluency": cell["judge_fluency"].mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["size_b", "model"])


def add_fit(ax, x, y, color):
    if x.nunique() < 2:
        return
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, m * xx + b, color=color, linestyle="--", linewidth=1)


def write_score_size(store, out):
    points = collect_score_points(store)
    points.to_csv(out / "score_size_summary.csv", index=False)

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, ylabel, color) in zip(axes, SCORE_PANELS):
        data = points.dropna(subset=[col])
        ax.scatter(data["size_b"], data[col], s=42, color=color, alpha=0.9)
        add_fit(ax, data["size_b"], data[col], color)
        for row in data.itertuples(index=False):
            offset = SCORE_LABEL_OFFSETS.get(col, {}).get(row.run, (3, 3))
            ax.annotate(
                row.label,
                (row.size_b, getattr(row, col)),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title("inhouse", fontsize=18)
        ax.set_xlabel("Model size (B)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(0, max(15, points["size_b"].max() + 1))
        ax.set_xticks([0, 5, 10, 15])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=12)
        sns.despine(ax=ax, trim=True, offset=10)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.85, wspace=0.45)
    save_path = out / "score_size.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="render summary figures across result stores")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--figure",
        choices=(
            "all",
            "model_data",
            "calib_optimal",
            "calib_full",
            "calib_full_refuse",
            "calib_full_retain",
            "calib_full_fluency",
            "score_size",
        ),
        default="all",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if args.figure in ("all", "model_data"):
        written.append(write_model_data(args.store, args.out))
    if args.figure in ("all", "calib_optimal"):
        written.append(write_calib_optimal(args.store, args.out))
    if args.figure in ("all", "calib_full"):
        written.extend(write_calib_full(args.store, args.out))
    for title, metric, ylabel in FULL_METRICS:
        if args.figure == f"calib_full_{title}":
            written.append(write_calib_full_metric(args.store, args.out, title, metric, ylabel))
    if args.figure in ("all", "score_size"):
        written.append(write_score_size(args.store, args.out))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
