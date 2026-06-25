import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

from refuse.calibration import select_optimal_config

from .util import (
    AXIS_COLOR,
    CALIB_COLS,
    OUT,
    STORE,
    result_file,
    save_figure,
    setup_summary_style,
)


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
SCORE_MODEL_COLORS = {
    "llama32_1b": "#fdd0a2",
    "llama32_3b": "#f16913",
    "llama8b": "#7f2704",
    "mistral7b": "#4d4d4d",
    "qwen05b": "#c6dbef",
    "qwen3b": "#6baed6",
    "qwen7b": "#2171b5",
    "qwen14b": "#08306b",
    "phi4mini": "#cbc9e2",
    "phi4": "#6a51a3",
}
SCORE_REFUSAL_LEGEND_ORDER = (
    "llama32_1b",
    "llama32_3b",
    "llama8b",
    "mistral7b",
    "qwen05b",
    "qwen3b",
    "qwen7b",
    "qwen14b",
    "phi4mini",
    "phi4",
)
SCORE_REFUSAL_LEGEND_LABELS = {
    "llama32_1b": "Llama 3.2 1B",
    "llama32_3b": "Llama 3.2 3B",
    "llama8b": "Llama 3.1 8B",
    "mistral7b": "Mistral 7B",
    "qwen05b": "Qwen 0.5B",
    "qwen3b": "Qwen 3B",
    "qwen7b": "Qwen 7B",
    "qwen14b": "Qwen 14B",
    "phi4mini": "Phi-4 mini",
    "phi4": "Phi-4 14B",
}
SCORE_DATASETS = [
    ("inhouse", "inhouse"),
    ("mmlu", "MMLU"),
    ("rwku", "RWKU"),
    ("conceptvectors", "ConceptVectors"),
]

SCORE_PANELS = [
    ("judge_refusal", "Refusal rate", AXIS_COLOR["refusal"]),
    ("judge_retention", "Retain rate", AXIS_COLOR["retention"]),
    ("judge_fluency", "Fluency rate", AXIS_COLOR["fluency"]),
]
SCORE_GROUPS = [
    ("targeted", "Targeted", AXIS_COLOR["refusal"], AXIS_COLOR["refusal"]),
    ("untargeted", "Untargeted", "0.35", "black"),
]

SCORE_LABEL_OFFSETS = {
    "judge_refusal": {
        "llama32_3b_inhouse": (-26, 6),
        "qwen3b_inhouse": (4, 8),
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


def has_judged_metrics(csv):
    if not csv.exists():
        return False
    cols = pd.read_csv(csv, nrows=0).columns
    return all(col in cols for col in CALIB_COLS)


def optimal_cell(df):
    layers, scale = select_optimal_config(df)
    layer_key = str(layers)
    d = df[df["label"] == "intervention"]
    d = d[d["source_layer"].astype(str) == layer_key]
    d = d[np.isclose(d["scale"].astype(float), scale)]
    return d, layers, scale


def bar_cell(df, layers, scale, group):
    layer_key = str(layers)
    d = df[df["label"] == "intervention"]
    d = d[d["source_layer"].astype(str) == layer_key]
    d = d[d["target_layer"].astype(str) == layer_key]
    d = d[np.isclose(d["scale"].astype(float), scale)]
    if group == "targeted":
        d = d[d["concept"] == d["target"]]
    if group == "untargeted":
        d = d[d["concept"] != d["target"]]
    return d


def collect_score_points(store=STORE, group="targeted"):
    rows = []
    for spec in SCORE_RUNS:
        cal_csv = result_file(store / spec["run"], "calibration_judged.csv")
        bars_csv = result_file(store / spec["run"], "bars_judged.csv")
        if not has_judged_metrics(cal_csv) or not has_judged_metrics(bars_csv):
            continue
        cal_df = pd.read_csv(cal_csv)
        _, layers, scale = optimal_cell(cal_df)
        bars_df = pd.read_csv(bars_csv)
        cell = bar_cell(bars_df, layers, scale, group)
        row = dict(spec)
        row.update({
            "group": group,
            "optimal_layer": str(layers),
            "optimal_scale": scale,
            "n": len(cell),
            "judge_refusal": cell["judge_refusal"].mean(),
            "judge_retention": cell["judge_retention"].mean(),
            "judge_fluency": cell["judge_fluency"].mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["size_b", "model"])


def collect_split_score_points(store=STORE):
    return pd.concat(
        [collect_score_points(store, group=group) for group, _, _, _ in SCORE_GROUPS],
        ignore_index=True,
    )


def add_fit(ax, x, y, color):
    if x.nunique() < 2:
        return
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, m * xx + b, color=color, linestyle="--", linewidth=1)


def add_fit_with_stats(ax, x, y, color):
    if x.nunique() < 2:
        return
    fit = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, fit.slope * xx + fit.intercept, color=color, linestyle="--", linewidth=1)
    ax.text(
        0.52, 0.12, f"$R^2$={fit.rvalue ** 2:.2f}, p={fit.pvalue:.2g}",
        transform=ax.transAxes, ha="center", va="center", fontsize=8,
    )


def trim_y_axis(ax, y):
    lo = float(y.min())
    hi = float(y.max())
    pad = max((hi - lo) * 0.18, 0.03)
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))


def write_score_size(store=STORE, out=OUT):
    points = collect_split_score_points(store)
    points.to_csv(out / "score_size_summary.csv", index=False)

    setup_summary_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for row_idx, (group, group_label, point_color, fit_color) in enumerate(SCORE_GROUPS):
        group_points = points[points["group"] == group]
        for ax, (col, ylabel, _) in zip(axes[row_idx], SCORE_PANELS):
            data = group_points.dropna(subset=[col])
            ax.scatter(
                data["size_b"], data[col], s=42, color=point_color,
                edgecolor="black", linewidth=0.5, alpha=0.9,
            )
            add_fit(ax, data["size_b"], data[col], fit_color)
            for point in data.itertuples(index=False):
                offset = SCORE_LABEL_OFFSETS.get(col, {}).get(point.run, (3, 3))
                ax.annotate(
                    point.label,
                    (point.size_b, getattr(point, col)),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                )
            if row_idx == 0:
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
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.92, wspace=0.45, hspace=0.44)
    save_path = out / "score_size.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_score_size_refusal(store=STORE, out=OUT):
    rows = []
    for dataset, dataset_label in SCORE_DATASETS:
        for spec in SCORE_RUNS:
            model_key = spec["run"].removesuffix("_inhouse")
            run = f"{model_key}_{dataset}"
            cal_csv = store / run / "results" / "prefill_logit" / "calibration_judged.csv"
            bars_csv = store / run / "results" / "prefill_logit" / "bars_judged.csv"
            if not has_judged_metrics(cal_csv) or not has_judged_metrics(bars_csv):
                continue
            cal_df = pd.read_csv(cal_csv)
            _, layers, scale = optimal_cell(cal_df)
            bars_df = pd.read_csv(bars_csv)
            cell = bar_cell(bars_df, layers, scale, "targeted")
            row = dict(spec)
            row.update({
                "model_key": model_key,
                "run": run,
                "dataset": dataset,
                "dataset_label": dataset_label,
                "optimal_layer": str(layers),
                "optimal_scale": scale,
                "n": len(cell),
                "judge_refusal": cell["judge_refusal"].mean(),
            })
            rows.append(row)
    points = pd.DataFrame(rows).sort_values(["dataset", "size_b", "model"])
    points.to_csv(out / "score_size_refusal_summary.csv", index=False)

    col, ylabel, _ = SCORE_PANELS[0]

    setup_summary_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.4), sharey=True)
    flat_axes = axes.flat
    for idx, (ax, (dataset, dataset_label)) in enumerate(zip(flat_axes, SCORE_DATASETS)):
        data = points[points["dataset"] == dataset].dropna(subset=[col])
        ax.scatter(
            data["size_b"], data[col], s=28, color=AXIS_COLOR["refusal"],
            edgecolor="black", linewidth=0.5, alpha=0.9, clip_on=False,
        )
        add_fit_with_stats(ax, data["size_b"], data[col], AXIS_COLOR["refusal"])
        ax.set_title(dataset_label, fontsize=8, weight="bold", fontfamily="Arial", y=1.04)
        ax.set_xlabel("Model size (B)" if idx >= 2 else "", fontsize=7.5)
        ax.set_ylabel(ylabel if idx in (0, 2) else "", fontsize=7.5)
        ax.set_xlim(0, max(15, points["size_b"].max() + 1))
        ax.set_xticks([0, 5, 10, 15])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7)
        if idx < 2:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        if idx not in (0, 2):
            ax.tick_params(axis="y", left=False, labelleft=False)
        sns.despine(ax=ax, trim=True, offset=10)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.88, wspace=0.14, hspace=0.42)
    save_path = out / "score_size_refusal.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
