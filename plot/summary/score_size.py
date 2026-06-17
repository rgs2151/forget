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

SCORE_PANELS = [
    ("judge_refusal", "Refusal rate", AXIS_COLOR["refusal"]),
    ("judge_retention", "Retain rate", AXIS_COLOR["retention"]),
    ("judge_fluency", "Fluency rate", AXIS_COLOR["fluency"]),
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


def has_calibration(csv):
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


def collect_score_points(store=STORE):
    rows = []
    for spec in SCORE_RUNS:
        csv = result_file(store / spec["run"], "calibration_judged.csv")
        if not has_calibration(csv):
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


def add_fit_with_stats(ax, x, y, color):
    if x.nunique() < 2:
        return
    fit = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    label = f"linear fit ($R^2$={fit.rvalue ** 2:.2f}, p={fit.pvalue:.2g})"
    ax.plot(xx, fit.slope * xx + fit.intercept, color=color, linestyle="--", linewidth=1, label=label)


def trim_y_axis(ax, y):
    lo = float(y.min())
    hi = float(y.max())
    pad = max((hi - lo) * 0.18, 0.03)
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))


def write_score_size(store=STORE, out=OUT):
    points = collect_score_points(store)
    points.to_csv(out / "score_size_summary.csv", index=False)

    setup_summary_style()
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
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_score_size_refusal(store=STORE, out=OUT):
    points = collect_score_points(store)
    points = points[points["run"] != "phi4mini_inhouse"].copy()
    points.to_csv(out / "score_size_refusal_summary.csv", index=False)

    col, ylabel, color = SCORE_PANELS[0]
    data = points.dropna(subset=[col])

    setup_summary_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.8))
    ax.scatter(data["size_b"], data[col], s=42, color=color, alpha=0.9)
    add_fit_with_stats(ax, data["size_b"], data[col], color)
    for row in data.itertuples(index=False):
        offset = SCORE_LABEL_OFFSETS.get(col, {}).get(row.run, (3, 3))
        ax.annotate(
            row.label,
            (row.size_b, getattr(row, col)),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Model size (B)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0, max(15, points["size_b"].max() + 1))
    ax.set_xticks([0, 5, 10, 15])
    trim_y_axis(ax, data[col])
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=12)
    ax.legend(loc=(1, 0.5), frameon=False, fontsize=9)
    sns.despine(ax=ax, trim=True, offset=10)

    fig.subplots_adjust(left=0.19, right=0.73, bottom=0.2, top=0.97)
    save_path = out / "score_size_refusal.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
