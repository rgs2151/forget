import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from refuse.calibration import select_optimal_config

from .plot import AXIS_COLOR, setup_style


RUNS = [
    {"run": "qwen05b_inhouse", "model": "Qwen 0.5B", "label": "Qwen .5B", "size_b": 0.5},
    {"run": "llama32_1b_inhouse", "model": "Llama 3.2 1B", "label": "L3.2 1B", "size_b": 1.0},
    {"run": "llama32_3b_inhouse", "model": "Llama 3.2 3B", "label": "L3.2 3B", "size_b": 3.0},
    {"run": "qwen3b_inhouse", "model": "Qwen 3B", "label": "Qwen 3B", "size_b": 3.0},
    {"run": "mistral7b_inhouse", "model": "Mistral 7B", "label": "Mistral 7B", "size_b": 7.0},
    {"run": "qwen7b_inhouse", "model": "Qwen 7B", "label": "Qwen 7B", "size_b": 7.0},
    {"run": "llama8b_inhouse", "model": "Llama 3.1 8B", "label": "L3.1 8B", "size_b": 8.0},
    {"run": "phi4_inhouse", "model": "Phi-4 14B", "label": "Phi-4 14B", "size_b": 14.0},
    {"run": "qwen14b_inhouse", "model": "Qwen 14B", "label": "Qwen 14B", "size_b": 14.0},
]

PANEL = [
    ("judge_refusal", "Forget", AXIS_COLOR["refusal"]),
    ("judge_retention", "Retain", AXIS_COLOR["retention"]),
    ("judge_fluency", "Fluency", AXIS_COLOR["fluency"]),
]

LABEL_OFFSETS = {
    "judge_refusal": {
        "qwen7b_inhouse": (4, 4),
        "qwen14b_inhouse": (4, 2),
    },
    "judge_retention": {
        "llama32_1b_inhouse": (-10, -12),
        "llama32_3b_inhouse": (4, -12),
        "qwen05b_inhouse": (4, 5),
        "mistral7b_inhouse": (4, -12),
        "llama8b_inhouse": (4, 5),
        "phi4_inhouse": (4, -12),
        "qwen14b_inhouse": (4, 5),
    },
    "judge_fluency": {
        "mistral7b_inhouse": (4, 6),
        "qwen7b_inhouse": (4, -10),
        "qwen14b_inhouse": (4, -12),
    },
}


def optimal_cell(df):
    layers, scale = select_optimal_config(df)
    layer_key = str(layers)
    d = df[df["label"] == "intervention"]
    d = d[d["source_layer"].astype(str) == layer_key]
    d = d[np.isclose(d["scale"].astype(float), scale)]
    return d, layers, scale


def collect_points(store):
    rows = []
    for spec in RUNS:
        csv = store / spec["run"] / "calibration_judged.csv"
        if not csv.exists():
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


def plot_model_size_scores(points, save_path):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (col, title, color) in zip(axes, PANEL):
        data = points.dropna(subset=[col])
        ax.scatter(data["size_b"], data[col], s=42, color=color, alpha=0.9)
        add_fit(ax, data["size_b"], data[col], color)
        for row in data.itertuples(index=False):
            offset = LABEL_OFFSETS.get(col, {}).get(row.run, (3, 3))
            ax.annotate(
                row.label,
                (row.size_b, getattr(row, col)),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Model size (B)", fontsize=14)
        ax.set_xlim(0, max(15, points["size_b"].max() + 1))
        ax.set_xticks([0, 5, 10, 15])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=12)
        sns.despine(ax=ax, trim=True, offset=10)
    axes[0].set_ylabel("Judge score / refusal rate", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.85, wspace=0.32)
    fig.savefig(save_path, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="render summary figures across result stores")
    parser.add_argument("--store", default="store", type=Path)
    parser.add_argument("--out", default=Path("plot/figures"), type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    points = collect_points(args.store)
    csv_path = args.out / "model_size_summary.csv"
    fig_path = args.out / "model_size_scores.png"
    points.to_csv(csv_path, index=False)
    plot_model_size_scores(points, fig_path)
    plt.close()
    print(csv_path)
    print(fig_path)


if __name__ == "__main__":
    main()
