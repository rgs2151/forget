import pandas as pd
import matplotlib.pyplot as plt

from forget.plot.plot import PRIMARY_COLOR
from forget.plot.results import (
    DATASETS,
    EVALUATED_MODELS,
    STORE,
    draw_cross,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
)
from .paths import OUT


MODELS = EVALUATED_MODELS
METRICS = [
    ("Refusal rates", "judge_refusal"),
    ("Retention rates", "judge_retention"),
    ("Fluency rates", "judge_fluency"),
]


def _add_dataset_labels(fig, axes):
    fig.canvas.draw()
    for row, (dataset_label, _) in enumerate(DATASETS * len(METRICS)):
        bbox = axes[row, 0].get_position()
        fig.text(
            0.018,
            (bbox.y0 + bbox.y1) / 2,
            dataset_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            weight="bold",
        )


def _add_metric_headers(fig, axes):
    fig.canvas.draw()
    for metric_idx, (title, _) in enumerate(METRICS):
        start = metric_idx * len(DATASETS)
        left = axes[start, 0].get_position()
        right = axes[start, -1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            left.y1 + 0.010,
            title,
            ha="center",
            va="bottom",
            fontsize=15,
            weight="bold",
            fontfamily="Arial",
        )


def _add_bottom_model_labels(fig, axes):
    fig.canvas.draw()
    for col, model in enumerate(MODELS):
        bbox = axes[-1, col].get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - 0.020,
            model["full"],
            ha="center",
            va="top",
            fontsize=7.5,
            weight="bold",
        )


def _add_family_headers(fig, axes):
    fig.canvas.draw()
    start = 0
    for i, model in enumerate(MODELS + [{"family": None}]):
        if i == len(MODELS) or model["family"] != MODELS[start]["family"]:
            left = axes[0, start].get_position()
            right = axes[0, i - 1].get_position()
            x0, x1 = left.x0, right.x1
            y = axes[0, 0].get_position().y1 + 0.040
            fig.text(
                (x0 + x1) / 2,
                y,
                MODELS[start]["family"],
                ha="center",
                va="bottom",
                fontsize=14,
                weight="bold",
            )
            fig.add_artist(
                plt.Line2D(
                    [x0, x1],
                    [y - 0.004, y - 0.004],
                    transform=fig.transFigure,
                    color="0.3",
                    linewidth=0.8,
                )
            )
            start = i


def _draw_bars(ax, csv_path, metric):
    df = pd.read_csv(csv_path)
    if "label" in df:
        df = df[df["label"] == "intervention"]
    targeted = df[df["concept"] == df["target"]][metric].mean()
    untargeted = df[df["concept"] != df["target"]][metric].mean()

    ax.bar([0, 1], [targeted, untargeted], width=0.72,
           color=[PRIMARY_COLOR, "black"], edgecolor="black", linewidth=0.25)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.45)


def write_supp_bars(store=STORE, out=OUT):
    setup_summary_style()
    n_rows = len(DATASETS) * len(METRICS)
    n_cols = len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22.5, 25.5))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.115,
                        wspace=0.25, hspace=0.42)

    for metric_idx, (_, metric) in enumerate(METRICS):
        for dataset_idx, (_, dataset_key) in enumerate(DATASETS):
            row = metric_idx * len(DATASETS) + dataset_idx
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                bars_csv = result_file(store / f"{model['key']}_{dataset_key}", "bars_judged.csv")
                if has_cols(bars_csv, (metric,)):
                    _draw_bars(ax, bars_csv, metric)
                else:
                    draw_cross(ax)

    _add_family_headers(fig, axes)
    _add_dataset_labels(fig, axes)
    _add_metric_headers(fig, axes)
    _add_bottom_model_labels(fig, axes)

    save_path = out / "supp_bars.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
