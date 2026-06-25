import matplotlib.pyplot as plt
import seaborn as sns

from ..summary.util import (
    DATASETS,
    OUT,
    STORE,
    draw_cross,
    has_cols,
    heatmap_pivot,
    result_file,
    save_figure,
    setup_summary_style,
)
from .supp_optimal import MODELS


METRICS = [
    ("Refusal rates", "judge_refusal"),
    ("Retention rates", "judge_retention"),
    ("Fluency rates", "judge_fluency"),
]


def _draw_confusion(ax, csv_path, metric):
    scores, _ = heatmap_pivot(csv_path, metric)
    n = scores.shape[0]
    sns.heatmap(
        scores.fillna(0),
        ax=ax,
        cmap="Greys",
        square=True,
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        linewidths=0.20,
        linecolor="black",
    )
    ax.set_xticks([0.5, n - 0.5])
    ax.set_yticks([0.5, n - 0.5])
    ax.set_xticklabels(["$c_0$", f"$c_{{{n}}}$"], rotation=0)
    ax.set_yticklabels(["$c_0$", f"$c_{{{n}}}$"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.5, length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_dataset_labels(fig, axes):
    fig.canvas.draw()
    for row, (dataset_label, _) in enumerate(DATASETS * len(METRICS)):
        bbox = axes[row, 0].get_position()
        fig.text(0.018, (bbox.y0 + bbox.y1) / 2, dataset_label,
                 ha="center", va="center", rotation=90, fontsize=9, weight="bold")


def _add_metric_headers(fig, axes):
    fig.canvas.draw()
    for metric_idx, (title, _) in enumerate(METRICS):
        start = metric_idx * len(DATASETS)
        left = axes[start, 0].get_position()
        right = axes[start, -1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.010, title,
                 ha="center", va="bottom", fontsize=15, weight="bold", fontfamily="Arial")


def _add_bottom_model_labels(fig, axes):
    fig.canvas.draw()
    for col, model in enumerate(MODELS):
        bbox = axes[-1, col].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y0 - 0.020, model["full"],
                 ha="center", va="top", fontsize=7.5, weight="bold")


def _add_family_headers(fig, axes):
    fig.canvas.draw()
    start = 0
    for i, model in enumerate(MODELS + [{"family": None}]):
        if i == len(MODELS) or model["family"] != MODELS[start]["family"]:
            left = axes[0, start].get_position()
            right = axes[0, i - 1].get_position()
            x0, x1 = left.x0, right.x1
            y = axes[0, 0].get_position().y1 + 0.040
            fig.text((x0 + x1) / 2, y, MODELS[start]["family"],
                     ha="center", va="bottom", fontsize=14, weight="bold")
            fig.add_artist(plt.Line2D([x0, x1], [y - 0.004, y - 0.004],
                                      transform=fig.transFigure, color="0.3", linewidth=0.8))
            start = i


def write_supp_confusion(store=STORE, out=OUT):
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
                conf_csv = result_file(store / f"{model['key']}_{dataset_key}", "confusion_judged.csv")
                if has_cols(conf_csv, (metric,)):
                    _draw_confusion(ax, conf_csv, metric)
                else:
                    draw_cross(ax)

    _add_family_headers(fig, axes)
    _add_dataset_labels(fig, axes)
    _add_metric_headers(fig, axes)
    _add_bottom_model_labels(fig, axes)

    save_path = out / "supp_confusion.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
