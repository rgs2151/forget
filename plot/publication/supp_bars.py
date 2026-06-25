import pandas as pd
import matplotlib.pyplot as plt

from ..plot import PRIMARY_COLOR
from ..summary.util import (
    DATASETS,
    OUT,
    STORE,
    draw_cross,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
)
from .supp_confusion import (
    METRICS,
    _add_bottom_model_labels,
    _add_dataset_labels,
    _add_family_headers,
    _add_metric_headers,
)
from .supp_optimal import MODELS


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
