import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..summary.util import (
    DATASETS,
    OUT,
    STORE,
    draw_cross,
    draw_full_calibration,
    full_cmap,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
    tidy_grid_axis,
)


MODELS = [
    {"family": "Llama", "label": "Llama-3.2-1B-Instruct", "key": "llama32_1b"},
    {"family": "Llama", "label": "Llama-3.2-3B-Instruct", "key": "llama32_3b"},
    {"family": "Llama", "label": "Llama-3.1-8B-Instruct", "key": "llama8b"},
    {"family": "Mistral", "label": "Mistral-7B-Instruct-v0.3", "key": "mistral7b"},
    {"family": "Qwen", "label": "Qwen2.5-0.5B-Instruct", "key": "qwen05b"},
    {"family": "Qwen", "label": "Qwen2.5-3B-Instruct", "key": "qwen3b"},
    {"family": "Qwen", "label": "Qwen2.5-7B-Instruct", "key": "qwen7b"},
    {"family": "Qwen", "label": "Qwen2.5-14B-Instruct", "key": "qwen14b"},
    {"family": "Phi", "label": "Phi-4-mini-instruct", "key": "phi4mini"},
    {"family": "Phi", "label": "Phi-4", "key": "phi4"},
]


def _add_dataset_headers(fig, axes):
    fig.canvas.draw()
    for i, (label, _) in enumerate(DATASETS):
        bbox = axes[0, i].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.955, label,
                 ha="center", va="bottom", fontsize=12, weight="bold")


def _add_model_headers(fig, axes):
    fig.canvas.draw()
    for row, model in enumerate(MODELS):
        left = axes[row, 0].get_position()
        right = axes[row, -1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.0025, model["label"],
                 ha="center", va="bottom", fontsize=7.5, weight="bold")

    start = 0
    for i, model in enumerate(MODELS + [{"family": None}]):
        if i == len(MODELS) or model["family"] != MODELS[start]["family"]:
            first = axes[start, 0].get_position()
            last = axes[i - 1, 0].get_position()
            y0, y1 = last.y0, first.y1
            fig.text(0.025, (y0 + y1) / 2, MODELS[start]["family"],
                     ha="center", va="center", rotation=90, fontsize=14, weight="bold")
            fig.add_artist(plt.Line2D([0.045, 0.045], [y0, y1],
                                      transform=fig.transFigure, color="0.3", linewidth=0.8))
            start = i


def _compact_colorbar(fig, axes, cmap):
    bbox = axes[0, -1].get_position()
    cax = fig.add_axes([bbox.x1 + 0.012, bbox.y0 + 0.12 * bbox.height, 0.012, bbox.height * 0.72])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["early", "late"])
    cbar.set_label("Layer depth", rotation=270, labelpad=9, fontsize=9)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)


def _write_supp_metric(store, out, metric, ylabel, filename):
    setup_summary_style()
    n_rows, n_cols = len(MODELS), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 11.2))
    fig.subplots_adjust(left=0.13, right=0.93, top=0.92, bottom=0.045,
                        wspace=0.36, hspace=0.88)
    cmap = full_cmap()

    for row, model in enumerate(MODELS):
        for col, (_, dataset_key) in enumerate(DATASETS):
            ax = axes[row, col]
            cal_csv = result_file(store / f"{model['key']}_{dataset_key}", "calibration_judged.csv")
            if has_cols(cal_csv, (metric,)):
                draw_full_calibration(ax, cal_csv, metric, cmap)
                for line in ax.lines:
                    line.set_linewidth(0.35)
                tidy_grid_axis(ax, row, col, n_rows)
                ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
            else:
                draw_cross(ax)

    _add_dataset_headers(fig, axes)
    _add_model_headers(fig, axes)
    _compact_colorbar(fig, axes, cmap)

    save_path = out / filename
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_supp_refuse(store=STORE, out=OUT):
    return _write_supp_metric(store, out, "judge_refusal", "Refusal rate", "supp_refuse.png")


def write_supp_retain(store=STORE, out=OUT):
    return _write_supp_metric(store, out, "judge_retention", "Retention rate", "supp_retain.png")


def write_supp_fluency(store=STORE, out=OUT):
    return _write_supp_metric(store, out, "judge_fluency", "Fluency rate", "supp_fluency.png")
