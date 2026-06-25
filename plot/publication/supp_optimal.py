import matplotlib.pyplot as plt

from ..summary.util import (
    CALIB_COLS,
    DATASETS,
    OUT,
    STORE,
    add_family_headers,
    add_row_headers,
    draw_calibration,
    draw_cross,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
    tidy_grid_axis,
)


MODELS = [
    {"family": "Llama", "label": "1B", "key": "llama32_1b", "full": "Llama-3.2-1B"},
    {"family": "Llama", "label": "3B", "key": "llama32_3b", "full": "Llama-3.2-3B"},
    {"family": "Llama", "label": "8B", "key": "llama8b", "full": "Llama-3.1-8B"},
    {"family": "Mistral", "label": "7B", "key": "mistral7b", "full": "Mistral-7B-v0.3"},
    {"family": "Qwen", "label": "0.5B", "key": "qwen05b", "full": "Qwen-2.5-0.5B"},
    {"family": "Qwen", "label": "3B", "key": "qwen3b", "full": "Qwen-2.5-3B"},
    {"family": "Qwen", "label": "7B", "key": "qwen7b", "full": "Qwen-2.5-7B"},
    {"family": "Qwen", "label": "14B", "key": "qwen14b", "full": "Qwen-2.5-14B"},
    {"family": "Phi", "label": "mini", "key": "phi4mini", "full": "Phi-4-mini"},
    {"family": "Phi", "label": "14B", "key": "phi4", "full": "Phi-4"},
]


def _add_bottom_model_labels(fig, axes):
    fig.canvas.draw()
    for col, model in enumerate(MODELS):
        bbox = axes[-1, col].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.105, model["full"],
                 ha="center", va="top", fontsize=8, weight="bold")


def _add_family_headers(fig, axes):
    fig.canvas.draw()
    start = 0
    for i, model in enumerate(MODELS + [{"family": None}]):
        if i == len(MODELS) or model["family"] != MODELS[start]["family"]:
            left = axes[0, start].get_position()
            right = axes[0, i - 1].get_position()
            x0, x1 = left.x0, right.x1
            fig.text((x0 + x1) / 2, 0.845, MODELS[start]["family"],
                     ha="center", va="bottom", fontsize=14, weight="bold")
            fig.add_artist(plt.Line2D([x0, x1], [0.838, 0.838],
                                      transform=fig.transFigure, color="0.3", linewidth=0.8))
            start = i


def write_supp_optimal(store=STORE, out=OUT):
    setup_summary_style()
    n_rows, n_cols = len(DATASETS), len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.25, 10.5))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.20,
                        wspace=0.40, hspace=0.55)
    legend_drawn = False

    for row, (_, dataset_key) in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            cal_csv = result_file(store / f"{model['key']}_{dataset_key}", "calibration_judged.csv")
            if has_cols(cal_csv, CALIB_COLS):
                draw_calibration(ax, cal_csv, legend=not legend_drawn)
                legend_drawn = True
                tidy_grid_axis(ax, row, col, n_rows)
            else:
                draw_cross(ax)

    _add_family_headers(fig, axes)
    add_row_headers(fig, axes, DATASETS, x=-0.035)
    _add_bottom_model_labels(fig, axes)

    save_path = out / "supp_optimal.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
