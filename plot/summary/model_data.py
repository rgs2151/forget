import matplotlib.pyplot as plt

from .util import (
    CALIB_COLS,
    DATA_MODELS,
    DATASETS,
    OUT,
    STORE,
    draw_bars,
    draw_cross,
    draw_heatmap,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
)


def write_model_data(store=STORE, out=OUT):
    setup_summary_style()
    fig, axes = plt.subplots(4, 6, figsize=(20, 13))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.55, hspace=0.55)

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, (_, model_key) in enumerate(DATA_MODELS):
            store_dir = store / f"{model_key}_{row_key}"
            conf_csv = result_file(store_dir, "confusion_judged.csv")
            bars_csv = result_file(store_dir, "bars_judged.csv")
            ax_left = axes[r, mi * 2]
            ax_right = axes[r, mi * 2 + 1]

            if has_cols(conf_csv, ("judge_refusal",)):
                draw_heatmap(ax_left, conf_csv, "judge_refusal", "Refusal")
            else:
                draw_cross(ax_left)

            if row_key == "inhouse":
                if has_cols(conf_csv, ("judge_retention",)):
                    draw_heatmap(ax_right, conf_csv, "judge_retention", "Retain")
                else:
                    draw_cross(ax_right)
            elif has_cols(bars_csv, CALIB_COLS):
                draw_bars(ax_right, bars_csv)
            else:
                draw_cross(ax_right)

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
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path

