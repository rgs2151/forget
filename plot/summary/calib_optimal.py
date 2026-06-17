import matplotlib.pyplot as plt

from .util import (
    CALIB_COLS,
    CALIB_MODELS,
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


def write_calib_optimal(store=STORE, out=OUT):
    setup_summary_style()
    n_rows, n_cols = len(DATASETS), len(CALIB_MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10.5))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.84, bottom=0.07,
                        wspace=0.40, hspace=0.55)
    legend_drawn = False

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, model in enumerate(CALIB_MODELS):
            model_key = model["key"]
            cal_csv = result_file(store / f"{model_key}_{row_key}", "calibration_judged.csv")
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
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path

