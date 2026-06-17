import matplotlib.pyplot as plt

from .util import (
    CALIB_MODELS,
    DATASETS,
    FULL_METRICS,
    OUT,
    STORE,
    add_family_headers,
    add_row_headers,
    draw_cross,
    draw_full_calibration,
    full_cmap,
    has_cols,
    layer_colorbar,
    result_file,
    save_figure,
    setup_summary_style,
    tidy_grid_axis,
)


def write_calib_full_metric(store=STORE, out=OUT, title="refuse", metric="judge_refusal", ylabel="Refusal rate"):
    setup_summary_style()
    n_rows, n_cols = len(DATASETS), len(CALIB_MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10.8))
    fig.subplots_adjust(left=0.05, right=0.955, top=0.82, bottom=0.07,
                        wspace=0.40, hspace=0.55)
    cmap = full_cmap()

    for r, (_, row_key) in enumerate(DATASETS):
        for mi, model in enumerate(CALIB_MODELS):
            model_key = model["key"]
            cal_csv = result_file(store / f"{model_key}_{row_key}", "calibration_judged.csv")
            ax = axes[r, mi]
            if has_cols(cal_csv, (metric,)):
                draw_full_calibration(ax, cal_csv, metric, cmap)
                if mi == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                tidy_grid_axis(ax, r, mi, n_rows)
            else:
                draw_cross(ax)

    add_family_headers(fig, axes, CALIB_MODELS, family_y=0.94, model_y=0.885)
    add_row_headers(fig, axes, DATASETS, x=0.018)
    fig.suptitle(title, fontsize=22, weight="bold", y=0.992)
    layer_colorbar(fig, cmap)

    save_path = out / f"calib_full_{title}.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_calib_full(store=STORE, out=OUT):
    return [write_calib_full_metric(store, out, title, metric, ylabel)
            for title, metric, ylabel in FULL_METRICS]

