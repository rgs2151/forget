import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from refuse.calibration import select_optimal_config

from ..plot import AXES, AXIS_COLOR, HARMONIC_COLOR, SECONDARY_COLOR, custom_cmap, harmonic_refusal_fluency
from ..summary.util import OUT, STORE, result_file, save_figure, setup_summary_style


MODELS = (
    ("Llama-3.1-8B", "llama8b_inhouse"),
    ("Mistral-7B-v0.3", "mistral7b_inhouse"),
    ("Qwen-2.5-7B", "qwen7b_inhouse"),
)
METRIC_LABELS = {
    "refusal": "Refusal rate",
    "retention": "Retention rate",
    "fluency": "Fluency rate",
}


def _layer_value(source_layer):
    layers = ast.literal_eval(source_layer) if isinstance(source_layer, str) else list(source_layer)
    return float(sum(layers) / len(layers))


def _load_calibration(csv_path):
    df = pd.read_csv(csv_path)
    return df[df["label"] == "intervention"].copy()


def _draw_across_scale(ax, df, axes=AXES, show_harmonic=True, legend=True, star_y="harmonic"):
    layers, scale = select_optimal_config(df)
    d = df[df["source_layer"].astype(str) == str(layers)].copy()
    d = d.assign(judge_harmonic=harmonic_refusal_fluency(d))
    mx = float(d["scale"].max())

    for axis in axes:
        sns.lineplot(
            data=d, x="scale", y=f"judge_{axis}", ax=ax,
            color=AXIS_COLOR[axis], label=axis.title(),
            estimator="mean", errorbar=("ci", 95),
        )

    if show_harmonic:
        sns.lineplot(
            data=d, x="scale", y="judge_harmonic", ax=ax,
            color=HARMONIC_COLOR, label="Harmonic (R, F)",
            estimator="mean", errorbar=("ci", 95), linestyle="--",
        )

    means = d.groupby("scale", as_index=False)["judge_harmonic"].mean()
    peak = means[means["scale"] == scale].iloc[0]
    if star_y == "refusal":
        peak_y = d[d["scale"] == scale]["judge_refusal"].mean()
    else:
        peak_y = peak["judge_harmonic"]
    ax.plot(
        peak["scale"], peak_y,
        marker="*", color=SECONDARY_COLOR, markersize=8,
        fillstyle="none", linestyle="None",
    )

    ax.set_xlabel("Scale $s$", fontsize=7)
    ax.set_ylabel("Rate", fontsize=7)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=6, length=2)
    ax.set_box_aspect(1)
    if legend:
        ax.legend(loc="center right", fontsize=3.6, frameon=False)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax, trim=True, offset=4)


def _draw_layer_metric(ax, df, metric, cmap, norm, ylabel):
    mx = float(df["scale"].max())
    for _, layer_df in df.groupby("source_layer"):
        value = _layer_value(layer_df["source_layer"].iloc[0])
        trace = layer_df.groupby("scale", as_index=False)[metric].mean()
        ax.plot(trace["scale"], trace[metric], color=cmap(norm(value)), linewidth=0.55)

    ax.set_xlabel("Scale $s$", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=6, length=2)
    ax.set_box_aspect(1)
    sns.despine(ax=ax, trim=True, offset=4)


def _add_colorbar(fig, row_axes, cmap, norm):
    bbox = row_axes[-1].get_position()
    cax = fig.add_axes([bbox.x1 + 0.010, bbox.y0 + 0.14 * bbox.height, 0.008, bbox.height * 0.62])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels([f"{int(norm.vmin)}", f"{int(norm.vmax)}"])
    cbar.set_label("layers", rotation=270, labelpad=7, fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)


def _write_publish_params(store, out, filename, axes_to_plot, scale_axes, show_harmonic, star_y):
    setup_summary_style()
    ncols = 1 + len(axes_to_plot)
    figsize = (6.2, 4.65) if ncols == 4 else (4.7, 4.65)
    left = 0.13 if ncols == 4 else 0.20
    right = 0.91 if ncols == 4 else 0.89
    hspace = 0.82 if ncols == 4 else 0.54
    model_label_offset = 0.070 if ncols == 4 else 0.105
    fig, axes = plt.subplots(
        3, ncols, figsize=figsize,
        gridspec_kw={"width_ratios": [1.02] + [1] * len(axes_to_plot)},
    )
    fig.subplots_adjust(left=left, right=right, top=0.90, bottom=0.10, wspace=0.66, hspace=hspace)
    if ncols == 3:
        for ax in axes[:, 1:].flat:
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.030, pos.y0, pos.width, pos.height])

    for row, (model_label, run_key) in enumerate(MODELS):
        csv_path = result_file(store / run_key, "calibration_judged.csv")
        df = _load_calibration(csv_path)
        layer_values = df["source_layer"].map(_layer_value)
        cmap = custom_cmap(int(layer_values.nunique()))
        norm = Normalize(vmin=float(layer_values.min()), vmax=float(layer_values.max()))

        _draw_across_scale(
            axes[row, 0], df, axes=scale_axes,
            show_harmonic=show_harmonic, legend=row == 1, star_y=star_y,
        )
        for col, axis in enumerate(axes_to_plot, start=1):
            _draw_layer_metric(axes[row, col], df, f"judge_{axis}", cmap, norm, METRIC_LABELS[axis])
        _add_colorbar(fig, axes[row, 1:], cmap, norm)

        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - model_label_offset, (bbox.y0 + bbox.y1) / 2, model_label,
            ha="center", va="center", rotation=90, fontsize=8, weight="bold",
            fontfamily="Arial",
        )

    left = axes[0, 0].get_position()
    fig.text((left.x0 + left.x1) / 2, 0.94, "Across scale",
             ha="center", va="bottom", fontsize=9, weight="bold", fontfamily="Arial")
    right_left = axes[0, 1].get_position()
    right_right = axes[0, -1].get_position()
    fig.text((right_left.x0 + right_right.x1) / 2, 0.94, "Across layers",
             ha="center", va="bottom", fontsize=9, weight="bold", fontfamily="Arial")

    save_path = out / filename
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_publish_params(store=STORE, out=OUT):
    return _write_publish_params(
        store, out, "publish_params.png",
        axes_to_plot=AXES,
        scale_axes=AXES,
        show_harmonic=True,
        star_y="harmonic",
    )


def write_publish_params_min(store=STORE, out=OUT):
    return _write_publish_params(
        store, out, "publish_params_min.png",
        axes_to_plot=("refusal", "retention"),
        scale_axes=("refusal", "retention"),
        show_harmonic=False,
        star_y="refusal",
    )
