import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from refuse.calibration import select_optimal_config

from .util import (
    AXES,
    AXIS_COLOR,
    HARMONIC_COLOR,
    OUT,
    STORE,
    custom_cmap,
    draw_cross,
    harmonic_refusal_fluency,
    has_cols,
    result_file,
    save_figure,
    setup_summary_style,
)


SCALE_LAYER_MODELS = [
    ("Llama3", "llama8b_inhouse"),
    ("Mistral", "mistral7b_inhouse"),
]
LAYER_AXES = ("refusal", "retention")


def layer_value(source_layer):
    layers = ast.literal_eval(source_layer) if isinstance(source_layer, str) else list(source_layer)
    return float(sum(layers) / len(layers))


def load_calibration(csv_path):
    df = pd.read_csv(csv_path)
    return df[df["label"] == "intervention"].copy()


def draw_across_scale(ax, df, legend=True):
    layers, scale = select_optimal_config(df)
    layer_key = str(layers)
    d = df[df["source_layer"].astype(str) == layer_key].copy()
    d = d.assign(judge_harmonic=harmonic_refusal_fluency(d))
    mx = float(d["scale"].max())

    for axis in AXES:
        sns.lineplot(
            data=d, x="scale", y=f"judge_{axis}", ax=ax,
            color=AXIS_COLOR[axis], label=axis.title(),
            estimator="mean", errorbar=("ci", 95),
        )

    sns.lineplot(
        data=d, x="scale", y="judge_harmonic", ax=ax,
        color=HARMONIC_COLOR, label="Harmonic (R, F)",
        estimator="mean", errorbar=("ci", 95), linestyle="--",
    )

    means = d.groupby("scale", as_index=False)["judge_harmonic"].mean()
    peak = means[means["scale"] == scale].iloc[0]
    ax.plot(
        peak["scale"], peak["judge_harmonic"],
        marker="*", color="black", markersize=12,
        fillstyle="none", linestyle="None",
    )

    ax.set_xlabel("Scale $s$", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)
    if legend:
        ax.legend(loc="center right", fontsize=6, frameon=False)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax, trim=True, offset=8)


def draw_layer_metric(ax, df, metric, cmap, norm):
    df = df.assign(_layer=df["source_layer"].map(layer_value))
    mx = float(df["scale"].max())
    for layer, layer_df in df.groupby("source_layer"):
        value = layer_value(layer)
        trace = layer_df.groupby("scale", as_index=False)[metric].mean()
        ax.plot(trace["scale"], trace[metric], color=cmap(norm(value)), linewidth=0.8)

    ax.set_xlabel("Scale $s$", fontsize=13)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, mx])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)
    sns.despine(ax=ax, trim=True, offset=8)


def add_row_colorbar(fig, row_axes, cmap, norm):
    bbox = row_axes[-1].get_position()
    cax = fig.add_axes([bbox.x1 + 0.018, bbox.y0 + 0.05, 0.010, bbox.height * 0.70])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels([f"{int(norm.vmin)}", f"{int(norm.vmax)}"])
    cbar.set_label("layers", rotation=270, labelpad=12, fontsize=13)
    cbar.ax.tick_params(labelsize=12)


def write_calib_scale_layers(store=STORE, out=OUT):
    setup_summary_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(9.7, 6.8),
        gridspec_kw={"width_ratios": [1.05, 1, 1]},
    )
    fig.subplots_adjust(left=0.11, right=0.91, top=0.84, bottom=0.10,
                        wspace=0.68, hspace=0.86)

    for r, (model_label, run_key) in enumerate(SCALE_LAYER_MODELS):
        csv_path = result_file(store / run_key, "calibration_judged.csv")
        if has_cols(csv_path, ("judge_refusal", "judge_retention", "judge_fluency")):
            df = load_calibration(csv_path)
            layer_values = df["source_layer"].map(layer_value)
            cmap = custom_cmap(int(layer_values.nunique()))
            norm = Normalize(vmin=float(layer_values.min()), vmax=float(layer_values.max()))

            draw_across_scale(axes[r, 0], df, legend=True)
            for ax, axis in zip(axes[r, 1:], LAYER_AXES):
                draw_layer_metric(ax, df, f"judge_{axis}", cmap, norm)
                ax.set_title(axis.title(), fontsize=14)
            axes[r, 1].set_ylabel("Rate", fontsize=13)
            axes[r, 2].set_ylabel("")
            add_row_colorbar(fig, axes[r, 1:], cmap, norm)
        else:
            for ax in axes[r]:
                draw_cross(ax)

        bbox = axes[r, 0].get_position()
        fig.text(
            bbox.x0 - 0.045, bbox.y1 + 0.015, model_label,
            ha="left", va="bottom", fontsize=14, weight="bold", style="italic",
        )

    left = axes[0, 0].get_position()
    fig.text((left.x0 + left.x1) / 2, 0.925, "Across scale",
             ha="center", va="bottom", fontsize=16, weight="bold", style="italic")
    right_left = axes[0, 1].get_position()
    right_right = axes[0, 2].get_position()
    fig.text((right_left.x0 + right_right.x1) / 2, 0.925, "Across layers",
             ha="center", va="bottom", fontsize=16, weight="bold", style="italic")

    save_path = out / "calib_scale_layers.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
