import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from ..plot import PRIMARY_COLOR
from ..summary.util import OUT, STORE, save_figure, setup_summary_style
from .publish_bar import DATASETS, MODELS


LINE_COLORS = {
    "inhouse": "#4c78a8",
    "MMLU": "#f58518",
    "RWKU": "#54a24b",
    "CV": "#b279a2",
}
ARIAL_BOLD = FontProperties(family="Arial", weight="bold", size=5.8)


def _rates(store, model, dataset, metric, result):
    bars_csv = store / f"{model}_{dataset}" / "results" / result / "bars_judged.csv"
    if not bars_csv.exists():
        return None
    df = pd.read_csv(bars_csv)
    df = df[df["label"] == "intervention"]
    targeted = df[df["concept"] == df["target"]][metric].mean()
    untargeted = df[df["concept"] != df["target"]][metric].mean()
    return targeted, untargeted


def _write_paired_plot(store, out, metric, result, ylabel, filename, legend_loc="upper left"):
    setup_summary_style()
    fig, ax = plt.subplots(1, 1, figsize=(1.55, 2.25))
    fig.subplots_adjust(left=0.38, right=0.94, bottom=0.34, top=0.96)

    x = [0, 0.155]
    for dataset, label in DATASETS:
        targeted = []
        untargeted = []
        for model, _ in MODELS:
            rates = _rates(store, model, dataset, metric, result)
            if rates is None:
                continue
            target_rate, untarget_rate = rates
            targeted.append(target_rate)
            untargeted.append(untarget_rate)

        y = [sum(targeted) / len(targeted), sum(untargeted) / len(untargeted)]
        ax.plot(x, y, color=LINE_COLORS[label], linewidth=1.35, zorder=1)
        ax.scatter([x[0]], [y[0]], s=78, color=PRIMARY_COLOR, edgecolor="black", linewidth=0.48, zorder=2)
        ax.scatter([x[1]], [y[1]], s=78, color="black", edgecolor="black", linewidth=0.48, zorder=2)

    handles = [Line2D([0], [0], color=LINE_COLORS[label], linewidth=1.35, label=label) for _, label in DATASETS]
    ax.legend(handles=handles, loc=legend_loc, frameon=False, prop=ARIAL_BOLD,
              handlelength=1.0, labelspacing=0.2, borderaxespad=0.2)

    ax.set_xlim(-0.08, 0.235)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(["targeted", "untargeted"])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=3, pad=2)
    ax.tick_params(axis="y", labelsize=7, length=3)
    sns.despine(ax=ax, trim=True, offset=5)

    plt.sca(ax)
    plt.xticks(rotation=30, ha="right")
    for tick in ax.get_xticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontweight("bold")

    save_path = out / filename
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_publish_disruption(store=STORE, out=OUT, result="prefill_logit"):
    return _write_paired_plot(store, out, "judge_retention", result, "Retention rate", "publish_disruption.png")


def write_publish_fluency(store=STORE, out=OUT, result="main"):
    return _write_paired_plot(store, out, "judge_fluency", result, "Fluency rate", "publish_fluency.png")


def write_publish_refusal(store=STORE, out=OUT, result="prefill_logit"):
    return _write_paired_plot(
        store, out, "judge_refusal", result, "Refusal rate", "publish_refusal.png",
        legend_loc="lower left",
    )
