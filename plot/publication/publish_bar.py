import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

from ..plot import PRIMARY_COLOR
from ..summary.util import OUT, STORE, result_file, save_figure, setup_summary_style


MODELS = (
    ("llama8b", "Llama-3.1-8B"),
    ("qwen7b", "Qwen-2.5-7B"),
    ("mistral7b", "Mistral-7B-v0.3"),
    ("phi4", "Phi-4"),
)
DATASETS = (
    ("inhouse", "inhouse"),
    ("mmlu", "MMLU"),
    ("rwku", "RWKU"),
    ("conceptvectors", "CV"),
)
EDGE_WIDTH = 0.48
ARIAL_BOLD = FontProperties(family="Arial", weight="bold", size=5.5)


def _refusal_rates(store, model, dataset):
    bars_csv = result_file(store / f"{model}_{dataset}", "bars_judged.csv")
    df = pd.read_csv(bars_csv)
    df = df[df["label"] == "intervention"]
    diagonal = df[df["concept"] == df["target"]]["judge_refusal"].mean()
    off_diagonal = df[df["concept"] != df["target"]]["judge_refusal"].mean()
    return diagonal, off_diagonal


def write_publish_bar(store=STORE, out=OUT):
    setup_summary_style()
    fig, axes = plt.subplots(1, 4, figsize=(6.0, 2.25), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.42, top=0.80, wspace=0.18)

    x = np.arange(len(DATASETS)) * 1.25
    width = 0.38

    for idx, (model, label) in enumerate(MODELS):
        ax = axes[idx]
        diagonal = []
        off_diagonal = []
        for dataset, _ in DATASETS:
            diag, off = _refusal_rates(store, model, dataset)
            diagonal.append(diag)
            off_diagonal.append(off)

        ax.bar(x - width / 2, diagonal, width=width, color=PRIMARY_COLOR,
               edgecolor="black", linewidth=EDGE_WIDTH)
        ax.bar(x + width / 2, off_diagonal, width=width, color="black",
               edgecolor="black", linewidth=EDGE_WIDTH)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in DATASETS])
        ax.tick_params(axis="x", labelsize=6, length=3, pad=2)
        ax.tick_params(axis="y", labelsize=7, length=3)
        if idx == 0:
            ax.set_ylabel("Refusal rate", fontsize=8)
            handles = [
                Patch(facecolor=PRIMARY_COLOR, edgecolor="black", linewidth=EDGE_WIDTH, label="targeted"),
                Patch(facecolor="black", edgecolor="black", linewidth=EDGE_WIDTH, label="untargeted"),
            ]
            ax.legend(handles=handles, loc="upper right",
                      ncol=1, frameon=False, prop=ARIAL_BOLD, handlelength=0.8,
                      handleheight=0.55, labelspacing=0.25, borderaxespad=0.2)
            sns.despine(ax=ax, trim=True, offset=5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
            sns.despine(ax=ax, left=True, trim=True, offset=5)

        plt.sca(ax)
        plt.xticks(rotation=30, ha="right")
        for tick in ax.get_xticklabels():
            tick.set_fontfamily("Arial")
            tick.set_fontweight("bold")

    save_path = out / "publish_bar.png"
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path
