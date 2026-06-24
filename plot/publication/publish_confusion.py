import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from ..summary.util import OUT, STORE, heatmap_pivot, result_file, save_figure, setup_summary_style


MODELS = ("llama8b", "qwen7b", "mistral7b", "phi4")
DATASET = "inhouse"
GRID_LINEWIDTH = 0.48


def _format_concept(name):
    label = str(name).replace("_", " ").lower()
    return "USA" if label == "united states" else label


def _write_confusion_matrix(metric, color, filename, store=STORE, out=OUT):
    setup_summary_style()
    cmap = LinearSegmentedColormap.from_list("white_primary", ["white", color])

    base = 1.5
    fig, axes = plt.subplots(1, 4, figsize=(4 * base, base))
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.16, top=0.98, wspace=0.10)

    for idx, (model_key, ax) in enumerate(zip(MODELS, axes)):
        conf_csv = result_file(store / f"{model_key}_{DATASET}", "confusion_judged.csv")
        scores, concepts = heatmap_pivot(conf_csv, metric)
        labels = [_format_concept(c) for c in concepts]
        concept_labels = [f"$c_{{{i + 1}}}$" for i in range(len(labels))]
        ylabels = [f"{label} ({concept})" for label, concept in zip(labels, concept_labels)]

        sns.heatmap(
            scores,
            ax=ax,
            cmap=cmap,
            square=True,
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            linewidths=GRID_LINEWIDTH,
            linecolor="black",
        )
        n = len(labels)
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(concept_labels, rotation=0)
        ax.set_yticklabels(ylabels if idx == 0 else [], rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, length=0, pad=1)
        ax.tick_params(axis="y", labelsize=7, rotation=0, length=0)
        ax.add_patch(Rectangle((0, 0), n, n, fill=False, edgecolor="black", linewidth=GRID_LINEWIDTH, clip_on=False))
        for spine in ax.spines.values():
            spine.set_visible(False)

    save_path = out / filename
    save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def write_publish_confusion(store=STORE, out=OUT):
    return _write_confusion_matrix("judge_refusal", "black", "publish_confusion.png", store, out)


def write_publish_confusion_ret(store=STORE, out=OUT):
    return _write_confusion_matrix("judge_retention", "black", "publish_confusion_ret.png", store, out)
