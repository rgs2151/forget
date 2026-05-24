import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
from matplotlib.colors import ListedColormap
from sklearn.metrics import auc, roc_curve

from .activations import pool_activation_dict
from .calibration import is_refusal_output


PRIMARY_COLOR = "darkred"
SECONDARY_COLOR = "black"


def setup_style():
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["figure.figsize"] = [3.0, 3.0]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "lower right"


def custom_cmap():
    colors = mpl.colormaps["RdYlBu_r"](np.linspace(0, 1, 10))
    return ListedColormap(colors)


def plot_heatmap(scored, concepts, intervention_layers, scale, metric, save_path=None):
    src = str(intervention_layers)
    tgt = str(intervention_layers)
    df = scored[
        (scored["label"] == "intervention")
        & (scored["scale"] == scale)
        & (scored["source_layer"].astype(str) == src)
        & (scored["target_layer"].astype(str) == tgt)
    ]
    scores = (
        df.pivot_table(index="concept", columns="target", values=metric, aggfunc="mean")
        .reindex(index=concepts, columns=concepts)
        .fillna(0)
    )

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        scores,
        cmap="Greys",
        square=True,
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.8, "ticks": [0, 1]},
    )
    ax.set_xlabel("Concept $c$")
    ax.set_ylabel("Target Concept $c'$")
    ax.set_xticks([0, len(concepts) - 1], labels=[1, len(concepts)])
    ax.set_yticks([0, len(concepts) - 1], labels=[1, len(concepts)])
    plt.xticks(rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(metric.replace("_", " ").title(), rotation=270, labelpad=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_calibration(calibration_results, refusal_string="I don't know.", save_path=None):
    df = calibration_results.copy()
    if "label" in df:
        df = df[df["label"] == "intervention"]
    df = df.assign(
        is_refusal=df["model_output"].fillna("").apply(
            lambda s: is_refusal_output(s, refusal_string=refusal_string)
        )
    )

    plt.figure()
    sns.lineplot(
        data=df, x="scale", y="is_refusal",
        color=PRIMARY_COLOR, marker="o",
        estimator="mean", errorbar=("ci", 95),
    )

    rates = df.groupby("scale", as_index=False)["is_refusal"].mean()
    best = rates.sort_values(["is_refusal", "scale"], ascending=[False, True]).iloc[0]
    plt.plot(
        best["scale"], best["is_refusal"],
        marker="*", color=SECONDARY_COLOR,
        markersize=14, fillstyle="none", linestyle="None",
    )

    plt.xlabel("Scale $s$")
    plt.ylabel("Refusal rate")
    plt.ylim(-0.05, 1.05)
    sns.despine(trim=True, offset=10)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_detection_roc(baseline_acts, baseline_masks, v_detect, thresholds,
                       concepts, intervention_layers, save_path=None):
    acts_pooled = pool_activation_dict(baseline_acts, baseline_masks)
    layer_idx = intervention_layers[-1]

    plt.figure()
    aucs = []
    for concept in concepts:
        v = v_detect[concept][layer_idx, 0].float()
        pos_acts = acts_pooled[concept][:, layer_idx, 0, :].float()
        neg_acts = t.cat([
            acts_pooled[other][:, layer_idx, 0, :].float()
            for other in concepts if other != concept
        ], dim=0)

        pos_proj = (pos_acts @ v).cpu().numpy()
        neg_proj = (neg_acts @ v).cpu().numpy()

        scores = np.concatenate([pos_proj, neg_proj])
        labels = np.concatenate([np.ones(len(pos_proj)), np.zeros(len(neg_proj))])
        fpr, tpr, _ = roc_curve(labels, scores)
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, color=PRIMARY_COLOR, alpha=0.4, linewidth=1)

    plt.plot([0, 1], [0, 1], color=SECONDARY_COLOR, linestyle="--", linewidth=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", fontsize=12)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    sns.despine(trim=True, offset=10)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def make_all(
    save_dir,
    concepts,
    intervention_layers,
    scale,
    *,
    scored=None,
    calibration_results=None,
    baseline_acts=None,
    baseline_masks=None,
    v_detect=None,
    thresholds=None,
):
    setup_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    if scored is not None:
        plot_heatmap(scored, concepts, intervention_layers, scale,
                     "refusal_score", save_path=save_dir / "heatmap_refusal.png")
        plt.close()
        plot_heatmap(scored, concepts, intervention_layers, scale,
                     "retention_score", save_path=save_dir / "heatmap_retention.png")
        plt.close()

    if calibration_results is not None and not calibration_results.empty:
        plot_calibration(calibration_results, save_path=save_dir / "calibration.png")
        plt.close()

    if (baseline_acts is not None and baseline_masks is not None
            and v_detect is not None and thresholds is not None):
        plot_detection_roc(baseline_acts, baseline_masks, v_detect, thresholds,
                           concepts, intervention_layers,
                           save_path=save_dir / "detection_roc.png")
        plt.close()
