import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as t
from matplotlib.colors import ListedColormap
from sklearn.metrics import auc, roc_curve


PRIMARY_COLOR = "darkred"
SECONDARY_COLOR = "black"

AXES = ("refusal", "retention", "fluency")
AXIS_COLOR = {
    "refusal":   "darkred",
    "retention": "midnightblue",
    "fluency":   "darkgreen",
}
AXIS_LABEL = {axis: axis.title() for axis in AXES}


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


def _derive_intervention_layers(df):
    raw = df["source_layer"].iloc[0]
    return list(ast.literal_eval(raw)) if isinstance(raw, str) else list(raw)


def _derive_scale(df):
    nonzero = sorted(s for s in df.loc[df["label"] == "intervention", "scale"].unique() if s != 0)
    if not nonzero:
        raise ValueError("no non-zero intervention scale found in judged dataframe")
    return float(nonzero[0])


def plot_calibration(calibration_judged_csv, save_path=None):
    df = pd.read_csv(calibration_judged_csv)
    if "label" in df:
        df = df[df["label"] == "intervention"]

    plt.figure()
    for axis in AXES:
        col = f"judge_{axis}"
        if col not in df.columns:
            continue
        sns.lineplot(
            data=df, x="scale", y=col,
            color=AXIS_COLOR[axis], label=AXIS_LABEL[axis],
            estimator="mean", errorbar=("ci", 95),
        )

    means = df.groupby("scale", as_index=False)["judge_refusal"].mean()
    peak = means.sort_values(["judge_refusal", "scale"], ascending=[False, True]).iloc[0]
    plt.plot(
        peak["scale"], peak["judge_refusal"],
        marker="*", color=SECONDARY_COLOR,
        markersize=14, fillstyle="none", linestyle="None",
    )

    plt.xlabel("Scale $s$")
    plt.ylabel("Score")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best")
    sns.despine(trim=True, offset=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_heatmap(judged_csv, save_path=None, metric="judge_refusal",
                 scale=None, intervention_layers=None, concepts=None):
    df = pd.read_csv(judged_csv)
    if intervention_layers is None:
        intervention_layers = _derive_intervention_layers(df)
    if scale is None:
        scale = _derive_scale(df)
    if concepts is None:
        concepts = list(df["concept"].unique())

    src = str(intervention_layers)
    tgt = str(intervention_layers)
    plot_df = df[
        (df["label"] == "intervention")
        & (df["scale"] == scale)
        & (df["source_layer"].astype(str) == src)
        & (df["target_layer"].astype(str) == tgt)
    ]
    scores = (
        plot_df.pivot_table(index="concept", columns="target", values=metric, aggfunc="mean")
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

    label = metric.replace("judge_", "").replace("_", " ").title()
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(label, rotation=270, labelpad=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_detection_roc(baseline_test_acts_pt, baseline_test_masks_pt, v_detect_pt,
                       save_path=None, layer_idx=None, intervention_layers=None):
    acts = t.load(baseline_test_acts_pt, weights_only=False)
    masks = t.load(baseline_test_masks_pt, weights_only=False)
    v_detect = t.load(v_detect_pt, weights_only=False)
    concepts = list(v_detect.keys())

    if layer_idx is None:
        if intervention_layers is None:
            n_layers = v_detect[concepts[0]].shape[0]
            layer_idx = n_layers - 1
        else:
            layer_idx = intervention_layers[-1]

    pooled = {c: _mean_pool(acts[c], masks[c]) for c in concepts}
    cmap = custom_cmap()

    plt.figure()
    aucs = []
    for i, concept in enumerate(concepts):
        v = v_detect[concept][layer_idx, 0].float()
        pos = pooled[concept][:, layer_idx, 0, :].float() @ v
        neg = t.cat([pooled[other][:, layer_idx, 0, :].float() @ v
                     for other in concepts if other != concept], dim=0)
        scores = np.concatenate([pos.cpu().numpy(), neg.cpu().numpy()])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        fpr, tpr, _ = roc_curve(labels, scores)
        aucs.append(auc(fpr, tpr))
        color = cmap(i / max(len(concepts) - 1, 1))
        plt.plot(fpr, tpr, color=color, linewidth=1.2, alpha=0.85)

    plt.plot([0, 1], [0, 1], color=SECONDARY_COLOR, linestyle="--", linewidth=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (n={len(concepts)}, layer {layer_idx})",
              fontsize=10)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    sns.despine(trim=True, offset=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def _mean_pool(acts, token_mask):
    if token_mask is None:
        if acts.shape[2] == 1:
            return acts
        return acts.mean(dim=2, keepdim=True)
    mask = token_mask[:, None, :, None].to(acts.device, dtype=acts.dtype)
    denom = mask.sum(dim=2, keepdim=True).clamp_min(1)
    return (acts * mask).sum(dim=2, keepdim=True) / denom


def make_all(store, save_dir=None):
    store = Path(store)
    save_dir = Path(save_dir) if save_dir is not None else store / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    cal = store / "calibration_judged.csv"
    judged = store / "judged.csv"
    btacts = store / "baseline_answer_acts_test.pt"
    btmasks = store / "baseline_answer_masks_test.pt"
    vdet = store / "v_detect.pt"

    written = []
    if cal.exists():
        plot_calibration(cal, save_path=save_dir / "calibration.png")
        plt.close()
        written.append("calibration.png")

    if judged.exists():
        for axis in AXES:
            col = f"judge_{axis}"
            df_head = pd.read_csv(judged, nrows=1)
            if col in df_head.columns:
                plot_heatmap(judged, save_path=save_dir / f"heatmap_{axis}.png", metric=col)
                plt.close()
                written.append(f"heatmap_{axis}.png")

    if btacts.exists() and btmasks.exists() and vdet.exists():
        plot_detection_roc(btacts, btmasks, vdet, save_path=save_dir / "detection_roc.png")
        plt.close()
        written.append("detection_roc.png")

    return written
