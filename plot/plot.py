import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


PRIMARY_COLOR = "darkred"
SECONDARY_COLOR = "black"
HARMONIC_COLOR = "purple"

AXES = ("refusal", "retention", "fluency")
AXIS_COLOR = {
    "refusal":   "darkred",
    "retention": "midnightblue",
    "fluency":   "darkgreen",
}
AXIS_LABEL = {axis: axis.title() for axis in AXES}
EPS = 1e-9


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


def harmonic_refusal_fluency(df):
    return 2 * df["judge_refusal"] * df["judge_fluency"] / (df["judge_refusal"] + df["judge_fluency"] + EPS)


def plot_calibration(calibration_judged_csv, save_path=None):
    df = pd.read_csv(calibration_judged_csv)
    if "label" in df:
        df = df[df["label"] == "intervention"]
    df = df.assign(judge_harmonic=harmonic_refusal_fluency(df))

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

    sns.lineplot(
        data=df, x="scale", y="judge_harmonic",
        color=HARMONIC_COLOR, label="Harmonic (R, F)",
        estimator="mean", errorbar=("ci", 95), linestyle="--",
    )

    harmonic_means = df.groupby("scale", as_index=False)["judge_harmonic"].mean()
    peak = harmonic_means.sort_values(["judge_harmonic", "scale"], ascending=[False, True]).iloc[0]
    plt.plot(
        peak["scale"], peak["judge_harmonic"],
        marker="*", color=SECONDARY_COLOR,
        markersize=14, fillstyle="none", linestyle="None",
    )

    plt.xlabel("Scale $s$")
    plt.ylabel("Score")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best", fontsize=8)
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


def plot_detection_roc(calibration_judged_csv, save_path=None):
    """ROC across scales: TPR = refusal when target==concept, FPR = refusal when target!=concept."""
    df = pd.read_csv(calibration_judged_csv)
    if "label" in df:
        df = df[df["label"] == "intervention"]
    df = df.assign(is_positive=(df["concept"] == df["target"]).astype(int))

    scales = sorted(df["scale"].unique())
    tprs, fprs = [], []
    for s in scales:
        sdf = df[df["scale"] == s]
        pos = sdf[sdf["is_positive"] == 1]["judge_refusal"]
        neg = sdf[sdf["is_positive"] == 0]["judge_refusal"]
        tprs.append(float(pos.mean()) if len(pos) else 0.0)
        fprs.append(float(neg.mean()) if len(neg) else 0.0)

    points = sorted(zip(fprs, tprs))
    fprs_s = [0.0] + [p[0] for p in points] + [1.0]
    tprs_s = [0.0] + [p[1] for p in points] + [1.0]
    auc_val = float(np.trapezoid(tprs_s, fprs_s))

    plt.figure()
    plt.plot(fprs_s, tprs_s, color=PRIMARY_COLOR, marker="o", linewidth=1.5, markersize=4)
    plt.plot([0, 1], [0, 1], color=SECONDARY_COLOR, linestyle="--", linewidth=0.5)
    plt.xlabel("FPR (refusal on $c\\neq$target)")
    plt.ylabel("TPR (refusal on $c=$target)")
    plt.title(f"AUC {auc_val:.3f}", fontsize=12)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    sns.despine(trim=True, offset=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def make_all(store, save_dir=None):
    store = Path(store)
    save_dir = Path(save_dir) if save_dir is not None else store / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    cal = store / "calibration_judged.csv"
    judged = store / "judged.csv"

    written = []
    if cal.exists():
        plot_calibration(cal, save_path=save_dir / "calibration.png")
        plt.close()
        written.append("calibration.png")

        plot_detection_roc(cal, save_path=save_dir / "detection_roc.png")
        plt.close()
        written.append("detection_roc.png")

    if judged.exists():
        df_head = pd.read_csv(judged, nrows=1)
        for axis in AXES:
            col = f"judge_{axis}"
            if col in df_head.columns:
                plot_heatmap(judged, save_path=save_dir / f"heatmap_{axis}.png", metric=col)
                plt.close()
                written.append(f"heatmap_{axis}.png")

    return written
