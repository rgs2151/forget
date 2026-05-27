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
    plt.yticks([0, 1])
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
    ax.set_xlabel("Concepts")
    ax.set_ylabel("Target Concepts")
    n = len(concepts)
    ax.set_xticks([0.5, n - 0.5], labels=[0, n - 1])
    ax.set_yticks([0.5, n - 0.5], labels=[0, n - 1])
    plt.xticks(rotation=0)

    label = metric.replace("judge_", "").replace("_", " ").title()
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(label, rotation=270, labelpad=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_bars(judged_csv, save_path=None):
    df = pd.read_csv(judged_csv)
    if "label" in df:
        df = df[df["label"] == "intervention"]
    df = df.assign(kind=np.where(df["concept"] == df["target"], "Target", "Untargeted"))

    score_cols = [f"judge_{axis}" for axis in AXES if f"judge_{axis}" in df.columns]
    long_df = df[["kind"] + score_cols].melt(id_vars="kind", var_name="axis", value_name="score")
    long_df["axis"] = long_df["axis"].str.replace("judge_", "").str.title()

    plt.figure(figsize=(5, 3))
    ax = sns.barplot(
        data=long_df, x="axis", y="score", hue="kind",
        order=[AXIS_LABEL[a] for a in AXES if f"judge_{a}" in df.columns],
        hue_order=["Target", "Untargeted"],
        palette={"Target": PRIMARY_COLOR, "Untargeted": SECONDARY_COLOR},
        errorbar=("ci", 95),
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.legend(loc="best", fontsize=8, title="")
    sns.despine(trim=True, offset=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def _plot_confusion(csv, save_dir, name):
    head = pd.read_csv(csv, nrows=1)
    written = []
    for axis in AXES:
        col = f"judge_{axis}"
        if col not in head.columns:
            continue
        out = save_dir / f"{name}_heatmap_{axis}.png"
        plot_heatmap(csv, save_path=out, metric=col)
        plt.close()
        written.append(out.name)
    return written


def _plot_bars(csv, save_dir, name):
    out = save_dir / f"{name}.png"
    plot_bars(csv, save_path=out)
    plt.close()
    return [out.name]


EVAL_PLOTTERS = {
    "confusion": _plot_confusion,
    "bars": _plot_bars,
}


def make_all(store, save_dir=None):
    store = Path(store)
    save_dir = Path(save_dir) if save_dir is not None else store / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    written = []
    cal = store / "calibration_judged.csv"
    if cal.exists():
        plot_calibration(cal, save_path=save_dir / "calibration.png")
        plt.close()
        written.append("calibration.png")

    for csv in sorted(store.glob("*_judged.csv")):
        if csv.name == "calibration_judged.csv":
            continue
        name = csv.stem[:-len("_judged")]
        plotter = EVAL_PLOTTERS.get(name)
        if plotter is None:
            continue
        written.extend(plotter(csv, save_dir, name))

    return written
