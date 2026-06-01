from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "figs"

RED = "darkred"
BLUE = "midnightblue"
GREEN = "darkgreen"
PURPLE = "purple"
BLACK = "#222222"
GRAY = "#9a9a9a"


def setup_style():
    sns.set_theme(context="paper", style="ticks", palette="dark")
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["legend.frameon"] = False


def read_summary(path):
    return pd.read_csv(ROOT / path)


def best_idk(path):
    df = read_summary(path)
    return df.sort_values(["idk_rate", "scale"], ascending=[False, True]).iloc[0]


def plot_prefill_evidence():
    phi_assistant = best_idk("debug/phi_debug/structured_steering/phi_store_all_layers_v2/summary.csv")
    phi_prefill = best_idk("debug/phi_debug/structured_steering/phi_all_content_best_validation_v1/summary.csv")
    qwen_assistant = best_idk("debug/qwen_debug/runs/qwen_clean_vectors_additive_assistant_control_v1/summary.csv")
    qwen_prefill = best_idk("debug/qwen_debug/runs/qwen_clean_vectors_additive_all10_validation_v1/summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    width = 0.55

    ax = axes[0]
    vals = [phi_assistant["idk_rate"], phi_prefill["idk_rate"]]
    labels = ["assistant\nmarker", "all-content\nprefill"]
    ax.bar(labels, vals, width=width, color=[GRAY, RED])
    ax.set_title("Phi: timing unlocks refusal")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.text(0, vals[0] + 0.04, f"{vals[0]:.2f}", ha="center", fontsize=13)
    ax.text(1, vals[1] - 0.12, f"{vals[1]:.2f}", ha="center", fontsize=13, color="white")

    ax = axes[1]
    x = [0, 1]
    idk_vals = [qwen_assistant["idk_rate"], qwen_prefill["idk_rate"]]
    start_vals = [0.01, 1.00]
    ax.bar([p - 0.17 for p in x], idk_vals, width=0.32, color=BLUE, label="IDK substring")
    ax.bar([p + 0.17 for p in x], start_vals, width=0.32, color=RED, label="refusal starts")
    ax.set_xticks(x, ["assistant\nmarker", "all-content\nprefill"])
    ax.set_title("Qwen: prefill fixes start")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    for p, val in zip([p - 0.17 for p in x], idk_vals):
        y = val - 0.12 if val > 0.75 else val + 0.04
        color = "white" if val > 0.75 else BLACK
        ax.text(p, y, f"{val:.2f}", ha="center", fontsize=12, color=color)
    for p, val in zip([p + 0.17 for p in x], start_vals):
        y = val - 0.12 if val > 0.9 else val + 0.04
        color = "white" if val > 0.9 else BLACK
        ax.text(p, y, f"{val:.2f}", ha="center", fontsize=12, color=color)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.6), fontsize=9)

    for ax in axes:
        sns.despine(ax=ax, trim=True, offset=10)
    fig.subplots_adjust(bottom=0.22, wspace=0.42)
    fig.savefig(OUT / "prefill_evidence.png", bbox_inches="tight")
    plt.close(fig)


def plot_layer_traces(ax, path, title, note):
    df = read_summary(path)
    layers = sorted(df["layer"].unique())
    cmap = mpl.colormaps["RdYlBu_r"]
    norm = Normalize(vmin=min(layers), vmax=max(layers))
    span = df["scale"].max() - df["scale"].min()
    for layer in layers:
        trace = df[df["layer"] == layer].sort_values("scale")
        color = cmap(norm(layer))
        ax.plot(trace["scale"], trace["idk_rate"], color=color, linewidth=2, label=f"L{int(layer)}")

    best = df.sort_values(["idk_rate", "scale"], ascending=[False, True]).iloc[0]
    ax.plot(
        best["scale"], best["idk_rate"],
        marker="*", color=BLACK, markersize=16,
        fillstyle="none", linestyle="None",
    )
    ax.text(
        0.04, 0.88,
        f"best: L{int(best['layer'])}, s={best['scale']:.0f}, {best['idk_rate']:.2f}",
        ha="left", transform=ax.transAxes, fontsize=10,
    )
    ax.set_title(title)
    ax.set_xlabel("Scale $s$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_xlim(df["scale"].min(), df["scale"].max() + span * 0.12)
    ax.set_xticks([df["scale"].min(), df["scale"].max()])
    ax.text(0.04, 0.08, note, ha="left", va="bottom", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="lower right", title="layer", fontsize=8, title_fontsize=8)
    sns.despine(ax=ax, trim=True, offset=10)


def plot_model_summary():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    plot_layer_traces(
        axes[0],
        "debug/phi_debug/structured_steering/phi_all_content_best_validation_v1/summary.csv",
        "Phi all-content prefill sweep",
        "store vector + EOS pad",
    )
    plot_layer_traces(
        axes[1],
        "debug/qwen_debug/runs/qwen_clean_vectors_additive_all10_validation_v1/summary.csv",
        "Qwen clean additive prefill sweep",
        "clean vector + additive mode",
    )
    axes[0].set_ylabel("IDK rate")
    axes[1].set_ylabel("")

    fig.subplots_adjust(bottom=0.22, right=0.96, wspace=0.34)
    fig.savefig(OUT / "phi_qwen_debug_sweeps.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_prefill_evidence()
    plot_model_summary()


if __name__ == "__main__":
    main()
