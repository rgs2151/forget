import numpy as np
import matplotlib.pyplot as plt


def paper_concept_label(concept):
    return concept.replace("_", " ").title()


def shared_concept_order(concepts, *frames):
    available = set()
    for frame in frames:
        available.update(frame["concept"].dropna().tolist())
        available.update(frame["target"].dropna().tolist())
    ordered = [concept for concept in concepts if concept in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def pivot_metric(df, metric, concept_order, target_order):
    return df.pivot_table(
        index="concept",
        columns="target",
        values=metric,
        aggfunc="mean",
    ).reindex(index=concept_order, columns=target_order)


def plot_effect_heatmaps(
    steered_df,
    baseline_df,
    concepts,
    source_layer,
    target_layer,
    left_metric,
    right_metric,
    left_title,
    right_title,
    left_label,
    right_label,
):
    src_str = str(source_layer)
    tgt_str = str(target_layer)
    steered_mask = (steered_df["source_layer"].astype(str) == src_str) & (steered_df["target_layer"].astype(str) == tgt_str)
    baseline_mask = (baseline_df["source_layer"].astype(str) == src_str) & (baseline_df["target_layer"].astype(str) == tgt_str)

    plot_df = steered_df[steered_mask].copy()
    baseline_plot_df = baseline_df[baseline_mask].copy()
    if plot_df.empty:
        raise ValueError("No steered QA results found for the requested source/target layers.")
    if baseline_plot_df.empty:
        raise ValueError("No baseline QA results found for the requested source/target layers.")

    concept_order = shared_concept_order(concepts, plot_df, baseline_plot_df)
    target_order = concept_order
    display_targets = [paper_concept_label(target) for target in target_order]
    display_concepts = [paper_concept_label(concept) for concept in concept_order]

    left_effect = (pivot_metric(plot_df, left_metric, concept_order, target_order) - pivot_metric(baseline_plot_df, left_metric, concept_order, target_order)).fillna(0)
    right_effect = (pivot_metric(plot_df, right_metric, concept_order, target_order) - pivot_metric(baseline_plot_df, right_metric, concept_order, target_order)).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    left_im = axes[0].imshow(left_effect.to_numpy(), cmap="gray_r", aspect="equal")
    right_im = axes[1].imshow(right_effect.to_numpy(), cmap="gray_r", aspect="equal")

    for ax, title in zip(axes, [left_title, right_title]):
        ax.set_xticks(range(len(display_targets)))
        ax.set_xticklabels(display_targets, rotation=45, ha="right")
        ax.set_yticks(range(len(display_concepts)))
        ax.set_yticklabels(display_concepts)
        ax.set_xlabel("Steered concept")
        ax.set_ylabel("Question concept")
        ax.set_title(title)

    fig.colorbar(left_im, ax=axes[0], fraction=0.046, pad=0.04).set_label(left_label)
    fig.colorbar(right_im, ax=axes[1], fraction=0.046, pad=0.04).set_label(right_label)
    plt.show()


def plot_metric_by_scale(df, source_layer, target_layer, metric_col, label, show_std=False, ylabel=None):
    scales = sorted(df["scale"].unique())
    src_str = str(source_layer)
    tgt_str = str(target_layer)
    mask_layer = (df["source_layer"].astype(str) == src_str) & (df["target_layer"].astype(str) == tgt_str)

    concepts = df["concept"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for color, concept in zip(colors, concepts):
        means = []
        sems = []
        for scale in scales:
            sub = df[mask_layer & (df["scale"] == scale) & (df["concept"] == concept)]
            n_items = len(sub)
            means.append(sub[metric_col].mean())
            sems.append(sub[metric_col].std() / np.sqrt(n_items) if n_items > 1 else 0)
        ax.plot(scales, means, color=color, label=concept)
        if show_std:
            ax.fill_between(scales, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=color)

    ax.set_xlabel("scale")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(label)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()