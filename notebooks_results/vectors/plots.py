import matplotlib.pyplot as plt
import torch as t


def plot_vdetect_similarity(v_detect, concepts, target="obama"):
    cos = t.nn.CosineSimilarity(dim=-1)
    target_flat = v_detect[target].float().squeeze()
    num_layers = target_flat.shape[0]

    fig, ax = plt.subplots(figsize=(6, 3))
    for concept in concepts:
        if concept == target:
            continue
        sim = cos(target_flat, v_detect[concept].float().squeeze()).numpy()
        ax.plot(range(num_layers), sim, label=f"{target} vs {concept}")
    ax.set_xlim(0, num_layers - 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_vdetect_gate(target, concepts, know_acts_train, know_acts_val, v_detect):
    gate = v_detect[target].float().squeeze()
    num_layers = gate.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for ax, acts, title in [(ax1, know_acts_train, "train"), (ax2, know_acts_val, "test")]:
        for concept in concepts:
            projections = (acts[concept].float().squeeze(2) * gate.unsqueeze(0)).sum(-1)
            line, = ax.plot(
                range(num_layers),
                projections.mean(0).numpy(),
                label=concept,
                linewidth=2 if concept == target else 0.8,
                zorder=2 if concept == target else 1,
            )
            ax.fill_between(
                range(num_layers),
                (projections.mean(0) - projections.std(0)).numpy(),
                (projections.mean(0) + projections.std(0)).numpy(),
                alpha=0.15,
                color=line.get_color(),
            )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlim(0, num_layers - 1)
        ax.set_xlabel("Layer")
        ax.set_title(f"{target} gate - {title}")
    ax1.set_ylabel(f"Projection onto v_detect[{target}]")
    ax1.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_gated_separation(target, concepts, know_acts_train, know_acts_val, v_detect, thresholds):
    gate = v_detect[target].float().squeeze()
    tau = thresholds[target]
    if not isinstance(tau, t.Tensor):
        tau = t.tensor(tau).float()
    tau = tau.float().squeeze()
    num_layers = gate.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, acts, title in [(axes[0], know_acts_train, "train"), (axes[1], know_acts_val, "test")]:
        for concept in concepts:
            projections = (acts[concept].float().squeeze(2) * gate.unsqueeze(0)).sum(-1) - tau.unsqueeze(0)
            mean = projections.mean(0).numpy()
            std = projections.std(0).numpy()
            is_target = concept == target
            line, = ax.plot(
                range(num_layers),
                mean,
                label=concept,
                linewidth=2.5 if is_target else 0.8,
                zorder=3 if is_target else 1,
            )
            ax.fill_between(
                range(num_layers),
                mean - std,
                mean + std,
                alpha=0.25 if is_target else 0.08,
                color=line.get_color(),
            )
        ax.axhline(0, color="red", linewidth=1, ls="--", label="gate threshold")
        ax.set_xlim(0, num_layers - 1)
        ax.set_xlabel("Layer")
        ax.set_title(f"{target} gated separation - {title}")
    axes[0].set_ylabel(f"(acts @ v_detect[{target}]) - tau")
    axes[0].legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_forget_vector_similarity(v_forget_per, concepts):
    cos = t.nn.CosineSimilarity(dim=-1)
    pairs = [(i, j) for i in range(len(concepts)) for j in range(i + 1, len(concepts))]
    sims = t.stack([
        cos(v_forget_per[concepts[i]].float().squeeze(), v_forget_per[concepts[j]].float().squeeze())
        for i, j in pairs
    ])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(sims.shape[1]), sims.mean(0).numpy(), color="darkred", label="mean")
    ax.fill_between(
        range(sims.shape[1]),
        (sims.mean(0) - sims.std(0)).numpy(),
        (sims.mean(0) + sims.std(0)).numpy(),
        alpha=0.5,
        color="darkred",
        label="+-1 std",
    )
    ax.fill_between(
        range(sims.shape[1]),
        sims.min(0).values.numpy(),
        sims.max(0).values.numpy(),
        alpha=0.15,
        color="darkred",
        label="min-max range",
    )
    ax.set_xlim(0, sims.shape[1] - 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("v_forget_per: all-pairs cosine similarity across layers")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()