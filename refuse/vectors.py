import gc

import torch as t
from tqdm.auto import tqdm

from .paths import cached_pt


def diffed_vectors(know_acts, forget_acts, concepts, show_progress=True):
    v_detect = {}
    for concept in tqdm(concepts, desc="diffed_vectors detect", disable=not show_progress):
        others_mean = t.stack([
            know_acts[other].mean(0)
            for other in concepts if other != concept
        ]).mean(0)
        diffs = know_acts[concept] - others_mean.unsqueeze(0)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_detect[concept] = diffs.mean(0).unsqueeze(1)
        v_detect[concept] = v_detect[concept] / v_detect[concept].norm(dim=-1, keepdim=True)

    all_diffs = t.stack([
        forget_acts[concept].mean(0) - know_acts[concept].mean(0)
        for concept in concepts
    ])
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget


def projected_vectors(know_acts, forget_acts, concepts, show_progress=True):
    class_means = {concept: know_acts[concept].mean(0) for concept in concepts}

    v_detect = {}
    for concept in tqdm(concepts, desc="projected_vectors detect", disable=not show_progress):
        others = [class_means[other] for other in concepts if other != concept]
        n_layers, hidden = class_means[concept].shape
        steering = []
        for layer_index in range(n_layers):
            matrix = t.stack([other[layer_index] for other in others]).float()
            q_mat, _ = t.linalg.qr(matrix.T)
            projector = t.eye(hidden, device=q_mat.device) - q_mat @ q_mat.T
            activations = know_acts[concept][:, layer_index, :].float()
            projected = activations @ projector
            vector = projected.mean(0)
            vector = vector / vector.norm()
            steering.append(vector)
        v_detect[concept] = t.stack(steering).unsqueeze(1)

    all_diffs = t.stack([
        (forget_acts[concept].mean(0) - know_acts[concept].mean(0)).float()
        for concept in concepts
    ])
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget


def lda_vectors(know_acts, forget_acts, concepts, show_progress=True, device=None):
    """LDA on per-example pooled activations. Shapes: know_acts[c] = [N_c, L, H]."""
    if device is None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    device = t.device(device)

    counts, x_sums = {}, {}
    total_xx_sum, total_x_sum, hidden = None, None, None
    for concept in tqdm(concepts, desc="lda_vectors totals", disable=not show_progress):
        acts = know_acts[concept].to(device, non_blocking=True).float()
        layer_first = acts.permute(1, 0, 2)  # [L, N, H]
        counts[concept] = layer_first.shape[1]
        x_sums[concept] = layer_first.sum(dim=1)  # [L, H]
        xx_c = layer_first.transpose(1, 2) @ layer_first  # [L, H, H]
        if total_xx_sum is None:
            hidden = layer_first.shape[2]
            total_xx_sum = xx_c.clone()
            total_x_sum = x_sums[concept].clone()
        else:
            total_xx_sum += xx_c
            total_x_sum += x_sums[concept]
        del xx_c, layer_first, acts

    N = sum(counts.values())
    reg = (1e-2 * t.eye(hidden, device=device)).unsqueeze(0)

    v_detect, thresholds = {}, {}
    for target in tqdm(concepts, desc="lda_vectors detect", disable=not show_progress):
        acts = know_acts[target].to(device, non_blocking=True).float()
        layer_first = acts.permute(1, 0, 2)
        xx_target = layer_first.transpose(1, 2) @ layer_first

        n_c = counts[target]
        n_other = N - n_c
        mu_target = x_sums[target] / n_c
        mu_other = (total_x_sum - x_sums[target]) / n_other

        mm_target = t.einsum("lh,lk->lhk", mu_target, mu_target)
        mm_other = t.einsum("lh,lk->lhk", mu_other, mu_other)
        scatter_target = (xx_target - n_c * mm_target) / n_c
        scatter_other = ((total_xx_sum - xx_target) - n_other * mm_other) / n_other
        scatter = scatter_target + scatter_other + reg

        diff = mu_target - mu_other
        chol = t.linalg.cholesky(scatter)
        weights = t.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
        weights = weights / weights.norm(dim=-1, keepdim=True)
        tau = ((weights * mu_target).sum(-1) + (weights * mu_other).sum(-1)) / 2

        v_detect[target] = weights.unsqueeze(1).cpu()
        thresholds[target] = tau.cpu()
        del layer_first, xx_target, scatter, scatter_target, scatter_other, mm_target, mm_other, acts

    del total_xx_sum, total_x_sum, x_sums
    gc.collect()
    t.cuda.empty_cache()

    all_diffs = t.stack([
        (forget_acts[concept].mean(0) - know_acts[concept].mean(0)).float()
        for concept in concepts
    ])
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget, thresholds


def cached_lda_vectors(know_acts, forget_acts, concepts, paths, device=None):
    paths_dict = {
        "v_detect": paths.v_detect,
        "v_refuse": paths.v_refuse,
        "thresholds": paths.thresholds,
    }
    def compute():
        v_detect, v_refuse, thresholds = lda_vectors(know_acts, forget_acts, concepts, device=device)
        return {"v_detect": v_detect, "v_refuse": v_refuse, "thresholds": thresholds}
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"], loaded["thresholds"]


def cached_diffed_vectors(know_acts, forget_acts, concepts, paths, device=None):
    paths_dict = {"v_detect": paths.v_detect, "v_refuse": paths.v_refuse}
    def compute():
        v_detect, v_refuse = diffed_vectors(know_acts, forget_acts, concepts)
        return {"v_detect": v_detect, "v_refuse": v_refuse}
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"]


def cached_projected_vectors(know_acts, forget_acts, concepts, paths, device=None):
    paths_dict = {"v_detect": paths.v_detect, "v_refuse": paths.v_refuse}
    def compute():
        v_detect, v_refuse = projected_vectors(know_acts, forget_acts, concepts)
        return {"v_detect": v_detect, "v_refuse": v_refuse}
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"]
