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


def lda_vectors(know_acts, forget_acts, concepts, show_progress=True, device=None, layer_chunk=1):
    """LDA on per-example pooled activations. Shapes: know_acts[c] = [N_c, L, H].

    Layers are independent, so the covariance math is computed in layer chunks.
    """
    if device is None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    device = t.device(device)
    if device.type == "cuda":
        t.backends.cuda.preferred_linalg_library("cusolver")

    counts, x_sums = {}, {}
    total_xx_chunks, total_x_sum, hidden, n_layers = None, None, None, None
    for concept in tqdm(concepts, desc="lda_vectors totals", disable=not show_progress):
        concept_acts = know_acts[concept]
        counts[concept] = concept_acts.shape[0]
        if total_xx_chunks is None:
            n_layers, hidden = concept_acts.shape[1], concept_acts.shape[2]
            total_xx_chunks = [None for _ in range(0, n_layers, layer_chunk)]
            total_x_sum = t.zeros(n_layers, hidden)
        x_sums[concept] = t.empty(n_layers, hidden)

        for chunk_index, start in enumerate(range(0, n_layers, layer_chunk)):
            sl = slice(start, start + layer_chunk)
            acts = concept_acts[:, sl, :].to(device, non_blocking=True).float()
            layer_first = acts.permute(1, 0, 2)  # [chunk, N, H]
            x_sum = layer_first.sum(dim=1).cpu()
            xx_c = (layer_first.transpose(1, 2) @ layer_first).cpu()
            x_sums[concept][sl] = x_sum
            total_x_sum[sl] += x_sum
            if total_xx_chunks[chunk_index] is None:
                total_xx_chunks[chunk_index] = xx_c
            else:
                total_xx_chunks[chunk_index] += xx_c
            del xx_c, x_sum, layer_first, acts

    N = sum(counts.values())
    reg = (1e-2 * t.eye(hidden, device=device)).unsqueeze(0)

    v_detect, thresholds = {}, {}
    for target in tqdm(concepts, desc="lda_vectors detect", disable=not show_progress):
        n_c = counts[target]
        n_other = N - n_c
        layer_weights, layer_tau = [], []
        for chunk_index, start in enumerate(range(0, n_layers, layer_chunk)):
            sl = slice(start, start + layer_chunk)
            total_xx = total_xx_chunks[chunk_index].to(device, non_blocking=True)
            total_x = total_x_sum[sl].to(device, non_blocking=True)
            target_x_sum = x_sums[target][sl].to(device, non_blocking=True)
            acts = know_acts[target][:, sl, :].to(device, non_blocking=True).float()
            layer_first = acts.permute(1, 0, 2)
            xx_target = layer_first.transpose(1, 2) @ layer_first

            mu_target = target_x_sum / n_c
            mu_other = (total_x - target_x_sum) / n_other

            mm_target = t.einsum("lh,lk->lhk", mu_target, mu_target)
            mm_other = t.einsum("lh,lk->lhk", mu_other, mu_other)
            scatter_target = (xx_target - n_c * mm_target) / n_c
            scatter_other = ((total_xx - xx_target) - n_other * mm_other) / n_other
            scatter = scatter_target + scatter_other + reg

            diff = mu_target - mu_other
            weights = t.linalg.solve(scatter, diff.unsqueeze(-1)).squeeze(-1)
            weights = weights / weights.norm(dim=-1, keepdim=True)
            tau = ((weights * mu_target).sum(-1) + (weights * mu_other).sum(-1)) / 2

            layer_weights.append(weights.cpu())
            layer_tau.append(tau.cpu())
            del acts, layer_first, xx_target, scatter, scatter_target, scatter_other
            del mm_target, mm_other, weights, tau, diff, mu_target, mu_other
            del total_xx, total_x, target_x_sum

        v_detect[target] = t.cat(layer_weights, dim=0).unsqueeze(1)
        thresholds[target] = t.cat(layer_tau, dim=0)

    del total_xx_chunks, total_x_sum, x_sums
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
