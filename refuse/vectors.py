import gc

import torch as t
from tqdm.auto import tqdm

from .activations import flatten_token_rows, masked_mean_acts
from .paths import cached_pt


def diffed_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    v_detect = {}
    for concept in tqdm(concepts, desc="diffed_vectors detect", disable=not show_progress):
        others_mean = t.stack([
            flatten_token_rows(know_acts[other], know_masks.get(other)).mean(0)
            for other in concepts if other != concept
        ]).mean(0)
        diffs = flatten_token_rows(know_acts[concept], know_masks.get(concept)) - others_mean.unsqueeze(0)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_detect[concept] = diffs.mean(0).unsqueeze(1)
        v_detect[concept] = v_detect[concept] / v_detect[concept].norm(dim=-1, keepdim=True)

    all_diffs = t.cat([
        masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
        - masked_mean_acts(know_acts[concept], know_masks.get(concept))
        for concept in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget


def projected_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    class_means = {
        concept: flatten_token_rows(know_acts[concept], know_masks.get(concept)).mean(0).unsqueeze(1)
        for concept in concepts
    }

    v_detect = {}
    for concept in tqdm(concepts, desc="projected_vectors detect", disable=not show_progress):
        others = [class_means[other] for other in concepts if other != concept]
        n_layers, _, hidden = class_means[concept].shape
        steering = []
        for layer_index in range(n_layers):
            matrix = t.stack([other[layer_index, 0] for other in others]).float()
            q_mat, _ = t.linalg.qr(matrix.T)
            projector = t.eye(hidden, device=q_mat.device) - q_mat @ q_mat.T
            activations = flatten_token_rows(know_acts[concept], know_masks.get(concept))[:, layer_index, :].float()
            projected = activations @ projector
            vector = projected.mean(0)
            vector = vector / vector.norm()
            steering.append(vector)
        v_detect[concept] = t.stack(steering).unsqueeze(1)

    all_diffs = t.cat([
        (
            masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
            - masked_mean_acts(know_acts[concept], know_masks.get(concept))
        ).float()
        for concept in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget


def lda_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None,
                show_progress=True, device=None):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}
    if device is None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    device = t.device(device)

    counts, x_sums = {}, {}
    total_xx_sum, total_x_sum, hidden = None, None, None
    for concept in tqdm(concepts, desc="lda_vectors totals", disable=not show_progress):
        acts = know_acts[concept].to(device, non_blocking=True)
        mask = know_masks.get(concept)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        rows = flatten_token_rows(acts, mask).float()
        layer_first = rows.permute(1, 0, 2)
        counts[concept] = layer_first.shape[1]
        x_sums[concept] = layer_first.sum(dim=1)
        xx_c = layer_first.transpose(1, 2) @ layer_first
        if total_xx_sum is None:
            hidden = layer_first.shape[2]
            total_xx_sum = xx_c.clone()
            total_x_sum = x_sums[concept].clone()
        else:
            total_xx_sum += xx_c
            total_x_sum += x_sums[concept]
        del xx_c, rows, layer_first, acts

    N = sum(counts.values())
    reg = (1e-4 * t.eye(hidden, device=device)).unsqueeze(0)

    v_detect, thresholds = {}, {}
    for target in tqdm(concepts, desc="lda_vectors detect", disable=not show_progress):
        acts = know_acts[target].to(device, non_blocking=True)
        mask = know_masks.get(target)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        rows = flatten_token_rows(acts, mask).float()
        layer_first = rows.permute(1, 0, 2)
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
        del rows, layer_first, xx_target, scatter, scatter_target, scatter_other, mm_target, mm_other, acts

    del total_xx_sum, total_x_sum, x_sums
    gc.collect()
    t.cuda.empty_cache()

    all_diff_parts = []
    for concept in tqdm(concepts, desc="lda_vectors forget", disable=not show_progress):
        forget = forget_acts[concept]
        know = know_acts[concept]
        fmask = forget_masks.get(concept)
        kmask = know_masks.get(concept)
        forget_mean = masked_mean_acts(forget, fmask).float()
        know_mean = masked_mean_acts(know, kmask).float()
        d = forget_mean - know_mean
        all_diff_parts.append(d / d.norm(dim=-1, keepdim=True))
        del forget_mean, know_mean, d
    all_diffs = t.cat(all_diff_parts, dim=0)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget, thresholds


def cached_lda_vectors(know_acts, forget_acts, concepts, paths, know_masks=None, forget_masks=None, device=None):
    paths_dict = {
        "v_detect": paths.v_detect,
        "v_refuse": paths.v_refuse,
        "thresholds": paths.thresholds,
    }
    def compute():
        v_detect, v_refuse, thresholds = lda_vectors(
            know_acts, forget_acts, concepts, know_masks, forget_masks, device=device,
        )
        return {
            "v_detect": v_detect,
            "v_refuse": v_refuse,
            "thresholds": thresholds,
        }
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"], loaded["thresholds"]


def cached_diffed_vectors(know_acts, forget_acts, concepts, paths, know_masks=None, forget_masks=None, device=None):
    paths_dict = {
        "v_detect": paths.v_detect,
        "v_refuse": paths.v_refuse,
    }
    def compute():
        v_detect, v_refuse = diffed_vectors(
            know_acts, forget_acts, concepts, know_masks, forget_masks,
        )
        return {"v_detect": v_detect, "v_refuse": v_refuse}
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"]


def cached_projected_vectors(know_acts, forget_acts, concepts, paths, know_masks=None, forget_masks=None, device=None):
    paths_dict = {
        "v_detect": paths.v_detect,
        "v_refuse": paths.v_refuse,
    }
    def compute():
        v_detect, v_refuse = projected_vectors(
            know_acts, forget_acts, concepts, know_masks, forget_masks,
        )
        return {"v_detect": v_detect, "v_refuse": v_refuse}
    loaded = cached_pt(paths_dict, compute)
    return loaded["v_detect"], loaded["v_refuse"]
