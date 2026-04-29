import torch as t
from tqdm.auto import tqdm

from activations import flatten_token_rows, masked_mean_acts


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

    v_forget_per = {}
    for concept in tqdm(concepts, desc="diffed_vectors forget", disable=not show_progress):
        know_mean = masked_mean_acts(know_acts[concept], know_masks.get(concept))
        forget_mean = masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
        diffs = forget_mean - know_mean
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[concept] = diffs.mean(0)
        v_forget_per[concept] = v_forget_per[concept] / v_forget_per[concept].norm(dim=-1, keepdim=True)

    all_diffs = t.cat([
        masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
        - masked_mean_acts(know_acts[concept], know_masks.get(concept))
        for concept in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    return v_detect, v_forget_per, v_forget


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

    v_forget_per = {}
    for concept in tqdm(concepts, desc="projected_vectors forget", disable=not show_progress):
        diffs = (
            masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
            - masked_mean_acts(know_acts[concept], know_masks.get(concept))
        ).float()
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[concept] = diffs.mean(0)
        v_forget_per[concept] = v_forget_per[concept] / v_forget_per[concept].norm(dim=-1, keepdim=True)

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
    return v_detect, v_forget_per, v_forget


def lda_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    class_means = {
        concept: flatten_token_rows(know_acts[concept], know_masks.get(concept)).mean(0).unsqueeze(1)
        for concept in concepts
    }
    _, _, hidden = class_means[concepts[0]].shape
    reg = 1e-4 * t.eye(hidden).unsqueeze(0)

    v_detect = {}
    thresholds = {}
    for target in tqdm(concepts, desc="lda_vectors detect", disable=not show_progress):
        target_rows = flatten_token_rows(know_acts[target], know_masks.get(target)).float()
        other_rows = t.cat([
            flatten_token_rows(know_acts[concept], know_masks.get(concept))
            for concept in concepts if concept != target
        ], dim=0).float()

        target_layers = target_rows.permute(1, 0, 2)
        other_layers = other_rows.permute(1, 0, 2)

        mu_target = target_layers.mean(1)
        mu_other = other_layers.mean(1)

        delta_target = target_layers - mu_target.unsqueeze(1)
        delta_other = other_layers - mu_other.unsqueeze(1)
        scatter = (
            (delta_target.transpose(1, 2) @ delta_target) / target_layers.shape[1]
            + (delta_other.transpose(1, 2) @ delta_other) / other_layers.shape[1]
            + reg.to(delta_target.device)
        )

        diff = mu_target - mu_other
        weights = t.linalg.solve(scatter, diff)
        weights = weights / weights.norm(dim=-1, keepdim=True)
        tau = ((weights * mu_target).sum(-1) + (weights * mu_other).sum(-1)) / 2

        v_detect[target] = weights.unsqueeze(1)
        thresholds[target] = tau

    v_forget_per = {}
    for concept in tqdm(concepts, desc="lda_vectors forget", disable=not show_progress):
        diffs = (
            masked_mean_acts(forget_acts[concept], forget_masks.get(concept))
            - masked_mean_acts(know_acts[concept], know_masks.get(concept))
        ).float()
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[concept] = diffs.mean(0)
        v_forget_per[concept] = v_forget_per[concept] / v_forget_per[concept].norm(dim=-1, keepdim=True)

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
    return v_detect, v_forget_per, v_forget, thresholds
