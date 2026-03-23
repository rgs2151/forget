import torch as t
from tqdm.auto import tqdm

def _masked_mean_acts(acts, token_mask=None):
    if token_mask is None:
        if acts.shape[2] == 1:
            return acts
        return acts.mean(dim=2, keepdim=True)

    mask = token_mask[:, None, :, None].to(acts.device, dtype=acts.dtype)
    denom = mask.sum(dim=2, keepdim=True).clamp_min(1)
    return (acts * mask).sum(dim=2, keepdim=True) / denom


def _flatten_token_rows(acts, token_mask=None):
    if token_mask is None:
        if acts.shape[2] == 1:
            return acts[:, :, 0, :]
        n, layers, seq_len, hidden = acts.shape
        return acts.permute(0, 2, 1, 3).reshape(n * seq_len, layers, hidden)

    token_mask = token_mask.to(acts.device).bool()
    token_first = acts.permute(0, 2, 1, 3)
    return token_first[token_mask]


def diffed_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    v_detect = {}
    for c in tqdm(concepts, desc="diffed_vectors detect", disable=not show_progress):
        others_mean = t.stack([
            _flatten_token_rows(know_acts[o], know_masks.get(o)).mean(0)
            for o in concepts if o != c
        ]).mean(0)
        diffs = _flatten_token_rows(know_acts[c], know_masks.get(c)) - others_mean.unsqueeze(0)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_detect[c] = diffs.mean(0).unsqueeze(1)
        v_detect[c] = v_detect[c] / v_detect[c].norm(dim=-1, keepdim=True)
        
    v_forget_per = {}
    for c in tqdm(concepts, desc="diffed_vectors forget", disable=not show_progress):
        know_mean = _masked_mean_acts(know_acts[c], know_masks.get(c))
        forget_mean = _masked_mean_acts(forget_acts[c], forget_masks.get(c))
        diffs = forget_mean - know_mean
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([
        _masked_mean_acts(forget_acts[c], forget_masks.get(c)) - _masked_mean_acts(know_acts[c], know_masks.get(c))
        for c in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    
    return v_detect, v_forget_per, v_forget

def projected_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    # precompute per-class means: (layers, 1, hidden)
    class_means = {
        c: _flatten_token_rows(know_acts[c], know_masks.get(c)).mean(0).unsqueeze(1)
        for c in concepts
    }

    v_detect = {}
    for c in tqdm(concepts, desc="projected_vectors detect", disable=not show_progress):
        others = [class_means[o] for o in concepts if o != c]
        L, _, H = class_means[c].shape
        steering = []
        for l in range(L):
            M = t.stack([o[l, 0] for o in others]).float()  # (k, hidden)
            Q, _ = t.linalg.qr(M.T)  # Q: (hidden, k)
            P = t.eye(H, device=Q.device) - Q @ Q.T  # nullspace projector
            a_c = _flatten_token_rows(know_acts[c], know_masks.get(c))[:, l, :].float()
            projected = a_c @ P  # (n, hidden)
            v = projected.mean(0)
            v = v / v.norm()
            steering.append(v)
        v_detect[c] = t.stack(steering).unsqueeze(1)  # (layers, 1, hidden)

    v_forget_per = {}
    for c in tqdm(concepts, desc="projected_vectors forget", disable=not show_progress):
        diffs = (
            _masked_mean_acts(forget_acts[c], forget_masks.get(c)) -
            _masked_mean_acts(know_acts[c], know_masks.get(c))
        ).float()
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([
        (
            _masked_mean_acts(forget_acts[c], forget_masks.get(c)) -
            _masked_mean_acts(know_acts[c], know_masks.get(c))
        ).float()
        for c in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)

    return v_detect, v_forget_per, v_forget


def lda_vectors(know_acts, forget_acts, concepts, know_masks=None, forget_masks=None, show_progress=True):
    know_masks = know_masks or {}
    forget_masks = forget_masks or {}

    class_means = {
        c: _flatten_token_rows(know_acts[c], know_masks.get(c)).mean(0).unsqueeze(1)
        for c in concepts
    }
    L, _, H = class_means[concepts[0]].shape
    reg = 1e-4 * t.eye(H).unsqueeze(0)  # (1, H, H)
    
    v_detect = {}
    thresholds = {}
    
    for target in tqdm(concepts, desc="lda_vectors detect", disable=not show_progress):
        X_t = _flatten_token_rows(know_acts[target], know_masks.get(target)).float()
        X_o = t.cat([
            _flatten_token_rows(know_acts[c], know_masks.get(c))
            for c in concepts if c != target
        ], dim=0).float()

        # transpose to (L, n, H) for batched ops
        X_t_l = X_t.permute(1, 0, 2)  # (L, n_t, H)
        X_o_l = X_o.permute(1, 0, 2)  # (L, n_o, H)

        mu_t = X_t_l.mean(1)  # (L, H)
        mu_o = X_o_l.mean(1)  # (L, H)

        # within-class scatter: (L, H, H)
        d_t = X_t_l - mu_t.unsqueeze(1)
        d_o = X_o_l - mu_o.unsqueeze(1)
        S_w = (d_t.transpose(1, 2) @ d_t) / X_t_l.shape[1] + \
              (d_o.transpose(1, 2) @ d_o) / X_o_l.shape[1] + reg.to(d_t.device)

        # batched solve: S_w @ w = diff → w = S_w^{-1} diff
        diff = mu_t - mu_o  # (L, H)
        w = t.linalg.solve(S_w, diff)  # (L, H)
        w = w / w.norm(dim=-1, keepdim=True)

        # threshold at midpoint of projected means
        tau = ((w * mu_t).sum(-1) + (w * mu_o).sum(-1)) / 2  # (L,)

        v_detect[target] = w.unsqueeze(1)      # (L, 1, H)
        thresholds[target] = tau                # (L,)
        
    v_forget_per = {}
    for c in tqdm(concepts, desc="lda_vectors forget", disable=not show_progress):
        diffs = (
            _masked_mean_acts(forget_acts[c], forget_masks.get(c)) -
            _masked_mean_acts(know_acts[c], know_masks.get(c))
        ).float()
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([
        (
            _masked_mean_acts(forget_acts[c], forget_masks.get(c)) -
            _masked_mean_acts(know_acts[c], know_masks.get(c))
        ).float()
        for c in concepts
    ], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    
    return v_detect, v_forget_per, v_forget, thresholds