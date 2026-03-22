import torch as t

def diffed_vectors(know_acts, forget_acts, concepts):
    v_detect = {}
    for c in concepts:
        others_mean = t.stack([know_acts[o].mean(0) for o in concepts if o != c]).mean(0)
        diffs = know_acts[c] - others_mean.unsqueeze(0)  # (n, layers, 1, hidden)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_detect[c] = diffs.mean(0)
        v_detect[c] = v_detect[c] / v_detect[c].norm(dim=-1, keepdim=True)
        
    v_forget_per = {}
    for c in concepts:
        diffs = forget_acts[c] - know_acts[c]  # (n, layers, 1, hidden)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([forget_acts[c] - know_acts[c] for c in concepts], dim=0)  # (n_total, layers, 1, hidden)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    
    return v_detect, v_forget_per, v_forget

def projected_vectors(know_acts, forget_acts, concepts):
    # precompute per-class means: (layers, 1, hidden)
    class_means = {c: know_acts[c].mean(0) for c in concepts}

    v_detect = {}
    for c in concepts:
        others = [class_means[o] for o in concepts if o != c]
        L, _, H = class_means[c].shape
        steering = []
        for l in range(L):
            M = t.stack([o[l, 0] for o in others]).float()  # (k, hidden)
            Q, _ = t.linalg.qr(M.T)  # Q: (hidden, k)
            P = t.eye(H, device=Q.device) - Q @ Q.T  # nullspace projector
            a_c = know_acts[c][:, l, 0, :].float()  # (n, hidden)
            projected = a_c @ P  # (n, hidden)
            v = projected.mean(0)
            v = v / v.norm()
            steering.append(v)
        v_detect[c] = t.stack(steering).unsqueeze(1)  # (layers, 1, hidden)

    v_forget_per = {}
    for c in concepts:
        diffs = (forget_acts[c] - know_acts[c]).float()  # (n, layers, 1, hidden)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([(forget_acts[c] - know_acts[c]).float() for c in concepts], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)

    return v_detect, v_forget_per, v_forget


def lda_vectors(know_acts, forget_acts, concepts):
    class_means = {c: know_acts[c].mean(0) for c in concepts}  # (layers, 1, hidden)
    L, _, H = class_means[concepts[0]].shape
    
    v_detect = {}
    thresholds = {}
    
    for target in concepts:
        steering = []
        taus = []
        for l in range(L):
            # target vs rest
            X_t = know_acts[target][:, l, 0, :].float()   # (n_t, H)
            X_o = t.cat([know_acts[c][:, l, 0, :] for c in concepts if c != target], dim=0).float()  # (n_o, H)
            
            mu_t = X_t.mean(0)
            mu_o = X_o.mean(0)
            
            # within-class scatter
            S_t = (X_t - mu_t).T @ (X_t - mu_t) / X_t.shape[0]
            S_o = (X_o - mu_o).T @ (X_o - mu_o) / X_o.shape[0]
            S_w = S_t + S_o  # (H, H)
            
            # regularize — S_w is rank-deficient with few samples
            S_w += 1e-4 * t.eye(H, device=S_w.device)
            
            # fisher direction: S_w^{-1} (mu_t - mu_o)
            diff = mu_t - mu_o
            w = t.linalg.solve(S_w, diff)
            w = w / w.norm()
            
            # threshold at midpoint of projected means
            tau = (w @ mu_t + w @ mu_o) / 2
            
            steering.append(w)
            taus.append(tau)
        
        v_detect[target] = t.stack(steering).unsqueeze(1)  # (layers, 1, hidden)
        thresholds[target] = t.stack(taus)                  # (layers,)
        
    v_forget_per = {}
    for c in concepts:
        diffs = (forget_acts[c] - know_acts[c]).float()  # (n, layers, 1, hidden)
        diffs = diffs / diffs.norm(dim=-1, keepdim=True)
        v_forget_per[c] = diffs.mean(0)
        v_forget_per[c] = v_forget_per[c] / v_forget_per[c].norm(dim=-1, keepdim=True)
        
    all_diffs = t.cat([(forget_acts[c] - know_acts[c]).float() for c in concepts], dim=0)
    all_diffs = all_diffs / all_diffs.norm(dim=-1, keepdim=True)
    v_forget = all_diffs.mean(0)
    v_forget = v_forget / v_forget.norm(dim=-1, keepdim=True)
    
    return v_detect, v_forget_per, v_forget, thresholds