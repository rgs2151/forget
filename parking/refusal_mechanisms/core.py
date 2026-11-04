"""Unit-local ReFT-r1 math and conditional intervention.

Training follows AxBench's TopKReLUIntervention and LsReFT objective at the
commit recorded in config.json. Padding is excluded explicitly. No upstream
packages or shared model/GPU code are modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
CACHE = UNIT / 'cache' / 'axbench_small'
PLOTS = UNIT / 'plots'


def config():
    return json.loads((UNIT / 'config.json').read_text())


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def prefix_length(llm, template):
    from forget.refuse.prompts import BASELINE_SYSTEM
    a = llm.tokenize(template.render(BASELINE_SYSTEM, '1'))
    b = llm.tokenize(template.render(BASELINE_SYSTEM, '2'))
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


def content_mask(attention_mask, prefix):
    return attention_mask.bool() & (attention_mask.long().cumsum(1) > prefix)


def split_data(train, test, cfg):
    train = train.copy().reset_index(names='source_row')
    test = test.copy().reset_index(names='source_row')
    train['source_split'], test['source_split'] = 'train', 'test'
    train = train.drop_duplicates('question')
    test = test[~test.question.isin(train.question)].drop_duplicates('question')
    parts = []
    for concept in sorted(train.concept.unique()):
        n_train, n_val = cfg['train_positive'], cfg['validation_per_concept']
        selected = train[train.concept == concept].sample(n_train + n_val, random_state=cfg['split_seed'])
        for split, sub in [('train', selected.iloc[:n_train]), ('validation', selected.iloc[n_train:]),
                           ('test', test[test.concept == concept].sample(cfg['test_per_concept'], random_state=cfg['split_seed']))]:
            parts.append(sub.assign(split=split))
    frame = pd.concat(parts, ignore_index=True).fillna('')
    frame['row_id'] = frame.source_split + ':' + frame.source_row.astype(str)
    assert frame.question.is_unique
    return frame


def evaluation_jobs(rows, concepts):
    """One target per test question: half targeted, half balanced off-target."""
    parts = []
    for i, concept in enumerate(concepts):
        frame = rows[rows.concept == concept].copy().reset_index(drop=True)
        half = len(frame) // 2
        assert len(frame) % 2 == 0
        frame['target'] = [concept] * half + [concepts[(i + 1 + j % (len(concepts) - 1)) % len(concepts)]
                                              for j in range(half)]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def hoyer(values):
    values = np.abs(np.asarray(values, dtype=float)).ravel()
    norm = np.linalg.norm(values)
    if norm == 0 or len(values) < 2:
        return float('nan')
    return float((np.sqrt(len(values)) - values.sum() / norm) / (np.sqrt(len(values)) - 1))


def choose_threshold(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    candidates = np.unique(np.r_[0., scores])
    balanced = [0.5 * ((scores[labels] > tau).mean() + (scores[~labels] <= tau).mean())
                for tau in candidates]
    return float(candidates[np.flatnonzero(np.isclose(balanced, max(balanced)))[-1]])


class RankOne(torch.nn.Module):
    def __init__(self, hidden_size, top_k=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((hidden_size,), 0.01))
        self.top_k = top_k

    def forward(self, hidden, mask):
        latent = torch.relu(hidden.float() @ self.weight) * mask
        k = min(self.top_k, int(mask.sum(1).min()))
        values, indices = latent.topk(k, dim=1)
        remaining = latent.scatter(1, indices, 0)
        penalty = (remaining.sum(1) / mask.sum(1).clamp_min(1)).mean()
        delta = values.mean(1)[:, None, None] * self.weight
        return hidden + (delta * mask[:, :, None]).to(hidden.dtype), penalty


class RecordedGate:
    def __init__(self, detector, direction, threshold, strength, prompt_mask):
        self.detector, self.direction = detector, direction
        self.threshold, self.strength = threshold, strength
        self.prompt_mask = prompt_mask
        self.alive = torch.ones(prompt_mask.shape[0], dtype=torch.bool, device=prompt_mask.device)
        self.records = {}

    def __call__(self, hidden):
        scores = torch.relu(hidden.float() @ self.detector.float()) if self.detector.ndim == 1 else torch.relu(
            torch.einsum('bsh,bh->bs', hidden.float(), self.detector.float()))
        threshold = torch.as_tensor(self.threshold, device=hidden.device).reshape(-1, 1)
        phase = 'prefill' if hidden.shape[1] > 1 else 'decode'
        mask = self.prompt_mask.to(hidden.device) if phase == 'prefill' else self.alive[:, None].to(hidden.device)
        gate = (scores > threshold) & mask
        active = (gate & mask).sum(1).detach().cpu().numpy()
        total = mask.sum(1).detach().cpu().numpy()
        maximum = scores.masked_fill(~mask.bool(), -torch.inf).max(1).values.detach().cpu().numpy()
        if phase not in self.records:
            self.records[phase] = [active, total, maximum]
        else:
            self.records[phase][0] += active
            self.records[phase][1] += total
            self.records[phase][2] = np.maximum(self.records[phase][2], maximum)
        strength = torch.as_tensor(self.strength, device=hidden.device, dtype=hidden.dtype).reshape(-1, 1, 1)
        return gate[:, :, None].to(hidden.dtype) * strength * self.direction.to(hidden.dtype)

    def summary(self):
        n = self.prompt_mask.shape[0]
        rows = [{} for _ in range(n)]
        for phase in ('prefill', 'decode'):
            active, total, maximum = self.records.get(phase, [np.zeros(n), np.zeros(n), np.zeros(n)])
            for i, row in enumerate(rows):
                row[f'{phase}_gate_fraction'] = float(active[i] / max(total[i], 1))
                row[f'{phase}_gate_any'] = int(active[i] > 0)
                row[f'{phase}_max_score'] = float(maximum[i])
        return rows


def make_pool(model, gpu_ids):
    from dotenv import load_dotenv
    from forget.llm.gpu import GPUPool
    load_dotenv(ROOT / '.env')
    pool = GPUPool.from_model_path(model, gpu_ids)
    for llm in pool.llms.values():
        llm.model.to(dtype=torch.bfloat16).eval()
        llm.set_capture_activations(False)
    return pool


def clean_output(llm, template, raw):
    text = template.trim_to_last_assistant(raw)
    for special in llm.tokenizer.all_special_tokens:
        text = text.replace(special, '')
    return text.strip()
