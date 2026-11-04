"""Train independent ReFT-r1 directions, sharded by layer across two GPUs."""

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from forget.refuse.prompts import BASELINE_SYSTEM
from .core import CACHE, RankOne, choose_threshold, config, content_mask, hoyer, make_pool, prefix_length, save_csv


REFUSALS = ["I don't know.", "I don't have that information.",
            "I don't know the answer.", "I cannot answer that question."]


def training_examples(rows, target, cfg):
    train = rows[rows.split == 'train']
    if target == '__refusal__':
        n = cfg['refusal_training_questions'] // train.concept.nunique()
        positive = train.groupby('concept', group_keys=False).sample(n, random_state=cfg['split_seed'])
        negative = positive
    else:
        positive = train[train.concept == target]
        others = train[train.concept != target]
        per_concept = math.ceil(cfg['train_negative'] / others.concept.nunique())
        negative = others.groupby('concept', group_keys=False).sample(per_concept, random_state=cfg['split_seed']).sample(
            cfg['train_negative'], random_state=cfg['split_seed'])
    examples = []
    for polarity, frame in [(1, positive), (0, negative)]:
        for i, row in enumerate(frame.itertuples()):
            answer = REFUSALS[i % len(REFUSALS)] if target == '__refusal__' and polarity else row.baseline_output
            examples.append((row.question, answer))
    return examples


def encode_examples(llm, template, examples, cfg):
    encoded = []
    for question, answer in examples:
        prefix = llm.tokenize(template.render(BASELINE_SYSTEM, question))
        completion = llm.tokenize(answer + template.assistant_end_marker)
        ids = (prefix + completion)[:cfg['max_training_tokens']]
        assert len(ids) > len(prefix)
        encoded.append((ids, [-100] * len(prefix) + ids[len(prefix):]))
    return encoded


def fit(llm, template, examples, layer, seed, cfg, smoke=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoded = encode_examples(llm, template, examples, cfg)
    direction = RankOne(llm.model.config.hidden_size, cfg['top_k']).to(llm.device)
    optimizer = torch.optim.AdamW(direction.parameters(), lr=cfg['learning_rate'], weight_decay=0.)
    n_steps = cfg['epochs'] * math.ceil(len(encoded) / cfg['train_batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1 - step / n_steps))
    state, losses = {}, []
    prefix = prefix_length(llm, template)

    def intervention(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed, penalty = direction(hidden, state['mask'])
        state['penalty'] = penalty
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = llm.model.model.layers[layer].register_forward_hook(intervention)
    started, step = time.monotonic(), 0
    for epoch in range(cfg['epochs']):
        indices = rng.permutation(len(encoded))
        for start in range(0, len(indices), cfg['train_batch_size']):
            batch = [encoded[i] for i in indices[start:start + cfg['train_batch_size']]]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([ids + [llm.pad_token_id] * (width - len(ids)) for ids, _ in batch], device=llm.device)
            labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch], device=llm.device)
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch], device=llm.device)
            state['mask'] = content_mask(mask, prefix)
            optimizer.zero_grad(set_to_none=True)
            outputs = llm.model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            loss = outputs.loss + step / n_steps * cfg['sparsity_coefficient'] * state['penalty']
            assert torch.isfinite(loss), f'Non-finite loss at layer {layer}'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(direction.parameters(), 1.)
            with torch.no_grad():
                direction.weight.div_(direction.weight.norm() + torch.finfo(torch.float32).eps)
                direction.weight.grad.sub_((direction.weight.grad @ direction.weight) * direction.weight)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            step += 1
            del outputs, loss
            state.pop('penalty')
            if smoke and step == 2:
                break
        if smoke:
            break
    handle.remove()
    weight = torch.nn.functional.normalize(direction.weight.detach(), dim=0).cpu()
    return weight, {'first_loss': losses[0], 'last_loss': losses[-1],
                    'mean_loss': float(np.mean(losses)), 'steps': step,
                    'seconds': time.monotonic() - started}


@torch.no_grad()
def score_rows(llm, template, rows, layer, detectors, refusal, examples):
    frame = rows[rows.split.isin(['validation', 'test'])].reset_index(drop=True)
    weights = torch.stack(detectors).to(llm.device).float()
    state = {}
    prefix = prefix_length(llm, template)

    def capture(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        scores = torch.relu(hidden.float() @ weights.T)
        state['scores'] = scores.masked_fill(~state['mask'][:, :, None], -torch.inf).max(1).values.cpu()
        pooled = (hidden.float() * state['mask'][:, :, None]).sum(1) / state['mask'].sum(1)[:, None]
        n = math.sqrt(pooled.shape[1])
        state['activation_sparsity'] = ((n - pooled.abs().sum(1) / pooled.norm(dim=1).clamp_min(1e-12)) / (n - 1)).cpu()

    handle = llm.model.model.layers[layer].register_forward_hook(capture)
    collected, activation_sparsity = [], []
    for start in range(0, len(frame), 32):
        batch = frame.iloc[start:start + 32]
        prompts = [template.render(BASELINE_SYSTEM, q) for q in batch.question]
        tokenized = llm.tokenize_batch(prompts)
        state['mask'] = content_mask(tokenized['attention_mask'], prefix).to(llm.device)
        llm.model.model(**{k: v.to(llm.device) for k, v in tokenized.items()}, use_cache=False)
        collected.append(state['scores'])
        activation_sparsity.extend(state['activation_sparsity'].tolist())
    handle.remove()
    scores = torch.cat(collected).numpy()
    frame['activation_sparsity'] = activation_sparsity

    def refusal_amplitude(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        latent = torch.relu(hidden.float() @ refusal.to(llm.device))
        state['amplitude'] = max(state.get('amplitude', 0.), float(latent[state['mask']].max()))

    handle = llm.model.model.layers[layer].register_forward_hook(refusal_amplitude)
    positive_examples = examples[:len(examples) // 2]
    for start in range(0, len(positive_examples), 16):
        prompts = [template.render(BASELINE_SYSTEM, q, a) for q, a in positive_examples[start:start + 16]]
        batch = llm.tokenize_batch(prompts)
        state['mask'] = content_mask(batch['attention_mask'], prefix).to(llm.device)
        llm.model.model(**{k: v.to(llm.device) for k, v in batch.items()}, use_cache=False)
    handle.remove()
    return frame, scores, state['amplitude']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = config()
    rows = pd.read_csv(CACHE / 'rows.csv', keep_default_na=False)
    concepts = sorted(rows.concept.unique())
    pool = make_pool(cfg['model'], [args.gpu])
    llm = pool.llms[args.gpu]
    for parameter in llm.model.parameters():
        parameter.requires_grad_(False)
    layers = [9] if args.smoke else cfg['layers'][args.gpu::len(cfg['gpus'])]
    for seed in cfg['seeds'][:1] if args.smoke else cfg['seeds']:
        for layer in layers:
            directory = CACHE / ('smoke' if args.smoke else f'seed_{seed}') / f'layer_{layer:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'bundle.pt').exists():
                print(f'[gpu {args.gpu}] seed={seed} layer={layer} cached', flush=True)
                continue
            detectors, training = [], []
            for target in concepts + ['__refusal__']:
                path = directory / f'{target}.pt'
                if path.exists():
                    weight = torch.load(path, weights_only=True)
                else:
                    weight, stats = fit(llm, pool.template, training_examples(rows, target, cfg), layer, seed, cfg, args.smoke)
                    torch.save(weight, path)
                    training.append(dict(seed=seed, layer=layer, target=target, **stats))
                    save_csv(pd.DataFrame(training), directory / 'training.csv')
                    print(f'[gpu {args.gpu}] seed={seed} layer={layer} {target}: {stats}', flush=True)
                if target != '__refusal__':
                    detectors.append(weight)
                else:
                    refusal = weight
                if args.smoke:
                    break
            if args.smoke:
                return
            frame, scores, amplitude = score_rows(llm, pool.template, rows, layer, detectors, refusal,
                                                  training_examples(rows, '__refusal__', cfg))
            assert amplitude >= 0
            thresholds, predictions, metrics = [], [], []
            validation = frame.split.eq('validation').to_numpy()
            for j, target in enumerate(concepts):
                labels = frame.concept.eq(target).to_numpy()
                tau = choose_threshold(labels[validation], scores[validation, j])
                thresholds.append(tau)
                for split in ['validation', 'test']:
                    sub = frame.split.eq(split).to_numpy()
                    y, s = labels[sub], scores[sub, j]
                    pred = s > tau
                    metrics.append(dict(seed=seed, layer=layer, target=target, split=split,
                        auroc=roc_auc_score(y, s), balanced_accuracy=balanced_accuracy_score(y, pred),
                        false_positive_rate=pred[~y].mean(), false_negative_rate=(~pred[y]).mean(),
                        threshold=tau, coordinate_sparsity=hoyer(detectors[j]),
                        concept_sparsity=hoyer([s[frame.loc[sub, 'concept'].eq(c).to_numpy()].mean() for c in concepts])))
                for row, s in zip(frame.to_dict('records'), scores[:, j]):
                    predictions.append(dict(row_id=row['row_id'], concept=row['concept'], split=row['split'],
                        seed=seed, layer=layer, target=target, detector_score=float(s), detector_positive=int(s > tau),
                        activation_sparsity=row['activation_sparsity']))
            save_csv(pd.DataFrame(predictions), directory / 'classification.csv')
            save_csv(pd.DataFrame(metrics), directory / 'classification_metrics.csv')
            torch.save(dict(concepts=concepts, detectors=torch.stack(detectors), refusal=refusal,
                            thresholds=torch.tensor(thresholds), amplitude=amplitude), directory / 'bundle.pt')
            print(f'[gpu {args.gpu}] COMPLETE seed={seed} layer={layer}', flush=True)


if __name__ == '__main__':
    main()
