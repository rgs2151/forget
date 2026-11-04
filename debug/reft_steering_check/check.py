"""Inspect real generations and measured hook effects; no judge or retraining."""

from pathlib import Path

import pandas as pd
import torch

from forget.refuse.prompts import BASELINE_SYSTEM
from parking.refusal_mechanisms.core import CACHE as SOURCE, RecordedGate, clean_output, config, content_mask, make_pool, prefix_length


UNIT = Path(__file__).resolve().parent


def inspect(llm, template, row, bundle, factor):
    llm.reset_all()
    prompt = template.render(BASELINE_SYSTEM, row.question)
    mask = content_mask(llm.tokenize_batch([prompt])['attention_mask'], prefix_length(llm, template)).to(llm.device)
    j = bundle['concepts'].index(row.concept)
    gate = RecordedGate(bundle['detectors'][j].to(llm.device), bundle['refusal'].to(llm.device),
                        bundle['thresholds'][j].to(llm.device), factor * bundle['amplitude'], mask)
    effects, state = [], {}

    def operation(hidden):
        delta = gate(hidden)
        state['before'] = hidden.detach().clone()
        state['expected'] = hidden + delta
        return delta

    def observed(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = (hidden - state.pop('before')).float().norm(dim=-1)
        error = float((hidden - state.pop('expected')).abs().max())
        effects.append(dict(tokens=delta.numel(), changed=int((delta > 0).sum()),
                            max_delta=float(delta.max()), mean_delta=float(delta.mean()), error=error))

    def next_token(_module, _args, logits):
        if 'p_I' in state:
            return
        probabilities = logits[0, -1].float().softmax(-1)
        state['p_I'] = float(probabilities[llm.tokenize('I')[0]])
        state['top_token'] = llm.tokenizer.decode([int(probabilities.argmax())])

    llm.set_steering_op(6, operation)
    observer = llm.model.model.layers[6].register_forward_hook(observed)
    head = llm.model.lm_head.register_forward_hook(next_token)
    output = llm.batch_generate([prompt], max_new_tokens=48, do_sample=False, temperature=1., intervention_start='prefill')[0]
    observer.remove()
    head.remove()
    llm.reset_all()
    return dict(row_id=row.row_id, split=row.split, concept=row.concept, question=row.question,
        factor=factor, physical_strength=factor * bundle['amplitude'],
        output=clean_output(llm, template, output), p_first_I=state['p_I'], first_token=state['top_token'],
        changed_tokens=sum(x['changed'] for x in effects),
        total_tokens=sum(x['tokens'] for x in effects), max_actual_delta=max(x['max_delta'] for x in effects),
        max_hook_error=max(x['error'] for x in effects), **gate.summary()[0])


def main():
    torch.set_num_threads(4)
    cfg = config()
    rows = pd.read_csv(SOURCE / 'rows.csv', keep_default_na=False)
    selected = rows[rows.concept.isin(['bacteria', 'cats']) & rows.split.isin(['train', 'test'])].groupby(
        ['split', 'concept'], group_keys=False).head(1).reset_index(drop=True)
    bundle = torch.load(SOURCE / 'seed_42/layer_06/bundle.pt', weights_only=True)
    pool = make_pool(cfg['model'], [0, 1])

    def worker(llm, shard):
        result = []
        for row in shard.itertuples():
            for factor in [0., 2., 5.]:
                record = inspect(llm, pool.template, row, bundle, factor)
                result.append(record)
                print(f"{row.split} {row.concept} factor={factor}: {record['output']}", flush=True)
        return result

    records = sum(pool.map(worker, [selected.iloc[::2], selected.iloc[1::2]]), [])
    output = UNIT / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / 'generations.csv', index=False)
    print('DONE: 12 generations; same gate, vectors, and layer; no judge calls', flush=True)


if __name__ == '__main__':
    main()
