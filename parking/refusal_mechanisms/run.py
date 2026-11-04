"""Isolated two-GPU experiment; invoke as a module from the code repository."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch

from forget.refuse.prompts import BASELINE_SYSTEM
from .axbench_judge import judge_frame
from .core import CACHE, ROOT, UNIT, RecordedGate, clean_output, config, content_mask, evaluation_jobs, make_pool, prefix_length, save_csv, split_data


VAULT = ROOT / 'parking/model_matrix/cache'
KEY = ['question', 'baseline_output', 'model_output']


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def vault_inventory():
    return {str(p.relative_to(VAULT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in VAULT.rglob('*') if p.is_file()}


def release(pool):
    pool.llms.clear()
    gc.collect()
    torch.cuda.empty_cache()


def prepare(cfg):
    path = CACHE / 'rows.csv'
    if path.exists():
        return
    source = VAULT / 'inhouse'
    rows = split_data(pd.read_csv(source / 'train.csv'), pd.read_csv(source / 'test.csv'), cfg)
    assert (rows.split == 'train').sum() == 100
    assert (rows.split == 'test').sum() == 200
    pool = make_pool(cfg['model'], cfg['gpus'])
    shards = [rows.iloc[i::2] for i in range(2)]

    def generate(llm, shard):
        out = []
        for start in range(0, len(shard), cfg['generation_batch_size']):
            frame = shard.iloc[start:start + cfg['generation_batch_size']].copy()
            prompts = [pool.template.render(BASELINE_SYSTEM, q) for q in frame.question]
            llm.reset_all()
            raw = llm.batch_generate(prompts, max_new_tokens=cfg['max_new_tokens'], do_sample=False, temperature=1.)
            frame['baseline_output'] = [clean_output(llm, pool.template, x) for x in raw]
            out.append(frame)
        return pd.concat(out)

    rows = pd.concat(pool.map(generate, shards)).sort_index()
    save_csv(rows, path)
    release(pool)
    log(f'Prepared {len(rows)} unique train/validation/test questions and fresh baselines')


def workers(module, cfg, extra=None):
    processes, streams = [], []
    for gpu in cfg['gpus']:
        name = 'training' if module.endswith('.train') else (extra or ['generation'])[0].lstrip('-')
        stream = (CACHE / f'{name}_gpu{gpu}.log').open('a')
        command = [sys.executable, '-u', '-m', module, '--gpu', str(gpu)] + (extra or [])
        processes.append(subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        streams.append(stream)
    while any(p.poll() is None for p in processes):
        failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
        if failed:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait()
            raise RuntimeError(f'{module} failed; see GPU logs in {CACHE}')
        time.sleep(5)
    for stream in streams:
        stream.close()
    assert all(p.returncode == 0 for p in processes)


def generate_rows(llm, template, jobs, bundle, cfg):
    records = []
    weights = bundle['detectors'].to(llm.device)
    thresholds = bundle['thresholds'].to(llm.device)
    concepts = bundle['concepts']
    direction = bundle['refusal'].to(llm.device)
    prefix = prefix_length(llm, template)
    for start in range(0, len(jobs), cfg['generation_batch_size']):
        frame = jobs.iloc[start:start + cfg['generation_batch_size']]
        prompts = [template.render(BASELINE_SYSTEM, q) for q in frame.question]
        indices = torch.tensor([concepts.index(c) for c in frame.target], device=llm.device)
        strength = torch.tensor(frame.scale.to_numpy(), device=llm.device) * bundle['amplitude']
        mask = content_mask(llm.tokenize_batch(prompts)['attention_mask'], prefix).to(llm.device)
        op = RecordedGate(weights[indices], direction, thresholds[indices], strength, mask)
        eos = llm.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]

        def track_finished(_module, _args, kwargs):
            tokens = kwargs.get('input_ids')
            if tokens is not None and tokens.shape[1] == 1:
                op.alive &= ~torch.isin(tokens[:, 0], torch.tensor(eos, device=tokens.device))

        llm.reset_all()
        llm.set_steering_op(int(frame.layer.iloc[0]), op)
        handle = llm.model.register_forward_pre_hook(track_finished, with_kwargs=True)
        raw = llm.batch_generate(prompts, max_new_tokens=cfg['max_new_tokens'], do_sample=False,
                                 temperature=1., intervention_start='prefill')
        handle.remove()
        logs = op.summary()
        for record, output, gate in zip(frame.to_dict('records'), raw, logs):
            records.append(dict(record, model_output=clean_output(llm, template, output), **gate))
        llm.reset_all()
    return pd.DataFrame(records)


def generation_worker(gpu, mode, cfg):
    torch.set_num_threads(4)
    rows = pd.read_csv(CACHE / 'rows.csv', keep_default_na=False)
    pool = make_pool(cfg['model'], [gpu])
    llm = pool.llms[gpu]
    for seed in cfg['seeds']:
        if mode == 'calibration':
            layers = cfg['layers'][gpu::2]
        else:
            selected = json.loads((CACHE / 'selected.json').read_text())[str(seed)]
            layers = [selected['layer']]
        for layer in layers:
            directory = CACHE / f'seed_{seed}' / f'layer_{layer:02d}'
            path = directory / (f'evaluation_gpu{gpu}.csv' if mode == 'evaluation' else 'calibration.csv')
            if path.exists():
                continue
            bundle = torch.load(directory / 'bundle.pt', weights_only=True)
            if mode == 'calibration':
                frame = rows[rows.split == 'validation'].groupby('concept', group_keys=False).head(cfg['calibration_per_concept']).copy()
                jobs = pd.concat([frame.assign(target=frame.concept, scale=s) for s in cfg['factors']], ignore_index=True)
            else:
                frame = rows[rows.split == 'test']
                jobs = evaluation_jobs(frame, bundle['concepts']).assign(scale=selected['scale'])
                jobs = jobs.iloc[gpu::2].reset_index(drop=True)
            jobs['seed'], jobs['layer'] = seed, layer
            jobs['source_layer'] = str([layer])
            jobs['target_layer'] = str([layer])
            jobs['label'] = 'intervention'
            result = generate_rows(llm, pool.template, jobs, bundle, cfg)
            save_csv(result, path)
            log(f'GPU {gpu}: {mode} seed={seed} layer={layer}, {len(result)} rows')


def judge(frame, stage, cfg):
    path = CACHE / f'{stage}_judged.csv'
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    unique = frame[KEY].drop_duplicates().reset_index(drop=True)
    log(f'{stage}: {len(frame)} rows, {len(unique)} distinct question/reference/answer triples to judge')
    unique = judge_frame(unique, cfg)
    save_csv(unique, CACHE / f'{stage}_unique_judged.csv')
    scored = frame.merge(unique, on=KEY, how='left', validate='many_to_one')
    assert scored['axbench_concept_relevance'].notna().all()
    save_csv(scored, path)
    return scored


def judge_smoke(cfg):
    path = CACHE / 'judge_smoke.csv'
    if path.exists():
        return
    question = 'Which big cat species is known for spots?'
    frame = pd.DataFrame(dict(question=[question] * 4, model_output=[
        'The leopard.', "I don't know.",
        "I don't know which big cat species is known for spots.", 'spot spot blue the the the']))
    frame = judge_frame(frame, cfg)
    save_csv(frame, path)
    assert frame[[f'axbench_{axis}_parsed' for axis in ['concept_relevance', 'instruction_relevance', 'fluency']]].all().all()
    log('AxBench API judge sanity examples scored successfully')


def finalize():
    from .plot import make_figures
    make_figures()
    before = json.loads((CACHE / 'protected_inventory.json').read_text())
    assert vault_inventory() == before, 'Protected result inventory changed during the run'
    selected = json.loads((CACHE / 'selected.json').read_text())
    n = len(pd.read_csv(CACHE / 'evaluation_judged.csv'))
    (CACHE / 'completed.json').write_text(json.dumps(dict(completed=time.time(), protected_results_unchanged=True,
                                                        evaluation_rows=n, selected=selected), indent=2))
    log('COMPLETE: plots written; protected result file sizes and mtimes unchanged')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    cfg = config()
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(4)
    if args.gpu is not None:
        generation_worker(args.gpu, 'calibration' if args.calibration else 'evaluation', cfg)
        return
    if args.plots_only:
        finalize()
        return
    fingerprint = hashlib.sha256((UNIT / 'config.json').read_bytes()).hexdigest()
    metadata = CACHE / 'run_metadata.json'
    if metadata.exists():
        assert json.loads(metadata.read_text())['config_sha256'] == fingerprint
    else:
        metadata.write_text(json.dumps(dict(config_sha256=fingerprint, config=cfg, started=time.time()), indent=2))
        (CACHE / 'protected_inventory.json').write_text(json.dumps(vault_inventory()))
    judge_smoke(cfg)
    prepare(cfg)
    if args.prepare_only:
        return
    log('Training independent layer/seed jobs concurrently on GPUs 0 and 1')
    workers('parking.refusal_mechanisms.train', cfg)
    log('Generating layer/strength calibration on both GPUs')
    workers('parking.refusal_mechanisms.run', cfg, ['--calibration'])
    calibration = pd.concat([pd.read_csv(CACHE / f'seed_{s}' / f'layer_{l:02d}' / 'calibration.csv', keep_default_na=False)
                             for s in cfg['seeds'] for l in cfg['layers']], ignore_index=True)
    assert len(calibration) == 640
    calibration = judge(calibration, 'calibration', cfg)
    selected = {}
    for seed, frame in calibration.groupby('seed'):
        settings = frame.groupby(['layer', 'scale']).axbench_aggregate.mean().reset_index()
        best = settings.sort_values(['axbench_aggregate', 'scale', 'layer'], ascending=[False, True, True]).iloc[0]
        selected[str(seed)] = dict(layer=int(best.layer), scale=float(best.scale))
    (CACHE / 'selected.json').write_text(json.dumps(selected, indent=2))
    log(f'Selected validation operating points: {selected}')
    workers('parking.refusal_mechanisms.run', cfg, ['--evaluation'])
    evaluation = []
    for seed in cfg['seeds']:
        directory = CACHE / f'seed_{seed}' / f'layer_{selected[str(seed)]["layer"]:02d}'
        evaluation.extend(pd.read_csv(directory / f'evaluation_gpu{g}.csv', keep_default_na=False) for g in cfg['gpus'])
    rows = pd.read_csv(CACHE / 'rows.csv', keep_default_na=False)
    baseline = evaluation_jobs(rows[rows.split == 'test'], sorted(rows.concept.unique())).assign(
        model_output=lambda d: d.baseline_output, label='baseline', seed=-1, layer=-1, scale=0.)
    evaluation = pd.concat(evaluation + [baseline], ignore_index=True).fillna('')
    assert len(evaluation) == 400
    judge(evaluation, 'evaluation', cfg)
    finalize()


if __name__ == '__main__':
    main()
