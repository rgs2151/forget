import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch as t


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

from llm import GPUPool, TEMPLATES, detect_template

from .activations import cached_concept_activations
from .baseline import generate_baseline
from .calibration import calibration_generate, calibration_score_select
from .intervention import GatedSteering, make_generation_jobs, run_jobs, sample_per_concept
from .paths import Paths
from .prompts import BASELINE_SYSTEM, refuse_system
from .vectors import cached_diffed_vectors, cached_lda_vectors, cached_projected_vectors

from judge import add_judge_scores
from plot import make_all as make_plots


CALIBRATION_SCALES = [round(i * 0.5, 1) for i in range(1, 31)]

VECTOR_METHODS = {
    "lda": cached_lda_vectors,
    "diffed": cached_diffed_vectors,
    "projected": cached_projected_vectors,
}


def default_intervention_layers(num_layers):
    fractions = [15 / 32, 18 / 32, 21 / 32, 24 / 32]
    return sorted({round(f * num_layers) for f in fractions})


def run(
    model_path,
    data_root,
    result_root,
    *,
    template=None,
    method="lda",
    gpu_ids=(0,),
    intervention_layers=None,
    calibration_scales=CALIBRATION_SCALES,
    calibration_frac=0.10,
    validation_frac=0.10,
    hf_token=None,
    plot=True,
    verbose=False,
    judge_model=None,
    judge_gpu_ids=None,
    judge_template=None,
    judge_max_retries=25,
):
    def log(msg):
        if verbose:
            print(f"[refuse] {msg}", flush=True)

    def hit(path):
        return "cached" if path.exists() else "compute"

    def free(name):
        gc.collect()
        t.cuda.empty_cache()
        log(f"freed {name}")

    t0 = time.perf_counter()
    if template is None:
        template = detect_template(model_path)
    template_name = next((k for k, v in TEMPLATES.items() if v is template), "custom")
    log(f"template={template_name} for {model_path!r}")

    if judge_gpu_ids is None:
        judge_gpu_ids = gpu_ids

    paths = Paths(root=Path(result_root), data_root=Path(data_root))
    log_file = open(paths.root / "pipeline.log", "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    df_train = pd.read_csv(paths.train)
    df_test = pd.read_csv(paths.test)
    concepts = df_train["concept"].unique().tolist()
    log(f"data: train={len(df_train)} test={len(df_test)} concepts={len(concepts)}")

    log(f"loading main on gpus={list(gpu_ids)}")
    pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)
    num_layers = len(pool.llms[pool.gpu_ids[0]].model.model.layers)
    if intervention_layers is None:
        intervention_layers = default_intervention_layers(num_layers)
    log(f"num_layers={num_layers}  intervention_layers={intervention_layers}")

    log(f"[3a] baseline_train ({hit(paths.baseline_train)})")
    baseline_train = generate_baseline(pool, df_train, paths.baseline_train, template)
    log(f"[3b] baseline_test ({hit(paths.baseline_test)})")
    baseline_test = generate_baseline(pool, df_test, paths.baseline_test, template)

    log(f"[4a] baseline activations ({hit(paths.baseline_acts)})")
    baseline_acts, baseline_masks = cached_concept_activations(
        pool, baseline_train,
        prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
        answer_fn=lambda row: row.baseline_output,
        acts_path=paths.baseline_acts, masks_path=paths.baseline_masks,
    )
    log(f"[4b] refuse activations ({hit(paths.refuse_acts)})")
    refuse_acts, refuse_masks = cached_concept_activations(
        pool, baseline_train,
        prompt_fn=lambda row, ans: template.render(refuse_system(row.concept), row.question, ans),
        answer_fn=lambda _row: template.idk_answer,
        acts_path=paths.refuse_acts, masks_path=paths.refuse_masks,
    )
    log(f"[4c] baseline_test activations ({hit(paths.baseline_test_acts)})")
    cached_concept_activations(
        pool, baseline_test,
        prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
        answer_fn=lambda row: row.baseline_output,
        acts_path=paths.baseline_test_acts, masks_path=paths.baseline_test_masks,
    )

    del pool
    free("main")

    log(f"[5] vectors method={method} ({hit(paths.v_detect)})")
    out = VECTOR_METHODS[method](
        baseline_acts, refuse_acts, concepts, paths,
        know_masks=baseline_masks, forget_masks=refuse_masks,
    )
    if method == "lda":
        v_detect, v_refuse, thresholds = out
    else:
        v_detect, v_refuse = out
        thresholds = {c: t.zeros(v_detect[c].shape[0]) for c in concepts}

    steering = GatedSteering(intervention_layers, intervention_layers, v_detect, v_refuse, thresholds)
    result_metadata = {"source_layer": intervention_layers, "target_layer": intervention_layers}

    log(f"loading main on gpus={list(gpu_ids)} (for calibration generation)")
    pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)
    log(f"[5.5a] calibration generate ({hit(paths.calibration)}, {int(calibration_frac*100)}% of test × {len(calibration_scales)} scales)")
    calibration_results = calibration_generate(
        pool, baseline_test, calibration_scales, steering, BASELINE_SYSTEM, template,
        sample_frac=calibration_frac,
        cache_path=paths.calibration,
        result_metadata=result_metadata,
    )
    del pool
    free("main")

    log(f"[5.5b] calibration score → select scale")
    log(f"loading judge on gpus={list(judge_gpu_ids)}")
    judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)
    refusal_fn = lambda df: add_judge_scores(
        judge_pool, df, cache_path=paths.calibration_judged, max_retries=judge_max_retries,
    )
    scale = calibration_score_select(calibration_results, refusal_fn)
    del judge_pool
    free("judge")
    log(f"  selected scale={scale}")

    log(f"loading main on gpus={list(gpu_ids)} (for steered generation)")
    pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)
    log(f"[6+7] steered generation ({hit(paths.results)})")
    if paths.results.exists():
        results = pd.read_csv(paths.results)
    else:
        n_per_concept = max(1, int(round(len(baseline_test) * validation_frac / len(concepts))))
        df_gen = sample_per_concept(baseline_test, n_per_concept=n_per_concept).reset_index(drop=True)
        prompts = [template.render(BASELINE_SYSTEM, row.question)
                   for row in df_gen.itertuples(index=False)]
        jobs = make_generation_jobs(df_gen, prompts, targets=concepts, scales=[scale])
        log(f"  {len(jobs)} jobs across {len(pool)} gpus")
        results = run_jobs(
            pool, jobs, steering,
            generation_kwargs={"max_new_tokens": 64, "do_sample": False, "temperature": 1.0},
            batch_size=128,
            trim_fn=template.trim_to_last_assistant,
            result_metadata=result_metadata,
        )
        results.to_csv(paths.results, index=False)
    del pool
    free("main")

    if judge_model is None:
        log("[8] judge skipped (no --judge-model provided); returning unscored results")
        if plot:
            log(f"[9] plots → {paths.root / 'plots'}")
            make_plots(paths.root)
        log(f"done in {time.perf_counter() - t0:.1f}s")
        return results

    log(f"loading judge on gpus={list(judge_gpu_ids)} (for final scoring)")
    judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)
    log(f"[8] judge ({hit(paths.judged)})")
    scored = add_judge_scores(
        judge_pool, results, cache_path=paths.judged, max_retries=judge_max_retries,
    )
    del judge_pool
    free("judge")

    if plot:
        log(f"[9] plots → {paths.root / 'plots'}")
        make_plots(paths.root)

    log(f"done in {time.perf_counter() - t0:.1f}s")
    return scored
