import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch as t
from transformers import AutoConfig

from llm import GPUPool, detect_template

from .activations import cached_concept_activations
from .baseline import generate_baseline
from .calibration import calibration_generate, select_refusal_scale
from .intervention import GatedSteering, make_generation_jobs, run_jobs, sample_per_concept
from .paths import Paths
from .prompts import BASELINE_SYSTEM, refuse_system
from .vectors import cached_diffed_vectors, cached_lda_vectors, cached_projected_vectors

from judge import add_judge_scores
from plot import make_all as make_plots


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


CALIBRATION_SCALES = [round(i * 0.5, 1) for i in range(1, 31)]
JUDGE_AXES = ("refusal", "retention", "fluency")
EPS = 1e-9

VECTOR_METHODS = {
    "lda": cached_lda_vectors,
    "diffed": cached_diffed_vectors,
    "projected": cached_projected_vectors,
}


def default_intervention_layers(num_layers):
    fractions = [15 / 32, 18 / 32, 21 / 32, 24 / 32]
    return sorted({round(f * num_layers) for f in fractions})


def _csv_complete(path, col):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    return col in df.columns and df[col].notna().all()


def _judge_complete(path):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    for axis in JUDGE_AXES:
        col = f"judge_{axis}_completion"
        if col not in df.columns or df[col].isna().any():
            return False
    return True


def _ensure_vectors(method, concepts, paths, baseline_acts=None, refuse_acts=None, device=None):
    """Load vectors from cache or compute them. Loads acts from disk if needed."""
    vec_cached = paths.v_detect.exists() and paths.v_refuse.exists() and (
        method != "lda" or paths.thresholds.exists()
    )
    if not vec_cached:
        if baseline_acts is None:
            baseline_acts = t.load(paths.baseline_acts)
        if refuse_acts is None:
            refuse_acts = t.load(paths.refuse_acts)

    out = VECTOR_METHODS[method](baseline_acts, refuse_acts, concepts, paths, device=device)
    if method == "lda":
        v_detect, v_refuse, thresholds = out
    else:
        v_detect, v_refuse = out
        thresholds = {c: t.zeros(v_detect[c].shape[0]) for c in concepts}
    return v_detect, v_refuse, thresholds


def run(
    model_path,
    data_root,
    result_root,
    *,
    method="lda",
    gpu_ids=(0,),
    intervention_layers=None,
    calibration_scales=CALIBRATION_SCALES,
    train_frac=1.0,
    calibration_frac=0.10,
    validation_frac=0.10,
    hf_token=None,
    plot=True,
    verbose=False,
    judge_model=None,
    judge_gpu_ids=None,
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
    template = detect_template(model_path)
    judge_template = detect_template(judge_model) if judge_model is not None else None
    log(f"model={model_path!r}")

    if judge_gpu_ids is None:
        judge_gpu_ids = gpu_ids

    paths = Paths(root=Path(result_root), data_root=Path(data_root))
    log_file = open(paths.root / "pipeline.log", "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    df_train = pd.read_csv(paths.train)
    df_test = pd.read_csv(paths.test)
    concepts = df_train["concept"].unique().tolist()
    if train_frac < 1.0:
        n_per_concept = max(1, int(round(len(df_train) * train_frac / len(concepts))))
        df_train = sample_per_concept(df_train, n_per_concept=n_per_concept).reset_index(drop=True)
    log(f"data: train={len(df_train)} test={len(df_test)} concepts={len(concepts)}")

    num_layers = AutoConfig.from_pretrained(model_path, token=hf_token).num_hidden_layers
    if intervention_layers is None:
        intervention_layers = default_intervention_layers(num_layers)
    log(f"num_layers={num_layers}  intervention_layers={intervention_layers}")
    result_metadata = {"source_layer": intervention_layers, "target_layer": intervention_layers}

    need_baselines = not (
        _csv_complete(paths.baseline_train, "baseline_output")
        and _csv_complete(paths.baseline_test, "baseline_output")
    )
    need_acts = not (
        paths.baseline_acts.exists()
        and paths.refuse_acts.exists()
        and paths.baseline_test_acts.exists()
    )
    need_calibration = not paths.calibration.exists()
    need_results = not paths.results.exists()
    need_calibration_judge = (
        (judge_model is not None) and need_results and not _judge_complete(paths.calibration_judged)
    )
    need_final_judge = (judge_model is not None) and not _judge_complete(paths.judged)

    # === Block 1: main model for baselines + activations ===
    baseline_train = baseline_test = None
    baseline_acts = refuse_acts = None
    calibration_results = None
    steering = None

    if need_baselines or need_acts:
        log(f"loading main on gpus={list(gpu_ids)}")
        pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)

        log(f"[3a] baseline_train ({hit(paths.baseline_train)})")
        baseline_train = generate_baseline(pool, df_train, paths.baseline_train, template)
        log(f"[3b] baseline_test ({hit(paths.baseline_test)})")
        baseline_test = generate_baseline(pool, df_test, paths.baseline_test, template)

        if need_acts:
            log(f"[4a] baseline activations ({hit(paths.baseline_acts)})")
            baseline_acts = cached_concept_activations(
                pool, baseline_train,
                prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
                answer_fn=lambda row: row.baseline_output,
                acts_path=paths.baseline_acts,
            )
            log(f"[4b] refuse activations ({hit(paths.refuse_acts)})")
            refuse_acts = cached_concept_activations(
                pool, baseline_train,
                prompt_fn=lambda row, ans: template.render(refuse_system(row.concept), row.question, ans),
                answer_fn=lambda _row: template.idk_answer,
                acts_path=paths.refuse_acts,
            )
            log(f"[4c] baseline_test activations ({hit(paths.baseline_test_acts)})")
            cached_concept_activations(
                pool, baseline_test,
                prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
                answer_fn=lambda row: row.baseline_output,
                acts_path=paths.baseline_test_acts,
            )

        del pool
        free("main")
    else:
        baseline_train = pd.read_csv(paths.baseline_train)
        baseline_test = pd.read_csv(paths.baseline_test)

    # === Block 2: main model for calibration generation (needs vectors → free GPU first) ===
    if need_calibration:
        log(f"[5] vectors method={method} ({hit(paths.v_detect)})")
        v_detect, v_refuse, thresholds = _ensure_vectors(
            method, concepts, paths, baseline_acts, refuse_acts,
        )
        del baseline_acts, refuse_acts
        baseline_acts = refuse_acts = None
        free("acts")
        steering = GatedSteering(intervention_layers, intervention_layers, v_detect, v_refuse, thresholds)

        log(f"loading main on gpus={list(gpu_ids)} (for calibration generation)")
        pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)
        log(f"[5.5a] calibration generate (compute, "
            f"{int(calibration_frac*100)}% of test × {len(calibration_scales)} scales)")
        calibration_results = calibration_generate(
            pool, baseline_test, calibration_scales, steering, BASELINE_SYSTEM, template,
            sample_frac=calibration_frac,
            cache_path=paths.calibration,
            result_metadata=result_metadata,
        )
        del pool
        free("main")
    elif paths.calibration.exists():
        calibration_results = pd.read_csv(paths.calibration)

    # === Block 2: judge for calibration scoring → select scale ===
    scale = None
    if need_results:
        if calibration_results is None:
            calibration_results = pd.read_csv(paths.calibration)

        if need_calibration_judge:
            log(f"loading judge on gpus={list(judge_gpu_ids)}")
            judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)
            log(f"[5.5b] calibration judge ({hit(paths.calibration_judged)})")
            scored = add_judge_scores(
                judge_pool, calibration_results,
                cache_path=paths.calibration_judged, max_retries=judge_max_retries,
            )
            del judge_pool
            free("judge")
        else:
            scored = pd.read_csv(paths.calibration_judged)

        scored = scored.assign(
            judge_harmonic=2 * scored["judge_refusal"] * scored["judge_fluency"]
            / (scored["judge_refusal"] + scored["judge_fluency"] + EPS)
        )
        scale = select_refusal_scale(scored, score_col="judge_harmonic")
        log(f"  selected scale={scale}")

    # === Block 3: main model for steered generation ===
    if need_results:
        v_detect, v_refuse, thresholds = _ensure_vectors(
            method, concepts, paths, baseline_acts, refuse_acts,
        )
        steering = GatedSteering(intervention_layers, intervention_layers, v_detect, v_refuse, thresholds)

        log(f"loading main on gpus={list(gpu_ids)} (for steered generation)")
        pool = GPUPool.from_model_path(model_path, gpu_ids, template=template, hf_token=hf_token)
        log(f"[6+7] steered generation (compute)")
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
    else:
        results = pd.read_csv(paths.results)

    # === Block 4: judge for final scoring ===
    if need_final_judge:
        log(f"loading judge on gpus={list(judge_gpu_ids)} (for final scoring)")
        judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)
        log(f"[8] judge ({hit(paths.judged)})")
        scored = add_judge_scores(
            judge_pool, results, cache_path=paths.judged, max_retries=judge_max_retries,
        )
        del judge_pool
        free("judge")
    elif judge_model is not None:
        scored = pd.read_csv(paths.judged)
    else:
        scored = results

    if plot:
        log(f"[9] plots → {paths.root / 'plots'}")
        make_plots(paths.root)

    log(f"done in {time.perf_counter() - t0:.1f}s")
    return scored
