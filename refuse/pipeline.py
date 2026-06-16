import gc
import shlex
import sys
import time
from pathlib import Path

import pandas as pd
import torch as t
import yaml
from transformers import AutoConfig

from llm import GPUPool, detect_template

from .activations import cached_concept_activations, clean_answer_text
from .baseline import generate_baseline
from .calibration import (
    build_grid,
    calibration_sweep,
    select_optimal_config,
)
from .evaluations import EVALUATIONS
from .intervention import GatedSteering, sample_per_concept
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


JUDGE_AXES = ("refusal", "retention", "fluency")
EPS = 1e-9

VECTOR_METHODS = {
    "lda": cached_lda_vectors,
    "diffed": cached_diffed_vectors,
    "projected": cached_projected_vectors,
}


def _csv_complete(path, col):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    return col in df.columns and df[col].notna().all()


def _judge_complete(path, mode="reasoning"):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    for axis in JUDGE_AXES:
        if mode == "reasoning":
            cols = [f"judge_{axis}_completion"]
        elif mode == "logit":
            cols = [f"judge_{axis}", f"judge_{axis}_p1", f"judge_{axis}_p2"]
        else:
            raise ValueError(f"unknown judge mode {mode!r}")
        for col in cols:
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
    layers="default",
    scales=15,
    scale_window="mid",
    train_frac=1.0,
    test_frac=1.0,
    calibration_n=10,
    calibration_concepts="all",
    evaluations=(),
    hf_token=None,
    plot=True,
    verbose=False,
    judge_model=None,
    judge_gpu_ids=None,
    judge_max_retries=25,
    judge_mode="reasoning",
    batch_size=64,
    judge_batch_size=32,
    trust_remote_code=False,
    result_name=None,
    artifact_cache="main",
    clean_activation_answers=True,
    intervention_start="assistant",
    config_snapshot=None,
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

    paths = Paths(
        root=Path(result_root),
        data_root=Path(data_root),
        result=result_name,
        artifact_cache=artifact_cache,
    )
    log_file = open(paths.pipeline_log, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    argv = " ".join(shlex.quote(a) for a in sys.argv)
    resolved = {
        "model": model_path, "data": str(data_root), "out": str(result_root),
        "result": result_name,
        "method": method, "gpus": list(gpu_ids),
        "train_frac": train_frac, "test_frac": test_frac,
        "calibration_n": calibration_n,
        "calibration_concepts": calibration_concepts,
        "layers": layers,
        "scales": scales,
        "scale_window": scale_window,
        "evaluations": list(evaluations),
        "judge_model": judge_model,
        "judge_gpus": list(judge_gpu_ids) if judge_gpu_ids is not None else None,
        "judge_max_retries": judge_max_retries,
        "judge_mode": judge_mode,
        "batch_size": batch_size,
        "judge_batch_size": judge_batch_size,
        "trust_remote_code": trust_remote_code,
        "artifact_cache": artifact_cache,
        "clean_activation_answers": clean_activation_answers,
        "intervention_start": intervention_start,
    }
    with open(paths.arguments_log, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {argv}\n")
        f.write(f"  resolved: {resolved}\n")
    with open(paths.config, "w") as f:
        yaml.safe_dump(config_snapshot or resolved, f, sort_keys=False)

    df_train = pd.read_csv(paths.train)
    df_test = pd.read_csv(paths.test)
    concepts = df_train["concept"].unique().tolist()
    if train_frac < 1.0:
        n_per_concept = max(1, int(round(len(df_train) * train_frac / len(concepts))))
        df_train = sample_per_concept(df_train, n_per_concept=n_per_concept).reset_index(drop=True)
    if test_frac < 1.0:
        n_per_concept = max(1, int(round(len(df_test) * test_frac / len(concepts))))
        df_test = sample_per_concept(df_test, n_per_concept=n_per_concept).reset_index(drop=True)
    log(f"data: train={len(df_train)} test={len(df_test)} concepts={len(concepts)}")

    num_layers = AutoConfig.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    ).num_hidden_layers
    grid = build_grid(num_layers, layers=layers, scales=scales, scale_window=scale_window)
    distinct_layers = {tuple(p["source_layers"]) for p in grid}
    log(f"num_layers={num_layers}  calibration grid={len(grid)} points "
        f"({len(distinct_layers)} layer configs)")

    evaluations = list(evaluations)
    for name, _ in evaluations:
        if name not in EVALUATIONS:
            raise ValueError(f"unknown evaluation {name!r}; known: {list(EVALUATIONS)}")
    eval_paths = {name: paths.eval_path(name) for name, _ in evaluations}
    eval_judged_paths = {name: paths.eval_judged_path(name) for name, _ in evaluations}
    need_eval = {name: not eval_paths[name].exists() for name, _ in evaluations}
    need_eval_judge = {
        name: judge_model is not None and not _judge_complete(eval_judged_paths[name], judge_mode)
        for name, _ in evaluations
    }
    any_eval_pending = any(need_eval.values())
    any_eval_judge_pending = any(need_eval_judge.values())

    if any_eval_pending and judge_model is None and not paths.calibration_judged.exists():
        raise ValueError(
            "evaluations need a calibration-selected scale; "
            "pass --judge-model or precompute calibration_judged.csv"
        )

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
    need_calibration_judge = (
        judge_model is not None
        and not _judge_complete(paths.calibration_judged, judge_mode)
    )

    # === Block 1: main model for baselines + activations ===
    baseline_train = baseline_test = None
    baseline_acts = refuse_acts = None
    calibration_results = None

    if need_baselines or need_acts:
        log(f"loading main on gpus={list(gpu_ids)}")
        pool = GPUPool.from_model_path(
            model_path,
            gpu_ids,
            template=template,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        log(f"[3a] baseline_train ({hit(paths.baseline_train)})")
        baseline_train = generate_baseline(pool, df_train, paths.baseline_train, template, batch_size=batch_size)
        log(f"[3b] baseline_test ({hit(paths.baseline_test)})")
        baseline_test = generate_baseline(pool, df_test, paths.baseline_test, template, batch_size=batch_size)

        if need_acts:
            answer_cleaner = None
            if clean_activation_answers:
                llm = next(iter(pool.llms.values()))
                answer_cleaner = lambda text: clean_answer_text(
                    llm.tokenizer,
                    text,
                    template.assistant_end_marker,
                )
            log(f"[4a] baseline activations ({hit(paths.baseline_acts)})")
            baseline_acts = cached_concept_activations(
                pool, baseline_train,
                prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
                answer_fn=lambda row: row.baseline_output,
                acts_path=paths.baseline_acts,
                batch_size=batch_size,
                answer_cleaner=answer_cleaner,
            )
            log(f"[4b] refuse activations ({hit(paths.refuse_acts)})")
            refuse_acts = cached_concept_activations(
                pool, baseline_train,
                prompt_fn=lambda row, ans: template.render(refuse_system(row.concept), row.question, ans),
                answer_fn=lambda _row: template.idk_answer,
                acts_path=paths.refuse_acts,
                batch_size=batch_size,
                answer_cleaner=answer_cleaner,
            )
            log(f"[4c] baseline_test activations ({hit(paths.baseline_test_acts)})")
            cached_concept_activations(
                pool, baseline_test,
                prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
                answer_fn=lambda row: row.baseline_output,
                acts_path=paths.baseline_test_acts,
                batch_size=batch_size,
                answer_cleaner=answer_cleaner,
            )

        del pool
        free("main")
    else:
        baseline_train = pd.read_csv(paths.baseline_train)
        baseline_test = pd.read_csv(paths.baseline_test)

    # === Block 2a: main model for calibration generation ===
    if need_calibration:
        log(f"[5] vectors method={method} ({hit(paths.v_detect)})")
        v_detect, v_refuse, thresholds = _ensure_vectors(
            method, concepts, paths, baseline_acts, refuse_acts,
        )
        del baseline_acts, refuse_acts
        baseline_acts = refuse_acts = None
        free("acts")

        log(f"loading main on gpus={list(gpu_ids)} (for calibration generation)")
        pool = GPUPool.from_model_path(
            model_path,
            gpu_ids,
            template=template,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        calibration_desc = (
            f"{calibration_n} random questions"
            if calibration_concepts == "random"
            else f"{calibration_n}/concept"
        )
        log(f"[5.5a] calibration sweep (compute, {len(grid)} grid points × "
            f"{calibration_desc})")
        calibration_results = calibration_sweep(
            pool, baseline_test, grid,
            v_detect, v_refuse, thresholds, BASELINE_SYSTEM, template,
            sample_n=calibration_n,
            concept_mode=calibration_concepts,
            cache_path=paths.calibration,
            batch_size=batch_size,
            intervention_start=intervention_start,
            log=log,
        )
        del pool
        free("main")
    elif paths.calibration.exists():
        calibration_results = pd.read_csv(paths.calibration)

    # === Block 2b: judge for calibration scoring ===
    scored = None
    if need_calibration_judge:
        if calibration_results is None:
            calibration_results = pd.read_csv(paths.calibration)
        log(f"loading judge on gpus={list(judge_gpu_ids)}")
        judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)
        log(f"[5.5b] calibration judge ({hit(paths.calibration_judged)})")
        scored = add_judge_scores(
            judge_pool, calibration_results,
            cache_path=paths.calibration_judged, max_retries=judge_max_retries,
            batch_size=judge_batch_size, mode=judge_mode,
        )
        del judge_pool
        free("judge")
    elif paths.calibration_judged.exists():
        scored = pd.read_csv(paths.calibration_judged)

    # === Optimal config: the (layer, scale) the calibration sweep found best ===
    optimal_layers = None
    scale = None
    if any_eval_pending:
        if scored is None:
            raise ValueError(
                "evaluations need the calibration optimum but calibration_judged.csv is "
                "unavailable (pass --judge-model to compute it)"
            )
        optimal_layers, scale = select_optimal_config(scored)
        log(f"  optimal config: layers={optimal_layers} scale={scale}")

    # === Block 3: main model for all pending evaluations (at the calibration optimum) ===
    if any_eval_pending:
        v_detect, v_refuse, thresholds = _ensure_vectors(
            method, concepts, paths, baseline_acts, refuse_acts,
        )
        steering = GatedSteering(optimal_layers, optimal_layers, v_detect, v_refuse, thresholds)
        result_metadata = {"source_layer": optimal_layers, "target_layer": optimal_layers}

        log(f"loading main on gpus={list(gpu_ids)} (for evaluations)")
        pool = GPUPool.from_model_path(
            model_path,
            gpu_ids,
            template=template,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        for name, kwargs in evaluations:
            if not need_eval[name]:
                log(f"[6:{name}] cached")
                continue
            log(f"[6:{name}] running {kwargs}")
            df = EVALUATIONS[name](
                pool, baseline_test, steering, scale,
                system_prompt=BASELINE_SYSTEM, template=template,
                result_metadata=result_metadata,
                batch_size=batch_size,
                intervention_start=intervention_start,
                **kwargs,
            )
            df.to_csv(eval_paths[name], index=False)

        del pool
        free("main")

    # === Block 4: judge for all pending evaluation scoring ===
    if any_eval_judge_pending:
        log(f"loading judge on gpus={list(judge_gpu_ids)} (for evaluation scoring)")
        judge_pool = GPUPool.from_model_path(judge_model, judge_gpu_ids, template=judge_template, hf_token=hf_token)

        for name, _ in evaluations:
            if not need_eval_judge[name]:
                log(f"[7:{name}] cached")
                continue
            log(f"[7:{name}] judging ({hit(eval_judged_paths[name])})")
            results = pd.read_csv(eval_paths[name])
            add_judge_scores(
                judge_pool, results,
                cache_path=eval_judged_paths[name], max_retries=judge_max_retries,
                batch_size=judge_batch_size, mode=judge_mode,
            )

        del judge_pool
        free("judge")

    if plot:
        log(f"[9] plots → {paths.result_root / 'plots'}")
        make_plots(paths.result_root)

    log(f"done in {time.perf_counter() - t0:.1f}s")
