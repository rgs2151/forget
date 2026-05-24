import time
from pathlib import Path

import pandas as pd
import torch as t

from .activations import cached_concept_activations
from .baseline import generate_baseline
from .calibration import select_scale
from .chat_templates import TEMPLATES, detect_template
from .gpu import GPUPool
from .intervention import GatedSteering, load_or_empty_results, make_generation_jobs, sample_per_concept
from .model import load_llm
from .paths import Paths
from .plots import make_all as make_plots
from .prompts import BASELINE_SYSTEM, refuse_system
from .scoring import add_acceptability_column, add_refusal_column, add_retention_column
from .vectors import cached_diffed_vectors, cached_lda_vectors, cached_projected_vectors


CALIBRATION_SCALES = [1, 2, 3, 5, 8, 13, 21]

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
    n_per_concept=25,
    hf_token=None,
    plot=True,
    verbose=False,
):
    def log(msg):
        if verbose:
            print(f"[refuse] {msg}", flush=True)

    def hit(path):
        return "cached" if path.exists() else "compute"

    t0 = time.perf_counter()
    if template is None:
        template = detect_template(model_path)
    template_name = next((k for k, v in TEMPLATES.items() if v is template), "custom")
    log(f"template={template_name} for {model_path!r}")

    paths = Paths(root=Path(result_root), data_root=Path(data_root))
    pool = GPUPool(
        lambda gid: load_llm(model_path, gid, template, hf_token),
        gpu_ids,
        template=template,
    )
    sample_llm = pool.llms[pool.gpu_ids[0]]
    num_layers = len(sample_llm.model.model.layers)
    log(f"model loaded on gpus={list(gpu_ids)}  num_layers={num_layers}")

    if intervention_layers is None:
        intervention_layers = default_intervention_layers(num_layers)
    log(f"intervention_layers={intervention_layers}")

    df_train = pd.read_csv(paths.train)
    df_test = pd.read_csv(paths.test)
    concepts = df_train["concept"].unique().tolist()
    log(f"data: train={len(df_train)} test={len(df_test)} concepts={len(concepts)}")

    log(f"[3a] baseline_train ({hit(paths.baseline_train)})")
    baseline_train = generate_baseline(pool, df_train, paths.baseline_train, template)
    log(f"[3b] baseline_test ({hit(paths.baseline_test)})")
    baseline_test = generate_baseline(pool, df_test, paths.baseline_test, template)

    log(f"[4a] baseline activations ({hit(paths.baseline_acts)})")
    baseline_acts, baseline_masks = cached_concept_activations(
        pool,
        baseline_train,
        prompt_fn=lambda row, ans: template.render(BASELINE_SYSTEM, row.question, ans),
        answer_fn=lambda row: row.baseline_output,
        acts_path=paths.baseline_acts,
        masks_path=paths.baseline_masks,
    )
    log(f"[4b] refuse activations ({hit(paths.refuse_acts)})")
    refuse_acts, refuse_masks = cached_concept_activations(
        pool,
        baseline_train,
        prompt_fn=lambda row, ans: template.render(refuse_system(row.concept), row.question, ans),
        answer_fn=lambda _row: template.idk_answer,
        acts_path=paths.refuse_acts,
        masks_path=paths.refuse_masks,
    )

    log(f"[5] vectors method={method} ({hit(paths.v_detect)})")
    cached_method = VECTOR_METHODS[method]
    out = cached_method(
        baseline_acts, refuse_acts, concepts, paths,
        know_masks=baseline_masks, forget_masks=refuse_masks,
    )
    if method == "lda":
        v_detect, v_refuse_per, v_refuse, thresholds = out
    else:
        v_detect, v_refuse_per, v_refuse = out
        thresholds = {c: t.zeros(v_detect[c].shape[0]) for c in concepts}

    steering = GatedSteering(
        intervention_layers, intervention_layers,
        v_detect, v_refuse, thresholds,
    )
    result_metadata = {
        "source_layer": intervention_layers,
        "target_layer": intervention_layers,
    }

    log(f"[5.5] calibration ({hit(paths.calibration)})")
    scale = select_scale(
        pool, baseline_test, calibration_scales, steering, BASELINE_SYSTEM, template,
        cache_path=paths.calibration,
        result_metadata=result_metadata,
    )
    log(f"  selected scale={scale}")

    log(f"[6+7] steered generation ({hit(paths.results)})")
    if paths.results.exists():
        results = pd.read_csv(paths.results)
    else:
        df_gen = sample_per_concept(baseline_test, n_per_concept=n_per_concept).reset_index(drop=True)
        prompts = [
            template.render(BASELINE_SYSTEM, row.question)
            for row in df_gen.itertuples(index=False)
        ]
        jobs = make_generation_jobs(df_gen, prompts, targets=concepts, scales=[scale])
        log(f"  {len(jobs)} jobs across {len(pool)} gpus")
        results = pool.run_jobs(
            jobs, steering,
            generation_kwargs={"max_new_tokens": 64, "do_sample": False, "temperature": 1.0},
            batch_size=128,
            trim_fn=template.trim_to_last_assistant,
            result_metadata=result_metadata,
        )
        results.to_csv(paths.results, index=False)

    log(f"[8] scoring ({hit(paths.scored)})")
    if paths.scored.exists():
        scored = pd.read_csv(paths.scored)
    else:
        scored = add_retention_column(results)
        scored = add_refusal_column(scored)
        scored = add_acceptability_column(scored)
        scored.to_csv(paths.scored, index=False)

    if plot:
        log(f"[9] plots → {paths.root / 'plots'}")
        make_plots(
            save_dir=paths.root / "plots",
            concepts=concepts,
            intervention_layers=intervention_layers,
            scale=scale,
            scored=scored,
            calibration_results=load_or_empty_results(paths.calibration, text_columns=["model_output"]),
            baseline_acts=baseline_acts,
            baseline_masks=baseline_masks,
            v_detect=v_detect,
            thresholds=thresholds,
        )

    log(f"done in {time.perf_counter() - t0:.1f}s")
    return scored
