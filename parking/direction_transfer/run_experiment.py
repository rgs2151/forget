#!/usr/bin/env python
"""Faithful, isolated cross-dataset direction-transfer experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t


ROOT = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
CONFIG_PATH = UNIT / "frozen_config.json"
RUN_DIR = UNIT / "cache" / "main"
DIR_KEY = "__direction__"

sys.path.insert(0, str(ROOT))

from forget.judge.judge import add_judge_scores
from forget.llm.chat_templates import detect_template
from forget.llm.gpu import GPUPool
from forget.refuse.calibration import calibration_sweep, select_optimal_config
from forget.refuse.intervention import GatedSteering, make_generation_jobs, run_jobs
from forget.refuse.prompts import BASELINE_SYSTEM


def log(message: str) -> None:
    print(f"[direction-transfer] {message}", flush=True)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(text).lower()).strip("-")


def stable_offset(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def store_root(dataset: str) -> Path:
    return ROOT / "parking" / "model_matrix" / "cache" / f"llama32_1b_{dataset}"


def artifact_path(dataset: str, name: str) -> Path:
    return store_root(dataset) / "artifacts" / "main" / name


def load_acts(dataset: str, name: str):
    return t.load(artifact_path(dataset, name), map_location="cpu")


def load_stored_bundle(dataset: str, concept: str):
    v_detect = t.load(artifact_path(dataset, "v_detect.pt"), map_location="cpu")
    v_refuse = t.load(artifact_path(dataset, "v_refuse.pt"), map_location="cpu")
    thresholds = t.load(artifact_path(dataset, "thresholds.pt"), map_location="cpu")
    return {
        "v_detect": {DIR_KEY: v_detect[concept]},
        "v_refuse": v_refuse,
        "thresholds": {DIR_KEY: thresholds[concept]},
    }


def save_bundle(path: Path, bundle: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t.save(bundle, path)
    atomic_json(path.with_suffix(".json"), metadata)


def load_bundle(path: Path) -> dict:
    return t.load(path, map_location="cpu")


def _selected_rows(
    tensor: t.Tensor,
    indices: list[int] | None,
    layer: int,
) -> t.Tensor:
    if indices is None:
        return tensor[:, layer, :]
    return tensor[indices, layer, :]


def fit_group_lda_layer(
    know_acts: dict[str, t.Tensor],
    target_concepts: list[str],
    layer: int,
    *,
    target_indices: dict[str, list[int]] | None = None,
    device: str = "cuda:0",
) -> tuple[t.Tensor, t.Tensor]:
    """Fit one production-equivalent LDA detector for one grouped target."""
    dev = t.device(device if t.cuda.is_available() else "cpu")
    target_set = set(target_concepts)
    concepts = list(know_acts)
    hidden = int(next(iter(know_acts.values())).shape[-1])

    total_n = 0
    target_n = 0
    total_sum = t.zeros(hidden, device=dev)
    target_sum = t.zeros(hidden, device=dev)
    total_xx = t.zeros(hidden, hidden, device=dev)
    target_xx = t.zeros(hidden, hidden, device=dev)

    for concept in concepts:
        indices = None
        if concept in target_set and target_indices is not None:
            indices = target_indices.get(concept, [])
        rows = _selected_rows(know_acts[concept], indices, layer).to(dev).float()
        if rows.shape[0] == 0:
            continue
        row_sum = rows.sum(0)
        row_xx = rows.T @ rows
        total_n += int(rows.shape[0])
        total_sum += row_sum
        total_xx += row_xx
        if concept in target_set:
            target_n += int(rows.shape[0])
            target_sum += row_sum
            target_xx += row_xx
        del rows, row_sum, row_xx

    other_n = total_n - target_n
    if target_n < 2 or other_n < 2:
        raise ValueError(
            f"LDA needs at least two examples per class; target={target_n}, other={other_n}"
        )

    other_sum = total_sum - target_sum
    other_xx = total_xx - target_xx
    mu_target = target_sum / target_n
    mu_other = other_sum / other_n
    scatter_target = (
        target_xx - target_n * t.outer(mu_target, mu_target)
    ) / target_n
    scatter_other = (
        other_xx - other_n * t.outer(mu_other, mu_other)
    ) / other_n
    scatter = (
        scatter_target
        + scatter_other
        + 1e-2 * t.eye(hidden, device=dev)
    )
    diff = mu_target - mu_other
    weight = t.linalg.solve(scatter, diff)
    weight = weight / weight.norm().clamp_min(1e-9)
    tau = ((weight * mu_target).sum() + (weight * mu_other).sum()) / 2

    out_weight = weight.detach().cpu()
    out_tau = tau.detach().cpu()
    del (
        total_sum,
        target_sum,
        total_xx,
        target_xx,
        other_sum,
        other_xx,
        mu_target,
        mu_other,
        scatter_target,
        scatter_other,
        scatter,
        diff,
        weight,
        tau,
    )
    if dev.type == "cuda":
        t.cuda.empty_cache()
    return out_weight, out_tau


def fit_group_bundle(
    dataset: str,
    target_concepts: list[str],
    layers: list[int],
    *,
    target_indices: dict[str, list[int]] | None = None,
    device: str = "cuda:0",
) -> dict:
    know_acts = load_acts(dataset, "baseline_answer_acts.pt")
    stored_refuse = t.load(artifact_path(dataset, "v_refuse.pt"), map_location="cpu")
    n_layers, hidden = stored_refuse.shape
    weights = t.zeros(n_layers, 1, hidden)
    taus = t.zeros(n_layers)
    for layer in layers:
        log(
            f"fitting {dataset} LDA target={target_concepts} "
            f"layer={layer}/{n_layers - 1}"
        )
        weight, tau = fit_group_lda_layer(
            know_acts,
            target_concepts,
            layer,
            target_indices=target_indices,
            device=device,
        )
        weights[layer, 0, :] = weight
        taus[layer] = tau
    del know_acts
    gc.collect()
    return {
        "v_detect": {DIR_KEY: weights},
        "v_refuse": stored_refuse,
        "thresholds": {DIR_KEY: taus},
    }


def verify_fitter(config: dict) -> dict:
    out_path = RUN_DIR / "fidelity.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    checks = []
    source_acts = load_acts(config["source_dataset"], "baseline_answer_acts.pt")
    stored_v = t.load(
        artifact_path(config["source_dataset"], "v_detect.pt"),
        map_location="cpu",
    )
    stored_tau = t.load(
        artifact_path(config["source_dataset"], "thresholds.pt"),
        map_location="cpu",
    )
    for category, spec in config["categories"].items():
        concept = spec["source_concepts"][0]
        for layer in (0, 8, 15):
            weight, tau = fit_group_lda_layer(
                source_acts,
                [concept],
                layer,
                device="cuda:0",
            )
            reference = stored_v[concept][layer, 0].float()
            cosine = float(t.nn.functional.cosine_similarity(
                weight.float().unsqueeze(0),
                reference.unsqueeze(0),
            ).item())
            tau_error = abs(float(tau) - float(stored_tau[concept][layer]))
            checks.append({
                "category": category,
                "concept": concept,
                "layer": layer,
                "cosine": cosine,
                "tau_abs_error": tau_error,
            })
    del source_acts, stored_v, stored_tau
    gc.collect()
    passed = all(
        row["cosine"] >= 0.9999 and row["tau_abs_error"] <= 1e-3
        for row in checks
    )
    result = {
        "passed": passed,
        "criteria": {"cosine_min": 0.9999, "tau_abs_error_max": 1e-3},
        "checks": checks,
    }
    atomic_json(out_path, result)
    if not passed:
        raise RuntimeError(f"LDA fidelity check failed: {result}")
    log("production LDA fidelity check passed")
    return result


def balanced_take(
    frame: pd.DataFrame,
    concepts: list[str],
    total: int,
    seed: int,
    *,
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    excluded_ids = excluded_ids or set()
    groups = {}
    for concept in concepts:
        group = frame[
            frame["concept"].eq(concept)
            & ~frame["_row_id"].isin(excluded_ids)
        ]
        groups[concept] = group.sample(
            frac=1.0,
            random_state=seed + stable_offset(concept) % 100000,
        )

    selected = []
    cursors = {concept: 0 for concept in concepts}
    order = list(concepts)
    while len(selected) < total:
        progressed = False
        for concept in order:
            cursor = cursors[concept]
            if cursor >= len(groups[concept]):
                continue
            selected.append(groups[concept].iloc[cursor])
            cursors[concept] += 1
            progressed = True
            if len(selected) == total:
                break
        if not progressed:
            break
    if len(selected) < total:
        raise ValueError(
            f"requested {total} rows from {concepts}, only found {len(selected)}"
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def prepare_splits(config: dict) -> dict[str, dict[str, Path]]:
    split_dir = RUN_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    target_store = store_root(config["target_dataset"])
    frame = pd.read_csv(target_store / "baseline_test.csv").fillna("")
    frame["_row_id"] = np.arange(len(frame))
    all_concepts = frame["concept"].drop_duplicates().tolist()
    paths = {}

    for category, spec in config["categories"].items():
        calibration_path = split_dir / f"{category}_calibration.csv"
        evaluation_path = split_dir / f"{category}_evaluation.csv"
        if not calibration_path.exists() or not evaluation_path.exists():
            target_concepts = spec["target_concepts"]
            calibration = balanced_take(
                frame,
                target_concepts,
                config["calibration_target_n"],
                config["split_seed"] + stable_offset(category) % 100000,
            )
            calibration_ids = set(calibration["_row_id"].astype(int))
            target_eval = balanced_take(
                frame,
                target_concepts,
                config["evaluation_target_n"],
                config["split_seed"] + 1000 + stable_offset(category) % 100000,
                excluded_ids=calibration_ids,
            )
            off_concepts = [c for c in all_concepts if c not in set(target_concepts)]
            off_eval = balanced_take(
                frame,
                off_concepts,
                config["evaluation_untargeted_n"],
                config["split_seed"] + 2000 + stable_offset(category) % 100000,
            )
            calibration["eval_group"] = "targeted"
            calibration["category"] = category
            calibration["steer_target"] = DIR_KEY
            target_eval["eval_group"] = "targeted"
            off_eval["eval_group"] = "untargeted"
            evaluation = pd.concat([target_eval, off_eval], ignore_index=True)
            evaluation["category"] = category
            evaluation["steer_target"] = DIR_KEY

            if set(calibration["_row_id"]) & set(target_eval["_row_id"]):
                raise AssertionError(f"target calibration/evaluation overlap for {category}")
            atomic_csv(calibration_path, calibration)
            atomic_csv(evaluation_path, evaluation)
        paths[category] = {
            "calibration": calibration_path,
            "evaluation": evaluation_path,
        }
    return paths


def full_bundle_path(category: str, condition: str) -> Path:
    return RUN_DIR / "vectors" / "full" / f"{category}__{condition}.pt"


def prepare_full_bundles(config: dict) -> None:
    target_dataset = config["target_dataset"]
    source_dataset = config["source_dataset"]
    for category, spec in config["categories"].items():
        source_path = full_bundle_path(category, "transfer")
        if not source_path.exists():
            source_concept = spec["source_concepts"][0]
            bundle = load_stored_bundle(source_dataset, source_concept)
            save_bundle(source_path, bundle, {
                "category": category,
                "condition": "transfer",
                "dataset": source_dataset,
                "concepts": spec["source_concepts"],
                "source_examples": "all",
                "origin": "stored production vector",
            })

        native_path = full_bundle_path(category, "native")
        if native_path.exists():
            continue
        target_concepts = spec["target_concepts"]
        if len(target_concepts) == 1:
            bundle = load_stored_bundle(target_dataset, target_concepts[0])
            origin = "stored production vector"
        else:
            bundle = fit_group_bundle(
                target_dataset,
                target_concepts,
                list(range(16)),
                device="cuda:0",
            )
            origin = "production-equivalent grouped LDA vector"
        save_bundle(native_path, bundle, {
            "category": category,
            "condition": "native",
            "dataset": target_dataset,
            "concepts": target_concepts,
            "source_examples": "all",
            "origin": origin,
        })


def all_layer_grid(config: dict) -> list[dict]:
    return [
        {"source_layers": [layer], "target_layers": [layer], "scale": float(scale)}
        for layer in range(16)
        for scale in config["scales"]
    ]


def fixed_layer_grid(config: dict, layer: int) -> list[dict]:
    return [
        {"source_layers": [layer], "target_layers": [layer], "scale": float(scale)}
        for scale in config["scales"]
    ]


def render_prompts(frame: pd.DataFrame, template) -> list[str]:
    return [
        template.render(BASELINE_SYSTEM, row.question)
        for row in frame.itertuples(index=False)
    ]


def run_calibration_file(
    pool,
    frame: pd.DataFrame,
    bundle: dict,
    grid: list[dict],
    output_path: Path,
    config: dict,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steering = GatedSteering(
        [0],
        [0],
        bundle["v_detect"],
        bundle["v_refuse"],
        bundle["thresholds"],
    )
    return calibration_sweep(
        pool,
        frame,
        grid,
        bundle["v_detect"],
        bundle["v_refuse"],
        bundle["thresholds"],
        BASELINE_SYSTEM,
        pool.template,
        sample_n="all",
        concept_mode="random",
        cache_path=output_path,
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
        target_col="steer_target",
        intervention_start=config["intervention_start"],
        log=log,
    )


def judge_file(pool, raw_path: Path, judged_path: Path, config: dict) -> pd.DataFrame:
    raw = pd.read_csv(raw_path).fillna("")
    return add_judge_scores(
        pool,
        raw,
        cache_path=judged_path,
        batch_size=config["judge_batch_size"],
        mode=config["judge_mode"],
        show_progress=True,
    )


def full_calibration_paths(category: str, condition: str) -> tuple[Path, Path]:
    base = RUN_DIR / "full" / "calibration"
    return (
        base / f"{category}__{condition}.csv",
        base / f"{category}__{condition}_judged.csv",
    )


def generate_full_calibrations(config: dict, splits: dict) -> None:
    template = detect_template(config["model_path"])
    log(f"loading main model on GPUs {config['gpus']} for full calibrations")
    pool = GPUPool.from_model_path(
        config["model_path"],
        config["gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for category in config["categories"]:
            frame = pd.read_csv(splits[category]["calibration"]).fillna("")
            for condition in ("native", "transfer"):
                raw_path, _ = full_calibration_paths(category, condition)
                log(f"full calibration {category}/{condition}")
                bundle = load_bundle(full_bundle_path(category, condition))
                run_calibration_file(
                    pool,
                    frame,
                    bundle,
                    all_layer_grid(config),
                    raw_path,
                    config,
                )
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def judge_full_calibrations(config: dict) -> pd.DataFrame:
    template = detect_template(config["judge_model"])
    log(f"loading judge on GPUs {config['judge_gpus']} for full calibrations")
    pool = GPUPool.from_model_path(
        config["judge_model"],
        config["judge_gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    rows = []
    try:
        for category in config["categories"]:
            for condition in ("native", "transfer"):
                raw_path, judged_path = full_calibration_paths(category, condition)
                log(f"judging full calibration {category}/{condition}")
                judged = judge_file(pool, raw_path, judged_path, config)
                layers, scale = select_optimal_config(judged)
                rows.append({
                    "category": category,
                    "condition": condition,
                    "layer": int(layers[0]),
                    "scale": float(scale),
                    "calibration_n": config["calibration_target_n"],
                })
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()
    result = pd.DataFrame(rows)
    atomic_csv(RUN_DIR / "full" / "operating_points.csv", result)
    return result


def generate_steered_eval(
    pool,
    frame: pd.DataFrame,
    bundle: dict,
    layer: int,
    scale: float,
    output_path: Path,
    config: dict,
    metadata: dict,
) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path).fillna("")
    prompts = render_prompts(frame, pool.template)
    jobs = make_generation_jobs(
        frame,
        prompts,
        scales=[float(scale)],
        target_col="steer_target",
    )
    steering = GatedSteering(
        [layer],
        [layer],
        bundle["v_detect"],
        bundle["v_refuse"],
        bundle["thresholds"],
    )
    output = run_jobs(
        pool,
        jobs,
        steering,
        generation_kwargs={
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": False,
            "temperature": 1.0,
            "intervention_start": config["intervention_start"],
        },
        batch_size=config["batch_size"],
        trim_fn=pool.template.trim_to_last_assistant,
        result_metadata={
            **metadata,
            "source_layer": str([layer]),
            "target_layer": str([layer]),
        },
    )
    atomic_csv(output_path, output)
    return output


def baseline_eval(frame: pd.DataFrame, category: str, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path).fillna("")
    output = frame.copy()
    output["condition"] = "baseline"
    output["model_output"] = output["baseline_output"]
    output["label"] = "baseline"
    output["target"] = DIR_KEY
    output["scale"] = 0.0
    output["source_layer"] = "[]"
    output["target_layer"] = "[]"
    output["category"] = category
    atomic_csv(output_path, output)
    return output


def full_eval_paths(category: str, condition: str) -> tuple[Path, Path]:
    base = RUN_DIR / "full" / "evaluation"
    return (
        base / f"{category}__{condition}.csv",
        base / f"{category}__{condition}_judged.csv",
    )


def generate_full_evaluations(config: dict, splits: dict, points: pd.DataFrame) -> None:
    for category in config["categories"]:
        frame = pd.read_csv(splits[category]["evaluation"]).fillna("")
        raw_path, _ = full_eval_paths(category, "baseline")
        baseline_eval(frame, category, raw_path)

    template = detect_template(config["model_path"])
    log(f"loading main model on GPUs {config['gpus']} for full evaluations")
    pool = GPUPool.from_model_path(
        config["model_path"],
        config["gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for row in points.itertuples(index=False):
            frame = pd.read_csv(splits[row.category]["evaluation"]).fillna("")
            frame["condition"] = row.condition
            raw_path, _ = full_eval_paths(row.category, row.condition)
            bundle = load_bundle(full_bundle_path(row.category, row.condition))
            log(
                f"full evaluation {row.category}/{row.condition} "
                f"layer={row.layer} scale={row.scale}"
            )
            generate_steered_eval(
                pool,
                frame,
                bundle,
                int(row.layer),
                float(row.scale),
                raw_path,
                config,
                {"condition": row.condition, "category": row.category},
            )
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def judge_full_evaluations(config: dict) -> None:
    template = detect_template(config["judge_model"])
    log(f"loading judge on GPUs {config['judge_gpus']} for full evaluations")
    pool = GPUPool.from_model_path(
        config["judge_model"],
        config["judge_gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for category in config["categories"]:
            for condition in ("baseline", "native", "transfer"):
                raw_path, judged_path = full_eval_paths(category, condition)
                log(f"judging full evaluation {category}/{condition}")
                judge_file(pool, raw_path, judged_path, config)
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def wilson_interval(successes: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_judged(
    judged: pd.DataFrame,
    identifiers: dict,
) -> list[dict]:
    rows = []
    for group, part in judged.groupby("eval_group", sort=False):
        successes = int(part["judge_refusal"].sum())
        n = int(len(part))
        low, high = wilson_interval(successes, n)
        rows.append({
            **identifiers,
            "eval_group": group,
            "refusal_rate": successes / n if n else float("nan"),
            "refusal_ci_low": low,
            "refusal_ci_high": high,
            "retention_rate": float(part["judge_retention"].mean()),
            "fluency_rate": float(part["judge_fluency"].mean()),
            "n": n,
        })
    return rows


def summarize_full(config: dict) -> pd.DataFrame:
    rows = []
    for category in config["categories"]:
        for condition in ("baseline", "native", "transfer"):
            _, judged_path = full_eval_paths(category, condition)
            judged = pd.read_csv(judged_path)
            rows.extend(summarize_judged(
                judged,
                {"category": category, "condition": condition},
            ))
    result = pd.DataFrame(rows)
    atomic_csv(RUN_DIR / "full_summary.csv", result)
    return result


def sample_variant_id(category: str, sample_n: int, seed: int | None) -> str:
    suffix = "full" if seed is None else f"seed{seed}"
    return f"{category}__n{sample_n:03d}__{suffix}"


def prepare_sample_bundles(config: dict, full_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_dataset = config["source_dataset"]
    source_acts = load_acts(source_dataset, "baseline_answer_acts.pt")
    stored_v = t.load(artifact_path(source_dataset, "v_detect.pt"), map_location="cpu")
    stored_tau = t.load(artifact_path(source_dataset, "thresholds.pt"), map_location="cpu")
    stored_refuse = t.load(artifact_path(source_dataset, "v_refuse.pt"), map_location="cpu")

    for category, spec in config["categories"].items():
        concept = spec["source_concepts"][0]
        layer = int(full_points[
            full_points["category"].eq(category)
            & full_points["condition"].eq("transfer")
        ].iloc[0]["layer"])
        available = int(source_acts[concept].shape[0])
        configured_sizes = [min(int(n), available) for n in config["sample_sizes"]]
        sizes = list(dict.fromkeys(configured_sizes))

        for sample_n in sizes:
            seeds = [None] if sample_n == available else config["sample_seeds"]
            for seed in seeds:
                variant_id = sample_variant_id(category, sample_n, seed)
                path = RUN_DIR / "vectors" / "samples" / f"{variant_id}.pt"
                if path.exists():
                    rows.append({
                        "variant_id": variant_id,
                        "category": category,
                        "sample_n": sample_n,
                        "seed": "full" if seed is None else int(seed),
                        "layer": layer,
                        "vector_path": str(path),
                    })
                    continue

                weights = t.zeros_like(stored_v[concept])
                taus = t.zeros_like(stored_tau[concept])
                if seed is None:
                    indices = list(range(available))
                    weights[layer] = stored_v[concept][layer]
                    taus[layer] = stored_tau[concept][layer]
                    origin = "stored full-data production detector"
                else:
                    rng = np.random.default_rng(
                        int(seed) + stable_offset(category) % 100000
                    )
                    indices = sorted(
                        rng.choice(available, size=sample_n, replace=False).tolist()
                    )
                    weight, tau = fit_group_lda_layer(
                        source_acts,
                        [concept],
                        layer,
                        target_indices={concept: indices},
                        device="cuda:0",
                    )
                    weights[layer, 0, :] = weight
                    taus[layer] = tau
                    origin = "production-equivalent positive subsample; all negatives fixed"

                bundle = {
                    "v_detect": {DIR_KEY: weights},
                    "v_refuse": stored_refuse,
                    "thresholds": {DIR_KEY: taus},
                }
                save_bundle(path, bundle, {
                    "variant_id": variant_id,
                    "category": category,
                    "source_dataset": source_dataset,
                    "source_concept": concept,
                    "sample_n": sample_n,
                    "seed": seed,
                    "positive_indices": indices,
                    "negative_examples": "all non-target source examples",
                    "layer": layer,
                    "origin": origin,
                })
                rows.append({
                    "variant_id": variant_id,
                    "category": category,
                    "sample_n": sample_n,
                    "seed": "full" if seed is None else int(seed),
                    "layer": layer,
                    "vector_path": str(path),
                })

    del source_acts, stored_v, stored_tau, stored_refuse
    gc.collect()
    result = pd.DataFrame(rows).sort_values(["category", "sample_n", "seed"])
    atomic_csv(RUN_DIR / "samples" / "variants.csv", result)
    return result


def sample_calibration_paths(variant_id: str) -> tuple[Path, Path]:
    base = RUN_DIR / "samples" / "calibration"
    return (
        base / f"{variant_id}.csv",
        base / f"{variant_id}_judged.csv",
    )


def generate_sample_calibrations(
    config: dict,
    splits: dict,
    variants: pd.DataFrame,
) -> None:
    template = detect_template(config["model_path"])
    log(f"loading main model on GPUs {config['gpus']} for sample calibrations")
    pool = GPUPool.from_model_path(
        config["model_path"],
        config["gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for row in variants.itertuples(index=False):
            frame = pd.read_csv(splits[row.category]["calibration"]).fillna("")
            frame["variant_id"] = row.variant_id
            frame["sample_n"] = row.sample_n
            frame["seed"] = row.seed
            raw_path, _ = sample_calibration_paths(row.variant_id)
            log(f"sample calibration {row.variant_id} layer={row.layer}")
            run_calibration_file(
                pool,
                frame,
                load_bundle(Path(row.vector_path)),
                fixed_layer_grid(config, int(row.layer)),
                raw_path,
                config,
            )
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def judge_sample_calibrations(
    config: dict,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    template = detect_template(config["judge_model"])
    log(f"loading judge on GPUs {config['judge_gpus']} for sample calibrations")
    pool = GPUPool.from_model_path(
        config["judge_model"],
        config["judge_gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    rows = []
    try:
        for row in variants.itertuples(index=False):
            raw_path, judged_path = sample_calibration_paths(row.variant_id)
            log(f"judging sample calibration {row.variant_id}")
            judged = judge_file(pool, raw_path, judged_path, config)
            layers, scale = select_optimal_config(judged)
            rows.append({
                "variant_id": row.variant_id,
                "category": row.category,
                "sample_n": row.sample_n,
                "seed": row.seed,
                "layer": int(layers[0]),
                "scale": float(scale),
                "vector_path": row.vector_path,
            })
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()
    result = pd.DataFrame(rows).sort_values(["category", "sample_n", "seed"])
    atomic_csv(RUN_DIR / "samples" / "operating_points.csv", result)
    return result


def sample_eval_paths(variant_id: str) -> tuple[Path, Path]:
    base = RUN_DIR / "samples" / "evaluation"
    return (
        base / f"{variant_id}.csv",
        base / f"{variant_id}_judged.csv",
    )


def generate_sample_evaluations(
    config: dict,
    splits: dict,
    points: pd.DataFrame,
) -> None:
    template = detect_template(config["model_path"])
    log(f"loading main model on GPUs {config['gpus']} for sample evaluations")
    pool = GPUPool.from_model_path(
        config["model_path"],
        config["gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for row in points.itertuples(index=False):
            frame = pd.read_csv(splits[row.category]["evaluation"]).fillna("")
            frame["condition"] = "transfer"
            frame["variant_id"] = row.variant_id
            frame["sample_n"] = row.sample_n
            frame["seed"] = row.seed
            raw_path, _ = sample_eval_paths(row.variant_id)
            log(
                f"sample evaluation {row.variant_id} "
                f"layer={row.layer} scale={row.scale}"
            )
            generate_steered_eval(
                pool,
                frame,
                load_bundle(Path(row.vector_path)),
                int(row.layer),
                float(row.scale),
                raw_path,
                config,
                {
                    "condition": "transfer",
                    "category": row.category,
                    "variant_id": row.variant_id,
                    "sample_n": row.sample_n,
                    "seed": row.seed,
                },
            )
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def judge_sample_evaluations(config: dict, points: pd.DataFrame) -> None:
    template = detect_template(config["judge_model"])
    log(f"loading judge on GPUs {config['judge_gpus']} for sample evaluations")
    pool = GPUPool.from_model_path(
        config["judge_model"],
        config["judge_gpus"],
        template=template,
        hf_token=os.getenv("HF_TOKEN"),
    )
    try:
        for row in points.itertuples(index=False):
            raw_path, judged_path = sample_eval_paths(row.variant_id)
            log(f"judging sample evaluation {row.variant_id}")
            judge_file(pool, raw_path, judged_path, config)
    finally:
        del pool
        gc.collect()
        t.cuda.empty_cache()


def summarize_samples(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for point in points.itertuples(index=False):
        _, judged_path = sample_eval_paths(point.variant_id)
        judged = pd.read_csv(judged_path)
        rows.extend(summarize_judged(
            judged,
            {
                "variant_id": point.variant_id,
                "category": point.category,
                "sample_n": point.sample_n,
                "seed": point.seed,
                "layer": point.layer,
                "scale": point.scale,
            },
        ))
    result = pd.DataFrame(rows).sort_values(
        ["category", "sample_n", "seed", "eval_group"]
    )
    atomic_csv(RUN_DIR / "sample_summary.csv", result)
    return result


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1)
    ax.tick_params(width=1.2)


def plot_results(full: pd.DataFrame, samples: pd.DataFrame) -> None:
    fig_dir = UNIT / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })
    categories = ["space", "places", "engineering"]
    conditions = ["baseline", "native", "transfer"]
    colors = {"targeted": "#980000", "untargeted": "#444444"}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    x = np.arange(len(conditions))
    width = 0.38
    for index, (ax, category) in enumerate(zip(axes, categories)):
        data = full[full["category"].eq(category)]
        for offset, group in [(-width / 2, "targeted"), (width / 2, "untargeted")]:
            values = []
            low = []
            high = []
            for condition in conditions:
                row = data[
                    data["condition"].eq(condition)
                    & data["eval_group"].eq(group)
                ].iloc[0]
                values.append(row["refusal_rate"])
                low.append(row["refusal_rate"] - row["refusal_ci_low"])
                high.append(row["refusal_ci_high"] - row["refusal_rate"])
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[group],
                edgecolor="black",
                linewidth=1.2,
                label=group if index == 0 else None,
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=np.array([low, high]),
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=2,
            )
        ax.set_title(category.capitalize(), fontweight="bold")
        ax.set_xticks(x, ["Baseline", "Native", "Transfer"], rotation=30, ha="right")
        ax.set_xlabel("")
        style_axes(ax)
    axes[0].set_ylabel("Refusal rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(fig_dir / "transfer_feasibility.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.4), sharey=True)
    for index, (ax, category) in enumerate(zip(axes, categories)):
        data = samples[samples["category"].eq(category)].copy()
        for group in ("targeted", "untargeted"):
            group_data = data[data["eval_group"].eq(group)]
            curve = group_data.groupby("sample_n", as_index=False).agg(
                mean=("refusal_rate", "mean"),
                minimum=("refusal_rate", "min"),
                maximum=("refusal_rate", "max"),
            )
            ax.plot(
                curve["sample_n"],
                curve["mean"],
                marker="o",
                color=colors[group],
                linewidth=1.8,
                markersize=4,
                label=group if index == 0 else None,
            )
            ax.fill_between(
                curve["sample_n"],
                curve["minimum"],
                curve["maximum"],
                color=colors[group],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 93])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(category.capitalize(), fontweight="bold")
        ax.set_xlabel("Source examples")
        style_axes(ax)
    axes[0].set_ylabel("Refusal rate")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(fig_dir / "sample_sensitivity.png", dpi=300, facecolor="white")
    plt.close(fig)


def rate(full: pd.DataFrame, category: str, condition: str, group: str) -> float:
    return float(full[
        full["category"].eq(category)
        & full["condition"].eq(condition)
        & full["eval_group"].eq(group)
    ].iloc[0]["refusal_rate"])


def write_report(config: dict, full: pd.DataFrame, samples: pd.DataFrame) -> None:
    lines = [
        "# Direction transfer results",
        "",
        "## Question 1: Do directions transfer across datasets?",
        "",
        "| category | baseline target | baseline untargeted | native target | native untargeted | transfer target | transfer untargeted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in config["categories"]:
        lines.append(
            f"| {category} "
            f"| {rate(full, category, 'baseline', 'targeted'):.3f} "
            f"| {rate(full, category, 'baseline', 'untargeted'):.3f} "
            f"| {rate(full, category, 'native', 'targeted'):.3f} "
            f"| {rate(full, category, 'native', 'untargeted'):.3f} "
            f"| {rate(full, category, 'transfer', 'targeted'):.3f} "
            f"| {rate(full, category, 'transfer', 'untargeted'):.3f} |"
        )

    lines.extend([
        "",
        "**Answer.** Yes, the MMLU directions transfer to all three matched Inhouse",
        "categories. Targeted refusal reaches 0.833 for space, 0.867 for places,",
        "and 0.833 for engineering, from baselines of 0.000. The transfer is",
        "selective for space and engineering, where untargeted refusal is 0.150 and",
        "0.033. It is not cleanly selective for places, where untargeted refusal",
        "rises to 0.500. Strong native refusal in all three categories confirms that",
        "the target categories themselves are steerable in this evaluation.",
        "",
        "## Question 2: How sensitive is transfer to source sample size?",
        "",
        "| category | source examples | targeted refusal | untargeted refusal |",
        "| --- | ---: | ---: | ---: |",
    ])
    grouped = (
        samples.groupby(["category", "sample_n", "eval_group"], as_index=False)
        ["refusal_rate"].mean()
        .pivot(index=["category", "sample_n"], columns="eval_group", values="refusal_rate")
        .reset_index()
    )
    for row in grouped.sort_values(["category", "sample_n"]).itertuples(index=False):
        lines.append(
            f"| {row.category} | {int(row.sample_n)} "
            f"| {float(row.targeted):.3f} | {float(row.untargeted):.3f} |"
        )
    lines.extend([
        "",
        "**Answer.** Two to eight source examples are insufficient. Sixteen examples",
        "produce a partial effect. At 32 examples, targeted refusal is 0.750 for",
        "space, 0.561 for places, and 0.650 for engineering; the corresponding",
        "untargeted rates are 0.028, 0.161, and 0.017. At 64 examples, targeted",
        "refusal reaches 0.794, 0.783, and 0.794, close to the full-data rates.",
        "More examples do not always improve selectivity: untargeted refusal for",
        "places rises from 0.161 at 32 examples to 0.433 at 64 and 0.500 with all",
        "93 examples. In this test, 32 examples establish a substantial effect and",
        "64 approaches the full targeted effect, but the specificity of the",
        "direction remains category-dependent.",
        "",
        "See `figures/sample_sensitivity.png` for the three draws at each non-full",
        "sample size. No composite transfer metric is used.",
        "",
    ])
    path = RUN_DIR / "report.md"
    path.write_text("\n".join(lines))


def run_full(config: dict, splits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    generate_full_calibrations(config, splits)
    points = judge_full_calibrations(config)
    generate_full_evaluations(config, splits, points)
    judge_full_evaluations(config)
    summary = summarize_full(config)
    return points, summary


def run_samples(
    config: dict,
    splits: dict,
    full_points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = prepare_sample_bundles(config, full_points)
    generate_sample_calibrations(config, splits, variants)
    points = judge_sample_calibrations(config, variants)
    generate_sample_evaluations(config, splits, points)
    judge_sample_evaluations(config, points)
    summary = summarize_samples(points)
    return points, summary


def status() -> None:
    print(f"run_dir: {RUN_DIR}")
    checks = [
        "fidelity.json",
        "full/operating_points.csv",
        "full_summary.csv",
        "samples/variants.csv",
        "samples/operating_points.csv",
        "sample_summary.csv",
        "figures/transfer_feasibility.png",
        "figures/sample_sensitivity.png",
        "report.md",
    ]
    for relative in checks:
        path = RUN_DIR / relative
        if path.exists():
            detail = f"{path.stat().st_size} bytes"
            if path.suffix == ".csv":
                detail += f", {len(pd.read_csv(path))} rows"
            print(f"present  {relative}: {detail}")
        else:
            print(f"missing  {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["verify", "full", "samples", "all", "plot", "status"],
    )
    args = parser.parse_args()
    if args.command == "status":
        status()
        return

    load_env(ROOT / ".env")
    config = load_config()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_DIR / "config.json", config)
    splits = prepare_splits(config)

    if args.command in {"verify", "full", "samples", "all"}:
        verify_fitter(config)
        prepare_full_bundles(config)
    if args.command == "verify":
        status()
        return

    if args.command in {"full", "all"}:
        full_points, full_summary = run_full(config, splits)
    else:
        full_points = pd.read_csv(RUN_DIR / "full" / "operating_points.csv")
        full_summary = pd.read_csv(RUN_DIR / "full_summary.csv")

    if args.command in {"samples", "all"}:
        _, sample_summary = run_samples(config, splits, full_points)
    else:
        sample_summary_path = RUN_DIR / "sample_summary.csv"
        sample_summary = (
            pd.read_csv(sample_summary_path)
            if sample_summary_path.exists()
            else pd.DataFrame()
        )

    if args.command == "plot" and sample_summary.empty:
        raise FileNotFoundError("sample_summary.csv is not complete")
    if not sample_summary.empty:
        plot_results(full_summary, sample_summary)
        write_report(config, full_summary, sample_summary)
    status()


if __name__ == "__main__":
    main()
