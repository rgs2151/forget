#!/usr/bin/env python
"""Cross-dataset direction transfer debug runner.

This script intentionally lives under debug. It reads existing store artifacts
and writes all new outputs under debug/cross_dataset_transfer/runs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as t


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from judge.judge import add_judge_scores
from llm.chat_templates import detect_template
from llm.gpu import GPUPool
from refuse.intervention import GatedSteering, make_generation_jobs, run_jobs
from refuse.prompts import BASELINE_SYSTEM
from steering.steering import GatedSteer


MODEL_KEY = "llama32_1b"
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
JUDGE_MODEL = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
DIRECTION_KEY = "__source_direction__"

DATASET_RUN = {
    "inhouse": "llama32_1b_inhouse",
    "mmlu": "llama32_1b_mmlu",
    "rwku": "llama32_1b_rwku",
    "conceptvectors": "llama32_1b_conceptvectors",
}

TOPICS = {
    "space": {
        "inhouse": ["the_moon"],
        "mmlu": ["astronomy"],
    },
    "places": {
        "inhouse": ["paris", "united_states"],
        "mmlu": ["high_school_geography"],
        "conceptvectors": [
            "Germany",
            "India",
            "Greece",
            "Netherlands",
            "New Zealand",
            "United Kingdom",
            "Italy",
            "Egypt",
            "London",
            "Poland",
        ],
    },
    "engineering": {
        "inhouse": ["lasers"],
        "mmlu": ["electrical_engineering"],
        "conceptvectors": ["CT scan", "Silicon Valley", "TensorFlow"],
    },
    "computing": {
        "mmlu": ["computer_security"],
        "conceptvectors": [
            "HTTP cookie",
            "TensorFlow",
            "Array (data structure)",
            "Artificial intelligence",
            "Blockchain",
        ],
    },
    "religion": {
        "mmlu": ["world_religions"],
        "conceptvectors": ["Islam", "Bible", "Jesus", "Catholic Church", "Jewish culture"],
    },
    "people": {
        "inhouse": ["people", "obama"],
        "rwku": [
            "Donald Trump",
            "Thomas Jefferson",
            "Jimmy Carter",
            "Franklin D. Roosevelt",
            "Jill Biden",
            "Socrates",
            "Aristotle",
            "Karl Marx",
            "Leonardo da Vinci",
            "Taylor Swift",
        ],
        "conceptvectors": ["Julius Caesar", "William Shakespeare", "Sherlock Holmes"],
    },
}


REFUSAL_PATTERNS = [
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "i'm not sure",
    "i am not sure",
    "not sure",
    "i can't",
    "i cannot",
    "can't help",
    "cannot help",
    "not familiar",
    "no information",
    "not aware",
    "not recall",
]


@dataclass(frozen=True)
class Cell:
    topic: str
    source_dataset: str
    target_dataset: str
    requested_k: int
    source_concepts: tuple[str, ...]
    target_concepts: tuple[str, ...]

    @property
    def id(self) -> str:
        parts = [self.topic, self.source_dataset, self.target_dataset, f"k{self.requested_k}"]
        return "__".join(slug(p) for p in parts)


class TransferSteering(GatedSteering):
    def _make_op(self, src_layer, target, scale, device):
        if target not in self.v_detect:
            raise ValueError(f"Missing detect vectors for target {target!r}.")
        if target not in self.thresholds:
            raise ValueError(f"Missing thresholds for target {target!r}.")
        return GatedSteer(
            v_detect=self.v_detect[target][src_layer].to(device),
            v_steer=self.v_steer[src_layer].to(device),
            tau=float(self.thresholds[target][src_layer]),
            scale=scale,
        )


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_or_replace(path: Path, rows: list[dict], key_cols: list[str]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for col in key_cols:
            if col not in old:
                old[col] = pd.NA
        old_key = old[key_cols].astype(str).agg("||".join, axis=1)
        new_key = new[key_cols].astype(str).agg("||".join, axis=1)
        old = old[~old_key.isin(set(new_key))]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)


def dataset_store(dataset: str) -> Path:
    return ROOT / "store" / DATASET_RUN[dataset]


def load_frame(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_csv(dataset_store(dataset) / f"baseline_{split}.csv").fillna("")


def load_acts(dataset: str, name: str):
    return t.load(dataset_store(dataset) / "artifacts" / "main" / name, map_location="cpu")


def balanced_indices(n_by_concept: dict[str, int], concepts: list[str], total: int, seed: int):
    rng = random.Random(seed)
    indices = {concept: [] for concept in concepts}
    if total <= 0 or not concepts:
        return indices
    available = {concept: n_by_concept[concept] for concept in concepts}
    order = list(concepts)
    rng.shuffle(order)
    while sum(len(v) for v in indices.values()) < total:
        progressed = False
        for concept in order:
            if len(indices[concept]) >= available[concept]:
                continue
            used = set(indices[concept])
            remaining = [i for i in range(available[concept]) if i not in used]
            if not remaining:
                continue
            indices[concept].append(rng.choice(remaining))
            progressed = True
            if sum(len(v) for v in indices.values()) >= total:
                break
        if not progressed:
            break
    return indices


def gather_acts(acts_by_concept, concepts: list[str], total: int, seed: int):
    n_by_concept = {concept: int(acts_by_concept[concept].shape[0]) for concept in concepts}
    idx = balanced_indices(n_by_concept, concepts, total, seed)
    parts = []
    for concept in concepts:
        chosen = idx[concept]
        if chosen:
            parts.append(acts_by_concept[concept][chosen])
    if not parts:
        raise ValueError(f"no activations sampled for {concepts}")
    return t.cat(parts, dim=0), {concept: len(idx[concept]) for concept in concepts}


def fit_binary_direction(pos_know, neg_know, pos_refuse, device: str):
    dev = t.device(device if t.cuda.is_available() else "cpu")
    n_layers = pos_know.shape[1]
    hidden = pos_know.shape[2]
    reg_eye = 1e-2 * t.eye(hidden, device=dev)
    weights, taus, refuse_vecs = [], [], []
    for layer in range(n_layers):
        xp = pos_know[:, layer, :].to(dev).float()
        xn = neg_know[:, layer, :].to(dev).float()
        rp = pos_refuse[:, layer, :].to(dev).float()

        mu_p = xp.mean(0)
        mu_n = xn.mean(0)
        cp = (xp - mu_p).T @ (xp - mu_p) / max(1, xp.shape[0])
        cn = (xn - mu_n).T @ (xn - mu_n) / max(1, xn.shape[0])
        scatter = cp + cn + reg_eye
        diff = mu_p - mu_n
        w = t.linalg.solve(scatter, diff)
        w = w / w.norm().clamp_min(1e-9)
        tau = ((w * mu_p).sum() + (w * mu_n).sum()) / 2

        rv = (rp.mean(0) - xp.mean(0))
        rv = rv / rv.norm().clamp_min(1e-9)

        weights.append(w.detach().cpu())
        taus.append(tau.detach().cpu())
        refuse_vecs.append(rv.detach().cpu())
        del xp, xn, rp, mu_p, mu_n, cp, cn, scatter, diff, w, tau, rv
        if dev.type == "cuda":
            t.cuda.empty_cache()
    return (
        {DIRECTION_KEY: t.stack(weights).unsqueeze(1)},
        t.stack(refuse_vecs),
        {DIRECTION_KEY: t.stack(taus)},
    )


def build_or_load_direction(cell: Cell, run_dir: Path, seed: int, lda_device: str):
    vec_dir = run_dir / "vector_cache"
    vec_dir.mkdir(parents=True, exist_ok=True)
    vec_path = vec_dir / f"{cell.id}.pt"
    meta_path = vec_dir / f"{cell.id}.json"
    if vec_path.exists() and meta_path.exists():
        obj = t.load(vec_path, map_location="cpu")
        meta = json.loads(meta_path.read_text())
        return obj["v_detect"], obj["v_refuse"], obj["thresholds"], meta

    source_dataset = cell.source_dataset
    know = load_acts(source_dataset, "baseline_answer_acts.pt")
    refuse = load_acts(source_dataset, "refuse_answer_acts.pt")
    source_concepts = list(cell.source_concepts)
    all_concepts = sorted(know)
    negative_concepts = [c for c in all_concepts if c not in set(source_concepts)]

    pos_know, pos_counts = gather_acts(know, source_concepts, cell.requested_k, seed)
    pos_refuse, _ = gather_acts(refuse, source_concepts, int(pos_know.shape[0]), seed)
    neg_know, neg_counts = gather_acts(know, negative_concepts, int(pos_know.shape[0]), seed + 10000)

    v_detect, v_refuse, thresholds = fit_binary_direction(pos_know, neg_know, pos_refuse, lda_device)
    meta = {
        "cell_id": cell.id,
        "topic": cell.topic,
        "source_dataset": cell.source_dataset,
        "target_dataset": cell.target_dataset,
        "requested_k": cell.requested_k,
        "actual_k": int(pos_know.shape[0]),
        "negative_k": int(neg_know.shape[0]),
        "source_counts": pos_counts,
        "negative_counts": neg_counts,
        "source_concepts": list(cell.source_concepts),
        "target_concepts": list(cell.target_concepts),
    }
    t.save({"v_detect": v_detect, "v_refuse": v_refuse, "thresholds": thresholds}, vec_path)
    write_json(meta_path, meta)
    return v_detect, v_refuse, thresholds, meta


def sample_target_frame(
    dataset: str,
    topic: str,
    concepts: tuple[str, ...],
    n_pos: int,
    n_off: int,
    seed: int,
) -> pd.DataFrame:
    df = load_frame(dataset, "test")
    pos = df[df["concept"].isin(concepts)].copy()
    off = df[~df["concept"].isin(concepts)].copy()
    if pos.empty:
        raise ValueError(f"no target rows for {dataset}/{topic}/{concepts}")
    pos = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    off = off.sample(n=min(n_off, len(off)), random_state=seed + 10000)
    pos["eval_group"] = "target"
    off["eval_group"] = "untargeted"
    out = pd.concat([pos, off], ignore_index=True)
    out["target_topic"] = topic
    out["steer_target"] = DIRECTION_KEY
    return out.reset_index(drop=True)


def rank_target_layers_by_gate(cell: Cell, args, v_detect, thresholds):
    target_acts = load_acts(cell.target_dataset, "baseline_answer_acts_test.pt")
    all_concepts = sorted(target_acts)
    target_concepts = list(cell.target_concepts)
    off_concepts = [concept for concept in all_concepts if concept not in set(target_concepts)]
    pos, _ = gather_acts(
        target_acts,
        target_concepts,
        max(1, args.calib_target_n),
        args.seed + cell.requested_k + 70000,
    )
    neg, _ = gather_acts(
        target_acts,
        off_concepts,
        max(1, args.calib_untargeted_n),
        args.seed + cell.requested_k + 80000,
    )
    records = []
    detect = v_detect[DIRECTION_KEY].squeeze(1).float()
    tau = thresholds[DIRECTION_KEY].float()
    for layer in args.layers:
        w = detect[layer]
        p_score = (pos[:, layer, :].float() @ w) - tau[layer]
        n_score = (neg[:, layer, :].float() @ w) - tau[layer]
        target_gate = float((p_score > 0).float().mean().item())
        untargeted_gate = float((n_score > 0).float().mean().item())
        records.append({
            "layer": int(layer),
            "target_gate": target_gate,
            "untargeted_gate": untargeted_gate,
            "gate_score": target_gate * (1.0 - untargeted_gate),
        })
    ranked = pd.DataFrame(records).sort_values(
        ["gate_score", "target_gate", "untargeted_gate", "layer"],
        ascending=[False, False, True, True],
    )
    if args.top_layers_by_gate and args.top_layers_by_gate > 0:
        return ranked.head(args.top_layers_by_gate)["layer"].astype(int).tolist(), ranked
    return [int(x) for x in args.layers], ranked


def cheap_refusal(text: str) -> int:
    lowered = str(text).lower()
    return int(any(pattern in lowered for pattern in REFUSAL_PATTERNS))


def cheap_fluency(text: str) -> int:
    s = str(text).strip()
    if len(s) < 2:
        return 0
    if len(set(s)) <= 3 and len(s) > 8:
        return 0
    return 1


def build_prompts(df: pd.DataFrame, template):
    return [template.render(BASELINE_SYSTEM, row.question) for row in df.itertuples(index=False)]


def generate_cell_calibration(pool, cell: Cell, run_dir: Path, args, direction):
    cell_dir = run_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cell_dir / f"{cell.id}.calibration_raw.csv"
    summary_path = cell_dir / f"{cell.id}.calibration_summary.json"
    if raw_path.exists() and summary_path.exists():
        return pd.read_csv(raw_path), json.loads(summary_path.read_text())

    v_detect, v_refuse, thresholds, meta = direction
    candidate_layers, gate_rank = rank_target_layers_by_gate(cell, args, v_detect, thresholds)
    frame = sample_target_frame(
        cell.target_dataset,
        cell.topic,
        cell.target_concepts,
        args.calib_target_n,
        args.calib_untargeted_n,
        args.seed + cell.requested_k,
    )
    for key, value in meta.items():
        if isinstance(value, (str, int, float)):
            frame[key] = value
    frame["cell_id"] = cell.id
    prompts = build_prompts(frame, pool.template)
    rows = []
    for layer in candidate_layers:
        steering = TransferSteering([layer], [layer], v_detect, v_refuse, thresholds)
        jobs = make_generation_jobs(frame, prompts, target_col="steer_target", scales=args.scales)
        out = run_jobs(
            pool,
            jobs,
            steering,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
                "intervention_start": "prefill",
            },
            batch_size=args.batch_size,
            trim_fn=pool.template.trim_to_last_assistant,
            result_metadata={"source_layer": str([layer]), "target_layer": str([layer])},
        )
        rows.append(out)
    raw = pd.concat(rows, ignore_index=True)
    raw["cheap_refusal"] = raw["model_output"].map(cheap_refusal)
    raw["cheap_fluency"] = raw["model_output"].map(cheap_fluency)
    raw.to_csv(raw_path, index=False)

    grouped = (
        raw.groupby(["source_layer", "scale", "eval_group"], as_index=False)
        .agg(cheap_refusal=("cheap_refusal", "mean"), cheap_fluency=("cheap_fluency", "mean"))
    )
    pivot = grouped.pivot_table(
        index=["source_layer", "scale"],
        columns="eval_group",
        values=["cheap_refusal", "cheap_fluency"],
        fill_value=0.0,
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    for col in (
        "cheap_refusal_target",
        "cheap_refusal_untargeted",
        "cheap_fluency_target",
        "cheap_fluency_untargeted",
    ):
        if col not in pivot:
            pivot[col] = 0.0
    pivot["cheap_transfer_score"] = (
        pivot["cheap_refusal_target"]
        * (1.0 - pivot["cheap_refusal_untargeted"])
        * pivot["cheap_fluency_target"]
    )
    best = pivot.sort_values(
        ["cheap_transfer_score", "cheap_refusal_target", "cheap_refusal_untargeted", "scale"],
        ascending=[False, False, True, True],
    ).iloc[0]
    raw_layer = best["source_layer"]
    parsed_layer = ast.literal_eval(raw_layer) if isinstance(raw_layer, str) else [int(raw_layer)]
    summary = {
        **meta,
        "candidate_layers": candidate_layers,
        "best_layer": int(parsed_layer[0]),
        "best_scale": float(best["scale"]),
        "cheap_refusal_target": float(best["cheap_refusal_target"]),
        "cheap_refusal_untargeted": float(best["cheap_refusal_untargeted"]),
        "cheap_fluency_target": float(best["cheap_fluency_target"]),
        "cheap_fluency_untargeted": float(best["cheap_fluency_untargeted"]),
        "cheap_transfer_score": float(best["cheap_transfer_score"]),
    }
    gate_rank.to_csv(cell_dir / f"{cell.id}.gate_layers.csv", index=False)
    write_json(summary_path, summary)
    return raw, summary


def generate_cell_eval(pool, cell: Cell, run_dir: Path, args, direction, summary: dict):
    cell_dir = run_dir / "cells"
    raw_path = cell_dir / f"{cell.id}.eval_raw.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)
    v_detect, v_refuse, thresholds, meta = direction
    frame = sample_target_frame(
        cell.target_dataset,
        cell.topic,
        cell.target_concepts,
        args.eval_target_n,
        args.eval_untargeted_n,
        args.seed + cell.requested_k + 50000,
    )
    for key, value in meta.items():
        if isinstance(value, (str, int, float)):
            frame[key] = value
    frame["cell_id"] = cell.id
    frame["best_layer"] = summary["best_layer"]
    frame["best_scale"] = summary["best_scale"]
    prompts = build_prompts(frame, pool.template)
    layer = int(summary["best_layer"])
    scale = float(summary["best_scale"])
    steering = TransferSteering([layer], [layer], v_detect, v_refuse, thresholds)
    jobs = make_generation_jobs(frame, prompts, target_col="steer_target", scales=[scale])
    raw = run_jobs(
        pool,
        jobs,
        steering,
        generation_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "intervention_start": "prefill",
        },
        batch_size=args.batch_size,
        trim_fn=pool.template.trim_to_last_assistant,
        result_metadata={"source_layer": str([layer]), "target_layer": str([layer])},
    )
    raw["cheap_refusal"] = raw["model_output"].map(cheap_refusal)
    raw["cheap_fluency"] = raw["model_output"].map(cheap_fluency)
    raw.to_csv(raw_path, index=False)
    return raw


def validate_topics():
    missing = []
    for topic, datasets in TOPICS.items():
        for dataset, concepts in datasets.items():
            frame = load_frame(dataset, "train")
            present = set(frame["concept"])
            for concept in concepts:
                if concept not in present:
                    missing.append((topic, dataset, concept))
    if missing:
        raise ValueError(f"missing configured concepts: {missing}")


def build_cells(mode: str, sample_sizes: list[int]) -> list[Cell]:
    topic_names = ["space"] if mode == "scout" else list(TOPICS)
    cells = []
    for topic in topic_names:
        datasets = TOPICS[topic]
        for requested_k in sample_sizes:
            for source_dataset, source_concepts in datasets.items():
                for target_dataset, target_concepts in datasets.items():
                    cells.append(
                        Cell(
                            topic=topic,
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            requested_k=requested_k,
                            source_concepts=tuple(source_concepts),
                            target_concepts=tuple(target_concepts),
                        )
                    )
    return cells


def configure_args(args):
    if args.mode == "scout":
        args.sample_sizes = [5] if args.sample_sizes is None else args.sample_sizes
        args.layers = [5, 8, 9] if args.layers is None else args.layers
        args.scales = [4.0, 8.0, 10.0] if args.scales is None else args.scales
        args.top_layers_by_gate = min(args.top_layers_by_gate, len(args.layers))
        args.calib_target_n = min(args.calib_target_n, 2)
        args.calib_untargeted_n = min(args.calib_untargeted_n, 2)
        args.eval_target_n = min(args.eval_target_n, 4)
        args.eval_untargeted_n = min(args.eval_untargeted_n, 4)
    else:
        args.sample_sizes = (
            [2, 4, 8, 16, 32, 64, 128, 256, 512]
            if args.sample_sizes is None
            else args.sample_sizes
        )
        args.layers = list(range(16)) if args.layers is None else args.layers
        args.scales = [float(i) for i in range(1, 11)] if args.scales is None else args.scales
    return args


def run(args):
    load_env(ROOT / ".env")
    args = configure_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    n_threads = os.cpu_count() or 1
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    t.set_num_threads(n_threads)

    validate_topics()
    run_dir = ROOT / "debug" / "cross_dataset_transfer" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cells").mkdir(exist_ok=True)
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"func"} and isinstance(value, (str, int, float, bool, list, type(None)))
    }
    write_json(run_dir / "config.json", config)

    cells = build_cells(args.mode, args.sample_sizes)
    print(f"[transfer] run={args.run_name} mode={args.mode} cells={len(cells)}", flush=True)
    print(f"[transfer] gpus={args.gpus} judge_gpus={args.judge_gpus}", flush=True)

    template = detect_template(MODEL_PATH)
    pool = GPUPool.from_model_path(MODEL_PATH, args.gpus, template=template, hf_token=os.getenv("HF_TOKEN"))
    cell_summaries = []
    eval_parts = []
    for index, cell in enumerate(cells, 1):
        print(f"[transfer] cell {index}/{len(cells)} {cell.id}", flush=True)
        direction = build_or_load_direction(cell, run_dir, args.seed, args.lda_device)
        _, summary = generate_cell_calibration(pool, cell, run_dir, args, direction)
        cell_summaries.append(summary)
        eval_parts.append(generate_cell_eval(pool, cell, run_dir, args, direction, summary))
        append_or_replace(run_dir / "cell_summary.csv", [summary], ["cell_id"])
    del pool
    if t.cuda.is_available():
        t.cuda.empty_cache()

    eval_raw = pd.concat(eval_parts, ignore_index=True)
    eval_raw.to_csv(run_dir / "eval_raw.csv", index=False)

    if not args.skip_judge:
        judge_template = detect_template(JUDGE_MODEL)
        judge_pool = GPUPool.from_model_path(
            JUDGE_MODEL,
            args.judge_gpus,
            template=judge_template,
            hf_token=os.getenv("HF_TOKEN"),
        )
        judged = add_judge_scores(
            judge_pool,
            eval_raw,
            cache_path=run_dir / "eval_judged.csv",
            batch_size=args.judge_batch_size,
            mode="logit",
            show_progress=True,
        )
        del judge_pool
    else:
        judged = eval_raw.copy()
        judged["judge_refusal"] = judged["cheap_refusal"]
        judged["judge_retention"] = 0
        judged["judge_fluency"] = judged["cheap_fluency"]
        judged.to_csv(run_dir / "eval_judged.csv", index=False)

    summarize(run_dir)
    plot(run_dir)
    print(f"[transfer] done {run_dir}", flush=True)


def summarize(run_dir: Path):
    judged_path = run_dir / "eval_judged.csv"
    if not judged_path.exists():
        return
    df = pd.read_csv(judged_path)
    group_cols = [
        "cell_id",
        "topic",
        "source_dataset",
        "target_dataset",
        "requested_k",
        "actual_k",
        "best_layer",
        "best_scale",
        "eval_group",
    ]
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            refusal=("judge_refusal", "mean"),
            retention=("judge_retention", "mean"),
            fluency=("judge_fluency", "mean"),
            n=("judge_refusal", "size"),
            cheap_refusal=("cheap_refusal", "mean"),
        )
    )
    pivot = grouped.pivot_table(
        index=[
            "cell_id",
            "topic",
            "source_dataset",
            "target_dataset",
            "requested_k",
            "actual_k",
            "best_layer",
            "best_scale",
        ],
        columns="eval_group",
        values=["refusal", "retention", "fluency", "cheap_refusal", "n"],
        fill_value=0,
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns]
    pivot = pivot.reset_index()
    for col in ["refusal_target", "refusal_untargeted", "retention_untargeted", "fluency_target"]:
        if col not in pivot:
            pivot[col] = 0.0
    pivot["transfer_score"] = pivot["refusal_target"] * (1.0 - pivot["refusal_untargeted"]) * pivot["fluency_target"]
    pivot["is_native"] = pivot["source_dataset"] == pivot["target_dataset"]
    pivot.to_csv(run_dir / "transfer_summary.csv", index=False)


def plot(run_dir: Path):
    summary_path = run_dir / "transfer_summary.csv"
    if not summary_path.exists():
        summarize(run_dir)
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    df["selectivity"] = df["refusal_target"] - df["refusal_untargeted"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    sns.set_theme(context="talk", style="ticks")

    # 1. Sample-efficiency curves: off-diagonal transfer only.
    off = df[~df["is_native"]].copy()
    if not off.empty:
        curve = (
            off.groupby(["topic", "requested_k"], as_index=False)
            .agg(
                refusal_target=("refusal_target", "mean"),
                refusal_untargeted=("refusal_untargeted", "mean"),
                transfer_score=("transfer_score", "mean"),
            )
        )
        plt.figure(figsize=(7, 4.5), dpi=180)
        sns.lineplot(data=curve, x="requested_k", y="transfer_score", hue="topic", marker="o")
        plt.xscale("log", base=2)
        plt.ylim(0, 1)
        plt.xlabel("source examples")
        plt.ylabel("transfer score")
        plt.tight_layout()
        plt.savefig(fig_dir / "sample_efficiency_transfer_score.png")
        plt.close()

        plt.figure(figsize=(7, 4.5), dpi=180)
        sns.lineplot(data=curve, x="requested_k", y="refusal_target", hue="topic", marker="o")
        plt.xscale("log", base=2)
        plt.ylim(0, 1)
        plt.xlabel("source examples")
        plt.ylabel("targeted refusal")
        plt.tight_layout()
        plt.savefig(fig_dir / "sample_efficiency_targeted_refusal.png")
        plt.close()

    # 1b. Sample efficiency with native and transfer shown together.
    cond = df.copy()
    cond["condition"] = cond["is_native"].map({True: "native", False: "transfer"})
    cond_curve = (
        cond.groupby(["condition", "requested_k"], as_index=False)
        .agg(
            refusal_target=("refusal_target", "mean"),
            refusal_untargeted=("refusal_untargeted", "mean"),
            selectivity=("selectivity", "mean"),
        )
    )
    metrics = [
        ("refusal_target", "targeted refusal", (0, 1)),
        ("refusal_untargeted", "untargeted refusal", (0, 1)),
        ("selectivity", "target - untargeted", (-0.25, 1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=180, sharex=True)
    palette = {"native": "#555555", "transfer": "#980000"}
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for condition in ["native", "transfer"]:
            d = cond_curve[cond_curve["condition"] == condition]
            ax.plot(
                d["requested_k"],
                d[metric],
                marker="o",
                linewidth=2,
                markersize=5,
                color=palette[condition],
                label=condition,
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(*ylim)
        ax.set_xlabel("source examples")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "sample_efficiency_native_transfer.png")
    plt.close()

    # 2. Native vs transfer by topic at the largest requested K.
    max_k = int(df["requested_k"].max())
    top = df[df["requested_k"] == max_k].copy()
    if not top.empty:
        comp = (
            top.groupby(["topic", "is_native"], as_index=False)
            .agg(refusal_target=("refusal_target", "mean"), transfer_score=("transfer_score", "mean"))
        )
        comp["condition"] = comp["is_native"].map({True: "native", False: "transfer"})
        plt.figure(figsize=(8, 4.5), dpi=180)
        sns.barplot(data=comp, x="topic", y="refusal_target", hue="condition", edgecolor="black")
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("targeted refusal")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "native_vs_transfer_targeted_refusal.png")
        plt.close()

        # 3. Transfer selectivity matrices at largest K.
        dataset_labels = {
            "conceptvectors": "CV",
            "inhouse": "inhouse",
            "mmlu": "MMLU",
            "rwku": "RWKU",
        }
        topics = sorted(top["topic"].unique())
        ncols = 3
        nrows = int(np.ceil(len(topics) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.8, max(3.0, 2.55 * nrows)), dpi=180)
        axes = np.atleast_1d(axes).ravel()
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        for ax in axes[len(topics) :]:
            ax.axis("off")
        cbar_ax = fig.add_axes([0.92, 0.23, 0.018, 0.56])
        for idx, topic in enumerate(topics):
            ax = axes[idx]
            d = top[top["topic"] == topic]
            mat = d.pivot_table(
                index="source_dataset",
                columns="target_dataset",
                values="selectivity",
                aggfunc="mean",
            )
            mat = mat.rename(index=dataset_labels, columns=dataset_labels)
            sns.heatmap(
                mat,
                ax=ax,
                vmin=-0.5,
                vmax=1,
                center=0,
                cmap=cmap,
                cbar=idx == 0,
                cbar_ax=cbar_ax if idx == 0 else None,
                annot=True,
                fmt=".2f",
                linewidths=1,
                linecolor="black",
                square=True,
                annot_kws={"fontsize": 9},
            )
            ax.set_title(topic, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=0, labelsize=9)
            ax.tick_params(axis="y", labelrotation=0, labelsize=9)
        cbar_ax.set_ylabel("target - untargeted refusal", fontsize=9)
        cbar_ax.tick_params(labelsize=9)
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.9, wspace=0.4, hspace=0.55)
        plt.savefig(fig_dir / "topic_transfer_matrices_max_k.png")
        plt.close()

        # 4. Off-diagonal transfer feasibility at largest K.
        off_top = top[~top["is_native"]].copy()
        if not off_top.empty:
            plt.figure(figsize=(5.2, 4.8), dpi=180)
            ax = sns.scatterplot(
                data=off_top,
                x="refusal_untargeted",
                y="refusal_target",
                hue="topic",
                s=70,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.2, linestyle="--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("untargeted refusal")
            ax.set_ylabel("targeted refusal")
            ax.legend(frameon=False, fontsize=9, title="", loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(fig_dir / "transfer_refusal_vs_untargeted_max_k.png")
            plt.close()


def status(args):
    run_dir = ROOT / "debug" / "cross_dataset_transfer" / "runs" / args.run_name
    if not run_dir.exists():
        print(f"missing run: {run_dir}")
        return
    cells = list((run_dir / "cells").glob("*.calibration_summary.json"))
    evals = list((run_dir / "cells").glob("*.eval_raw.csv"))
    print(f"run_dir: {run_dir}")
    print(f"calibrated cells: {len(cells)}")
    print(f"evaluated cells: {len(evals)}")
    for name in ["cell_summary.csv", "eval_raw.csv", "eval_judged.csv", "transfer_summary.csv"]:
        p = run_dir / name
        if p.exists():
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = "?"
            print(f"{name}: present rows={n}")
        else:
            print(f"{name}: missing")


def parse_ints(text: str | None):
    if text is None:
        return None
    return [int(x) for x in text.replace(",", " ").split()]


def parse_floats(text: str | None):
    if text is None:
        return None
    return [float(x) for x in text.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--run-name", required=True)
    run_p.add_argument("--mode", choices=["scout", "full"], default="full")
    run_p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    run_p.add_argument("--judge-gpus", nargs="+", type=int, default=[0, 1])
    run_p.add_argument("--batch-size", type=int, default=32)
    run_p.add_argument("--judge-batch-size", type=int, default=8)
    run_p.add_argument("--max-new-tokens", type=int, default=48)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--calib-target-n", type=int, default=4)
    run_p.add_argument("--calib-untargeted-n", type=int, default=4)
    run_p.add_argument("--eval-target-n", type=int, default=16)
    run_p.add_argument("--eval-untargeted-n", type=int, default=16)
    run_p.add_argument("--sample-sizes", type=parse_ints, default=None)
    run_p.add_argument("--layers", type=parse_ints, default=None)
    run_p.add_argument("--scales", type=parse_floats, default=None)
    run_p.add_argument("--lda-device", default="cuda:0")
    run_p.add_argument("--top-layers-by-gate", type=int, default=4)
    run_p.add_argument("--skip-judge", action="store_true")
    run_p.set_defaults(func=run)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--run-name", required=True)
    plot_p.set_defaults(func=lambda args: plot(ROOT / "debug" / "cross_dataset_transfer" / "runs" / args.run_name))

    status_p = sub.add_parser("status")
    status_p.add_argument("--run-name", required=True)
    status_p.set_defaults(func=status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
