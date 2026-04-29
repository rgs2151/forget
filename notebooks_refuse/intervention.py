from dataclasses import dataclass

import pandas as pd
import torch
from tqdm.auto import tqdm

from forget.model.steering import GatedSteer

from activations import build_question_prompts


@dataclass(frozen=True)
class RunSpec:
    label: str
    target: str
    source_layer: object
    target_layer: object
    scale: float


def normalize_layers(layer):
    if isinstance(layer, int):
        return [layer]
    return list(layer)


def sample_per_concept(df, n_per_concept=None, random_state=42):
    if n_per_concept is None:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby("concept"):
        parts.append(group.sample(min(len(group), n_per_concept), random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def load_or_empty_results(csv_path, text_columns=None):
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for column in text_columns or []:
        if column in df:
            df[column] = df[column].fillna("")
    return df


def make_run_specs(objective_df, baseline_scale=0.0, steered_label="steered", baseline_label="baseline"):
    specs = []
    for row in objective_df.itertuples(index=False):
        specs.append(RunSpec(
            label=baseline_label,
            target=row.target,
            source_layer=row.source_layer,
            target_layer=row.target_layer,
            scale=baseline_scale,
        ))
        specs.append(RunSpec(
            label=steered_label,
            target=row.target,
            source_layer=row.source_layer,
            target_layer=row.target_layer,
            scale=row.best_scale,
        ))
    return specs


def _run_done(df, spec):
    if df.empty:
        return False
    mask = (
        (df["label"] == spec.label)
        & (df["target"] == spec.target)
        & (df["source_layer"].astype(str) == str(spec.source_layer))
        & (df["target_layer"].astype(str) == str(spec.target_layer))
        & (df["scale"] == spec.scale)
    )
    return mask.any()


def evaluate_qa_generation_batched(
    llm,
    df,
    prompts,
    target,
    steer_factory,
    source_layer,
    target_layer,
    scale,
    batch_size=32,
    generation_kwargs=None,
    trim_output_fn=None,
    show_progress=False,
    progress_desc="QA batches",
):
    src_layers = normalize_layers(source_layer)
    tgt_layers = normalize_layers(target_layer)
    assert len(src_layers) == len(tgt_layers)

    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("max_new_tokens", 128)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("temperature", 1.0)

    all_outputs = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    ):
        batch_prompts = prompts[start:start + batch_size]
        llm.reset_all()
        for source_layer_item, target_layer_item in zip(src_layers, tgt_layers):
            llm.set_steering_op(target_layer_item, steer_factory(source_layer_item, scale))
        outputs = llm.batch_generate(batch_prompts, **generation_kwargs)
        all_outputs.extend(outputs)
        llm.reset_all()

    rows = []
    for (_, row), raw in zip(df.iterrows(), all_outputs):
        response = trim_output_fn(raw) if trim_output_fn is not None else raw
        rows.append({
            "concept": row["concept"],
            "target": target,
            "question": row["question"],
            "correct_answer": row["answer"],
            "scale": scale,
            "source_layer": src_layers if len(src_layers) > 1 else src_layers[0],
            "target_layer": tgt_layers if len(tgt_layers) > 1 else tgt_layers[0],
            "label": None,
            "model_output": response,
        })
    return pd.DataFrame(rows)


def _normalize_tau(tau):
    if isinstance(tau, torch.Tensor) and tau.numel() == 1:
        return float(tau.item())
    return tau


def run_intervention_generations(
    llm,
    df,
    prompts,
    run_specs,
    csv_path,
    detect_vectors_by_target=None,
    steering_vector=None,
    thresholds_by_target=None,
    generation_kwargs=None,
    trim_output_fn=None,
    batch_size=64,
):
    results_df = load_or_empty_results(csv_path, text_columns=["model_output"])
    if len(prompts) != len(df):
        raise ValueError("prompts and df must have the same length.")

    detect_vectors_by_target = detect_vectors_by_target or {}
    thresholds_by_target = thresholds_by_target or {}

    pending_specs = [spec for spec in run_specs if not _run_done(results_df, spec)]
    generation_kwargs_local = dict(generation_kwargs or {})
    generation_kwargs_local.setdefault("max_new_tokens", 128)
    generation_kwargs_local.setdefault("do_sample", False)
    generation_kwargs_local.setdefault("temperature", 1.0)

    for spec in tqdm(pending_specs, desc="Intervention generation runs"):
        src_layers = normalize_layers(spec.source_layer)
        tgt_layers = normalize_layers(spec.target_layer)
        assert len(src_layers) == len(tgt_layers)

        all_outputs = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            llm.reset_all()
            if spec.scale != 0:
                if steering_vector is None:
                    raise ValueError("steering_vector is required for non-baseline runs.")
                if spec.target not in detect_vectors_by_target:
                    raise ValueError(f"Missing detect vectors for target {spec.target!r}.")
                if spec.target not in thresholds_by_target:
                    raise ValueError(f"Missing thresholds for target {spec.target!r}.")
                target_detect = detect_vectors_by_target[spec.target]
                target_thresholds = thresholds_by_target[spec.target]
                for source_layer_item, target_layer_item in zip(src_layers, tgt_layers):
                    llm.set_steering_op(
                        target_layer_item,
                        GatedSteer(
                            v_detect=target_detect[source_layer_item].to(llm.device),
                            v_steer=steering_vector[source_layer_item].to(llm.device),
                            tau=_normalize_tau(target_thresholds[source_layer_item]),
                            scale=spec.scale,
                        ),
                    )
            outputs = llm.batch_generate(batch_prompts, **generation_kwargs_local)
            all_outputs.extend(outputs)
            llm.reset_all()

        rows = []
        for prompt_index, ((_, row), raw) in enumerate(zip(df.iterrows(), all_outputs)):
            response = trim_output_fn(raw) if trim_output_fn is not None else raw
            rows.append({
                "prompt_index": prompt_index,
                "concept": row["concept"],
                "target": spec.target,
                "question": row["question"],
                "answer": row["answer"] if "answer" in row.index else None,
                "scale": spec.scale,
                "source_layer": src_layers if len(src_layers) > 1 else src_layers[0],
                "target_layer": tgt_layers if len(tgt_layers) > 1 else tgt_layers[0],
                "label": spec.label,
                "model_output": response,
            })
        run_df = pd.DataFrame(rows)
        run_df["label"] = spec.label
        results_df = pd.concat([results_df, run_df], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    return results_df


def run_qa_benchmark(
    llm,
    df,
    prompt_factory,
    run_specs,
    steer_factory_fn,
    csv_path,
    generation_kwargs=None,
    trim_output_fn=None,
    batch_size=64,
    question_col="question",
):
    results_df = load_or_empty_results(csv_path, text_columns=["model_output"])
    prompts = build_question_prompts(df, prompt_factory, question_col=question_col)
    pending_specs = [spec for spec in run_specs if not _run_done(results_df, spec)]

    for spec in tqdm(pending_specs, desc="QA benchmark runs"):
        factory = steer_factory_fn(spec.target)
        run_df = evaluate_qa_generation_batched(
            llm,
            df,
            prompts,
            target=spec.target,
            steer_factory=factory,
            source_layer=spec.source_layer,
            target_layer=spec.target_layer,
            scale=spec.scale,
            batch_size=batch_size,
            generation_kwargs=generation_kwargs,
            trim_output_fn=trim_output_fn,
        )
        run_df["label"] = spec.label
        results_df = pd.concat([results_df, run_df], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    return results_df
