from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm.auto import tqdm


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


def _normalize_tau(tau):
    return float(getattr(tau, "item", lambda: tau)())


def _job_id(label, target, scale, prompt_index):
    return f"{label}|{target}|{scale}|{prompt_index}"


def make_generation_jobs(df, prompts, targets=None, scales=None, target_col=None, label="intervention"):
    if len(prompts) != len(df):
        raise ValueError("prompts and df must have the same length.")
    if scales is None:
        scales = [1.0]
    elif not isinstance(scales, (list, tuple)):
        scales = [scales]

    rows = []
    frame = df.reset_index(drop=True)
    for prompt_index, row in frame.iterrows():
        base = row.to_dict()
        base["prompt_index"] = prompt_index
        base["prompt"] = prompts[prompt_index]

        row_targets = [row[target_col]] if target_col is not None else (targets or [])
        for target in row_targets:
            for scale in scales:
                rows.append({
                    **base,
                    "job_id": _job_id(label, target, scale, prompt_index),
                    "label": label,
                    "target": target,
                    "scale": scale,
                })

    return pd.DataFrame(rows)


def split_jobs_for_gpus(jobs, gpu_ids):
    prompt_indices = pd.Series(jobs["prompt_index"].drop_duplicates().tolist())
    splits = []
    for index, gpu_id in enumerate(gpu_ids):
        gpu_prompt_indices = prompt_indices.iloc[index::len(gpu_ids)]
        gpu_jobs = jobs[jobs["prompt_index"].isin(gpu_prompt_indices)].reset_index(drop=True)
        splits.append((gpu_id, gpu_jobs))
    return splits


def is_refusal_output(text, refusal_string="I don't know."):
    text = str(text).lower().replace("’", "'")
    refusal = refusal_string.lower().replace("’", "'").rstrip(".")
    return refusal in text


def select_refusal_scale(results, refusal_string="I don't know.", label="intervention"):
    df = results.copy()
    if "label" in df:
        df = df[df["label"] == label]
    if df.empty:
        raise ValueError("No rows available for scale selection.")

    scored = df.assign(
        is_refusal=df["model_output"].fillna("").apply(
            lambda text: is_refusal_output(text, refusal_string=refusal_string)
        )
    )
    rates = scored.groupby("scale", as_index=False)["is_refusal"].mean()
    rates = rates.sort_values(["is_refusal", "scale"], ascending=[False, True])
    return rates.iloc[0]["scale"]


def make_gated_steering_factory(source_layer, target_layer, detect_vectors, steer_vector, gate_thresholds):
    src_layers = normalize_layers(source_layer)
    tgt_layers = normalize_layers(target_layer)
    if len(src_layers) != len(tgt_layers):
        raise ValueError("source_layer and target_layer must have the same length.")

    def factory(llm, target, scale):
        from forget.model.steering import GatedSteer

        if target not in detect_vectors:
            raise ValueError(f"Missing detect vectors for target {target!r}.")
        if target not in gate_thresholds:
            raise ValueError(f"Missing gate thresholds for target {target!r}.")

        target_detect = detect_vectors[target]
        target_thresholds = gate_thresholds[target]
        for source_layer_item, target_layer_item in zip(src_layers, tgt_layers):
            llm.set_steering_op(
                target_layer_item,
                GatedSteer(
                    v_detect=target_detect[source_layer_item].to(llm.device),
                    v_steer=steer_vector[source_layer_item].to(llm.device),
                    tau=_normalize_tau(target_thresholds[source_layer_item]),
                    scale=scale,
                ),
            )

    return factory


def run_generation_jobs(
    llm,
    jobs,
    steering_factory,
    generation_kwargs=None,
    trim_output_fn=lambda text: text,
    batch_size=64,
    result_metadata=None,
):
    jobs = jobs.reset_index(drop=True)
    result_metadata = dict(result_metadata or {})
    generation_kwargs_local = dict(generation_kwargs or {})
    generation_kwargs_local.setdefault("max_new_tokens", 128)
    generation_kwargs_local.setdefault("do_sample", False)
    generation_kwargs_local.setdefault("temperature", 1.0)

    rows = []
    groups = list(jobs.groupby(["label", "target", "scale"], sort=False))
    for (label, target, scale), group in tqdm(groups, desc="Generation runs"):
        for start in range(0, len(group), batch_size):
            batch_jobs = group.iloc[start:start + batch_size]
            batch_prompts = batch_jobs["prompt"].tolist()

            llm.reset_all()
            steering_factory(llm, target, scale)
            outputs = llm.batch_generate(batch_prompts, **generation_kwargs_local)
            llm.reset_all()

            for job, raw in zip(batch_jobs.to_dict("records"), outputs):
                job.pop("prompt")
                response = trim_output_fn(raw)
                rows.append({
                    **job,
                    **result_metadata,
                    "model_output": response,
                })

    return pd.DataFrame(rows)


def run_generation_jobs_for_gpu(
    llm_factory,
    gpu_id,
    jobs,
    steering_factory,
    generation_kwargs=None,
    trim_output_fn=lambda text: text,
    batch_size=64,
    result_metadata=None,
):
    llm = llm_factory(gpu_id)
    return run_generation_jobs(
        llm,
        jobs,
        steering_factory=steering_factory,
        generation_kwargs=generation_kwargs,
        trim_output_fn=trim_output_fn,
        batch_size=batch_size,
        result_metadata=result_metadata,
    )


def run_generation_jobs_multi_gpu(
    llm_factory,
    jobs,
    gpu_ids,
    csv_path,
    steering_factory,
    generation_kwargs=None,
    trim_output_fn=lambda text: text,
    batch_size=64,
    result_metadata=None,
):
    gpu_jobs = split_jobs_for_gpus(jobs, gpu_ids)
    with ThreadPoolExecutor(len(gpu_ids)) as executor:
        futures = [
            executor.submit(
                run_generation_jobs_for_gpu,
                llm_factory,
                gpu_id,
                jobs_for_gpu,
                steering_factory=steering_factory,
                generation_kwargs=generation_kwargs,
                trim_output_fn=trim_output_fn,
                batch_size=batch_size,
                result_metadata=result_metadata,
            )
            for gpu_id, jobs_for_gpu in gpu_jobs
        ]
        frames = [future.result() for future in futures]

    results = pd.concat(frames, ignore_index=True)
    results = results.sort_values(["prompt_index", "target", "scale"]).reset_index(drop=True)
    results.to_csv(csv_path, index=False)
    return results
