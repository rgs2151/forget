import pandas as pd
from tqdm.auto import tqdm

from steering.steering import GatedSteer


def _normalize_layers(spec):
    if isinstance(spec, int):
        return [spec]
    return list(spec)


def _to_float(x):
    return float(x.item() if hasattr(x, "item") else x)


def _job_id(label, target, scale, prompt_index):
    return f"{label}|{target}|{scale}|{prompt_index}"


class Steering:
    def __init__(self, source_layers, target_layers):
        self.source_layers = _normalize_layers(source_layers)
        self.target_layers = _normalize_layers(target_layers)
        if len(self.source_layers) != len(self.target_layers):
            raise ValueError("source_layers and target_layers must have the same length.")

    def apply(self, llm, target, scale):
        for src, tgt in zip(self.source_layers, self.target_layers):
            llm.set_steering_op(tgt, self._make_op(src, target, scale, llm.device))

    def _make_op(self, src_layer, target, scale, device):
        raise NotImplementedError


class GatedSteering(Steering):
    def __init__(self, source_layers, target_layers, v_detect, v_steer, thresholds):
        super().__init__(source_layers, target_layers)
        self.v_detect = v_detect
        self.v_steer = v_steer
        self.thresholds = thresholds

    def _make_op(self, src_layer, target, scale, device):
        if target not in self.v_detect:
            raise ValueError(f"Missing detect vectors for target {target!r}.")
        if target not in self.thresholds:
            raise ValueError(f"Missing thresholds for target {target!r}.")
        return GatedSteer(
            v_detect=self.v_detect[target][src_layer].to(device),
            v_steer=self.v_steer[src_layer].to(device),
            tau=_to_float(self.thresholds[target][src_layer]),
            scale=scale,
        )


def sample_per_concept(df, n_per_concept=None, random_state=42, concept_col="concept"):
    if n_per_concept is None:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby(concept_col):
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


def run_generation_jobs(
    llm,
    jobs,
    steering,
    generation_kwargs=None,
    trim_output_fn=lambda text: text,
    batch_size=64,
    result_metadata=None,
):
    jobs = jobs.reset_index(drop=True)
    result_metadata = dict(result_metadata or {})
    gen_kwargs = dict(generation_kwargs or {})
    gen_kwargs.setdefault("max_new_tokens", 64)
    gen_kwargs.setdefault("do_sample", False)
    gen_kwargs.setdefault("temperature", 1.0)

    rows = []
    groups = list(jobs.groupby(["label", "target", "scale"], sort=False))
    for (_label, target, scale), group in tqdm(groups, desc="Generation runs"):
        for start in range(0, len(group), batch_size):
            batch_jobs = group.iloc[start:start + batch_size]
            batch_prompts = batch_jobs["prompt"].tolist()

            llm.reset_all()
            steering.apply(llm, target, scale)
            outputs = llm.batch_generate(batch_prompts, **gen_kwargs)
            llm.reset_all()

            for job, raw in zip(batch_jobs.to_dict("records"), outputs):
                job.pop("prompt")
                rows.append({
                    **job,
                    **result_metadata,
                    "model_output": trim_output_fn(raw),
                })

    return pd.DataFrame(rows)


def _split_jobs_for_gpus(jobs, n_gpus):
    if n_gpus <= 1:
        return [jobs.reset_index(drop=True)]
    prompt_indices = pd.Series(jobs["prompt_index"].drop_duplicates().tolist())
    shards = []
    for i in range(n_gpus):
        gpu_indices = prompt_indices.iloc[i::n_gpus]
        gpu_jobs = jobs[jobs["prompt_index"].isin(gpu_indices)].reset_index(drop=True)
        shards.append(gpu_jobs)
    return shards


def run_jobs(pool, jobs, steering, generation_kwargs=None, batch_size=64,
             trim_fn=None, result_metadata=None):
    shards = _split_jobs_for_gpus(jobs, len(pool.gpu_ids))

    def run(llm, jobs_shard):
        return run_generation_jobs(
            llm,
            jobs_shard,
            steering,
            generation_kwargs=generation_kwargs,
            trim_output_fn=trim_fn or (lambda x: x),
            batch_size=batch_size,
            result_metadata=result_metadata,
        )

    results = pool.map(run, shards)
    merged = pd.concat(results, ignore_index=True)
    return merged.sort_values(["prompt_index", "target", "scale"]).reset_index(drop=True)
