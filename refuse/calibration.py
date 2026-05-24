from .intervention import load_or_empty_results, make_generation_jobs, run_jobs


def select_refusal_scale(results, score_col="judge_refusal", label="intervention"):
    df = results.copy()
    if "label" in df:
        df = df[df["label"] == label]
    if df.empty:
        raise ValueError("No rows available for scale selection.")
    rates = df.groupby("scale", as_index=False)[score_col].mean()
    rates = rates.sort_values([score_col, "scale"], ascending=[False, True])
    return rates.iloc[0]["scale"]


def calibration_generate(
    pool,
    df,
    scales,
    steering,
    system_prompt,
    template,
    *,
    sample_frac=0.10,
    cache_path=None,
    batch_size=128,
    max_new_tokens=64,
    target_col="concept",
    random_state=42,
    result_metadata=None,
):
    if cache_path is not None and cache_path.exists():
        return load_or_empty_results(cache_path, text_columns=["model_output"])
    n = max(1, int(round(len(df) * sample_frac)))
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    prompts = [
        template.render(system_prompt, row.question)
        for row in sample.itertuples(index=False)
    ]
    jobs = make_generation_jobs(sample, prompts, scales=scales, target_col=target_col)
    results = run_jobs(
        pool,
        jobs,
        steering,
        generation_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
        },
        batch_size=batch_size,
        trim_fn=template.trim_to_last_assistant,
        result_metadata=result_metadata,
    )
    if cache_path is not None:
        results.to_csv(cache_path, index=False)
    return results


