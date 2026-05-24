from .intervention import load_or_empty_results, make_generation_jobs


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


def select_scale(
    pool,
    df,
    scales,
    steering,
    system_prompt,
    template,
    *,
    n_samples=20,
    cache_path=None,
    batch_size=128,
    max_new_tokens=64,
    target_col="concept",
    refusal_string="I don't know.",
    random_state=42,
    result_metadata=None,
):
    if cache_path is not None and cache_path.exists():
        results = load_or_empty_results(cache_path, text_columns=["model_output"])
    else:
        sample = df.sample(n=min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)
        prompts = [
            template.render(system_prompt, row.question)
            for row in sample.itertuples(index=False)
        ]
        jobs = make_generation_jobs(
            sample,
            prompts,
            scales=scales,
            target_col=target_col,
        )
        results = pool.run_jobs(
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

    return select_refusal_scale(results, refusal_string=refusal_string)
