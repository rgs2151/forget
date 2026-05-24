import re

import pandas as pd

from .prompts import (
    FLUENCY_TEMPLATE,
    JUDGE_SYSTEM,
    REFUSAL_TEMPLATE,
    RETENTION_TEMPLATE,
    RATING_VALUES,
    SCORE_MAP,
)


DEFAULT_RATING = float("nan")
AXES = ("refusal", "retention", "fluency")
AXIS_TEMPLATE = {
    "refusal": REFUSAL_TEMPLATE,
    "retention": RETENTION_TEMPLATE,
    "fluency": FLUENCY_TEMPLATE,
}
_RESULT_RE = re.compile(r"\*?\*?\s*Result:\s*\*?\*?\s*\[?\s*([12])(?![\d.])", re.IGNORECASE)


def _parse_score(completion):
    if not isinstance(completion, str):
        return DEFAULT_RATING
    match = _RESULT_RE.search(completion)
    if match is None:
        return DEFAULT_RATING
    value = int(match.group(1))
    if value in RATING_VALUES:
        return float(SCORE_MAP[value])
    return DEFAULT_RATING


def _needs_judge(text):
    if not isinstance(text, str) or pd.isna(text):
        return True
    return pd.isna(_parse_score(text))


def _harmonic_mean(scores):
    if any((s != s) or s <= 0 for s in scores):
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def _score_axis(pool, df, axis, *, prediction_col, reference_col, question_col,
                cache_path, batch_size, max_new_tokens, max_retries,
                retry_temperature, show_progress):
    template = pool.template
    axis_template = AXIS_TEMPLATE[axis]
    completion_col = f"judge_{axis}_completion"

    def render(rows_df):
        prompts = []
        for row in rows_df.itertuples(index=False):
            user = axis_template.format(
                question=getattr(row, question_col),
                reference=getattr(row, reference_col),
                answer=getattr(row, prediction_col),
            )
            prompts.append(template.render(JUDGE_SYSTEM, user))
        return prompts

    def run_pass(rows_df, do_sample, temperature):
        prompts = render(rows_df)
        return pool.generate(
            prompts,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
            },
            batch_size=batch_size,
            trim_fn=template.trim_to_last_assistant,
            show_progress=show_progress,
        )

    if completion_col not in df.columns:
        df[completion_col] = pd.NA
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if completion_col in cached.columns and len(cached) == len(df):
            df[completion_col] = cached[completion_col].to_numpy()

    for attempt in range(max_retries + 1):
        missing_mask = df[completion_col].apply(_needs_judge)
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            break
        if show_progress:
            label = "first pass" if attempt == 0 else f"retry {attempt}/{max_retries}"
            print(f"[judge:{axis}] {label}: {n_missing} rows", flush=True)
        do_sample = attempt > 0
        completions = run_pass(
            df[missing_mask],
            do_sample=do_sample,
            temperature=retry_temperature if do_sample else 1.0,
        )
        df.loc[missing_mask, completion_col] = completions
        if cache_path is not None:
            df.to_csv(cache_path, index=False)

    final_failures = int(df[completion_col].apply(_needs_judge).sum())
    if final_failures > 0 and show_progress:
        print(f"[judge:{axis}] WARNING: {final_failures} unparseable after {max_retries} retries", flush=True)

    df[f"judge_{axis}"] = df[completion_col].apply(_parse_score)
    return df


def add_judge_scores(
    pool,
    df,
    *,
    prediction_col="model_output",
    reference_col="baseline_output",
    question_col="question",
    cache_path=None,
    batch_size=32,
    max_new_tokens=64,
    max_retries=2,
    retry_temperature=0.7,
    show_progress=True,
):
    df = df.copy()
    for axis in AXES:
        df = _score_axis(
            pool, df, axis,
            prediction_col=prediction_col,
            reference_col=reference_col,
            question_col=question_col,
            cache_path=cache_path,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
            retry_temperature=retry_temperature,
            show_progress=show_progress,
        )

    eps = 1e-9
    df["judge_aggregate"] = (
        2 * df["judge_refusal"] * df["judge_fluency"]
        / (df["judge_refusal"] + df["judge_fluency"] + eps)
    )
    if cache_path is not None:
        df.to_csv(cache_path, index=False)
    return df
