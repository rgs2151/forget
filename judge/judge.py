import re

import pandas as pd

from .prompts import JUDGE_RUBRIC_TEMPLATE, JUDGE_SYSTEM, RATING_MAX, RATING_MIN


DEFAULT_RATING = float("nan")
AXES = ("refusal", "retention", "fluency")
_RATING_RE = re.compile(
    r"(refusal|retention|fluency)\s*[:=]\s*\[*\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_three_scores(completion):
    if not isinstance(completion, str):
        return {axis: DEFAULT_RATING for axis in AXES}
    found = {axis: DEFAULT_RATING for axis in AXES}
    for match in _RATING_RE.finditer(completion):
        axis = match.group(1).lower()
        value = float(match.group(2))
        if RATING_MIN <= value <= RATING_MAX:
            found[axis] = value / RATING_MAX
    return found


def _harmonic_mean(scores):
    if any((s != s) or s <= 0 for s in scores):
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def _needs_judge(text):
    if not isinstance(text, str) or pd.isna(text):
        return True
    parsed = _parse_three_scores(text)
    return any(pd.isna(v) for v in parsed.values())


def add_judge_scores(
    pool,
    df,
    *,
    prediction_col="model_output",
    reference_col="baseline_output",
    question_col="question",
    completion_col="judge_completion",
    cache_path=None,
    batch_size=64,
    max_new_tokens=64,
    max_retries=2,
    retry_temperature=0.7,
    show_progress=True,
):
    template = pool.template

    def render_prompts(rows_df):
        prompts = []
        for row in rows_df.itertuples(index=False):
            user = JUDGE_RUBRIC_TEMPLATE.format(
                question=getattr(row, question_col),
                reference=getattr(row, reference_col),
                answer=getattr(row, prediction_col),
            )
            prompts.append(template.render(JUDGE_SYSTEM, user))
        return prompts

    def run_pass(rows_df, do_sample, temperature):
        prompts = render_prompts(rows_df)
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

    df = df.copy()
    if completion_col not in df.columns:
        df[completion_col] = pd.NA
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(df) and completion_col in cached.columns:
            df[completion_col] = cached[completion_col].to_numpy()

    for attempt in range(max_retries + 1):
        missing_mask = df[completion_col].apply(_needs_judge)
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            break
        if show_progress:
            label = "first pass" if attempt == 0 else f"retry {attempt}/{max_retries}"
            print(f"[judge] {label}: {n_missing} rows", flush=True)
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
        print(f"[judge] WARNING: {final_failures} rows unparseable after {max_retries} retries", flush=True)

    parsed_list = df[completion_col].apply(_parse_three_scores).tolist()
    for axis in AXES:
        df[f"judge_{axis}"] = [p[axis] for p in parsed_list]
    df["judge_aggregate"] = [
        _harmonic_mean([row[f"judge_{a}"] for a in AXES])
        for _, row in df.iterrows()
    ]

    if cache_path is not None:
        df.to_csv(cache_path, index=False)
    return df
