from ..intervention import run_jobs

DEFAULT_GENERATION = {"max_new_tokens": 64, "do_sample": False, "temperature": 1.0}


def build_prompts(df, system_prompt, template):
    return [template.render(system_prompt, row.question) for row in df.itertuples(index=False)]


def run_eval(pool, jobs, steering, template, *, batch_size=128, result_metadata=None):
    return run_jobs(
        pool, jobs, steering,
        generation_kwargs=DEFAULT_GENERATION,
        batch_size=batch_size,
        trim_fn=template.trim_to_last_assistant,
        result_metadata=result_metadata,
    )
