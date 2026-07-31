from .paths import cached_csv_rows
from .prompts import BASELINE_SYSTEM


def generate_baseline(
    pool,
    df,
    csv_path,
    template,
    system_prompt=BASELINE_SYSTEM,
    batch_size=64,
    max_new_tokens=64,
):
    def compute_missing(batch_df):
        prompts = [
            template.render(system_prompt, row.question)
            for row in batch_df.itertuples(index=False)
        ]
        return pool.generate(
            prompts,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
            },
            batch_size=batch_size,
            trim_fn=template.trim_to_last_assistant,
            show_progress=False,
        )

    return cached_csv_rows(
        csv_path,
        df,
        compute_missing,
        key_col="baseline_output",
        batch_size=batch_size * len(pool),
        desc=csv_path.stem,
    )
