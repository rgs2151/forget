import pandas as pd

from ..intervention import make_generation_jobs
from .base import build_prompts, run_eval


def run_bars(pool, baseline_test, steering, scale, *, n,
             system_prompt, template, batch_size=128,
             result_metadata=None, seed=42, intervention_start="assistant"):
    concepts = baseline_test["concept"].unique().tolist()
    parts = []
    for i, target in enumerate(concepts):
        target_pool = baseline_test[baseline_test["concept"] == target]
        other_pool = baseline_test[baseline_test["concept"] != target]
        target_sample = target_pool.sample(n=min(n, len(target_pool)), random_state=seed + i)
        other_sample = other_pool.sample(n=min(n, len(other_pool)), random_state=seed + i + 10000)
        df_sub = pd.concat([target_sample, other_sample], ignore_index=True)
        df_sub["target"] = target
        parts.append(df_sub)
    df_all = pd.concat(parts, ignore_index=True)
    prompts = build_prompts(df_all, system_prompt, template)
    jobs = make_generation_jobs(df_all, prompts, target_col="target", scales=[scale])
    return run_eval(pool, jobs, steering, template,
                    batch_size=batch_size, result_metadata=result_metadata,
                    intervention_start=intervention_start)
