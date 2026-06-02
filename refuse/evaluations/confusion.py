import random

from ..intervention import make_generation_jobs, sample_per_concept
from .base import build_prompts, run_eval


def run_confusion(pool, baseline_test, steering, scale, *, c, n,
                  system_prompt, template, batch_size=128,
                  result_metadata=None, seed=42):
    all_concepts = baseline_test["concept"].unique().tolist()
    rng = random.Random(seed)
    concepts = rng.sample(all_concepts, c) if c < len(all_concepts) else all_concepts
    df = baseline_test[baseline_test["concept"].isin(concepts)]
    df_gen = sample_per_concept(df, n_per_concept=n, random_state=seed).reset_index(drop=True)
    prompts = build_prompts(df_gen, system_prompt, template)
    jobs = make_generation_jobs(df_gen, prompts, targets=concepts, scales=[scale])
    return run_eval(pool, jobs, steering, template,
                    batch_size=batch_size, result_metadata=result_metadata)
