# Parallel Logit Judge OOM Probe

Suspicion: logit scoring failed because the judge wrapper retained full layer activations while scoring.
Test: score 2560 real calibration judge prompts with AtlaAI/Selene-1-Mini-Llama-3.1-8B on GPUs [0, 1], batch size 8.
Result: completed all chunks without OOM.
Summary: `debug/judge_logit_oom/runs/parallel_pool_blocking_long_v1/summary.csv`
