# refuse

Refusal-vector steering pipeline. See [`design.md`](design.md) for the per-module map and the top-level [`../readme.md`](../readme.md) for the cross-package architecture and CLI reference.

## Data fractions

CLI knobs control how much data flows through each stage.

```
df_train (full)
  │
  ├─ [--train-frac]  ──► per-concept subsample
  │   This is what's used to fit the LDA. Default 1.0 = keep everything.
  │
  └─ baseline_train (kept set)
       │
       └─ activations on baseline + refusal answers, then LDA fit
           Produces v_detect, v_refuse, thresholds. No further subsampling.
```

```
df_test (full)
  │
  ├─ [--test-frac]  ──► per-concept subsample
  │   This is what "exists" downstream. Default 1.0 = keep everything.
  │
  └─ baseline_test (kept set)
       │
       ├─ [--calibration-frac]  ──► random subsample × 30 scales
       │   Generate steered output, judge, pick best scale.
       │   Default 0.1 = 10% of kept set.
       │
       └─ evaluations (zero or more):
            ├─ [--confusion C N]   c × c × n grid (all targets × subsampled concepts × n questions)
            └─ [--bars N]          per target: n target-concept + n untargeted-pool questions
```

`--train-frac` and `--test-frac` are mainly for debug — they shrink the data that gets baseline-generated and activation-collected. For production runs leave them at 1.0 and pick at least one of `--confusion` / `--bars` to actually evaluate.

## Original spec

The nine-step pipeline that drove the initial design:

1. Organize path of stores
2. Load the model to N GPUs
3. Take the train and test dfs, divide across GPUs, generate baseline responses
4. Generate the contrastive refusal response and capture activations under both
5. Compute the vectors via the chosen method (LDA / diffed / projected) and store them
5.5. Sweep `s` over a small sample to find the optimal steering scale
6. Build steering classes from the calculated vectors and push them to GPU
7. Run the test df through the prepared model and save the results
8. Score the steered outputs (now via LLM-as-judge)
9. Render diagnostic plots from the scored CSV
