# Refuse package design

`refuse` owns the experiment pipeline. It should orchestrate stages and cache
artifacts, but it should not hide model runtime details or plotting behavior
inside global state.

## Module map

| Module | Responsibility |
| --- | --- |
| `paths.py` | Store paths and cache helpers. |
| `prompts.py` | Model-agnostic baseline and refusal prompts. |
| `baseline.py` | Baseline answer generation with row-wise CSV resume. |
| `activations.py` | Answer-token activation collection grouped by concept. |
| `vectors.py` | `lda`, `diffed`, and `projected` vector construction plus cache wrappers. |
| `calibration.py` | Layer/scale grid construction, sweep generation, and optimal-cell selection. |
| `intervention.py` | `GatedSteering`, generation jobs, multi-GPU fan-out, and sampling helpers. |
| `evaluations/` | Pluggable evaluation panels. |
| `config.py` | YAML experiment resolution and subprocess matrix execution. |
| `pipeline.py` | End-to-end stage ordering, model lifecycle, logging, and plotting handoff. |
| `__main__.py` | CLI entry point. |

## Stage order

```text
[3a] baseline_train
[3b] baseline_test
[4a] baseline activations
[4b] refuse activations
[4c] baseline_test activations
[5]  vectors
[5.5a] calibration sweep
[5.5b] calibration judge
[6:*] pending evaluation generation
[7:*] pending evaluation judging
[9]  plots
```

Stages are gated by artifact presence. If the output file is present, the stage
usually loads it instead of recomputing it.

## Data fractions

`train_frac` and `test_frac` shrink the data before baseline generation. They are
debug knobs, not calibration knobs.

`calibration_n` samples from the post-`test_frac` test dataframe. It is stratified
per concept and reused at every layer/scale grid cell.

Evaluation sizes are independent:

| Eval | Size field |
| --- | --- |
| `bars` | `n` target questions plus `n` non-target questions per target. |
| `confusion` | `c` concepts by `c` targets by `n` questions per concept. |

## Adding an evaluation

1. Create `refuse/evaluations/<name>.py`.
2. Implement `run_<name>(pool, baseline_test, steering, scale, *, system_prompt,
   template, batch_size=128, result_metadata=None, **kwargs)`.
3. Return a dataframe with `question`, `concept`, `baseline_output`, `target`,
   `scale`, and `model_output`.
4. Register it in `refuse/evaluations/__init__.py`.
5. Add a CLI flag in `refuse/__main__.py`.
6. Optionally add a plotter in `plot/plot.py`.

The pipeline handles selected calibration config, judging, logging, and cache
paths for registered evals.

## Cache caveat

The package does not attempt provenance tracking. A cache file does not know
which model, prompt, grid, or code version produced it. Use a new store for a new
research condition when provenance matters.
