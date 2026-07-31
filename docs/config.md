# Configuration

This page is the operational reference for experiment matrices and one-off CLI
runs. The shortest rule: use YAML for research sweeps, and use explicit CLI flags
only for ad hoc debugging.

## Matrix Runs

```bash
python -m forget.refuse --config parking/model_matrix/experiments.yml --list
python -m forget.refuse --config parking/model_matrix/experiments.yml
python -m forget.refuse --config parking/model_matrix/experiments.yml --only qwen7b_inhouse
```

Every resolved experiment runs in its own subprocess. A failure in one run does
not stop the remaining runs, and each store keeps its own `pipeline.log` and
`arguments.log`.

## YAML Shape

```yaml
data_root: parking/model_matrix/cache
store_root: parking/model_matrix/cache

defaults:
  method: lda
  gpus: [0, 1]
  judge_model: AtlaAI/Selene-1-Mini-Llama-3.1-8B
  judge_gpus: [0, 1]
  judge_retries: 100
  train_frac: 1.0

models:
  llama8b:
    path: meta-llama/Llama-3.1-8B-Instruct
    layers: all
    scales: 10
    scale_window: mid
    batch_size: 8
    judge_batch_size: 8

datasets:
  inhouse:
    calibration_n: 100
    concept: random

runs:
  - { model: llama8b, data: inhouse }
```

Resolution order is:

```text
defaults <- models[model] <- datasets[data] <- per-run override
```

For a run `{ model: llama8b, data: inhouse }`, the resolved name is
`llama8b_inhouse`, the data path is `parking/model_matrix/cache/inhouse`, and the output path is
`parking/model_matrix/cache/llama8b_inhouse` unless the run overrides `name` or `out`.

## Fields

Top-level fields:

| Field | Meaning |
| --- | --- |
| `data_root` | Root directory containing dataset folders. |
| `store_root` | Root directory for run artifact folders. |
| `defaults` | Values shared by every run. |
| `models` | Model-specific settings, including calibration grid settings. |
| `datasets` | Dataset-specific settings, usually calibration sample size. |
| `runs` | Experiment matrix entries. |

Model fields:

| Field | Meaning |
| --- | --- |
| `path` | Exact Hugging Face model path. Required. |
| `layers` | Layer-set specification for calibration. |
| `scales` | Number of scale values in the selected window. |
| `scale_window` | Scale range: `small`, `mid`, `large`, `xlarge`, or `"lo:hi"`. |
| `batch_size` | Main-model batch size. |
| `judge_batch_size` | Judge batch size. |

Dataset fields:

| Field | Meaning |
| --- | --- |
| `calibration_n` | Samples per concept for `concept: all`, or total samples for `concept: random`. |
| `concept` | Calibration sampling mode: `all` or `random`. |
| `train_frac` | Optional debug subsample of train data. |
| `test_frac` | Optional debug subsample of test data. |

Run fields:

| Field | Meaning |
| --- | --- |
| `model` | Key into `models`. |
| `data` | Key into `datasets` and folder name under `data_root`. |
| `name` | Optional resolved run name. Defaults to `{model}_{data}`. |
| `out` | Optional artifact path. Defaults to `{store_root}/{name}`. |
| `bars` | Optional `bars` evaluation size. |
| `confusion` | Optional `[C, N]` confusion evaluation. |

## Layer Specs

Layer specs are resolved against the model's `num_hidden_layers` at runtime.

| Spec | Meaning |
| --- | --- |
| `default` | One canonical multi-layer set derived from `[15, 18, 21, 24] / 32`. |
| `all` | Every single layer as its own layer set. |
| `frac: 0,.25,.5,.75,1` | Single-layer sets at depth fractions. |
| `"3 7 15,18,21,24"` | Explicit sets. Spaces separate sets; commas combine layers inside a set. |

Examples for a 32-layer model:

| Spec | Resolved layer sets |
| --- | --- |
| `default` | `[[15, 18, 21, 24]]` |
| `all` | `[[0], [1], ..., [31]]` |
| `frac: 0,.5,1` | `[[0], [16], [31]]` |
| `"3 7 15,18,21,24"` | `[[3], [7], [15, 18, 21, 24]]` |

## Scale Windows

| Window | Range |
| --- | --- |
| `small` | `0.0` to `1.0` |
| `mid` | `0.0` to `10.0` |
| `large` | `0.0` to `100.0` |
| `xlarge` | `0.0` to `300.0` |
| `"lo:hi"` | Custom numeric range. |

`scales: 10` creates 10 nonzero values inside the range. For example,
`scale_window: mid` gives `1.0, 2.0, ..., 10.0`.

## Calibration Semantics

The calibration grid is:

```text
resolve_layers(layers, num_layers) x scale_grid(scale_window, scales)
```

For each `(layer_set, scale)` cell, the pipeline generates diagonal outputs where
`target == concept` over the configured calibration sample. The same sampled
questions are reused across every grid cell.

The judged calibration file records:

| Column | Meaning |
| --- | --- |
| `concept` | Question's true concept. |
| `target` | Steered target concept. Equal to `concept` during calibration. |
| `source_layer` | Detection layer set. |
| `target_layer` | Steering layer set. Current calibration uses the same set for both. |
| `scale` | Steering scale. |
| `baseline_output` | Unsteered reference output. |
| `model_output` | Steered output. |
| `judge_refusal` | Whether the model explicitly refused. |
| `judge_retention` | Whether output preserved baseline content. |
| `judge_fluency` | Whether output was readable. |

The selected configuration is the `(source_layer, scale)` cell with the highest
mean harmonic score over refusal and fluency:

```text
2 * refusal * fluency / (refusal + fluency)
```

Retention is available for analysis but is not part of the current selection
objective.

## Evaluation Semantics

Evaluations are optional. Omit `bars` and `confusion` for calibration-only runs.

`bars: N` creates a compact target-vs-untargeted panel. For each target concept,
it samples up to `N` questions from that target and up to `N` questions from the
pool of other concepts.

`confusion: [C, N]` samples `C` concepts and creates a full concept-by-target
grid, with up to `N` questions per selected concept.

When an evaluation is requested, the pipeline reads or computes
`calibration_judged.csv`, selects the best `(layer, scale)` cell, and uses that
configuration for every pending evaluation.

## Cache Rules

The pipeline is cache-first:

| Cache | Reused when present |
| --- | --- |
| `baseline_*.csv` | Baseline generation. |
| `*_acts.pt` | Activation collection. |
| `v_detect.pt`, `v_refuse.pt`, `thresholds.pt` | Vector fitting. |
| `calibration_results.csv` | Calibration generation. |
| `calibration_judged.csv` | Calibration judge scoring. |
| `{eval}.csv`, `{eval}_judged.csv` | Evaluation generation and judge scoring. |

There is no automatic invalidation. If you change the model, dataset, prompt,
vector method, layer grid, scale grid, or judge rubric, use a new result variant
or artifact cache. Do not delete protected results without explicit approval for
the exact artifact.

## Common Workflows

Run a full all-layer calibration sweep:

```bash
python -m forget.refuse --config parking/model_matrix/experiments.yml --only llama8b_inhouse
```

Run only plots from existing CSVs:

```bash
python -m forget.plot \
  --store parking/model_matrix/cache/llama8b_inhouse/results/prefill_logit
```

Add an evaluation to a YAML run:

```yaml
runs:
  - { model: llama8b, data: inhouse, bars: 10, confusion: [10, 10] }
```

Run a small debug job:

```bash
python -m forget.refuse \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data parking/model_matrix/cache/inhouse \
  --out debug/qwen_inhouse/cache \
  --gpus 0 \
  --layers "0 14 27" \
  --scale-window small \
  --scale-steps 3 \
  --calibration-n 2 \
  --train-frac 0.1 \
  --test-frac 0.1 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  -v
```
