# Configuration & sweeps

Experiments live in YAML under `configs/`. One file is the source of truth for the whole model × dataset matrix; the CLI runs it.

```bash
python -m refuse --config configs/experiments.yml              # run every entry in `runs`
python -m refuse --config configs/experiments.yml --only qwen7b_rwku qwen7b_mmlu
python -m refuse --config configs/experiments.yml --list       # print resolved configs, run nothing
```

Each entry runs as its **own subprocess** — process isolation, per-store `pipeline.log`, and one crash never aborts the batch. A master log is appended to `logs/experiments.log`.

## File structure

```yaml
data_root: store          # where train.csv/test.csv folders live
store_root: store         # where result folders are written

defaults:                 # apply to every run
  method: lda
  gpus: [0, 1]
  judge_model: AtlaAI/Selene-1-Mini-Llama-3.1-8B
  judge_gpus: [0, 1]
  judge_retries: 100
  train_frac: 1.0
  batch_size: 16
  judge_batch_size: 16

models:                   # per-model overrides; calibration lives here
  llama8b: { path: meta-llama/Llama-3.1-8B-Instruct,   layers: all, scales: 15, scale_window: mid }
  qwen7b:  { path: Qwen/Qwen2.5-7B-Instruct,           layers: all, scales: 15, scale_window: large }

datasets:                 # per-dataset overrides
  inhouse: { calibration_n: 10 }    # 10 samples per concept; `all` uses every question
  rwku:    { calibration_n: 10 }

runs:                     # the matrix — just {model, data} (+ evals)
  - { model: qwen7b,  data: rwku }
  - { model: llama8b, data: inhouse }
```

### Layered resolution

Each run merges, later layers winning:

```
defaults  <-  models[model]  <-  datasets[data]  <-  per-run override
```

So Qwen's `scale_window: large` and `16/16` batch sizes are stated once in `models`; `calibration_n` (samples per concept, or `all`) lives on each `dataset`. `name`/`out` auto-derive as `{model}_{data}` → `{store_root}/{model}_{data}`; `data` → `{data_root}/{data}`.

### Evaluations

A run gets an eval only if it names one — `bars: 10` and/or `confusion: [C, N]`. Omit both for calibration-only.

## The calibration grid

Calibration is model-level — three flat keys on the model:

- `layers` — the layer spec (grammar below)
- `scales` — number of scale steps within `scale_window`
- `scale_window` — the scale range (`small` / `mid` / `large` / `"lo:hi"`)

The grid is the product: `resolve_layers(layers) × scale_grid(scale_window, scales)`. Execution nests **layer-outer / scales-inner** (one `GatedSteering` vector-bank per layer; questions batched by concept). To add a future axis, add one key and one loop in `build_grid`.

### Layer spec

`layers` is **model-agnostic**, resolved against each model's `num_hidden_layers` at runtime:

| spec | meaning | llama (32) | qwen (28) |
|---|---|---|---|
| `default` | fractional canonical set, as one config | `[[15,18,21,24]]` | `[[13,16,18,21]]` |
| `all` | every single layer | 32 configs | 28 configs |
| `frac: 0,.25,.5,.75,1` | single layers at depth fractions (depth-comparable across models) | `[[0],[8],[16],[24],[31]]` | `[[0],[7],[14],[20],[27]]` |
| `"3 7 15,18,21,24"` | explicit; space = new config, comma = layers within one | `[[3],[7],[15,18,21,24]]` | errors if any ≥ 28 |

`default`/`all`/`frac:` can never be out of range; explicit indices `≥ num_layers` raise.

### Scale window

`scale_window` (on the model) is the range: `small` (0–5), `mid` (0–15), `large` (0–100), or `"lo:hi"`. `scales` is the step count. So Qwen's `large` + `scales: 15` → `6.67, 13.33, …, 100.0`.

## What the sweep produces

```
grid = resolve_layers(layers) × scale_grid(scale_window, scales)

for each (layer_set, scale) in grid:
    diagonal generation (target == concept) over the sampled questions
    → rows with source_layer = target_layer = layer_set, scale, model_output
```

Sampling is `calibration_n` questions per concept (stratified, fixed seed) — or every question per concept when `calibration_n: all` — drawn once and reused at every grid point. The grid defines the rows; execution **fills `model_output`** (generation) then the **`judge_*` columns** (judge). Output is one flat file (+ its judged twin):

`calibration_results.csv` / `calibration_judged.csv`

| column | meaning |
|---|---|
| `concept`, `question`, `baseline_output` | the sampled question + unsteered reference |
| `target` | concept steered toward (== `concept`, diagonal) |
| `source_layer`, `target_layer` | the layer-set for this row, e.g. `[0]` or `[15, 18, 21, 24]` |
| `scale` | steering scale |
| `model_output` | steered generation |
| `judge_refusal/retention/fluency` (+ `_completion`) | judge scores |

The sweep resumes by skipping `(source_layer, scale)` pairs already in the CSV.

## CLI single-run

For a one-off without YAML, pass the calibration flags directly:

```bash
python -m refuse --model ... --data ... --out ... \
  --layers all --scale-window large --scale-steps 15 \
  --judge-model ... -v
```

## Guards & caching

- **Multi-layer grid + evals is rejected.** A grid spanning more than one layer config combined with `bars`/`confusion` raises — scale selection is scale-only and would average across layers. Sweep first, select post-hoc, eval after.
- **No auto-invalidation.** A present `calibration_results.csv` is reused as-is regardless of the requested grid. To re-sweep a different grid, delete `calibration*.csv` in that store first.
- **Activations/vectors are one-time.** `baseline_*.csv`, `*_acts.pt`, `v_*.pt` are reused whenever present in the store; a sweep on top of them is pure generation + judge.
