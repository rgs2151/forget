# Design

Per-concept refusal-vector steering, organized as a small set of single-responsibility packages with a strict downhill dependency graph.

```
steering  ←  llm  ←  refuse  ←  pipeline
                  ←  judge  ←  pipeline
                                 ↓
                              plot (standalone — no GPU, reads CSVs)
```

`judge` never imports steering ops; `plot` never imports model code; `refuse` owns orchestration.

## Packages

| package | responsibility |
|---|---|
| `steering/` | HF model wrapper + per-layer forward hooks + steering ops (`GatedSteer`). The actual additive-steering math. |
| `llm/` | Shared runtime: `ChatTemplate` registry (per model family), `GPUPool` (multi-GPU `map`/`generate`), `load_llm`. Consumer-agnostic — never imports `refuse`. |
| `refuse/` | The pipeline: baselines → activations → vectors → calibration sweep → evaluations. Owns orchestration and caching. |
| `judge/` | LLM-as-judge: per-axis binary rubric (refusal / retention / fluency), Selene `**Result:**` parsing, retry-on-parse-fail. |
| `plot/` | Diagnostic plots. Standalone — reads only CSVs from a store, no GPU, no model imports. |
| `api/` | `InstructorLLM` wrapper (Anthropic / OpenAI / Together) for dataset generation. Auxiliary. |

## refuse modules

| module | responsibility |
|---|---|
| `paths.py` | `Paths` dataclass enumerating every cache-file location; `cached_pt` (all-or-nothing `.pt`) and `cached_csv_rows` (row-wise CSV resume). |
| `prompts.py` | Model-agnostic prompts: `BASELINE_SYSTEM`, `refuse_system(concept)`. Chat tokens live in `llm/chat_templates.py`. |
| `baseline.py` | `generate_baseline` — baseline generations, row-wise resume. |
| `activations.py` | Answer-token activation collection + `cached_concept_activations`. |
| `vectors.py` | `lda` / `diffed` / `projected` detector + refusal vectors and cached wrappers. Vectors are computed for **all layers** at once. |
| `calibration.py` | The sweep: `resolve_layers(spec, num_layers)`, `scale_grid(window, steps)`, `build_grid(num_layers, layers, scales, scale_window)` (the `layers × scales` product grid), `calibration_sweep` (fills the grid — layer-outer, scales-inner, diagonal, resume-aware), and `select_refusal_scale`. |
| `intervention.py` | `Steering` / `GatedSteering` (vector bank + per-layer apply) + `make_generation_jobs` + `run_jobs` (multi-GPU fan-out) + `sample_per_concept`. |
| `evaluations/` | Pluggable evals. `base.run_eval` shared; `confusion` (C×C×N grid) and `bars` (target vs. untargeted) registered in `EVALUATIONS`; each writes its own `<name>.csv`. |
| `config.py` | YAML experiment configs: layered-defaults merge, `to_run_kwargs`, and `run_experiments` (one subprocess per run). Kept separate so config parsing never tangles with the pipeline. |
| `pipeline.py` | `run(...)` — wires every stage with a sequential load → use → `del` model lifecycle; tees stdout/stderr to `pipeline.log`; appends each invocation to `arguments.log`. |
| `__main__.py` | CLI: `--config FILE` (run a matrix) or explicit single-run flags. |

## Why single-layer steering is cheap

`v_detect[concept]` has shape `(num_layers, 1, hidden)` and `GatedSteering._make_op` indexes it by source layer. So vectors for **every** layer already exist in `v_detect.pt` — sweeping layers needs no new vectors and no new activations, only generation. `GatedSteering(layer_set, layer_set, ...)` steers at any layer set; the canonical 4-layer set is just one value in that space. A calibration sweep is therefore pure generation + judge on top of the one-time `baseline_*.csv` / `*_acts.pt` / `v_*.pt` artifacts.

## Model lifecycle

Each phase loads → uses → `del`s its model, with `gc.collect()` + `torch.cuda.empty_cache()` between phases. No CPU↔GPU swap dance. Every load is strictly gated on a `need_*` precondition; nothing loads speculatively.

```
[reuse] baseline_*.csv, *_acts.pt, v_*.pt          (one-time, on disk)
[5.5a]  load main  → calibration sweep             → calibration_results.csv → del
[5.5b]  load judge → judge calibration             → calibration_judged   → del
[6:*]   load main  → all pending evals             → {eval}.csv           → del
[7:*]   load judge → judge all pending eval CSVs   → {eval}_judged.csv    → del
[9]     plots (no model)
```

## Outputs per store

| file | contents |
|---|---|
| `baseline_train.csv`, `baseline_test.csv` | baseline generations |
| `baseline_answer_acts.pt`, `refuse_answer_acts.pt`, `baseline_answer_acts_test.pt` | per-layer activations |
| `v_detect.pt`, `v_refuse.pt`, `thresholds.pt` | per-concept vectors (all layers) |
| `calibration_results.csv` / `calibration_judged.csv` | the flat layer × scale sweep + judge scores |
| `{eval}.csv` / `{eval}_judged.csv` | per-eval generations + judge scores |
| `plots/` | rendered diagnostics |
| `pipeline.log`, `arguments.log` | terminal transcript + resolved config per invocation |
