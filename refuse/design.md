# refuse — design

The refusal-vector pipeline. Each `.py` file has one responsibility. Cross-package shared infrastructure lives in `llm/` and `steering/`; see the top-level `readme.md` for the broader package map.

## Modules (this package)

| Step | Module | Responsibility |
|---|---|---|
| 1 | `paths.py` | `Paths` dataclass enumerating every cache file location, plus shared helpers `cached_pt` (all-or-nothing for `.pt` files) and `cached_csv_rows` (row-wise resume for CSVs). |
| — | `prompts.py` | Model-agnostic prompts: `BASELINE_SYSTEM` and `refuse_system(concept)`. (Llama-3 chat tokens live in `llm/chat_templates.py`.) |
| 3 | `baseline.py` | `generate_baseline(pool, df, csv_path, template)` — row-wise resume from CSV via `cached_csv_rows`. |
| 4 | `activations.py` | `collect_concept_activations` and friends + `cached_concept_activations(pool, df, prompt_fn, answer_fn, acts_path, masks_path)` wrapper. Also hosts the standalone `collect_activations(pool, ...)` (the per-concept fan-out used to live on `GPUPool.collect_activations`; moved here so `llm/` stays consumer-agnostic). |
| 5 | `vectors.py` | `lda_vectors` / `diffed_vectors` / `projected_vectors` + cached wrappers. Cholesky-based solve. Forget loop runs on CPU (avoids OOM during large `acts * mask` intermediates). |
| 5.5 | `calibration.py` | `calibration_generate(pool, df, scales, steering, ...)` writes the sweep CSV; `calibration_score_select(results, refusal_fn)` scores via judge and picks the scale maximizing harmonic mean of refusal and fluency. |
| 6+7 | `intervention.py` | `Steering` / `GatedSteering` class hierarchy + `make_generation_jobs` + `run_generation_jobs` (single-LLM) + `run_jobs(pool, ...)` (multi-GPU fan-out). |
| — | `pipeline.py` | `run(model_path, data_root, result_root, …)` — wires every stage end-to-end with sequential model load/use/del lifecycle. Tees `stdout`/`stderr` into `<out>/pipeline.log`. |
| — | `__main__.py` | `python -m refuse --model ... --data ... --out ... [--judge-model ...] [-v]` CLI. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before any torch import. |

## Conventions

- This package never sees `gpu_ids` directly — everything goes through a `GPUPool` constructed in `pipeline.run`.
- Functions that touch chat formatting **require** an explicit `template: ChatTemplate` argument (no Llama-3 default). The pool carries the template so it can pass `template.assistant_end_marker` into activation collection.
- `intervention_layers` defaults to `default_intervention_layers(num_layers)` — 4 layers at `[15, 18, 21, 24] · num_layers / 32`. Override via `intervention_layers=` kwarg.
- Each stage is idempotent on its cache file in `paths.<name>`.
- `pipeline.run(..., verbose=True)` / `python -m refuse -v` prints one `[refuse]` line per stage with cache-hit status.
- No try/except, no backwards-compat, minimal docstrings (per `CLAUDE.md`).

## Pipeline order

```
[3a] baseline_train (main)
[3b] baseline_test (main)
[4a] baseline activations (main)
[4b] refuse activations (main)
[4c] baseline_test activations (main)
                          ↓ del main
[5]  vectors (LDA on CPU after del, no model needed)
                          ↓ load main
[5.5a] calibration generate (main)
                          ↓ del main, load judge
[5.5b] calibration score → select scale (judge)
                          ↓ del judge, load main
[6+7] steered generation at selected scale (main)
                          ↓ del main, load judge
[8] judge final scoring (judge)
                          ↓ del judge
[9] plots (no model)
```

Five model loads total in a full judged run. Trade: ~30 s/load (≈ 2.5 min overhead) vs. complex offload/reload state management.

## Entry points

```bash
python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct  --gpus       0,1 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B  --judge-gpus 0,1 \
    --data store/concepts --out store/llama3_concepts -v
```

```python
from refuse import run
scored = run(model_path="…", data_root="…", result_root="…",
             gpu_ids=[0, 1], judge_model="…", judge_gpu_ids=[0, 1], verbose=True)
```
