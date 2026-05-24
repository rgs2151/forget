1. Organize path of stores
2. Load the model to n gpus
3. Take the train and test df, divide it by n gpus and then feed it to the model to create the baseline responses
4. Generate the contrastive refusal response to the baseline and then feed it to the model and catch the activation
5. Calculate the vectors with whatever method mentioned and store them
5.5 using very small randomly sampled tran set, do a sweep of the hyperparams like s
6. organize the steering classes with the calculated vectors, and then shove them to the gpus
7. now simply run the test df through this prepared model and save the results
8. add the score table based on this steered results
9. use the scored csv for the plots and stuffs


# refuse — design

Refusal-vector steering pipeline. Each `.py` file has one responsibility.

## Modules

| Step | Module | Responsibility |
|---|---|---|
| 1   | `paths.py`       | Centralizes cache file locations under a result root. Also hosts the two cache helpers (`cached_pt`, `cached_csv_rows`) other modules call. |
| 2a  | `model.py`       | `load_llm(model_path, gpu_id)` — thin `AutoModelForCausalLMWrapper` ctor with chat-aware defaults. |
| 2b  | `gpu.py`         | `GPUPool` — owns model placement and all parallelism. The only file that knows about `gpu_ids`, `ThreadPoolExecutor`, or sharding. Exposes `map` (primitive), `generate`, `collect_activations`, `run_jobs`. |
| —   | `chat_templates.py` | `ChatTemplate` dataclass + per-family instances (`LLAMA3`, `QWEN`, …) + `detect_template(model_path)`. Owns chat tokens, `render`, `trim_to_last_assistant`, and the assistant/instruction-end markers used by activation capture and steering. |
| —   | `prompts.py`     | Model-agnostic prompts: `BASELINE_SYSTEM` and `refuse_system(concept)`. |
| 3   | `baseline.py`    | `generate_baseline(pool, df, csv_path)` — row-wise resume from CSV. |
| 4   | `activations.py` | `collect_concept_activations` and friends + `cached_concept_activations` wrapper. |
| 5   | `vectors.py`     | `lda_vectors` / `diffed_vectors` / `projected_vectors` + cached wrappers. Renamed from old `steering.py` to avoid colliding with `forget/model/steering.py`. |
| 5.5 | `calibration.py` | `select_scale(pool, df, scales, steering, ...)` — sweeps a small sample and picks the scale with the highest refusal rate. |
| 6+7 | `intervention.py`| `Steering` / `GatedSteering` class hierarchy + `make_generation_jobs` + `run_generation_jobs` (single-LLM). No GPU plumbing. |
| 8   | `scoring.py`     | `add_retention_column`, `add_refusal_column`, `add_acceptability_column`. Renamed from old `score.py`. |
| 9   | `plots.py`       | Diagnostic figures: refusal/retention heatmaps, calibration sweep, detection ROC. `make_all(save_dir, ...)` writes PNGs. |
| —   | `pipeline.py`    | `run(model_path, data_root, result_root, method, gpu_ids, plot=True)` — wires every stage end-to-end. Idempotent on cache. |
| —   | `__main__.py`    | `python -m refuse --model ... --data ... --out ... [--no-plot]` CLI. |

## Conventions

- Modules other than `gpu.py` never see `gpu_ids`. They take a `GPUPool` and feed it work.
- Modules that touch chat formatting **require** a `template: ChatTemplate` argument (no Llama-3 default). The pool also holds the template so it knows the assistant-end marker for activation capture.
- `load_llm(model_path, ...)` auto-detects the template from `model_path` when none is passed.
- `GPUPool.from_model_path(model_path, gpu_ids)` is the convenience constructor that auto-detects the template and wires `load_llm`.
- `pipeline.run` derives `intervention_layers` from the loaded model's depth (4 layers in the upper-middle quarter), not from a hard-coded `[15, 18, 21, 24]`. Override via the `intervention_layers=` kwarg.
- Each module caches its own outputs; rerunning skips computed steps.
- `pipeline.run(..., verbose=True)` / `python -m refuse -v` prints one line per stage with cache-hit status, the selected scale, and total elapsed time.
- No try/except, no backwards-compat, minimal docstrings (per `CLAUDE.md`).

## Entry points

```bash
# CLI
python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/concepts \
    --out store/llama3_concepts \
    --method lda \
    --gpus 0,1 -v

python -m refuse \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data store/concepts \
    --out  store/mistral_concepts \
    --method lda \
    --gpus 0,1 -v

# Qwen/Qwen2.5-7B-Instruct
```

```python
# Python
from refuse import run
scored = run(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_root="store/concepts",
    result_root="store/llama3_concepts",
    method="lda",
    gpu_ids=[0, 1],
)
```

Notebooks (`llama3_concepts.ipynb`, `llama3_rwku.ipynb`) become thin drivers: one cell per stage so intermediate state stays inspectable, plus the plot cells.
