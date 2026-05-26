# Steering

A research toolkit for **per-concept refusal-vector steering** on open-weight LLMs. Given a labeled dataset of concept-tagged questions, this pipeline:

1. records hidden-state activations under baseline vs. refusal-prompted generations
2. fits an LDA detector per concept and a shared refusal direction
3. applies gated additive steering at inference time
4. calibrates the steering scale by sweeping a validation slice
5. evaluates refusal / retention / fluency with an LLM-as-judge (binary rubric)

Evaluation is pluggable: pick one or more of `--confusion C N` (full c×c grid) and `--bars N` (per-target target vs. untargeted, much cheaper). Each eval writes its own `{name}.csv` / `{name}_judged.csv`; the judge and plots discover them by name.

End-to-end runnable as `python -m refuse ...` and re-renderable as `python -m plot ...`.

---

## Package layout

```
api/         InstructorLLM wrapper (Anthropic / OpenAI / Together) — dataset generation
steering/    HF model wrapper + per-layer activation hooks + steering ops (GatedSteer, etc.)
llm/         ChatTemplate registry + GPUPool + load_llm — shared runtime infrastructure
refuse/      The refusal-vector pipeline (baselines → activations → vectors → calibration → generation)
judge/       LLM-as-judge evaluation (binary rubric, three-axis: refusal / retention / fluency)
plot/        Diagnostic plots (calibration curve, ROC, heatmaps). Standalone — reads CSVs from disk.
```

**Dependency graph** (strict downhill, no cycles):

```
steering  ←  llm  ←  refuse  ←  pipeline
                  ←  judge  ←  pipeline
                                 ↓
                              plot (standalone — no GPU)
```

`judge` never imports steering ops. `plot` never imports any model code. `refuse` owns the orchestration.

---

## Install

```bash
pip install -e .
```

Installs all six packages (`api`, `steering`, `llm`, `refuse`, `judge`, `plot`) plus dependencies. Requires Python ≥ 3.12.

Set `HF_TOKEN` in `.env` for gated models (Llama-3 family).

---

## Quick start

```bash
python -m refuse \
  --model       meta-llama/Llama-3.1-8B-Instruct  --gpus       0,1 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B --judge-gpus 0,1 \
  --data store/inhouse \
  --out  store/llama3_inhouse \
  --bars 20 \
  -v
```

Walks the full pipeline. Stages `[3a] … [9]` print one line each with cache-hit status. Everything is cached idempotently — re-running on a populated store skips computed steps.

**Outputs** in `<--out>/`:

| file | contents |
|---|---|
| `baseline_train.csv`, `baseline_test.csv` | model's baseline outputs (one per question) |
| `baseline_answer_acts.pt`, `_masks.pt` | per-layer activations on baseline prompts (train) |
| `refuse_answer_acts.pt`, `_masks.pt` | per-layer activations on refusal-prompted prompts (train) |
| `baseline_answer_acts_test.pt`, `_masks.pt` | baseline activations on test set (for ROC) |
| `v_detect.pt` | per-concept LDA detection vectors `(n_layers, 1, hidden)` |
| `v_refuse.pt` | averaged refusal direction `(n_layers, 1, hidden)` |
| `thresholds.pt` | per-concept per-layer LDA gating thresholds |
| `calibration_results.csv` | steered generations across `--calibration-frac` × 30 scales |
| `calibration_judged.csv` | calibration outputs + `judge_refusal/retention/fluency` columns |
| `{eval}.csv` | per-eval steered generations at the selected scale (e.g. `confusion.csv`, `bars.csv`) |
| `{eval}_judged.csv` | per-eval outputs + judge scores |
| `plots/` | `calibration.png` plus per-eval plots (`confusion_heatmap_*.png`, `bars.png`, …) |
| `pipeline.log` | tee'd terminal transcript |

---

## CLI reference

```
required
  --model PATH                      HF model path (must be in EXACT_MATCHES or pass --template)
  --data DIR                        folder with train.csv and test.csv
  --out  DIR                        artifact store

main model
  --gpus 0,1                        GPU ids
  --template {llama3,mistral,qwen}  override auto-detect

method + sampling
  --method {lda,diffed,projected}   vector method (default lda)
  --train-frac       1.0            per-concept subsample of train (default 1.0)
  --test-frac        1.0            per-concept subsample of test (default 1.0)
  --calibration-frac 0.10           fraction of (kept) test for calibration sweep

evaluations (any combination; none = skip eval stage)
  --confusion C N                   subsample C concepts, sweep all C targets × N questions/concept
  --bars      N                     per target: N target-concept + N untargeted-pool questions

judge (LLM-as-judge)
  --judge-model PATH                HF model for the judge
  --judge-gpus 0,1                  GPU ids for judge (default: same as --gpus)
  --judge-retries 25                retry budget for parse failures

misc
  --no-plot                         skip plot rendering at end
  -v, --verbose                     print stage progress
```

---

## Python API

```python
from refuse import run

run(
    model_path    = "meta-llama/Llama-3.1-8B-Instruct",
    data_root     = "store/concepts",
    result_root   = "store/llama3_concepts",
    gpu_ids       = [0, 1],
    judge_model   = "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
    judge_gpu_ids = [0, 1],
    evaluations   = [("bars", {"n": 20})],          # or ("confusion", {"c": 10, "n": 10}), or both
    verbose       = True,
)
```

Lower-level building blocks (when you want partial pipeline control):

```python
from llm import GPUPool, detect_template
from refuse import (
    Paths, generate_baseline, cached_concept_activations,
    cached_lda_vectors, GatedSteering, make_generation_jobs, run_jobs,
    calibration_generate, calibration_score_select,
)
from refuse.prompts import BASELINE_SYSTEM, refuse_system
from judge import add_judge_scores
from plot import make_all
```

Each stage is idempotent on its cache file.

---

## Re-rendering plots (no GPU, no model)

All four plots are pure functions of the on-disk store. Re-render any time:

```bash
python -m plot --store store/llama3_concepts
# or: python -m plot --store STORE --out OTHER_DIR
```

Reads `calibration_judged.csv` and any `{eval}_judged.csv` files in the store, writes PNGs to `<store>/plots/`. Plots that have no source CSV are silently skipped.

---

## Verified models

`llm.chat_templates.EXACT_MATCHES` only auto-detects exact paths. Anything else → `detect_template` raises a loud error (no fuzzy matching footguns).

| Model path | Template |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | LLAMA3 |
| `mistralai/Mistral-7B-Instruct-v0.3` | MISTRAL |
| `Qwen/Qwen2.5-7B-Instruct` | QWEN |
| `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | LLAMA3 |

To add a model, register it in `llm/chat_templates.py`:

```python
EXACT_MATCHES["my-org/my-model"] = LLAMA3   # or a custom ChatTemplate(...)
```

---

## Architecture notes

**Model lifecycle is sequential, not pooled.** Each pipeline phase loads → uses → `del`s its model. No CPU↔GPU swap dance, no shared-GPU branching. Trade ~30 s per disk reload for far simpler code. Up to five model loads in a full judged run (Blocks 3+4 share one main load and one judge load regardless of how many evals are enabled):

```
1. load main  →  baselines + activations  →  del
2. (LDA on CPU — no model needed)
3. load main  →  calibration generate     →  del
4. load judge →  calibration score        →  del
5. load main  →  all pending evaluations  →  del
6. load judge →  judge all pending evals  →  del
```

**`GPUPool` is just `map` + `generate`.** Multi-GPU is `pool.map(fn, shards)` running `fn(llm, shard)` in a `ThreadPoolExecutor`. `pool.generate(prompts, ...)` is the only convenience method. Activation collection and steered generation are standalone functions in `refuse/` that *take* a pool — `llm/` never imports from `refuse/`.

**Strict template matching.** `detect_template(model_path)` only matches exact paths. Substring matching (`"llama3" in name`) is a footgun; fixing one alias silently breaks similarly-named models.

**LLM-as-judge with binary Selene-style rubric.** Three separate per-axis prompts (refusal / retention / fluency), each following Selene's training format:

```
**Reasoning:** <one short sentence>

**Result:** <1 or 2>
```

Score 1 → 0 (negative), Score 2 → 1 (positive). Per-axis cache columns (`judge_<axis>_completion`). Parse failures retry with `do_sample=True, temperature=0.7` up to `--judge-retries` times.

**Calibration picks scale on harmonic mean** of refusal and fluency: `2·R·F/(R+F)`. Penalizes scales that achieve high refusal by destroying fluency.

**ROC measures end-to-end behaviour**, not classifier separability. For each calibration scale, plots `(FPR = refusal rate when target ≠ concept, TPR = refusal rate when target == concept)`. Each point on the curve is one scale. AUC over the sweep.

**Caching.** Every stage's output is a file under `<out>/`. On re-run, existing files are loaded and compute is skipped. Two helpers in `paths.py`:
- `cached_pt({"name": path, ...}, compute_fn)` — all-or-nothing for `.pt` files
- `cached_csv_rows(path, df, compute_missing_fn, key_col)` — row-wise resume for CSVs (used by `generate_baseline` and judge passes)

**Defaults tuned for an 8B model on 2× RTX 5090 (31 GB each)**:
- `add_judge_scores(batch_size=16)` — KV cache for retention prompts at batch=64 OOMs
- `cached_concept_activations(batch_size=64)` — hooks store 32 layer × `(batch, seq, hidden)` activations on GPU
- `--judge-retries 25` — `do_sample=True` retries virtually eliminate parse failures

---

## Design philosophy

From `CLAUDE.md`:
- No `try/except` unless explicitly asked.
- No backwards-compat shims.
- No defensive code for scenarios that can't happen.
- Minimal docstrings — one-liner or skip.
- Each `.py` file has one responsibility (see `refuse/design.md` for the per-module map).

Composition over magic: `pool`, `template`, `paths` are passed explicitly to every function that needs them.

---

## Datasets

Two example datasets live under `store/`:

- `store/concepts/` — 10 science / general-knowledge concepts (lasers, the_moon, paris, dogs, …), ~4.7k train / 1.2k test
- `store/RWKU/` — Real-World Knowledge Unlearning (200 famous people), larger

Schema: `concept, subtopic, question, answer` (subtopic optional, never read by the pipeline). Generated by the notebooks under `datasets/` using `api.InstructorLLM` against GPT or Claude.

To use a different dataset: drop `train.csv` and `test.csv` (with at minimum `concept` and `question` columns) in a folder and point `--data` at it.

---

## Repo conventions

- Path layout: `store/<dataset>/` for raw data, `store/<model>_<dataset>/` for results.
- All artifacts in `store/` are checkpointed and git-ignored.
- Notebooks live under `notebooks/`; they are drivers, not source of truth.
