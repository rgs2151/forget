# Refuse

Refuse is a research framework for **per-concept refusal-vector steering** in
open-weight instruction-tuned language models. Given a dataset of concept-labeled
questions, the framework learns a detector for each concept, learns a shared
refusal direction, calibrates where and how strongly to apply that direction, and
judges the resulting behavior.

The current branch is organized around calibration-first experiments: run an
all-layer steering sweep, judge the sweep, select the best `(layer, scale)` cell,
and then run optional evaluation panels from that calibrated configuration.

## What it does

The main pipeline:

1. Generate baseline answers for train and test questions.
2. Collect answer-token activations for baseline answers and refusal-prompted
   "I don't know" answers.
3. Fit per-concept detector vectors with LDA, or use `diffed` / `projected`
   alternatives.
4. Build a gated steering operator: if a hidden state crosses a concept-specific
   detector threshold, add the shared refusal direction.
5. Sweep layer sets and steering scales on a stratified calibration sample.
6. Judge refusal, retention, and fluency with an LLM-as-judge rubric.
7. Select the calibration cell with the highest harmonic mean of refusal and
   fluency.
8. Optionally run `bars` or `confusion` evaluations from the selected cell.
9. Render plots from CSV artifacts. Plotting does not load model code.

## Repository map

| Path | Purpose |
| --- | --- |
| `refuse/` | Pipeline orchestration: baseline generation, activations, vectors, calibration, evaluations, caches. |
| `steering/` | Hugging Face model wrapper, layer hooks, and steering operators. |
| `llm/` | Chat templates, model loading, and `GPUPool` multi-GPU helper. |
| `judge/` | Binary LLM-as-judge scoring for refusal, retention, and fluency. |
| `plot/` | Offline plotting from stored CSVs. No GPU dependency. |
| `api/` | Instructor-style wrapper used by dataset-generation notebooks. |
| `configs/` | YAML experiment matrices. |
| `doc/` | Sphinx documentation pages. |
| `ds/`, `notebooks/` | Dataset and exploratory notebooks. |
| `store/` | Datasets and experiment artifacts. This directory is large. |

## Install

Requires Python 3.12 or newer.

```bash
pip install -e .
```

Set `HF_TOKEN` in `.env` when using gated Hugging Face models such as the
Llama-3 family.

The full pipeline needs the heavy ML dependencies from `pyproject.toml`. Plotting
only needs the data/plotting stack and can run without loading a model.

## Data contract

Each dataset directory must contain:

```text
train.csv
test.csv
```

Minimum columns:

| Column | Required | Meaning |
| --- | --- | --- |
| `concept` | yes | Concept label used for fitting detectors and choosing steering targets. |
| `question` | yes | User prompt to answer. |
| `answer` | no | Optional reference metadata. The current pipeline judges retention against the generated baseline output. |
| `subtopic` | no | Optional metadata. Not used by the pipeline. |

Current dataset directories include `store/inhouse`, `store/rwku`, `store/mmlu`,
`store/concept10`, `store/concept500`, and `store/conceptvectors`.

## Quick start

The primary interface is the YAML matrix:

```bash
python -m refuse --config configs/experiments.yml --list
python -m refuse --config configs/experiments.yml
python -m refuse --config configs/experiments.yml --only qwen7b_inhouse
```

Each run in the matrix executes in its own subprocess. This gives process
isolation, appends a master transcript to `logs/experiments.log`, and appends
per-store transcripts to `<store>/pipeline.log`.

The current matrix is calibration-only for four models on `store/inhouse`:

```yaml
runs:
  - { model: qwen7b,    data: inhouse }
  - { model: llama8b,   data: inhouse }
  - { model: mistral7b, data: inhouse }
  - { model: phi4,      data: inhouse }
```

To run a one-off calibration:

```bash
python -m refuse \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data store/inhouse \
  --out store/llama8b_inhouse \
  --method lda \
  --gpus 0,1 \
  --layers all \
  --scale-window mid \
  --scale-steps 15 \
  --calibration-n 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  --judge-retries 100 \
  --batch-size 16 \
  --judge-batch-size 16 \
  -v
```

To run evaluations after calibration:

```bash
python -m refuse \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data store/inhouse \
  --out store/llama8b_inhouse \
  --method lda \
  --gpus 0,1 \
  --bars 10 \
  --confusion 10 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  -v
```

When evaluation CSVs are missing, the pipeline reads `calibration_judged.csv`,
selects the best `(layer, scale)` cell, generates the requested evals, judges
them, and refreshes plots.

## CLI reference

Config mode:

| Flag | Meaning |
| --- | --- |
| `--config FILE` | Load a YAML experiment matrix. |
| `--only NAME ...` | Run only selected resolved experiment names. |
| `--list` | Print resolved runs and exit. |

Single-run mode:

| Flag | Meaning |
| --- | --- |
| `--model PATH` | Exact Hugging Face model path registered in `llm.chat_templates.EXACT_MATCHES`. |
| `--data DIR` | Dataset folder with `train.csv` and `test.csv`. |
| `--out DIR` | Artifact store for this run. |
| `--method {lda,diffed,projected}` | Vector construction method. |
| `--gpus 0,1` | Comma-separated GPU IDs for the main model. |

Calibration:

| Flag | Meaning |
| --- | --- |
| `--layers SPEC` | `default`, `all`, `frac: ...`, or explicit layer sets such as `"3 7 15,18,21,24"`. |
| `--scale-window WINDOW` | `small`, `mid`, `large`, or `"lo:hi"`. |
| `--scale-steps N` | Number of scale values inside the window. |
| `--calibration-n N` | Samples per concept. Use `all` for every test question per concept. |
| `--train-frac F` | Per-concept training subsample for debug runs. |
| `--test-frac F` | Per-concept test subsample for debug runs. |

Evaluations:

| Flag | Meaning |
| --- | --- |
| `--bars N` | For each target concept, generate `N` target-concept questions and `N` non-target questions. |
| `--confusion C N` | Select `C` concepts and generate a full target-by-concept grid with `N` questions per concept. |

Judge and batch controls:

| Flag | Meaning |
| --- | --- |
| `--judge-model PATH` | Exact model path for judging. |
| `--judge-gpus 0,1` | GPU IDs for the judge. Defaults to `--gpus`. |
| `--judge-retries N` | Retry budget for unparseable judge completions. |
| `--batch-size N` | Main-model generation/activation batch size. |
| `--judge-batch-size N` | Judge generation batch size. |
| `--no-plot` | Skip plot rendering. |
| `-v`, `--verbose` | Print stage progress. |

## Outputs

Each run writes to `<out>/`.

| File | Contents |
| --- | --- |
| `baseline_train.csv`, `baseline_test.csv` | Baseline model outputs. |
| `baseline_answer_acts.pt` | Per-layer train activations on baseline answers. |
| `refuse_answer_acts.pt` | Per-layer train activations on refusal-prompted answers. |
| `baseline_answer_acts_test.pt` | Per-layer test activations for later analysis. |
| `v_detect.pt` | Per-concept detector vectors for every layer. |
| `v_refuse.pt` | Shared refusal direction for every layer. |
| `thresholds.pt` | Per-concept, per-layer detector thresholds. |
| `calibration_results.csv` | Flat layer x scale generation sweep. |
| `calibration_judged.csv` | Calibration sweep plus `judge_refusal`, `judge_retention`, and `judge_fluency`. |
| `{eval}.csv` | Optional steered generation outputs, for example `bars.csv`. |
| `{eval}_judged.csv` | Optional eval outputs plus judge scores. |
| `plots/` | Rendered calibration and eval plots. |
| `pipeline.log` | Per-store terminal transcript. |
| `arguments.log` | Resolved invocation history for this store. |

## Plotting

Plots are pure functions of the CSVs in a store:

```bash
python -m plot --store store/llama8b_inhouse
python -m plot --store store/llama8b_inhouse --out /tmp/llama8b_plots
```

The plot command writes any figures it can support from the files present. If a
store only has calibration CSVs, only calibration plots are produced.

## Interpreting Scores

The judge scores three binary axes:

| Score | Meaning |
| --- | --- |
| `judge_refusal` | `1` when the answer explicitly refuses. |
| `judge_retention` | `1` when the candidate preserves the baseline answer content. |
| `judge_fluency` | `1` when the candidate is readable rather than degenerate. |

Calibration selection uses the harmonic mean of refusal and fluency. Retention is
still recorded, but it is not part of the current selection objective.

## Verified model templates

Template detection is exact. Register new paths in `llm/chat_templates.py`.

| Model path | Template |
| --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | `LLAMA3` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `MISTRAL` |
| `Qwen/Qwen2.5-7B-Instruct` | `QWEN` |
| `microsoft/phi-4` | `PHI4` |
| `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `LLAMA3` |

## More documentation

| Page | Use it for |
| --- | --- |
| [Configuration](doc/config.md) | YAML structure, layer specs, scale windows, caching rules, and common workflows. |
| [Design](doc/design.md) | Package responsibilities and pipeline lifecycle. |
| [Results](doc/results.md) | Artifact semantics and result inspection. |
| [Troubleshooting](doc/troubleshooting.md) | Missing dependencies, stale caches, CUDA memory pressure, and judge parse retries. |
| [API reference](doc/api.rst) | Module-level public API inventory. |

## Current limitations

- There is no automatic cache invalidation. If you change a grid, model, dataset,
  prompt, or method but reuse the same store, existing artifacts are reused.
- The current calibration objective ignores retention.
- Exact chat-template matching prevents accidental model/template mismatches, but
  it also means new model aliases must be registered manually.
- There are no unit tests in this repository. Validation is currently done through
  command smoke checks, cache inspection, and artifact-level plots.
