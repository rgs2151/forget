# Design

Refuse is split into small packages with one-way dependencies. The goal is to
keep expensive model execution, research pipeline orchestration, judging, and
plotting from bleeding into each other.

```text
steering <- llm <- refuse
                  refuse -> judge
                  refuse -> plot
```

`plot` is intentionally offline: it reads CSVs and writes figures. It does not
import model code. `judge` scores text but does not import steering operations.

## Package responsibilities

| Package | Responsibility |
| --- | --- |
| `steering` | Hugging Face wrapper, per-layer hooks, activation capture, and steering operators. |
| `llm` | Exact chat-template registry, model loading, and `GPUPool`. |
| `refuse` | End-to-end research pipeline and cache ownership. |
| `judge` | LLM-as-judge prompts, parsing, retries, and score columns. |
| `plot` | Matplotlib/seaborn figures from existing CSV artifacts. |
| `api` | Dataset-generation helper around instructor-style APIs. |

## Pipeline lifecycle

`refuse.pipeline.run` uses a sequential model lifecycle:

```text
load main  -> baseline generation + activations -> del
compute vectors
load main  -> calibration generation            -> del
load judge -> calibration scoring               -> del
load main  -> pending evaluation generation     -> del
load judge -> pending evaluation scoring        -> del
plot from CSVs
```

This is slower than keeping models resident, but it keeps memory ownership clear
and makes cache resumes simpler.

## Cache ownership

Every expensive stage has a file under the output store:

| Stage | Cache |
| --- | --- |
| Baseline generation | `baseline_train.csv`, `baseline_test.csv` |
| Activation collection | `baseline_answer_acts.pt`, `refuse_answer_acts.pt`, `baseline_answer_acts_test.pt` |
| Vector fitting | `v_detect.pt`, `v_refuse.pt`, `thresholds.pt` |
| Calibration generation | `calibration_results.csv` |
| Calibration judging | `calibration_judged.csv` |
| Eval generation | `{eval}.csv` |
| Eval judging | `{eval}_judged.csv` |

The cache policy is deliberately simple: if the expected artifact exists and is
complete enough for that stage, it is reused.

## Calibration design

Vectors are computed for every layer once. Layer sweeps do not require new
activations or new vectors; they only require new generation and judge passes.

The calibration grid is the product of layer sets and scale values. Current
calibration uses the same layer set for detection and steering:

```text
source_layer == target_layer
```

The selected cell maximizes harmonic refusal/fluency. This favors answers that
refuse when targeted without becoming unreadable. Retention remains available as
a diagnostic and as an evaluation metric.

## LDA implementation

`refuse.vectors.lda_vectors` fits per-concept detector vectors from pooled
answer-token activations. It accumulates class sums and second moments, then
solves independent layer chunks with `torch.linalg.solve`. When CUDA is
available, it asks PyTorch to prefer cuSOLVER for the linear algebra backend.

The chunked solve bounds peak memory for wider models while preserving the same
per-layer solution.

## Evaluation design

Evaluations are pluggable functions registered in `refuse.evaluations`.

Current evals:

| Eval | Purpose |
| --- | --- |
| `bars` | Compact target-vs-untargeted comparison for each target concept. |
| `confusion` | Full concept-by-target grid over a sampled concept set. |

Both evals run from the selected calibration cell and write their own CSVs.
Adding a new eval should not require changing model lifecycle code.

## Judge design

The judge is a binary rubric over three axes:

| Axis | Positive score means |
| --- | --- |
| `refusal` | The answer explicitly refused. |
| `retention` | The answer preserved baseline content. |
| `fluency` | The answer remained readable. |

The parser looks for `Result: 1` or `Result: 2`. Unparseable rows are retried up
to the configured retry budget.

## Why exact templates

`detect_template(model_path)` requires an exact model path. This prevents quiet
template mismatches when two models have similar names but different chat
formats. The tradeoff is that new model aliases must be registered explicitly.
