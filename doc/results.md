# Results and artifacts

This page explains how to read a result store after a run has finished.

## Store layout

Every run writes artifacts to one directory, usually `store/{model}_{dataset}`.

```text
store/llama8b_inhouse/
  arguments.log
  pipeline.log
  baseline_train.csv
  baseline_test.csv
  baseline_answer_acts.pt
  refuse_answer_acts.pt
  baseline_answer_acts_test.pt
  v_detect.pt
  v_refuse.pt
  thresholds.pt
  calibration_results.csv
  calibration_judged.csv
  plots/
```

Evaluation runs may also add:

```text
bars.csv
bars_judged.csv
confusion.csv
confusion_judged.csv
```

## Logs

`arguments.log` records each invocation and the resolved settings used by
`pipeline.run`.

`pipeline.log` is the terminal transcript for that store. Search for `[refuse]`
to see stage boundaries and cache hits.

```bash
grep -n "\\[refuse\\]" store/llama8b_inhouse/pipeline.log
```

## Baseline files

`baseline_train.csv` and `baseline_test.csv` are the input dataframes plus
`baseline_output`. These files define the generated reference answers used for
activation collection and retention judging.

If a baseline file exists and has a complete `baseline_output` column, the
pipeline reuses it.

## Activation and vector files

The `.pt` files are Torch artifacts:

| File | Meaning |
| --- | --- |
| `baseline_answer_acts.pt` | Train activations from baseline answers, grouped by concept. |
| `refuse_answer_acts.pt` | Train activations from refusal-prompted answers, grouped by concept. |
| `baseline_answer_acts_test.pt` | Test activations from baseline answers. |
| `v_detect.pt` | Per-concept detector vectors for all layers. |
| `v_refuse.pt` | Shared refusal vector for all layers. |
| `thresholds.pt` | Per-concept LDA thresholds for all layers. |

These files are expensive to produce and are reused aggressively.

## Calibration files

`calibration_results.csv` contains steered generations for each sampled question
and each grid cell. It does not contain judge scores.

`calibration_judged.csv` contains the same rows plus:

| Column | Meaning |
| --- | --- |
| `judge_refusal` | `1` if the answer explicitly refused. |
| `judge_retention` | `1` if the answer preserved baseline content. |
| `judge_fluency` | `1` if the answer was readable. |
| `judge_aggregate` | Harmonic mean of refusal and fluency. |
| `judge_*_completion` | Raw judge completions used to parse scores. |

The current calibration objective selects the `(source_layer, scale)` cell with
the highest mean harmonic score over refusal and fluency.

Inspect the selected cell:

```bash
python - <<'PY'
import ast
import pandas as pd

path = "store/llama8b_inhouse/calibration_judged.csv"
df = pd.read_csv(path)
if "label" in df:
    df = df[df["label"] == "intervention"]

df = df.assign(
    harmonic=2 * df["judge_refusal"] * df["judge_fluency"]
    / (df["judge_refusal"] + df["judge_fluency"] + 1e-9)
)

summary = (
    df.groupby(["source_layer", "scale"], as_index=False)
      .agg(
          refusal=("judge_refusal", "mean"),
          retention=("judge_retention", "mean"),
          fluency=("judge_fluency", "mean"),
          harmonic=("harmonic", "mean"),
          n=("harmonic", "size"),
      )
      .sort_values(["harmonic", "scale"], ascending=[False, True])
)

print(summary.head(10).to_string(index=False))
PY
```

## Evaluation files

`bars.csv` and `confusion.csv` contain generated outputs at the selected
calibration cell.

Their judged counterparts add the judge score columns. The plotting code looks
for files ending in `_judged.csv` and renders any supported eval type.

## Plots

Run:

```bash
python -m plot --store store/llama8b_inhouse
```

Common outputs:

| Plot | Source |
| --- | --- |
| `calibration.png` | Best layer trace across scales. |
| `calibration_layers.png` | Per-layer calibration traces. |
| `bars.png` | `bars_judged.csv`, if present. |
| `confusion_heatmap_refusal.png` | `confusion_judged.csv`, if present. |
| `confusion_heatmap_retention.png` | `confusion_judged.csv`, if present. |
| `confusion_heatmap_fluency.png` | `confusion_judged.csv`, if present. |

## Reading the scores

High refusal with low fluency means the steering may be breaking generation.
High fluency with low refusal means the steering is too weak or the detector is
not activating. High refusal with low retention is expected for target-concept
questions, but it is undesirable for non-target questions in evaluation panels.

For calibration, retention is diagnostic. For evaluations, retention is central:
it tells whether non-target knowledge survived the intervention.
