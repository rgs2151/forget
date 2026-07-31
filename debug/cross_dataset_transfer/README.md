# Cross-dataset direction transfer debug

This tree is debug-only. It reads existing Llama-3.2-1B artifacts from
`store/`, but writes all new vectors, generations, judged CSVs, logs, and
figures under `debug/cross_dataset_transfer/`.

## Question

Can a source dataset/topic direction transfer to a semantically matched target
dataset/topic if layer and scale are optimized on the target?

Secondary question: how much source data is needed before transfer stabilizes?

## Design

Rows are source datasets and columns are target datasets within a semantic topic.
Each valid source-target cell is reversible. The diagonal is the native
within-dataset control. Off-diagonal cells are direction-transfer tests.

Layer and scale are tuned on the target side for every source-target cell.
For speed, all layers are first ranked by transferred gate separation on target
activations, then generation sweeps the top layer candidates across scale.
This experiment is not testing strict source-layer/source-scale transfer.

## Topics

| topic | datasets |
| --- | --- |
| `space` | Inhouse, MMLU |
| `places` | Inhouse, MMLU, ConceptVectors |
| `engineering` | Inhouse, MMLU, ConceptVectors |
| `computing` | MMLU, ConceptVectors |
| `religion` | MMLU, ConceptVectors |
| `people` | Inhouse, RWKU, ConceptVectors |

ConceptVectors is included as a stress-test dataset, not as the only evidence.

## Sample Sizes

Requested source-topic totals:

`2, 4, 8, 16, 32, 64, 128, 256, 512`

If a source topic has multiple concepts, samples are balanced across those
concepts. If the requested size exceeds available source examples, the run
records the capped `actual_k`.

## Outputs

| path | contents |
| --- | --- |
| `runs/<run_name>/config.json` | exact debug configuration |
| `runs/<run_name>/cell_summary.csv` | best layer/scale per cell from cheap calibration |
| `runs/<run_name>/eval_raw.csv` | selected operating-point generations |
| `runs/<run_name>/eval_judged.csv` | logit-judge results |
| `runs/<run_name>/transfer_summary.csv` | judged rates per cell |
| `runs/<run_name>/figures/` | diagnostic plots |

## Completed Run: `full_v1`

Status: complete.

- Cells: 351 calibrated and evaluated.
- Generations: 11,232 rows in `eval_raw.csv`.
- Judge: 11,232 rows in `eval_judged.csv`, with logit probabilities and binary decisions.
- Summary: 351 rows in `transfer_summary.csv`.
- Artifacts: all written under `runs/full_v1/`; no debug outputs were written to `store/`.

Largest requested source size was 512. Some cells are capped by available source
examples. MMLU tops out at 92-93 examples for the selected topic subsets;
ConceptVectors and Inhouse also cap for a few high-k topic cells.

## Main Readout

Direction transfer is feasible, but weaker than native within-dataset directions.
At the largest requested source size, off-diagonal transfer averages:

- targeted refusal: 0.430
- untargeted refusal: 0.188
- target-minus-untargeted selectivity: 0.242

Native within-dataset directions at the same setting average:

- targeted refusal: 0.558
- untargeted refusal: 0.100
- target-minus-untargeted selectivity: 0.458

For off-diagonal transfer at the largest requested source size, 20 of 24 cells
have positive selectivity. Ten of 24 cells reach targeted refusal at least 0.5,
and six of 24 reach targeted refusal at least 0.5 while keeping untargeted
refusal at or below 0.25.

Sample size helps early and then mostly plateaus. Mean off-diagonal transfer
selectivity is 0.096 at k=2, 0.240 at k=32, and 0.242 at the largest requested
k. In this debug setting, adding more source examples beyond roughly 32 does not
reliably close the gap to native directions.

Topic behavior is not uniform. Space transfers cleanly. Computing transfers
strongly but with more untargeted refusal. Engineering transfers moderately.
Places and religion are weak. People has some high targeted-refusal transfers,
but also substantial spillover for some dataset directions.

## Most Useful Figures

- `runs/full_v1/figures/sample_efficiency_native_transfer.png`: best summary for
  the sample-size question.
- `runs/full_v1/figures/transfer_refusal_vs_untargeted_max_k.png`: best summary
  for whether transfer is selective or mostly spillover.
- `runs/full_v1/figures/topic_transfer_matrices_max_k.png`: best source-target
  dataset view, showing asymmetric successes and failures by topic.

## Interpretation Caveat

This experiment optimizes layer and scale on the target dataset for each source
direction. It tests whether the source direction contains transferable concept
information. It does not test whether the exact source layer and scale can be
reused without retuning.
