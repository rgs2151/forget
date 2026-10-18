# selectivity

## Caption

Concept-selective refusal. Rows are queried concepts; columns are steering
targets. Each cell averages ten questions. Darker cells indicate higher refusal
(0–1). Off-diagonal entries show effects on other concepts. Each model uses its
calibrated layer and strength.

## Panel Notes

- Left to right: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B, Phi-4.

## Checks

- Visual encodings checked against `replot.py`, `plots/selectivity.png`.
- Statistics checked against saved `prefill_logit` confusion responses.
- Remaining uncertainty: judge labels are behavioral measurements.

# layer_metrics

## Caption

Joint characterization of the intervention. Columns are models; rows are
refusal, retention, and fluency. Each cell averages 100 targeted calibration
responses at one layer and strength. All color scales span 0–1. Retention here
concerns targeted answers, not preservation of other concepts. Strength ranges
follow each model's saved sweep.

## Panel Notes

- Columns: Llama, Mistral, Qwen. Rows: refusal, retention, fluency.

## Checks

- Visual encodings checked against `plots/layer_metrics.png` and `replot.py`.
- Statistics checked against `plots/layer_metrics.csv`; all 920 cells have 100
  responses and all three metric means.
- Remaining uncertainty: no off-target response grid is available in these
  calibration files; no such grid is inferred.

# semantic_cloud

## Caption

Semantic proximity and off-target refusal. A, three-dimensional UMAP projection
of MiniLM question embeddings and concept centroids; turquoise shells show
projected question density. Highlighted pairs illustrate high and low cosine
similarity. B, ConceptVectors responses split at median question-to-target
similarity (0.238; 450 observations per group/model). Bars: mean refusal. Error
bars: row-wise standard errors, without correction for repeated questions.

## Panel Notes

- A: original paper's turquoise-shell point cloud, unchanged within its crop.
- B: refusal rates recomputed from the matching `prefill_logit` variant, using
  the cached external-embedding similarities.

## Checks

- Visual encodings checked against `replot.py`, `plots/semantic_cloud.png`.
- Statistics checked against `semantic_near_far.csv`, `semantic_pairs.csv`,
  and exact one-to-one joins to the saved confusion results.
- Remaining uncertainty: descriptive association; the cloud does not depict
  the steered model's hidden states. Error bars ignore shared-question and
  shared-concept dependence.
