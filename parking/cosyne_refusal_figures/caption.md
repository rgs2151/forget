# selectivity

## Caption

Selectivity and spillover. A, exclusive refusal: matched concepts are refused while other concepts remain answerable. B, unrestricted refusal spreads to other concepts. A/B are illustrative schematics. C, model refusal matrices at calibrated settings; each cell averages ten questions. Rows are queried concepts; columns are steered concepts. Darker cells indicate higher refusal.

## Panel Notes

- A/B: original illustrative schematics, placed on the left.
- C: four observed model matrices on the right, Llama/Mistral above Qwen/Phi.

## Checks

- Visual encodings checked against `replot.py`, `plots/selectivity.png`.
- Statistics checked against saved `prefill_logit` confusion responses.
- Remaining uncertainty: judge labels are behavioral measurements.

# layer_metrics

## Caption

Refusal and retention across scale and layer. Columns show Llama, Mistral, and Qwen; rows show refusal and retention. Each colored line is one layer, identified by the color bar. Each point averages 100 targeted calibration responses. Scale s is intervention strength α. Retention here concerns targeted baseline answers, not preservation of other concepts.

## Panel Notes

- Columns: Llama, Mistral, Qwen. Rows: refusal, retention.

## Checks

- Visual encodings checked against `plots/layer_metrics.png` and `replot.py`.
- Statistics checked against `plots/layer_metrics.csv`; all 920 cells have 100
  responses and both metric means; layer colors match the original custom_cmap, with alpha 1.0.
- Remaining uncertainty: no off-target response grid is available in these
  calibration files; no such grid is inferred.

# semantic_cloud

## Caption

Semantic proximity and off-target refusal. A, MiniLM question embeddings and concept centroids in three-dimensional UMAP; turquoise shells show projected question density. Highlighted pairs illustrate high/low cosine similarity. B, ConceptVectors responses grouped below/above median question-to-target similarity (0.238; 450 observations/group/model). Bars: mean refusal; error bars: row-wise standard errors without correction for repeated questions.

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
