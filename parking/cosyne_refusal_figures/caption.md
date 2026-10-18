# refusal_example

## Caption

Illustrative answer before and after refusal steering.

## Panel Notes

- Figure 1. Single panel: user-supplied answer/refusal illustration, wrapped beside Methods.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- No statistics; illustrative output. Original supplied image is byte-identical.
- No additional results are asserted by this layout revision.

# selectivity

## Caption

Targeted refusal. A, schematic: target-concept questions are refused; other questions remain answerable. B–E, measured refusal at calibrated settings (ten questions/cell). Rows: queried concepts; columns: steered concepts. Darker cells indicate higher refusal (0–1).

## Panel Notes

- Figure 2. A: targeted-refusal illustration. B–E: measured Llama, Mistral, Qwen, Phi matrices in one row.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- Each measured cell averages ten responses; numeric selectivity.csv unchanged.
- No additional results are asserted by this layout revision.

# layer_metrics

## Caption

Layer and strength dependence. Refusal (top) and retention (bottom) versus scale s = α. Each line is one layer. Continuous colors progress from blue (early) through red (middle) to green (late), with one color bar per panel. Each point averages 100 target-concept responses. Retention measures agreement with the unsteered answer, not preservation of other concepts.

## Panel Notes

- Figure 3. Top: refusal. Bottom: retention. Columns: Llama, Mistral, Qwen.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- 920 settings, 100 targeted responses each; both means unchanged. Continuous linear-light RGB interpolation between blue, red, and green anchors at 0, 0.5, and 1; alpha 1.0; no numeric changes.
- No additional results are asserted by this layout revision.

# semantic_cloud

## Caption

Semantic similarity and off-target refusal (spillover). A, MiniLM question embeddings and concept centroids in three-dimensional UMAP; shells show projected density. Near and far examples illustrate high and low semantic similarity. B, question–target pairs below (Far) or at/above (Near) median similarity (450 observations/group/model). Bars: means ± row-wise standard errors, without correction for repeated questions.

## Panel Notes

- Figure 4. A: reconstructed original projected geometry without a frame, restored example-pair connections, inward Near/Far callouts, and a three-axis orientation key. Definitions remain in the caption. B: taller, narrow near/far bars with an internal legend and a single-line y-axis title. Figure 4 is approved and frozen.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- Near/far split and all means/SEs unchanged. Repeated questions are not independent; no inferential claim.
- No additional results are asserted by this layout revision.
