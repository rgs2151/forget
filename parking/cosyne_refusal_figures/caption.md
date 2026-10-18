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

Layer and strength dependence. Refusal (top) and retention (bottom) versus scale s = α. Each line is one layer; each point averages 100 target-concept responses. Retention measures agreement with the unsteered answer, not preservation of other concepts.

## Panel Notes

- Figure 3. Top: refusal. Bottom: retention. Columns: Llama, Mistral, Qwen.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- 920 settings, 100 targeted responses each; both means unchanged. Original layer colors, alpha 1.0.
- No additional results are asserted by this layout revision.

# semantic_cloud

## Caption

Semantic similarity and off-target refusal. A, MiniLM question embeddings and concept centroids in three-dimensional UMAP; shells show projected density. Labels mark example pairs and their cosine similarity. B, refusal below (“Far”) or above (“Near”) median question-to-target similarity (0.238; 450 observations/group/model). Bars: means ± row-wise standard errors, without correction for repeated questions.

## Panel Notes

- Figure 4. A: original point-cloud geometry with separated labels. B: narrow near/far bars.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- Near/far split and all means/SEs unchanged. Repeated questions are not independent; no inferential claim.
- No additional results are asserted by this layout revision.
