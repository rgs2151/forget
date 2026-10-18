# refusal_example

## Caption

Example response before and after refusal steering.

## Panel Notes

- Figure 1. Single panel: user-supplied answer/refusal illustration, wrapped beside Methods.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- No statistics; illustrative output. Original supplied image is byte-identical.
- Caption language simplified; figure images, data, and definitions preserved. No new result is asserted.

# selectivity

## Caption

Targeted refusal. A, intended refusal on target-concept questions only. B–E, measured refusal across models. Rows show question concepts; columns show steering targets. Darker cells indicate higher refusal.

## Panel Notes

- Figure 2. A: targeted-refusal illustration. B–E: measured Llama, Mistral, Qwen, Phi matrices in one row.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- Each measured cell averages ten responses; numeric selectivity.csv unchanged.
- Caption language simplified; figure images, data, and definitions preserved. No new result is asserted.

# layer_metrics

## Caption

Layer and strength dependence. Refusal (top) and retention (bottom) on target-concept questions across steering strengths. Lines are colored by layer. Retention is agreement with the answer before steering.

## Panel Notes

- Figure 3. Top: refusal. Bottom: retention. Columns: Llama, Mistral, Qwen.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- 920 settings, 100 targeted responses each; both means unchanged. Continuous linear-light RGB interpolation between blue, red, and green anchors at 0, 0.5, and 1 (midpoint #980000 matches Figures 2/4B); alpha 1.0; no numeric changes.
- Caption language simplified; figure images, data, and definitions preserved. No new result is asserted.

# semantic_cloud

## Caption

Semantic similarity and spillover. A, a semantic map of questions and concepts; shading shows question density. B, off-target refusal for questions with higher (Near) or lower (Far) similarity to the target, split at the median. Bars show means ± standard errors, unadjusted for repeated questions.

## Panel Notes

- Figure 4. A: reconstructed original projected geometry without a frame, restored example-pair connections, inward Near/Far callouts, and a three-axis orientation key. Definitions remain in the caption. B: taller, narrow near/far bars with an internal legend and a single-line y-axis title. Figure 4 is approved and frozen.

## Checks

- Visual encoding checked against the final Word PDF and unit plots.
- Near/far split and all means/SEs unchanged. Repeated questions are not independent; no inferential claim.
- Caption language simplified; figure images, data, and definitions preserved. No new result is asserted.
