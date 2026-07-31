# Cross-dataset direction transfer

This unit tests whether concept directions learned from MMLU transfer to
semantically matched Inhouse concepts in Llama-3.2-1B, and how transfer changes
with the number of positive source examples.

The design is frozen in `frozen_config.json`. The unit reads existing
activations and vectors from `parking/model_matrix/cache/llama32_1b_mmlu` and
`parking/model_matrix/cache/llama32_1b_inhouse`. It writes only under `parking/direction_transfer/cache`.

## Categories

| category | MMLU source | Inhouse target |
| --- | --- | --- |
| space | `astronomy` | `the_moon` |
| places | `high_school_geography` | `paris`, `united_states` |
| engineering | `electrical_engineering` | `lasers` |

## Fixed method

- Llama-3.2-1B with gated prefill steering.
- Existing LDA detector mathematics and universal refusal direction.
- Full-data calibration over all 16 layers and scales 1 through 10.
- Existing logit judge and rubric.
- Disjoint target calibration and evaluation questions.
- Direct reporting of targeted and untargeted refusal rates.
- Sample sizes `2, 4, 8, 16, 32, 64, 93`, with three draws except for the
  full-data condition.
- All non-target MMLU examples remain fixed while positive source examples are
  subsampled.

## Outputs

The completed run writes:

- `full_summary.csv`: baseline, native, and transferred refusal rates.
- `sample_summary.csv`: refusal rates by category, source size, and draw.
- `plots/transfer_feasibility.png`: full-data transfer results.
- `plots/sample_sensitivity.png`: source-sample sensitivity.
- `plots/concept_footprint.png`: refusal rates for each Inhouse concept.
- `plots/calibration_landscape.png`: native and transferred refusal across
  every calibrated layer and scale; stars mark the selected cells.
- `plots/sample_vector_diagnostics.png`: direction similarity and selected
  scale across source sample sizes.
- `report.md`: concise answers to the two research questions.

Raw generations, judged CSVs, fitted debug vectors, and logs remain under
`cache/` and are ignored by git.

The concept-footprint plot boxes the intended target concepts. Each target
category contributes 60 evaluation questions; the untargeted total is 60,
balanced across the remaining concepts. In the sample diagnostics, gray lines
are the three independent draws and the red line is their mean. The 93-example
point is the single full-data direction.

## Results

The full MMLU directions raise targeted Inhouse refusal to 0.833 for space,
0.867 for places, and 0.833 for engineering. Untargeted refusal is 0.150,
0.500, and 0.033, respectively. Direction transfer therefore works cleanly for
space and engineering, while the places direction transfers with substantial
untargeted refusal.

Two to eight positive source examples are insufficient. Thirty-two examples
produce targeted refusal rates of 0.750, 0.561, and 0.650 for space, places, and
engineering; 64 examples raise these to 0.794, 0.783, and 0.794. More examples
do not always improve specificity: places untargeted refusal rises from 0.161
at 32 examples to 0.433 at 64 and 0.500 with all 93 examples.

These results answer the two questions for one model and one MMLU-to-Inhouse
transfer design. They do not establish transfer across model families or every
possible source-target dataset pair.

## Diagnostic findings

- The engineering direction is the most selective. It refuses lasers at 0.833;
  among the other concepts, only bacteria shows a new refusal signal at 0.167.
- The space direction remains concentrated on the Moon at 0.833, with its
  largest off-target rates on Obama at 0.429 and dogs at 0.286.
- The places direction is broad. In addition to Paris at 0.933 and the United
  States at 0.800, it refuses cats and the Moon at 0.857, Obama at 0.750, and
  chess at 0.571.
- The transferred directions select layers 4 or 5, while the native Inhouse
  directions select layers 6 or 9. The layer-scale plots show coherent regions
  around the selected cells rather than isolated single-cell peaks.
- Mean cosine similarity to the full direction reaches 0.742--0.806 with 32
  source examples and 0.916--0.918 with 64. The selected scale also stabilizes
  near 5--6 for space and engineering, while places remains more variable.
