# transfer_feasibility

## Method

- Learn one gated refusal direction from each MMLU source category and apply it to semantically matched Inhouse target concepts in Llama-3.2-1B.
- Reuse the framework's LDA detector mathematics, shared refusal direction, logit judge, and prefill intervention.
- Optimize layer and scale on held-out target calibration questions, then evaluate disjoint targeted and untargeted questions.
- Write `plots/transfer_feasibility.png` and full run tables under `cache/main/`.

## Variables

- Space: MMLU astronomy to Inhouse `the_moon`.
- Places: MMLU high-school geography to Inhouse `paris` and `united_states`.
- Engineering: MMLU electrical engineering to Inhouse `lasers`.
- Layers: all 16 Llama-3.2-1B layers.
- Scales: integers 1 through 10.
- Evaluation: 60 targeted and 60 balanced untargeted questions per category.

## Statistics

- None; targeted and untargeted refusal are means of binary logit-judge decisions.
- Layer and scale are selected by the existing calibration objective on held-out questions.

## Legends

- Categories: space, places, and engineering.
- Conditions: baseline, native Inhouse direction, and transferred MMLU direction.
- Rates: targeted and untargeted judge refusal.

## Interpretation

- Full MMLU directions raised targeted Inhouse refusal to 0.833 for space, 0.867 for places, and 0.833 for engineering.
- Untargeted refusal was 0.150, 0.500, and 0.033, respectively.
- Direction transfer was selective for space and engineering and broad for places.

## Notes

- The frozen design is `frozen_config.json`.
- The experiment answers feasibility for one model and one MMLU-to-Inhouse design; it does not establish transfer across every dataset pair or model family.

## References

- Experiment: `run_experiment.py`.

# sample_sensitivity

## Method

- Subsample positive MMLU source examples while holding all non-target source examples fixed.
- Refit the source direction for each size and draw.
- Re-optimize target layer and scale and evaluate the same disjoint Inhouse protocol.
- Write `plots/sample_sensitivity.png`.

## Variables

- Positive sample sizes: 2, 4, 8, 16, 32, 64, and the full 93 examples.
- Draw seeds: 42, 314, and 2718 except for the single full-data condition.
- Outcomes: targeted and untargeted refusal.

## Statistics

- Lines summarize descriptive refusal rates by source sample size.
- Three independent draws are shown for subsampled conditions; the full-data point has one direction.

## Legends

- X axis: number of positive source examples.
- Y axis: judge refusal rate.
- Gray traces: individual draws.
- Dark-red trace: mean across draws.

## Interpretation

- Two to eight examples were insufficient.
- At 32 examples, targeted refusal was 0.750, 0.561, and 0.650 for space, places, and engineering.
- At 64 examples, the corresponding rates were 0.794, 0.783, and 0.794.
- More examples did not guarantee specificity; places untargeted refusal increased at larger sizes.

## Notes

- Negative sampling remains fixed across source-size conditions.

## References

- Experiment: `run_experiment.py`.

# concept_footprint

## Method

- Group judged Inhouse evaluations by queried concept for each transferred category and condition.
- Mark the intended target concepts and display refusal across the complete Inhouse inventory.
- Write `plots/concept_footprint.png` and the underlying diagnostic table under `cache/main/diagnostics/`.

## Variables

- Queried concepts: the ten Inhouse concepts.
- Conditions: baseline, native, and transferred.
- Measure: mean binary judge refusal.

## Statistics

- None; concept rates are descriptive means.

## Legends

- Boxes identify intended target concepts.
- Bar/point position identifies the queried Inhouse concept.

## Interpretation

- Engineering was concentrated on lasers; space was concentrated on the Moon with smaller off-target effects.
- Places affected several additional concepts and was the least selective transferred direction.

## Notes

- Each target category contributes 60 targeted questions; the untargeted total is 60.

## References

- Diagnostics: `plot_diagnostics.py`.

# calibration_landscape

## Method

- Read every native and transferred calibration cell.
- Arrange targeted refusal over all 16 layers and scales 1--10.
- Mark the selected operating cell and write `plots/calibration_landscape.png`.

## Variables

- Rows/curves: decoder layers.
- X axis: steering scale.
- Measure: targeted judge refusal.
- Conditions: native and transferred direction for each category.

## Statistics

- None; cells are descriptive calibration rates.

## Legends

- Color identifies layer depth.
- Star marks the selected layer-scale cell.

## Interpretation

- Transferred directions selected layers 4 or 5, while native Inhouse directions selected layers 6 or 9.
- High-performing cells formed coherent regions rather than isolated maxima.

## Notes

- Calibration questions are disjoint from the evaluation questions.

## References

- Diagnostics: `plot_diagnostics.py`.

# sample_vector_diagnostics

## Method

- Compare each subsampled source direction with the full-data direction.
- Record cosine similarity and the selected steering scale for every category, size, and draw.
- Write `plots/sample_vector_diagnostics.png`.

## Variables

- X axis: number of positive source examples.
- Measures: cosine similarity to the full direction and selected scale.
- Groups: category and draw.

## Statistics

- Gray traces are individual draws and the red trace is their mean.
- No inferential test is used.

## Legends

- Panels separate direction similarity and selected scale.
- Categories are shown separately.

## Interpretation

- Mean similarity reached 0.742--0.806 with 32 examples and 0.916--0.918 with 64.
- Scale stabilized near 5--6 for space and engineering while places remained more variable.

## Notes

- The 93-example condition is the single full-data direction.

## References

- Diagnostics: `plot_diagnostics.py`.
