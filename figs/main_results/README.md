# publish_bar

## Method

- Read judged bar evaluations for four models and four datasets.
- Average refusal separately for rows where the input concept matches the
  steering target and rows where it does not.
- Write `plots/publish_bar.png`.

## Variables

- Models: Llama-3.1-8B, Mistral-7B-v0.3, Qwen-2.5-7B, and Phi-4.
- Datasets: Inhouse, MMLU, RWKU, and ConceptVectors.
- Measure: judge refusal rate.

## Statistics

- None; bars are descriptive means.

## Legends

- Dark red: targeted concepts.
- Black: untargeted concepts.

## Interpretation

- Compare intended refusal with off-target refusal across datasets and models.

## Notes

- Input result resolution prefers `prefill_logit`, then `main`.

## References

- `publish_bar.py`

# publish_confusion

## Method

- Read Inhouse judged confusion evaluations and form concept-by-target matrices.
- Write refusal and retention matrices to `plots/publish_confusion.png` and
  `plots/publish_confusion_ret.png`.

## Variables

- Rows: input concepts.
- Columns: steering targets.
- Values: judge refusal or retention rates.

## Statistics

- None; cells are descriptive means.

## Legends

- White is 0 and black is 1.
- Concept indices follow the row labels on the first model.

## Interpretation

- Diagonal cells show targeted behavior; off-diagonal cells show effects on
  other concepts.

## Notes

- All matrices are square and share the same layout.

## References

- `publish_confusion.py`

# publish_disruption

## Method

- Average targeted and untargeted rates across the four main models for each
  dataset.
- Connect the two averages for each dataset.
- Write retention, fluency, and refusal views.

## Variables

- Outputs: `publish_disruption.png`, `publish_fluency.png`, and
  `publish_refusal.png`.
- Metrics: judge retention, fluency, and refusal.

## Statistics

- None; points are descriptive means across available models.

## Legends

- Line color identifies the dataset.
- Dark-red points are targeted; black points are untargeted.

## Interpretation

- The slope shows how each rate differs between targeted and untargeted inputs.

## Notes

- Retention and refusal use `prefill_logit`; fluency uses `main`.

## References

- `publish_disruption.py`

# publish_params

## Method

- Read judged calibration sweeps for the three main 7--8B models.
- Plot metric trajectories at the selected layer and layer-wise trajectories
  across scale.
- Write full and reduced views.

## Variables

- Outputs: `publish_params.png` and `publish_params_min.png`.
- Metrics: refusal, retention, and fluency; the reduced view omits fluency.
- Color: layer depth.

## Statistics

- Lines show mean rates; shaded intervals are 95% confidence intervals from
  Seaborn's bootstrap estimator.
- Stars mark the selected scale at the selected layer.

## Legends

- Metric colors follow `STYLE.md`.
- The sequential colorbar identifies layer depth.

## Interpretation

- The figure shows how refusal and response quality vary across scale and layer.

## Notes

- The plotting logic and current appearance are preserved from the original
  publication module.

## References

- `publish_params.py`
