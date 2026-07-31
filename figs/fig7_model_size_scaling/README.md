# score_size

## Method

- For each available Inhouse model run, select the calibrated layer and scale.
- Read the bar evaluation at that cell and compute targeted and untargeted means for refusal, retention, and fluency.
- Fit a separate least-squares line for each metric and pair group.
- Write `plots/score_size.png` and `plots/score_size_summary.csv`.

## Variables

- X axis: model parameter count in billions.
- Y axes: judge refusal, retention, and fluency rates.
- Rows: targeted and untargeted concept pairs.
- Labels: model names.

## Statistics

- Each dashed line is an unweighted first-degree least-squares fit.
- No hypothesis test or uncertainty interval is shown in this diagnostic view.

## Legends

- Dark red identifies targeted points and fits.
- Dark gray points with black fits identify untargeted results.

## Interpretation

- Compare metric trends with model size separately for matched and non-matched pairs.

## Notes

- This is a diagnostic companion to the dataset-resolved refusal figure.

## References

- Code: `model_size_scaling.py`.

# score_size_refusal

## Method

- For every available prefill-logit model-dataset run, select the calibrated layer and scale.
- Compute targeted refusal from `bars_judged.csv` at that cell.
- Fit refusal against model size separately for Inhouse, MMLU, RWKU, and ConceptVectors.
- Write `plots/score_size_refusal.png` and `plots/score_size_refusal_summary.csv`.

## Variables

- X axis: model parameter count in billions.
- Y axis: targeted judge refusal rate.
- Panels: one dataset per panel.
- Included models: every available checkpoint listed in `SCORE_RUNS`.

## Statistics

- Fit: SciPy ordinary least-squares linear regression.
- Null hypothesis: the slope is zero.
- Alternative hypothesis: the slope is nonzero.
- Reported quantities: coefficient of determination \(R^2\) and two-sided slope p-value.
- No multiple-comparison correction is applied across the four descriptive panel fits.

## Legends

- Dark-red points: model-level targeted refusal.
- Dark-red dashed line: dataset-specific linear fit.
- All panels share a 0--1 refusal axis.

## Interpretation

- Compare the association between model size and achieved targeted refusal across datasets.

## Notes

- The fit is descriptive and does not establish a causal effect of parameter count.

## References

- Code: `model_size_scaling.py`.

# publish_table_res

## Method

- Use the same selected prefill-logit layer and scale as `score_size_refusal`.
- Compute targeted and untargeted refusal and retention from the corresponding bar evaluation.
- Record total model layers, selected layer, selected scale, and the four rates.
- Write `plots/publish_table_res.csv`.

## Variables

- Rows: available model-dataset runs ordered by dataset and model family.
- Pair groups: matched and non-matched queried-concept/steering-target pairs.
- Rates: means of binary judge refusal and retention.

## Statistics

- None; table entries are descriptive means rounded to two decimal places.

## Legends

- `ell_star`: selected zero-indexed layer.
- `alpha_star`: selected steering scale.
- `Trg.` and `Untrg.`: matched and non-matched concept pairs.

## Interpretation

- The table provides the operating point and evaluation rates behind the model-size analysis.

## Notes

- Missing model-dataset evaluations are omitted.

## References

- Code: `model_size_scaling.py`.
- Table artifact: `plots/publish_table_res.csv`.
