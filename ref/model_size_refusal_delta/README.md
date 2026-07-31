# temp_refusal_delta

## Method

- Historical diagnostic comparing model size with the difference between targeted and untargeted refusal.
- The generating script was not retained.

## Variables

- Inputs: model parameter counts and refusal-rate differences recorded in `temp_refusal_delta_summary.csv`.
- Output: `temp_refusal_delta.png`.

## Statistics

- The table records a linear fit, coefficient of determination, and p-value.
- The calculation is retained for provenance but is not part of the active model-size analysis.

## Legends

- X axis: model size.
- Y axis: targeted refusal minus untargeted refusal.
- Points: model-level values.
- Line: historical linear regression fit.

## Interpretation

- This was an exploratory specificity-versus-size view.

## Notes

- Use the active figure rather than this diagnostic for paper results.

## References

- Active analysis: `figs/fig7_model_size_scaling/`.
