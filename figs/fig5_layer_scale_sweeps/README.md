# publish_params

## Method

- Read Inhouse judged calibration sweeps for Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B.
- Select each model's operating layer and scale with the framework calibration rule.
- Plot refusal, retention, and fluency across scale at the selected layer and across all calibrated layers.
- Write `plots/publish_params.png`.

## Variables

- Rows: the three model checkpoints.
- Left column: metric rates across scale at the selected layer.
- Remaining columns: one trajectory per layer for refusal, retention, and fluency.
- Star: selected scale, positioned at the refusal value.
- Layer color: `LAYER_CMAP_COLORS` from `forget.plot.plot`.

## Statistics

- Lines show mean binary judge rates.
- Shaded intervals in the selected-layer panels are Seaborn 95% bootstrap confidence intervals.
- No hypothesis test is used for operating-point selection in this figure.

## Legends

- X axis: steering scale.
- Y axis: rate from 0 to 1.
- Metric colors follow `STYLE.md`.
- Sequential line color identifies layer depth.

## Interpretation

- Compare how the three judged rates change with intervention strength and model layer.

## Notes

- Scale ranges follow each model's configured calibration sweep.

## References

- Code: `publish_params.py`.
- Selection rule: `forget.refuse.calibration.select_optimal_config`.

# publish_params_min

## Method

- Use the same calibration data and selected operating points as `publish_params`.
- Omit fluency and harmonic trajectories while retaining refusal, retention, and all-layer views.
- Write `plots/publish_params_min.png`.

## Variables

- Rows, models, scales, layers, and selected cells match `publish_params`.
- Star x position is the selected scale; star y position is the refusal rate.

## Statistics

- Mean rates and 95% bootstrap confidence intervals follow the full figure.

## Legends

- X axis: steering scale.
- Y axis: rate from 0 to 1.
- Sequential line color identifies layer depth.

## Interpretation

- This reduced view isolates refusal and retention sensitivity.

## Notes

- The middle-row legend is retained; the first and last row legends are omitted.

## References

- Code: `publish_params.py`.
