# supp_bars

## Method

- Plot targeted and untargeted judged means for refusal, retention, and fluency
  across every available model-dataset pair.

## Variables

- Rows group metrics and datasets; columns group models by family.

## Statistics

- None; bars are descriptive means.

## Legends

- Dark red is targeted and black is untargeted.

## Interpretation

- Compare on-target behavior and off-target effects across the full matrix.

## Notes

- Missing evaluations are marked with a cross.

## References

- `supp_bars.py`

# supp_confusion

## Method

- Plot square judged confusion matrices for refusal, retention, and fluency
  across every available model-dataset pair.

## Variables

- Rows group metrics and datasets; columns group models by family.

## Statistics

- None; cells are descriptive means.

## Legends

- White is 0 and black is 1.

## Interpretation

- Compare diagonal and off-diagonal behavior across the full matrix.

## Notes

- Missing evaluations are marked with a cross.

## References

- `supp_confusion.py`

# supp_optimal

## Method

- Plot the across-scale calibration summary at the selected layer for each
  available model-dataset pair.

## Variables

- Metrics: refusal, retention, and fluency.

## Statistics

- Lines show mean rates with 95% bootstrap confidence intervals.

## Legends

- Metric colors follow `STYLE.md`.

## Interpretation

- Compare selected operating regions across models and datasets.

## Notes

- Missing calibrations are marked with a cross.

## References

- `supp_optimal.py`

# supp_refuse

## Method

- Plot layer-wise calibration trajectories across scale separately for refusal,
  retention, and fluency.

## Variables

- Outputs: `supp_refuse.png`, `supp_retain.png`, and `supp_fluency.png`.
- Color: layer depth.

## Statistics

- Each line is the observed rate trajectory for one calibrated layer.

## Legends

- The sequential colorbar identifies early and late layers.

## Interpretation

- Compare layer sensitivity across the complete available matrix.

## Notes

- Missing calibrations are marked with a cross.

## References

- `supp_refuse.py`
