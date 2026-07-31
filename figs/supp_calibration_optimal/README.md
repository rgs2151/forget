# supp_optimal

## Method

- Read judged calibration results for every available model-dataset pair.
- Select the operating layer and scale with the framework calibration rule.
- Plot refusal, retention, and fluency across scale at the selected layer and mark the selected scale.
- Write `plots/supp_optimal.png`.

## Variables

- Rows: Inhouse, MMLU, RWKU, and ConceptVectors.
- Columns: model checkpoints grouped by family.
- X axis: configured steering scale.
- Y axis: judge rate from 0 to 1.

## Statistics

- Lines show mean binary judge rates.
- Shaded intervals are Seaborn 95% bootstrap confidence intervals.
- Selection uses `forget.refuse.calibration.select_optimal_config`.

## Legends

- Metric colors follow `STYLE.md`.
- Star: selected scale.
- Cross: missing calibration.

## Interpretation

- Compare the selected operating region across model-dataset pairs.

## Notes

- Scale ranges differ according to the configured model window.

## References

- Code: `supp_optimal.py`.
