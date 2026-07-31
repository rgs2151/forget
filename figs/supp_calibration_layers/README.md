# supp_refuse

## Method

- Read judged calibration results for every available model-dataset pair.
- Plot refusal across steering scale separately for every calibrated layer.
- Write `plots/supp_refuse.png`.

## Variables

- Columns: Inhouse, MMLU, RWKU, and ConceptVectors.
- Rows: model checkpoints grouped by family.
- Measure: mean `judge_refusal`.

## Statistics

- None; each trajectory is the observed descriptive rate for one layer.

## Legends

- X axis: steering scale.
- Y axis: refusal rate.
- Line color: layer depth.
- Cross: missing calibration.

## Interpretation

- Compare refusal sensitivity across layers and scales.

## Notes

- Model-specific scale windows are retained.

## References

- Code: `supp_refuse.py`.

# supp_retain

## Method

- Apply the same full layer-scale layout to `judge_retention`.
- Write `plots/supp_retain.png`.

## Variables

- Panels and scale ranges match `supp_refuse`.
- Measure: mean binary judge retention.

## Statistics

- None; each trajectory is a descriptive layer-wise rate.

## Legends

- X axis: steering scale.
- Y axis: retention rate.
- Line color: layer depth.
- Cross: missing calibration.

## Interpretation

- Compare answer-retention sensitivity across layers and scales.

## Notes

- None yet.

## References

- Code: `supp_refuse.py`.

# supp_fluency

## Method

- Apply the same full layer-scale layout to `judge_fluency`.
- Write `plots/supp_fluency.png`.

## Variables

- Panels and scale ranges match `supp_refuse`.
- Measure: mean binary judge fluency.

## Statistics

- None; each trajectory is a descriptive layer-wise rate.

## Legends

- X axis: steering scale.
- Y axis: fluency rate.
- Line color: layer depth.
- Cross: missing calibration.

## Interpretation

- Compare fluency sensitivity across layers and scales.

## Notes

- None yet.

## References

- Code: `supp_refuse.py`.
