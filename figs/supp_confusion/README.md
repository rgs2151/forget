# supp_confusion

## Method

- Read judged confusion evaluations for every available model-dataset pair.
- Average refusal, retention, and fluency for each queried-concept and steering-target pair.
- Arrange each metric as a square matrix and write `plots/supp_confusion.png`.

## Variables

- Matrix rows: queried concepts.
- Matrix columns: steering targets.
- Metric blocks: refusal, retention, and fluency.
- Dataset order: Inhouse, MMLU, RWKU, and ConceptVectors.

## Statistics

- None; every matrix cell is a descriptive mean.

## Legends

- White is 0 and black is 1.
- Columns group model checkpoints by family.
- Crossed panels indicate missing evaluations.
- Only endpoint concept indices are labeled.

## Interpretation

- Compare diagonal target behavior and off-diagonal effects across all evaluated model-dataset pairs.

## Notes

- All matrices are square.

## References

- Code: `supp_confusion.py`.
