# supp_bars

## Method

- Read judged bar evaluations for every available model-dataset pair.
- Keep intervention rows and average refusal, retention, and fluency separately for matched and non-matched concept pairs.
- Arrange datasets within metric blocks and models by family.
- Write `plots/supp_bars.png`.

## Variables

- Datasets: Inhouse, MMLU, RWKU, and ConceptVectors.
- Metrics: `judge_refusal`, `judge_retention`, and `judge_fluency`.
- Groups: targeted (`concept == target`) and untargeted (`concept != target`).
- Models: all checkpoints in `forget.plot.results.CALIB_MODELS`.

## Statistics

- None; bars are descriptive means over evaluation rows.

## Legends

- Columns: model checkpoints grouped by family.
- Rows: datasets repeated within refusal, retention, and fluency blocks.
- Dark red: targeted pairs.
- Black: untargeted pairs.
- Cross: missing evaluation.

## Interpretation

- Compare matched and non-matched behavior across the complete evaluated matrix.

## Notes

- Result resolution prefers `prefill_logit`, then `main`.

## References

- Code: `supp_bars.py`.
