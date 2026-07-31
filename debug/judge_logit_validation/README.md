# validate_logit_judge

## Method

- Define six hand-labeled responses covering clear refusal, direct answer, retained paraphrase, incorrect answer, short refusal, and degenerate text.
- Score the same cases with logit and reasoning judge modes.
- Compare binary refusal, retention, and fluency decisions with the expected labels.
- Write case-level and summary CSVs under `plots/`.

## Variables

- Judge: `AtlaAI/Selene-1-Mini-Llama-3.1-8B`.
- Axes: refusal, retention, and fluency.
- Outputs: manual judged tables, accuracy summaries, and prompt/suffix probes.

## Statistics

- Accuracy is the fraction of six expected binary labels matched on each axis.
- No inferential test is used because the cases are a fixed diagnostic set.

## Legends

- `expected_rate`: mean hand label.
- `judge_rate`: mean binary judge decision.
- `accuracy`: agreement with hand labels.

## Interpretation

- Refusal and retention matched all hand labels in both modes.
- Logit fluency marked the degenerate case as fluent, giving 5/6 agreement; reasoning mode matched all six.

## Notes

- This test checks clear cases, not calibrated judge accuracy on a representative sample.

## References

- Probe: `validate_logit_judge.py`.
- Outputs: `plots/*.csv`.
