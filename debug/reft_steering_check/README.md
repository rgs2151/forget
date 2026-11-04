# check

## Method
- Inspect four prompts directly: one training and one test prompt each for bacteria and cats.
- Reuse the small study's layer-6 directions and gate. Generate at factors 0, 2, and 5, giving 12 answers total on both GPUs. Factor zero is no intervention.
- Measure the actual block-output change after the shared steering hook, compare it with the requested change, and inspect the first-token prediction and generated text.

## Variables
- Input: `parking/refusal_mechanisms/cache/axbench_small/`.
- Model: Llama-3.2-1B-Instruct. No retraining, judge, or change to existing artifacts.
- Output: `plots/generations.csv`. Run: `python -m debug.reft_steering_check.check`.

## Statistics
- None. Direct generation and hook measurements only; no experimental performance claim.

## Legends
- `factor`: multiplier on the stored maximum-activation magnitude and learned unit direction.
- `changed_tokens`: token states with a nonzero measured block-output change, including prefill and generation.
- `max_actual_delta`: largest measured change norm; `max_hook_error`: largest difference between the requested and observed block output.
- `p_first_I`: probability of the first token of `I`, not a refusal score.

## Interpretation
- A nonzero measured change confirms the hook applies an intervention. It does not establish that the learned direction induces refusal.
- Completed check: the requested and observed block outputs matched exactly. Active changes had norms about 3.59 at factor 2 and 9.00 at factor 5, so the intervention was not silently disabled.
- None of these 12 generations produced refusal. Increasing the factor instead changed factual wording or corrupted answers; for the bacterial training prompt, outputs changed from `Sulfate-reducing bacteria.` to `Nitrogen fixation.` and then `Nit answer.`
- Two of four prompts activated the gate during prefill; the other two activated it only during generation. Nonzero factors changed only one to four token states per answer. These are direct observations on four prompts, not population estimates.
- Both selected training prompts occur in the shared refusal direction's actual 50-question training set. This is therefore not merely a held-out generalization failure.
- Inspection of `training_examples(..., '__refusal__', ...)` confirms that all 50 positive refusal questions are reused with normal-answer targets as negatives, under the same intervention with no distinguishing input condition. This mixes conflicting response targets for an intended refusal controller. It is a training-design problem, not proof of the sole cause of failure.

## Notes
- The concept gate stays enabled. This check does not introduce an ungated research condition.
- All diagnostic files stay in this folder. The main pipeline and completed results remain unchanged.
- The protected result inventory was checked after this run and remained unchanged. No additional scientific experiment or judge call was run.

## References
- AxBench `AdditionIntervention.forward`: `max_act * mag * weight`.
- Source: `/home/dev/forget/ref/repos/axbench/axbench/models/interventions.py` and `model.py`.
