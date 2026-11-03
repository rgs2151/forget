# store_vector_sweep

## Method

- Read stored Llama and Phi detector/refusal vectors from each run's `artifacts/main/`.
- Sweep every requested layer and scale with EOS padding and configurable assistant-boundary or prefill intervention.
- Generate a balanced Inhouse sample and measure explicit refusal phrases.
- Write per-run samples, sweep tables, summaries, configurations, and reports under `cache/structured_steering/`.

## Variables

- Models: Llama-3.1-8B and Phi-4.
- Timing: assistant boundary or all non-padding prompt tokens during prefill.
- Phi layer region examined most closely: layers 17--19.
- Phi scale region examined most closely: 60--80.

## Statistics

- None; IDK rate is the descriptive fraction of sampled generations matching a refusal phrase.

## Legends

- Layer and scale identify each sweep cell.
- Summary rows report the best observed IDK rate.

## Interpretation

- The assistant-boundary Phi sweep remained weak.
- Prefill intervention produced a broad high-refusal region, reaching 0.85 IDK at layer 19 and scale 70 in the 100-prompt validation.
- The result motivated the shared configurable prefill timing later tested across model families.

## Notes

- This unit records the diagnostic path; production experiments use the common gated pipeline and judge-scored evaluation.

## References

- Code: `scripts/store_vector_sweep.py`.

# oracle_refuse_prompt

## Method

- Generate from Phi-4 with an explicit refusal system prompt and no activation intervention.
- Measure the same refusal phrases used by the sweep diagnostics.

## Variables

- Model: Phi-4.
- Sample: 20 prompts in the completed control.
- Intervention: none.

## Statistics

- None; the reported rate is a sample proportion.

## Legends

- None; outputs are stored as tables under `cache/structured_steering/`.

## Interpretation

- Phi-4 can produce refusal behavior under the prompt format, so the weak assistant-boundary result was not an absence of refusal capability.

## Notes

- This is a capability control, not a steering result.

## References

- Code: `scripts/oracle_refuse_prompt.py`.

# boundary_vector_experiment

## Method

- Construct fixed vectors from selected assistant-boundary token representations.
- Apply them across configured layers and scales and compare with stored vectors.

## Variables

- Inputs: Inhouse train/test baselines and stored concept detectors.
- Outputs: calibration sample, sweep table, summary, and configuration under the unit cache.

## Statistics

- None; sweep cells report descriptive refusal-pattern rates.

## Legends

- Layer and scale identify each fixed-vector intervention.

## Interpretation

- Boundary-derived replacement vectors did not recover strong selective refusal.

## Notes

- This negative control was not promoted to the main framework.

## References

- Code: `scripts/boundary_vector_experiment.py`.

# logit_lens_probe

## Method

- Read stored Phi baseline-test activations and the refusal vector.
- Project vector-induced hidden-state changes through the model output head.
- Record token-level logit changes for refusal-related strings.

## Variables

- Inputs: `baseline_answer_acts_test.pt` and `v_refuse.pt` from `artifacts/main/`.
- Outputs: top changed tokens and refusal-string deltas.

## Statistics

- None; token-logit differences are direct projections.

## Legends

- Token rows are ranked by change in output logit.

## Interpretation

- The stored Phi vector did not produce a simple refusal-start token signature under this probe.

## Notes

- The probe is diagnostic and does not replace generation evaluation.

## References

- Code: `scripts/logit_lens_probe.py`.

# phi_token_debug

## Method

- Inspect special-token cleanup, padding, prompt boundaries, vector sources, and intervention positions in one exploratory harness.
- Run targeted sweeps and write token reports, combined sweep tables, configurations, and reports under `cache/phi_token_debug/`.

## Variables

- Padding: native or EOS.
- Answer cleanup: tokenizer-based removal of special tokens.
- Vector source: stored or fixed diagnostic vectors.
- Timing: assistant boundary or prefill.

## Statistics

- None; outputs are token diagnostics and descriptive refusal-pattern rates.

## Legends

- Run names encode the tested token, vector, timing, layer, and scale condition.

## Interpretation

- Token cleanup improved generation hygiene but was not sufficient by itself.
- Intervention timing was the dominant Phi improvement in this diagnostic sequence.

## Notes

- Production answer cleanup and prefill timing are implemented in the shared framework.

## References

- Code: `scripts/phi_token_debug.py`.
