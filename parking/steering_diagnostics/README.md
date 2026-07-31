# prefill_evidence

## Method

- Read completed Phi and Qwen assistant-boundary and prefill debug summaries.
- Select the directly comparable refusal-pattern rates.
- Plot the timing comparison and write `plots/prefill_evidence.png`.

## Variables

- Inputs: tracked run summaries under `parking/phi_steering/cache/` and `parking/qwen_steering/cache/`.
- Models: Phi-4 and Qwen2.5-7B.
- Timing conditions: assistant boundary and prefill.
- Measure: substring-matched IDK rate.

## Statistics

- None; bars/points are descriptive rates from fixed debug samples.

## Legends

- Dark red: prefill intervention.
- Gray: assistant-boundary intervention.
- Grouping: model.

## Interpretation

- Prefill timing produced the strongest common improvement observed in the two diagnostic investigations.

## Notes

- These are cheap debug metrics, not judge-scored publication results.

## References

- Code: `make_debug_figures.py`.
- Source units: `parking/phi_steering/` and `parking/qwen_steering/`.

# phi_qwen_debug_sweeps

## Method

- Read the best completed Phi and Qwen layer-scale debug sweeps.
- Plot refusal-pattern rate across scale with one trajectory per tested layer.
- Write `plots/phi_qwen_debug_sweeps.png`.

## Variables

- Phi input: `phi_all_content_best_validation_v1`.
- Qwen input: `qwen_clean_vectors_additive_all10_validation_v1`.
- X axis: steering scale.
- Y axis: substring-matched IDK rate.

## Statistics

- None; each line is a descriptive layer-wise trajectory.

## Legends

- Line color identifies layer.
- Panels separate Phi and Qwen.

## Interpretation

- The sweeps show where the diagnostic refusal behavior appeared across layer and scale.

## Notes

- The Qwen additive sweep is retained as a direction-strength control, not as the production gated method.

## References

- Code: `make_debug_figures.py`.
