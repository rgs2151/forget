# qwen_store_vector_sweep

## Method

- Read Qwen2.5-7B Inhouse baselines and the existing detector, refusal vector, and thresholds from `parking/model_matrix/cache/qwen7b_inhouse/artifacts/main/`.
- Sweep selected layers, scales, prompt-intervention positions, padding behavior, and fixed-vector controls.
- Measure explicit refusal phrases and whether the generated answer begins with refusal language.
- Write samples, sweep tables, summaries, configurations, and reports under `cache/`.

## Variables

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Concepts: configurable subset of the ten Inhouse concepts.
- Primary timing comparison: assistant boundary versus all non-padding prefill tokens.
- Controls: stored gated vectors, fixed steering vectors, additive vectors, and explicit refusal prompting.
- Debug metric: substring-matched IDK rate.

## Statistics

- None; rates are descriptive fractions of sampled generations matching the refusal patterns.

## Legends

- `layer`: zero-indexed decoder layer.
- `scale`: additive intervention strength.
- `idk_rate`: fraction containing a configured refusal phrase.
- `refusal_start_rate`: fraction beginning with refusal language.

## Interpretation

- Moving the intervention through prefill improved the stored gated-vector result.
- Very high scales could force refusal text but also produced malformed generations, so scale alone was not a valid fix.
- Additive ungated steering was a diagnostic control only; it is not the production method because it removes concept-conditioned gating.

## Notes

- The production framework retains `GatedSteering`.
- The historical sweep outputs remain in `cache/`.

## References

- Code: `qwen_store_vector_sweep.py`.
- Production artifacts: `parking/model_matrix/cache/qwen7b_inhouse/artifacts/main/`.

# qwen_clean_vector_experiment

## Method

- Rebuild activation pairs after removing decoded special-token pollution from baseline and refusal answers.
- Fit LDA concept detectors and a shared refusal direction with the same vector mathematics as the main framework.
- Compare gated prefill steering with diagnostic additive controls over all ten Inhouse concepts.
- Write run tables, summaries, configurations, and reports under `cache/`.

## Variables

- Baseline side: generated answers cleaned with the tokenizer.
- Refusal side: refusal answers cleaned before activation pooling.
- Steering timing: assistant boundary or prefill.
- Production-relevant condition: cleaned vectors with the concept gate preserved.

## Statistics

- None; IDK and refusal-start rates are descriptive sample proportions.

## Legends

- Run names identify vector construction, gating control, timing, layer, and scale.
- Summary CSVs report the best observed cells under each diagnostic condition.

## Interpretation

- Cleaning activation answers addressed Qwen-specific special-token pollution.
- Prefill timing improved refusal onset.
- Ungated additive controls demonstrated direction strength but did not test selective concept steering and were rejected as a production method.

## Notes

- `clean_activation_answers=true` is now the framework default.
- Main experiments share cleaned artifacts through `artifacts/main/` and keep gated intervention behavior.

## References

- Code: `qwen_clean_vector_experiment.py`.
- Cleaning implementation: `forget.refuse.activations.clean_answer_text`.
- Gated intervention: `forget.refuse.intervention.GatedSteering`.
