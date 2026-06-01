# Structured Steering Debug Hypotheses

This debug tree is for cheap experiments only. It may read existing `store/`
artifacts, but it must not modify `store/` or main pipeline code.

## Baseline Observation

Llama works strongly in the expensive calibration artifacts:

- best cell: layer 16, scale 13
- refusal: 0.93
- fluency: 0.99

Phi does not:

- best judged cell: layer 2, scale 9
- refusal: 0.26
- fluency: 0.32

## Hypothesis Loop

### H1: Phi is only failing because the previous debug probes missed the right layer.

Test: run a full single-layer scan for Phi using the existing full store vectors,
same cheap IDK metric, EOS padding to remove `<|dummy_85|>` output artifacts.

Prediction: if H1 is true, some Phi layer should show a clear IDK peak.

### H2: Llama's success has a layer-local refusal peak that should be visible under the same cheap IDK harness.

Test: run the same store-vector scan for Llama.

Prediction: if the cheap harness is meaningful, Llama should show a high-IDK
band near the known working layer 16.

### H3: Phi's problem is not just layer choice, but a different vector-to-generation signature.

Test: compare Llama vs Phi scan shapes and inspect outputs from their best cells.

Prediction: if H3 is true, Phi either has no strong IDK peak, or its peak
appears only with broken/repetitive text, unlike Llama.

Result: supported. With the same assistant-marker start, Llama had a clear
mid-layer refusal band up to 0.95 IDK, while Phi topped out at 0.10. Llama
outputs directly began with `I don't know...`; Phi outputs mostly clipped facts
or `know` fragments.

### H4: Phi can refuse, but the steering signal must enter during prompt processing.

Test: keep the same existing store vectors and concept gate, but move
`from_position` before the assistant marker so steering applies to all non-pad
prompt tokens during prefill as well as generated tokens.

Prediction: if H4 is true, Phi should recover a later layer band under
all-content prefill steering.

Result: supported. The first all-content scan improved Phi from 0.10 to 0.30.
A mid-layer refinement found layer 17, scale 60 at 0.85 IDK on 20 prompts.
Validation across all 10 concepts with 100 prompts and 64-token generation held
at 0.85 IDK, best layer 19 scale 70.

### H5: Phi lacks the ability to produce the desired refusal text.

Test: run the same calibration prompts with the explicit refuse system prompt
and no activation steering.

Prediction: if H5 is true, the oracle refuse prompt should fail too.

Result: rejected. Phi produced `I don't know.` for 20/20 oracle refuse prompts.
