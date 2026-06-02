# Phi steering debug

This tree is debug-only. It reads existing `store/phi4_inhouse` and Llama control artifacts, writes local sweep outputs, and does not change the main pipeline or expensive store results.

## Result

Phi can be steered, but not with the same timing that works for Llama.

Best debug candidate:

| setting | value |
| --- | --- |
| vector | existing Phi LDA store vector |
| padding | EOS padding |
| steering window | all non-pad prompt tokens during prefill, then generation |
| layer band | 17-19 |
| scale band | 60-80 |
| best validation cell | layer 19, scale 70 |
| cheap IDK rate | 0.85 over 100 prompts |

This is not production-clean yet. Refusal usually appears, but some outputs ramble after refusing and some concepts remain weak.

## Investigation chain

### 1. Suspicion: the cheap harness might be misleading

Test: run the same cheap substring-IDK sweep on Llama, where the expensive results already show strong steering.

Result: Llama reproduced the known band. `llama_store_all_layers_v1` reached `0.95` IDK at layer 17, scale 17.

Conclusion: the cheap harness is crude, but it can detect a real refusal band.

### 2. Suspicion: Phi only needed a full layer scan

Test: run Phi with existing store vectors, EOS padding, assistant-marker steering, and all layers.

Result: `phi_store_all_layers_v2` topped out at `0.10` IDK.

Conclusion: the earlier Phi failure was not just a missed layer.

### 3. Suspicion: Phi's vector was bad or pointed at the wrong token signature

Tests:

| test | result |
| --- | --- |
| token cleanup and EOS padding | cleaner text, no real refusal gain |
| `I don't know.` lm-head vector | `0.00` IDK |
| assistant-boundary vector | `0.00` IDK |
| logit-lens probe | Phi store vector did not create a clean refusal-start signature |

Conclusion: token artifacts were real but not the main failure. Simple replacement vectors were worse than the store vector.

### 4. Suspicion: Phi may not be capable of refusing under this prompt style

Test: use the explicit refuse system prompt with no activation steering.

Result: `phi_oracle_refuse_prompt_v1` produced IDK on `20/20` prompts.

Conclusion: Phi can refuse. The issue is injection timing, not model capability.

### 5. Suspicion: Phi needs steering before the assistant boundary

Test: keep the existing store vector and concept gate, but apply steering across all non-pad prompt content during prefill instead of starting at the assistant marker.

Result:

| run | best cell | IDK |
| --- | --- | ---: |
| `phi_store_all_content_v1` | layer 6, scale 20 | 0.30 |
| `phi_store_all_content_refine_v1` | layer 10, scale 40 | 0.40 |
| `phi_store_all_content_mid_layers_v1` | layer 17, scale 60 | 0.85 |
| `phi_all_content_best_validation_v1` | layer 19, scale 70 | 0.85 |

Conclusion: this is the first credible Phi fix. The production experiment to prioritize is making steering start configurable and testing all-content prefill steering for Phi before changing vector construction.

## Files

| path | contents |
| --- | --- |
| `scripts/` | Phi debug scripts used for sweeps and probes |
| `structured_steering/` | structured Llama/Phi control and hypothesis runs |
| `phi_token_debug/` | early exploratory token/vector runs |

