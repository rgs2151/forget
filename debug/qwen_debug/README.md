# Qwen steering debug

This tree is for Qwen-only debug experiments. It reads `store/qwen7b_inhouse`, writes local sweep artifacts, and does not modify the main pipeline or `store`.

## Result

Qwen has a credible debug fix, but it is different from Phi.

Best debug candidate:

| setting | value |
| --- | --- |
| vector | debug-rebuilt LDA vector from cleaned baseline answers and plain `I don't know.` refusal answers |
| application | ungated additive refusal vector |
| padding | EOS padding |
| steering window | all non-pad prompt tokens during prefill, then generation |
| layer band | 14, 15, or 17 |
| best scale | 70 |
| cheap IDK rate | 1.00 over 100 prompts |
| strict refusal-start rate | 1.00 over 100 prompts |

This is still a debug result. It uses cheap substring and start-pattern matching, not the judge.

## Investigation chain

### 1. Suspicion: the existing Qwen failure may be the same timing issue as Phi

Starting point: existing judged calibration for Qwen is poor. The best judged aggregate in `store/qwen7b_inhouse/calibration_judged.csv` is about `0.12`, with the best refusal cells around layer 14 at high scale.

Test: keep existing Qwen store vectors, use EOS padding, and move steering from assistant-marker-only to all-content prefill.

Result:

| run | setup | best IDK |
| --- | --- | ---: |
| `qwen_assistant_marker_mid_band_v1` | store vector, assistant marker | 0.10 |
| `qwen_store_all_content_v1` | store vector, all-content prefill | 0.35 |

Conclusion: the Phi timing lesson transfers partially. It helps, but it does not fix Qwen.

### 2. Suspicion: Qwen may not obey the refusal instruction

Test: use the explicit refuse system prompt with no activation steering.

Result: `qwen_oracle_refuse_prompt_v1` reached `1.00` IDK on 20 prompts.

Conclusion: Qwen can refuse. The problem is the steering setup, not model capability.

### 3. Suspicion: store-vector scale is too low

Test: keep existing store vectors and all-content prefill, then increase scale in the best layer band.

Result: `qwen_all_content_high_scale_v1` reached `0.50` IDK at layer 15, scale 200, but outputs were repetitive and malformed.

Conclusion: higher scale can force refusal substrings, but this is not a real fix.

### 4. Suspicion: Qwen store vectors are contaminated by answer-token pollution

Reason: Qwen baseline answers in `store/qwen7b_inhouse` contain repeated `<|endoftext|>` tokens. The original activation pooling treats those as answer tokens.

Test: rebuild Qwen vectors in debug from:

| side | answer text |
| --- | --- |
| baseline | baseline output decoded with special tokens removed |
| refusal | plain `I don't know.` |

Then test those vectors with the same gated all-content steering.

Result:

| run | scope | best IDK |
| --- | --- | ---: |
| `qwen_clean_answer_vectors_all_content_v1` | 4 concepts, 20 prompts | 0.65 |
| `qwen_clean_vectors_all10_validation_v1` | 10 concepts, 100 prompts | 0.57 |

Conclusion: clean vector construction is real progress, but gated steering still fails too often.

### 5. Suspicion: Qwen's concept gate is fragmenting the refusal intervention

Test: use the same clean debug vectors, but apply the refusal vector additively instead of through the concept gate.

Result:

| run | steering window | best IDK | refusal-start rate |
| --- | --- | ---: | ---: |
| `qwen_clean_vectors_additive_all10_validation_v1` | all-content prefill | 1.00 | 1.00 |
| `qwen_clean_vectors_additive_assistant_control_v1` | assistant marker | 0.84 | 0.01 |

Conclusion: Qwen needs both clean vectors and all-content prefill. Additive assistant-marker steering can insert refusal language after a factual start, but all-content prefill makes the answer begin as a refusal.

## Recommendation

Prioritize a Qwen main-pipeline experiment with:

| setting | value |
| --- | --- |
| model | `Qwen/Qwen2.5-7B-Instruct` |
| vector construction | clean baseline answers, plain `I don't know.` refusal answers |
| steering mode | additive refusal vector |
| steering start | all non-pad prompt content |
| generation padding | EOS |
| layer candidates | 14, 15, 17 |
| scale candidates | 60, 70 |

Do not prioritize fixed lm-head IDK vectors or high-scale existing store vectors. They looked weak or fake in debug.

## Files

| path | contents |
| --- | --- |
| `qwen_store_vector_sweep.py` | existing-store and oracle Qwen sweeps |
| `qwen_clean_vector_experiment.py` | debug vector rebuild and sweep |
| `runs/` | CSV outputs, summaries, and per-run reports |
