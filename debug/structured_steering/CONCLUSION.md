# Structured Phi Steering Debug Conclusion

This directory contains debug-only experiments. The scripts read existing
`store/*_inhouse` vectors and baselines, but they do not modify `store` or the
main pipeline.

## Control: Llama

The cheap IDK harness is meaningful: it rediscovers Llama's known working band.

| run | best layer | best scale | IDK rate | n |
| --- | ---: | ---: | ---: | ---: |
| `llama_store_all_layers_v1` | 17 | 17 | 0.95 | 20 |

Representative Llama outputs at the best cells start directly with refusal:

- `I don't know that term.`
- `I don't know that specific component.`
- `I don't know that specific information.`

So the harness is not simply too weak or too noisy to see refusal.

## Failed Phi Hypotheses

Assistant-marker-only steering with the existing Phi store vectors fails even
when every layer is scanned.

| run | steering start | best layer | best scale | IDK rate | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi_store_all_layers_v2` | assistant marker | 3 | 13 | 0.10 | 20 |

This rejects the earlier unstructured guess that we merely missed an early Phi
layer. The best assistant-marker outputs are mostly clipped facts or `know`
fragments, not clean refusals.

Other dead ends:

| run | hypothesis | best IDK rate |
| --- | --- | ---: |
| `phi_lm_head_idk_all_layers_v1` | steer with an `I don't know.` lm-head vector | 0.00 |
| `phi_boundary_vector_v1` | learn a vector from assistant-boundary prompt states | 0.00 |
| `phi_store_pad_eos_v1` | remove dummy-token padding only | 0.15 |

Phi can refuse when directly instructed:

| run | result |
| --- | ---: |
| `phi_oracle_refuse_prompt_v1` | 20/20 IDK |

So the problem is not that Phi cannot say `I don't know`; it is how the signal is
injected.

## Working Hypothesis

The key difference from Llama is steering timing. Llama works when steering
starts at the assistant marker. Phi needs the same existing store vector injected
through the whole non-pad prompt prefill, so the signal participates in system
and user-token processing before generation begins.

| run | steering start | best layer | best scale | IDK rate | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi_store_all_content_v1` | all content | 6 | 20 | 0.30 | 20 |
| `phi_store_all_content_refine_v1` | all content | 10 | 40 | 0.40 | 20 |
| `phi_store_all_content_mid_layers_v1` | all content | 17 | 60 | 0.85 | 20 |
| `phi_all_content_best_validation_v1` | all content | 19 | 70 | 0.85 | 100 |

Validation spread for the best cell, layer 19 scale 70:

| concept | IDK rate |
| --- | ---: |
| cats | 0.50 |
| bacteria | 0.70 |
| chess | 0.70 |
| dogs | 0.80 |
| lasers | 0.90 |
| the_moon | 0.90 |
| paris | 1.00 |
| obama | 1.00 |
| people | 1.00 |
| united_states | 1.00 |

The candidate is real but not production-clean yet. Many successful outputs
start with refusal and then ramble; examples include:

- `I don't know the specific answer, but ...`
- `I don't have the specific information ...`
- `I don't know the answer to that question.`

Non-IDK failures remain, especially cats and bacteria:

- `The pancreas neutralizes stomach acid entering the feline duodenum.`
- `The SI unit of laser power.`
- `I don.`

## Recommendation

Prioritize a main-pipeline experiment that makes steering start configurable.
For Phi, test all-content prefill steering with existing store vectors before
changing vector construction.

Concrete candidate:

- model: `microsoft/phi-4`
- vector source: existing LDA store vectors
- generation pad: EOS, not `<|dummy_85|>`
- steering start: all non-pad prompt content, not assistant marker
- layer candidates: 17-19
- scale candidates: 60-80
- current best debug cell: layer 19, scale 70

Token sanitation remains useful hygiene, but it was not the main fix. The main
fix discovered here is timing: Phi needs pre-assistant prompt steering.
