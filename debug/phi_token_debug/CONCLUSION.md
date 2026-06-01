# Phi Token Debug Conclusion

All experiments here are isolated under `debug/phi_token_debug`. They read from
`store/phi4_inhouse` but do not modify `store` or the main pipeline code.

## What Held Up

The Phi artifacts are polluted by pad tokens. In the selected inhouse concepts,
existing Phi baseline outputs contain `<|dummy_85|>` in large fractions of the
answer span:

| split | mean special fraction | median special fraction |
| --- | ---: | ---: |
| baseline train, selected concepts | 0.432-0.610 | 0.455-0.700 |
| baseline test, selected concepts | 0.426-0.634 | 0.444-0.725 |

Using EOS as the debug generation pad removes the visible dummy-token artifact:
the `pad_as_eos` sweeps have `special_fraction = 0.0`.

## What Did Not Hold Up

Token cleanup alone did not fix Phi steering. The simple variants in
`phi_fix_probe_v1` all topped out at `1/20` IDK matches:

| variant | best IDK rate |
| --- | ---: |
| current | 0.05 |
| clean_baseline | 0.05 |
| clean_idk | 0.05 |
| exclude_special_mask | 0.05 |
| clean_all | 0.05 |

The existing full Phi vectors from `store` also did not produce a strong IDK
rate under substring scoring. Best observed cell was `3/20`:

| run | best cell | IDK rate |
| --- | --- | ---: |
| `phi_store_vectors_v1` | layer 3, scale 11 | 0.15 |
| `phi_store_pad_eos_v1` | layer 3, scale 11 | 0.15 |

## Negative Probes

These did not improve Phi:

| run | change tested | best IDK rate |
| --- | --- | ---: |
| `phi_store_sign_scale_v1` | negative and large positive scales | 0.05 |
| `phi_store_add_v1` | ungated add of existing refusal vector | 0.00 |
| `phi_store_offset1_v1` | start steering after assistant marker | 0.15 |
| `phi_store_genonly_v1` | generated-token-only steering | 0.15 |
| `phi_detect_ablate_store_v1` | steer against detector direction | 0.00 |
| `phi_store_pad_eos_large_scale_v1` | EOS padding plus scales up to 100 | 0.00 |

Those results argue against the easy explanations: wrong sign, too-small scale,
marker-token prefill steering, dummy-token decoding, or a broken gate as the
sole cause.

## Best Candidate

The best debug candidate is:

- clean baseline answer spans
- exclude special tokens from activation pooling
- pool the first answer token instead of the mean answer span
- use concept-specific refusal vectors
- use EOS as generation pad
- layer 4, scale 13-20

It reached `4/20` IDK matches:

| run | best cell | IDK rate | special fraction |
| --- | --- | ---: | ---: |
| `phi_best_64tok_v1` | layer 4, scale 13 | 0.20 | 0.00 |
| `phi_first_token_layer_scan_v1` | layer 4, scale 20 | 0.20 | 0.00 |

This is an improvement over the current cheap probes, but it is not strong
enough to call a Phi fix.

## Recommendation

Do not launch an expensive Phi rerun expecting token sanitation alone to fix
steering. The sanitation change is still worth prioritizing as artifact hygiene,
because it prevents `<|dummy_85|>` from dominating baselines and activation
spans. But the debug evidence says Phi needs a deeper steering-objective change,
not just cleanup.

The most promising main-pipeline candidate from this debug pass is the small
bundle above: clean spans, special-token exclusion, first-token pooling, and
concept-specific refusal vectors. Treat it as a candidate for another targeted
debug iteration, not as a production change.
