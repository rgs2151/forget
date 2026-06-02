# Phi Token Debug

This directory is isolated from the main pipeline and `store/` artifacts.

## Setup

- model: `microsoft/phi-4`
- concepts: `bacteria, cats, dogs, united_states`
- train samples per concept: `6`
- calibration samples per concept: `5`
- layers: `0,2,3`
- scales: `5,9,13`
- max new tokens: `32`

## Baseline Token Pollution

| split | concept | rows | mean_special_fraction | median_special_fraction |
| --- | --- | --- | --- | --- |
| baseline_train.csv | bacteria | 457 | 0.463 | 0.529 |
| baseline_train.csv | cats | 469 | 0.432 | 0.455 |
| baseline_train.csv | dogs | 476 | 0.474 | 0.531 |
| baseline_train.csv | united_states | 472 | 0.610 | 0.700 |
| baseline_test.csv | bacteria | 137 | 0.488 | 0.545 |
| baseline_test.csv | cats | 125 | 0.426 | 0.444 |
| baseline_test.csv | dogs | 118 | 0.433 | 0.474 |
| baseline_test.csv | united_states | 122 | 0.634 | 0.725 |

## Best Cell Per Variant

| variant | layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- | --- |
| clean_all | 0 | 13.000 | 0.050 | 0.177 | 20 |
| clean_baseline | 0 | 13.000 | 0.050 | 0.204 | 20 |
| clean_idk | 0 | 9.000 | 0.050 | 0.197 | 20 |
| current | 0 | 9.000 | 0.050 | 0.211 | 20 |
| exclude_special_mask | 0 | 13.000 | 0.050 | 0.182 | 20 |

## Full Summary

| variant | layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- | --- |
| clean_all | 0 | 13.000 | 0.050 | 0.177 | 20 |
| exclude_special_mask | 0 | 13.000 | 0.050 | 0.182 | 20 |
| clean_idk | 0 | 9.000 | 0.050 | 0.197 | 20 |
| clean_baseline | 0 | 13.000 | 0.050 | 0.204 | 20 |
| current | 0 | 9.000 | 0.050 | 0.211 | 20 |
| current | 3 | 13.000 | 0.050 | 0.214 | 20 |
| exclude_special_mask | 0 | 9.000 | 0.050 | 0.230 | 20 |
| clean_all | 0 | 9.000 | 0.050 | 0.231 | 20 |
| clean_baseline | 0 | 9.000 | 0.050 | 0.264 | 20 |
| clean_idk | 0 | 13.000 | 0.050 | 0.269 | 20 |
| current | 3 | 9.000 | 0.000 | 0.146 | 20 |
| clean_all | 2 | 5.000 | 0.000 | 0.170 | 20 |
| current | 2 | 5.000 | 0.000 | 0.172 | 20 |
| clean_all | 3 | 5.000 | 0.000 | 0.172 | 20 |
| exclude_special_mask | 3 | 5.000 | 0.000 | 0.172 | 20 |
| exclude_special_mask | 2 | 5.000 | 0.000 | 0.172 | 20 |
| clean_idk | 3 | 5.000 | 0.000 | 0.173 | 20 |
| clean_baseline | 3 | 5.000 | 0.000 | 0.177 | 20 |
| clean_idk | 3 | 9.000 | 0.000 | 0.181 | 20 |
| clean_idk | 2 | 5.000 | 0.000 | 0.181 | 20 |
| current | 3 | 5.000 | 0.000 | 0.190 | 20 |
| clean_baseline | 0 | 5.000 | 0.000 | 0.192 | 20 |
| clean_baseline | 2 | 5.000 | 0.000 | 0.198 | 20 |
| exclude_special_mask | 0 | 5.000 | 0.000 | 0.201 | 20 |
| clean_idk | 3 | 13.000 | 0.000 | 0.201 | 20 |
| clean_idk | 2 | 13.000 | 0.000 | 0.207 | 20 |
| clean_all | 0 | 5.000 | 0.000 | 0.208 | 20 |
| clean_idk | 0 | 5.000 | 0.000 | 0.216 | 20 |
| clean_baseline | 2 | 9.000 | 0.000 | 0.217 | 20 |
| current | 0 | 5.000 | 0.000 | 0.218 | 20 |
| clean_all | 3 | 9.000 | 0.000 | 0.222 | 20 |
| clean_baseline | 3 | 13.000 | 0.000 | 0.229 | 20 |
| exclude_special_mask | 3 | 9.000 | 0.000 | 0.230 | 20 |
| clean_all | 2 | 9.000 | 0.000 | 0.241 | 20 |
| current | 2 | 13.000 | 0.000 | 0.246 | 20 |
| exclude_special_mask | 2 | 9.000 | 0.000 | 0.247 | 20 |
| current | 0 | 13.000 | 0.000 | 0.253 | 20 |
| exclude_special_mask | 3 | 13.000 | 0.000 | 0.263 | 20 |
| clean_all | 3 | 13.000 | 0.000 | 0.264 | 20 |
| clean_baseline | 2 | 13.000 | 0.000 | 0.266 | 20 |
| clean_all | 2 | 13.000 | 0.000 | 0.269 | 20 |
| exclude_special_mask | 2 | 13.000 | 0.000 | 0.269 | 20 |
| clean_idk | 2 | 9.000 | 0.000 | 0.281 | 20 |
| clean_baseline | 3 | 9.000 | 0.000 | 0.288 | 20 |
| current | 2 | 9.000 | 0.000 | 0.297 | 20 |
