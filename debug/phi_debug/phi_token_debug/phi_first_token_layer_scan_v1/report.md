# Phi Token Debug

This directory is isolated from the main pipeline and `store/` artifacts.

## Setup

- model: `microsoft/phi-4`
- concepts: `bacteria, cats, dogs, united_states`
- train samples per concept: `10`
- calibration samples per concept: `5`
- layers: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,30,39`
- scales: `9,13,20`
- max new tokens: `32`
- steering mode: `concept_gated`
- from position offset: `0`
- pad as eos: `True`
- activation pool mode: `first`

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
| clean_all | 4 | 20.000 | 0.200 | 0.000 | 20 |

## Full Summary

| variant | layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- | --- |
| clean_all | 4 | 20.000 | 0.200 | 0.000 | 20 |
| clean_all | 4 | 13.000 | 0.150 | 0.000 | 20 |
| clean_all | 2 | 9.000 | 0.150 | 0.002 | 20 |
| clean_all | 2 | 13.000 | 0.100 | 0.000 | 20 |
| clean_all | 3 | 13.000 | 0.100 | 0.000 | 20 |
| clean_all | 4 | 9.000 | 0.100 | 0.000 | 20 |
| clean_all | 0 | 13.000 | 0.050 | 0.000 | 20 |
| clean_all | 0 | 20.000 | 0.050 | 0.000 | 20 |
| clean_all | 8 | 20.000 | 0.050 | 0.002 | 20 |
| clean_all | 0 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 1 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 1 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 1 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 2 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 3 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 3 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 5 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 5 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 5 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 6 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 6 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 6 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 7 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 7 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 7 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 8 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 8 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 9 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 9 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 9 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 10 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 10 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 10 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 11 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 11 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 11 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 12 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 12 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 12 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 13 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 13 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 13 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 14 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 14 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 14 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 15 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 15 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 15 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 20 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 20 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 20 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 30 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 30 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 30 | 20.000 | 0.000 | 0.000 | 20 |
| clean_all | 39 | 9.000 | 0.000 | 0.000 | 20 |
| clean_all | 39 | 13.000 | 0.000 | 0.000 | 20 |
| clean_all | 39 | 20.000 | 0.000 | 0.000 | 20 |
