# Phi Token Debug

This directory is isolated from the main pipeline and `store/` artifacts.

## Setup

- model: `microsoft/phi-4`
- concepts: `bacteria, cats, dogs, united_states`
- train samples per concept: `6`
- calibration samples per concept: `5`
- layers: `0,2,3`
- scales: `7,9,11,13`
- max new tokens: `32`
- steering mode: `gated`
- from position offset: `10000`

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
| store_vectors | 0 | 11.000 | 0.150 | 0.280 | 20 |

## Full Summary

| variant | layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- | --- |
| store_vectors | 0 | 11.000 | 0.150 | 0.280 | 20 |
| store_vectors | 0 | 9.000 | 0.050 | 0.113 | 20 |
| store_vectors | 3 | 11.000 | 0.050 | 0.162 | 20 |
| store_vectors | 3 | 13.000 | 0.050 | 0.212 | 20 |
| store_vectors | 2 | 7.000 | 0.000 | 0.156 | 20 |
| store_vectors | 3 | 9.000 | 0.000 | 0.160 | 20 |
| store_vectors | 2 | 13.000 | 0.000 | 0.196 | 20 |
| store_vectors | 0 | 13.000 | 0.000 | 0.197 | 20 |
| store_vectors | 3 | 7.000 | 0.000 | 0.197 | 20 |
| store_vectors | 2 | 11.000 | 0.000 | 0.218 | 20 |
| store_vectors | 2 | 9.000 | 0.000 | 0.220 | 20 |
| store_vectors | 0 | 7.000 | 0.000 | 0.228 | 20 |
