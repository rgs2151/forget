# Phi Token Debug

This directory is isolated from the main pipeline and `store/` artifacts.

## Setup

- model: `microsoft/phi-4`
- concepts: `bacteria, cats, dogs, united_states`
- train samples per concept: `10`
- calibration samples per concept: `5`
- layers: `0,2,3`
- scales: `5,9,13`
- max new tokens: `32`
- steering mode: `gated`

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
| clean_all | 2 | 9.000 | 0.050 | 0.163 | 20 |
| phrase_cannot | 2 | 13.000 | 0.100 | 0.177 | 20 |
| phrase_no_info | 0 | 5.000 | 0.100 | 0.249 | 20 |
| phrase_sorry_idk | 3 | 13.000 | 0.150 | 0.221 | 20 |

## Full Summary

| variant | layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- | --- |
| phrase_sorry_idk | 3 | 13.000 | 0.150 | 0.221 | 20 |
| phrase_sorry_idk | 0 | 13.000 | 0.150 | 0.228 | 20 |
| phrase_sorry_idk | 2 | 13.000 | 0.150 | 0.237 | 20 |
| phrase_cannot | 2 | 13.000 | 0.100 | 0.177 | 20 |
| phrase_sorry_idk | 0 | 5.000 | 0.100 | 0.182 | 20 |
| phrase_sorry_idk | 0 | 9.000 | 0.100 | 0.184 | 20 |
| phrase_sorry_idk | 2 | 9.000 | 0.100 | 0.189 | 20 |
| phrase_no_info | 0 | 5.000 | 0.100 | 0.249 | 20 |
| clean_all | 2 | 9.000 | 0.050 | 0.163 | 20 |
| clean_all | 2 | 13.000 | 0.050 | 0.191 | 20 |
| phrase_sorry_idk | 3 | 9.000 | 0.050 | 0.193 | 20 |
| phrase_no_info | 3 | 13.000 | 0.050 | 0.220 | 20 |
| phrase_no_info | 2 | 13.000 | 0.050 | 0.233 | 20 |
| phrase_no_info | 3 | 9.000 | 0.050 | 0.233 | 20 |
| phrase_cannot | 0 | 9.000 | 0.000 | 0.152 | 20 |
| clean_all | 3 | 5.000 | 0.000 | 0.160 | 20 |
| phrase_sorry_idk | 3 | 5.000 | 0.000 | 0.161 | 20 |
| phrase_cannot | 3 | 5.000 | 0.000 | 0.172 | 20 |
| phrase_no_info | 3 | 5.000 | 0.000 | 0.180 | 20 |
| phrase_no_info | 2 | 5.000 | 0.000 | 0.192 | 20 |
| clean_all | 2 | 5.000 | 0.000 | 0.192 | 20 |
| phrase_cannot | 2 | 5.000 | 0.000 | 0.205 | 20 |
| phrase_cannot | 0 | 13.000 | 0.000 | 0.208 | 20 |
| phrase_no_info | 0 | 9.000 | 0.000 | 0.209 | 20 |
| phrase_sorry_idk | 2 | 5.000 | 0.000 | 0.212 | 20 |
| phrase_cannot | 3 | 9.000 | 0.000 | 0.219 | 20 |
| clean_all | 0 | 9.000 | 0.000 | 0.229 | 20 |
| phrase_no_info | 2 | 9.000 | 0.000 | 0.235 | 20 |
| phrase_cannot | 0 | 5.000 | 0.000 | 0.241 | 20 |
| phrase_cannot | 3 | 13.000 | 0.000 | 0.242 | 20 |
| phrase_cannot | 2 | 9.000 | 0.000 | 0.242 | 20 |
| clean_all | 3 | 13.000 | 0.000 | 0.242 | 20 |
| clean_all | 0 | 13.000 | 0.000 | 0.255 | 20 |
| clean_all | 3 | 9.000 | 0.000 | 0.257 | 20 |
| phrase_no_info | 0 | 13.000 | 0.000 | 0.272 | 20 |
| clean_all | 0 | 5.000 | 0.000 | 0.285 | 20 |
