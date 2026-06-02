# Store Vector Sweep: phi

This run uses existing store vectors and writes only debug artifacts.

## Setup

- model key: `phi`
- concepts: `bacteria,cats,dogs,united_states`
- samples per concept: `5`
- layers: `all`
- scales: `1,3,5,9,13,20`
- max new tokens: `32`
- from offset: `0`
- pad as eos: `True`
- vector source: `lm_head`
- steer text: `I don't know.`
- steer token mode: `mean`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 0.000 | 0.000 | 20 |
| 0 | 3.000 | 0.000 | 0.000 | 20 |
| 0 | 5.000 | 0.000 | 0.000 | 20 |
| 0 | 9.000 | 0.000 | 0.000 | 20 |
| 0 | 13.000 | 0.000 | 0.000 | 20 |
| 0 | 20.000 | 0.000 | 0.000 | 20 |
| 1 | 1.000 | 0.000 | 0.000 | 20 |
| 1 | 3.000 | 0.000 | 0.000 | 20 |
| 1 | 5.000 | 0.000 | 0.000 | 20 |
| 1 | 9.000 | 0.000 | 0.000 | 20 |
| 1 | 20.000 | 0.000 | 0.000 | 20 |
| 2 | 1.000 | 0.000 | 0.000 | 20 |
| 2 | 3.000 | 0.000 | 0.000 | 20 |
| 2 | 5.000 | 0.000 | 0.000 | 20 |
| 2 | 9.000 | 0.000 | 0.000 | 20 |
| 2 | 13.000 | 0.000 | 0.000 | 20 |
| 2 | 20.000 | 0.000 | 0.000 | 20 |
| 3 | 1.000 | 0.000 | 0.000 | 20 |
| 3 | 3.000 | 0.000 | 0.000 | 20 |
| 3 | 5.000 | 0.000 | 0.000 | 20 |
