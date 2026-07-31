# Store Vector Sweep: phi

This run uses existing store vectors and writes only debug artifacts.

## Setup

- model key: `phi`
- concepts: `bacteria,cats,dogs,united_states`
- samples per concept: `5`
- layers: `all`
- scales: `1,3,5,9,13,20`
- max new tokens: `32`
- from offset: `-10000`
- pad as eos: `True`
- vector source: `store`
- steer text: `I don't know.`
- steer token mode: `mean`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 6 | 20.000 | 0.300 | 0.000 | 20 |
| 1 | 13.000 | 0.250 | 0.000 | 20 |
| 2 | 9.000 | 0.200 | 0.000 | 20 |
| 4 | 13.000 | 0.150 | 0.012 | 20 |
| 2 | 13.000 | 0.100 | 0.000 | 20 |
| 3 | 9.000 | 0.100 | 0.000 | 20 |
| 7 | 20.000 | 0.100 | 0.000 | 20 |
| 8 | 20.000 | 0.100 | 0.000 | 20 |
| 5 | 20.000 | 0.100 | 0.002 | 20 |
| 4 | 20.000 | 0.100 | 0.005 | 20 |
| 0 | 9.000 | 0.050 | 0.000 | 20 |
| 0 | 20.000 | 0.050 | 0.000 | 20 |
| 1 | 9.000 | 0.050 | 0.000 | 20 |
| 1 | 20.000 | 0.050 | 0.000 | 20 |
| 2 | 20.000 | 0.050 | 0.000 | 20 |
| 3 | 13.000 | 0.050 | 0.000 | 20 |
| 4 | 9.000 | 0.050 | 0.000 | 20 |
| 6 | 13.000 | 0.050 | 0.000 | 20 |
| 0 | 1.000 | 0.000 | 0.000 | 20 |
| 0 | 3.000 | 0.000 | 0.000 | 20 |
