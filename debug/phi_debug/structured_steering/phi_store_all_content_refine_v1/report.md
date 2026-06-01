# Store Vector Sweep: phi

This run uses existing store vectors and writes only debug artifacts.

## Setup

- model key: `phi`
- concepts: `bacteria,cats,dogs,united_states`
- samples per concept: `5`
- layers: `1,2,3,4,5,6,7,8,9,10`
- scales: `15,20,25,30,40,60,80`
- max new tokens: `32`
- from offset: `-10000`
- pad as eos: `True`
- vector source: `store`
- steer text: `I don't know.`
- steer token mode: `mean`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 10 | 40.000 | 0.400 | 0.009 | 20 |
| 6 | 20.000 | 0.300 | 0.000 | 20 |
| 9 | 30.000 | 0.300 | 0.000 | 20 |
| 9 | 40.000 | 0.300 | 0.024 | 20 |
| 1 | 15.000 | 0.250 | 0.000 | 20 |
| 10 | 60.000 | 0.250 | 0.025 | 20 |
| 7 | 80.000 | 0.250 | 0.051 | 20 |
| 8 | 25.000 | 0.200 | 0.000 | 20 |
| 9 | 25.000 | 0.200 | 0.000 | 20 |
| 7 | 25.000 | 0.200 | 0.002 | 20 |
| 10 | 30.000 | 0.200 | 0.003 | 20 |
| 9 | 80.000 | 0.200 | 0.024 | 20 |
| 3 | 80.000 | 0.150 | 0.000 | 20 |
| 5 | 30.000 | 0.150 | 0.000 | 20 |
| 9 | 60.000 | 0.150 | 0.018 | 20 |
| 7 | 60.000 | 0.150 | 0.032 | 20 |
| 10 | 80.000 | 0.150 | 0.054 | 20 |
| 1 | 30.000 | 0.100 | 0.000 | 20 |
| 3 | 40.000 | 0.100 | 0.000 | 20 |
| 4 | 30.000 | 0.100 | 0.000 | 20 |
