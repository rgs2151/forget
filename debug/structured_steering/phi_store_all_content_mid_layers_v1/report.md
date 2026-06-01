# Store Vector Sweep: phi

This run uses existing store vectors and writes only debug artifacts.

## Setup

- model key: `phi`
- concepts: `bacteria,cats,dogs,united_states`
- samples per concept: `5`
- layers: `10,11,12,13,14,15,16,17,18,19,20`
- scales: `20,30,40,50,60,80`
- max new tokens: `32`
- from offset: `-10000`
- pad as eos: `True`
- vector source: `store`
- steer text: `I don't know.`
- steer token mode: `mean`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 17 | 60.000 | 0.850 | 0.009 | 20 |
| 16 | 80.000 | 0.800 | 0.023 | 20 |
| 18 | 80.000 | 0.750 | 0.000 | 20 |
| 19 | 80.000 | 0.750 | 0.002 | 20 |
| 19 | 60.000 | 0.650 | 0.002 | 20 |
| 16 | 60.000 | 0.650 | 0.016 | 20 |
| 15 | 50.000 | 0.600 | 0.002 | 20 |
| 15 | 60.000 | 0.600 | 0.002 | 20 |
| 20 | 80.000 | 0.600 | 0.014 | 20 |
| 18 | 60.000 | 0.550 | 0.002 | 20 |
| 13 | 40.000 | 0.550 | 0.003 | 20 |
| 20 | 60.000 | 0.450 | 0.011 | 20 |
| 14 | 50.000 | 0.400 | 0.003 | 20 |
| 10 | 40.000 | 0.400 | 0.009 | 20 |
| 10 | 50.000 | 0.400 | 0.009 | 20 |
| 11 | 40.000 | 0.350 | 0.000 | 20 |
| 11 | 50.000 | 0.350 | 0.003 | 20 |
| 15 | 80.000 | 0.350 | 0.003 | 20 |
| 16 | 50.000 | 0.350 | 0.013 | 20 |
| 13 | 50.000 | 0.300 | 0.000 | 20 |
