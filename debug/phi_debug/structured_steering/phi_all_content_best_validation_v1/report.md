# Store Vector Sweep: phi

This run uses existing store vectors and writes only debug artifacts.

## Setup

- model key: `phi`
- concepts: `united_states,people,lasers,dogs,obama,the_moon,chess,paris,cats,bacteria`
- samples per concept: `10`
- layers: `16,17,18,19`
- scales: `50,60,70,80`
- max new tokens: `64`
- from offset: `-10000`
- pad as eos: `True`
- vector source: `store`
- steer text: `I don't know.`
- steer token mode: `mean`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 19 | 70.000 | 0.850 | 0.010 | 100 |
| 18 | 70.000 | 0.840 | 0.003 | 100 |
| 17 | 60.000 | 0.830 | 0.016 | 100 |
| 16 | 70.000 | 0.820 | 0.023 | 100 |
| 19 | 80.000 | 0.800 | 0.007 | 100 |
| 16 | 60.000 | 0.800 | 0.019 | 100 |
| 19 | 60.000 | 0.760 | 0.003 | 100 |
| 17 | 70.000 | 0.750 | 0.024 | 100 |
| 18 | 60.000 | 0.740 | 0.003 | 100 |
| 16 | 80.000 | 0.730 | 0.036 | 100 |
| 18 | 80.000 | 0.710 | 0.004 | 100 |
| 17 | 50.000 | 0.670 | 0.009 | 100 |
| 16 | 50.000 | 0.540 | 0.015 | 100 |
| 18 | 50.000 | 0.480 | 0.002 | 100 |
| 19 | 50.000 | 0.480 | 0.003 | 100 |
| 17 | 80.000 | 0.460 | 0.038 | 100 |
