# Qwen clean-vector run: qwen_clean_vectors_additive_all10_validation_v1

This run rebuilds vectors in debug from cleaned Qwen answers.

## Setup

- concepts: `bacteria,cats,dogs,united_states,lasers,the_moon,paris,obama,people,chess`
- train per concept: `40`
- samples per concept: `10`
- layers: `14,15,17`
- scales: `40,50,60,70`
- from offset: `-10000`
- pad as eos: `True`
- steering mode: `add`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 14 | 70.000 | 1.000 | 0.000 | 100 |
| 15 | 70.000 | 1.000 | 0.000 | 100 |
| 17 | 70.000 | 1.000 | 0.000 | 100 |
| 14 | 60.000 | 0.990 | 0.000 | 100 |
| 15 | 60.000 | 0.980 | 0.000 | 100 |
| 17 | 60.000 | 0.980 | 0.000 | 100 |
| 14 | 50.000 | 0.820 | 0.000 | 100 |
| 17 | 50.000 | 0.770 | 0.000 | 100 |
| 15 | 50.000 | 0.640 | 0.000 | 100 |
| 17 | 40.000 | 0.280 | 0.000 | 100 |
| 14 | 40.000 | 0.270 | 0.000 | 100 |
| 15 | 40.000 | 0.180 | 0.000 | 100 |
