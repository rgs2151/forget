# Qwen clean-vector run: qwen_clean_vectors_all10_validation_v1

This run rebuilds vectors in debug from cleaned Qwen answers.

## Setup

- concepts: `bacteria,cats,dogs,united_states,lasers,the_moon,paris,obama,people,chess`
- train per concept: `40`
- samples per concept: `10`
- layers: `14,15,17`
- scales: `60,80,100,140`
- from offset: `-10000`
- pad as eos: `True`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 14 | 80.000 | 0.570 | 0.000 | 100 |
| 14 | 100.000 | 0.570 | 0.000 | 100 |
| 15 | 140.000 | 0.530 | 0.000 | 100 |
| 15 | 80.000 | 0.500 | 0.000 | 100 |
| 15 | 100.000 | 0.480 | 0.000 | 100 |
| 17 | 140.000 | 0.450 | 0.000 | 100 |
| 14 | 140.000 | 0.420 | 0.000 | 100 |
| 17 | 100.000 | 0.340 | 0.002 | 100 |
| 14 | 60.000 | 0.300 | 0.000 | 100 |
| 17 | 80.000 | 0.260 | 0.000 | 100 |
| 15 | 60.000 | 0.170 | 0.000 | 100 |
| 17 | 60.000 | 0.090 | 0.002 | 100 |
