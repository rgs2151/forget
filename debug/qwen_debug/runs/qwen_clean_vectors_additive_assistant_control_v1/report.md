# Qwen clean-vector run: qwen_clean_vectors_additive_assistant_control_v1

This run rebuilds vectors in debug from cleaned Qwen answers.

## Setup

- concepts: `bacteria,cats,dogs,united_states,lasers,the_moon,paris,obama,people,chess`
- train per concept: `40`
- samples per concept: `10`
- layers: `14,15,17`
- scales: `60,70`
- from offset: `0`
- pad as eos: `True`
- steering mode: `add`

## Top Cells

| layer | scale | idk_rate | special_fraction | n |
| --- | --- | --- | --- | --- |
| 15 | 70.000 | 0.840 | 0.000 | 100 |
| 15 | 60.000 | 0.610 | 0.000 | 100 |
| 14 | 70.000 | 0.330 | 0.000 | 100 |
| 17 | 70.000 | 0.310 | 0.003 | 100 |
| 14 | 60.000 | 0.260 | 0.000 | 100 |
| 17 | 60.000 | 0.190 | 0.000 | 100 |
