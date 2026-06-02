# refuse

The `refuse` package is the orchestration layer for refusal-vector steering
experiments. Use the top-level README for user-facing commands and this file for
package-local orientation.

## Public surface

Common imports:

```python
from refuse import run
from refuse import load_experiments, to_run_kwargs, run_experiments
from refuse import build_grid, calibration_sweep, select_optimal_config
from refuse import GatedSteering, make_generation_jobs, run_jobs
```

The CLI delegates to the same `run` function:

```bash
python -m refuse --config configs/experiments.yml
python -m refuse --model ... --data ... --out ...
```

## Pipeline inputs

`run` requires:

| Argument | Meaning |
| --- | --- |
| `model_path` | Exact Hugging Face path registered in `llm.chat_templates`. |
| `data_root` | Folder with `train.csv` and `test.csv`. |
| `result_root` | Artifact store. |

The dataframes must include `concept` and `question`.

## Pipeline outputs

`Paths` in `paths.py` is the source of truth for file names. Do not scatter
hard-coded artifact names through new modules.

Important outputs:

| Artifact | Stage |
| --- | --- |
| `baseline_*.csv` | Baseline generation. |
| `*_acts.pt` | Activation collection. |
| `v_detect.pt`, `v_refuse.pt`, `thresholds.pt` | Vector fitting. |
| `calibration_results.csv`, `calibration_judged.csv` | Calibration. |
| `{eval}.csv`, `{eval}_judged.csv` | Optional evaluations. |

## Implementation notes

- `pipeline.py` is responsible for model load/use/delete order.
- `calibration.py` owns layer and scale grids.
- `intervention.py` owns steered generation jobs.
- `vectors.py` owns vector math and cached vector construction.
- `judge` and `plot` remain separate packages so they can evolve independently.

Keep new orchestration logic in `pipeline.py`, not in `llm` or `steering`.
