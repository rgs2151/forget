# smoke_pipeline_models

## Method

- Detect each model's registered chat template and enter the standard pipeline on a 1% Inhouse subset.
- Generate baselines, collect activations, build vectors, and produce one no-judge calibration cell.
- Run each feasible model on GPU 0 and record whether `calibration_results.csv` is produced.
- Write raw runs under `runs/` and the tracked status table to `plots/smoke_summary.csv`.

## Variables

- Models: Llama-3.2-1B/3B, Qwen2.5-0.5B/3B/14B, Phi-4-mini, and Mistral-Small-24B.
- Layers: `frac:0.5`.
- Scale: one value in `0:1`.
- Sampling: `train_frac=0.01`, `test_frac=0.01`, and `calibration_n=1`.
- Judge and evaluations: disabled.

## Statistics

- None; pass means the pipeline produced the calibration output without an exception.

## Legends

- `status=passed`: the tiny pipeline smoke completed.
- `status=skipped_hardware`: the checkpoint was not loaded on current hardware.
- `calibration_output`: whether the expected CSV exists.

## Interpretation

- All feasible requested checkpoints entered the pipeline successfully.
- Mistral-Small-24B remained untested because one full-precision model copy exceeds a 32 GB GPU.

## Notes

- Raw runs are ignored by Git.
- Phi-4-mini uses the built-in Phi3 loader with `trust_remote_code=false`.

## References

- Script: `smoke_pipeline_models.py`.
- Summary: `plots/smoke_summary.csv`.
