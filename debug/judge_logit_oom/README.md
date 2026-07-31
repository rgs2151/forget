# probe_parallel_logit_judge

## Method

- Load real calibration prompts for the Selene logit judge.
- Score only the configured answer tokens at the final prompt position while activation capture is disabled.
- Probe the two-GPU pool over repeated batches and record probability ranges, elapsed time, and GPU memory.
- Write run logs, summaries, and reports under `runs/`.

## Variables

- Judge: `AtlaAI/Selene-1-Mini-Llama-3.1-8B`.
- GPUs: 0 and 1.
- Probe batch size: 8 per GPU in the completed long run.
- Outputs: `runs/<probe>/screen.log`, `summary.csv`, and `report.md`.

## Statistics

- None; the probe records runtime completion, memory use, and returned probability ranges.

## Legends

- `chunk`: sequential probe batch.
- `gpu_memory`: observed allocated/reserved memory.
- `p1` and `p2`: minimum and maximum judge option probabilities.

## Interpretation

- Final-position option scoring and disabled activation capture remove the full-vocabulary memory pressure.
- The blocking-CUDA runs are retained as intermediate diagnostics, not as the production configuration.

## Notes

- Production judging uses both GPUs, internal batching, and no `CUDA_LAUNCH_BLOCKING=1`.
- Historical blocking and non-blocking probe outputs remain under `runs/`.

## References

- Probe: `scripts/probe_parallel_logit_judge.py`.
- Production scorer: `forget.steering.base.LLM.batch_next_token_option_probs`.
