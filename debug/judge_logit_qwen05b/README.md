# qwen05b_logit_pipeline_smoke

## Method

- Run the standard refusal pipeline on Qwen2.5-0.5B with a 1% Inhouse subset.
- Build LDA vectors, generate one calibration cell, and score it with the Selene logit judge.
- Disable evaluation panels and plotting.
- Store the frozen run configuration at `frozen_config.yml`, judged tables under
  `plots/`, and raw pipeline artifacts under `cache/`.

## Variables

- Experimental model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Judge: `AtlaAI/Selene-1-Mini-Llama-3.1-8B`.
- GPU 0 for generation and judging, used sequentially by the pipeline.
- Calibration: one sampled example, one layer fraction, one scale in `0:1`.
- Batch sizes: 4 for generation and judging.

## Statistics

- None; this is a compatibility smoke test.
- Success criterion: the pipeline produces `calibration_results.csv` and `calibration_judged.csv` without an exception.

## Legends

- None; this unit has no figure.

## Interpretation

- The small Qwen checkpoint can complete the logit-judge pipeline path.

## Notes

- `frozen_config.yml` is the resolved configuration.
- `plots/calibration_results.csv` and `plots/calibration_judged.csv` are the
  tracked smoke outputs.
- The historical argument log retains the pre-refactor paths from the original run.

## References

- Pipeline: `forget.refuse.pipeline.run`.
