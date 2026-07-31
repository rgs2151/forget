# experiment_matrix

## Method

- Resolve explicit model, dataset, judge, intervention, artifact-cache, and result-variant rows from `experiments.yml`.
- Run baseline generation, activation collection, vector fitting, layer-scale calibration, judge scoring, and optional bar/confusion evaluations.
- Reuse existing artifacts by path and keep completed, skipped, planned, and active rows visible in one configuration.

## Variables

- Models: Llama, Mistral, Qwen, and Phi checkpoints listed in `experiments.yml`.
- Datasets: Inhouse, MMLU, RWKU, Concept10, Concept500, and ConceptVectors.
- Shared artifacts: `cache/<run>/artifacts/<artifact_cache>/`.
- Result variants: `cache/<run>/results/<result_variant>/`.
- Configuration: `experiments.yml`.

## Statistics

- Calibration evaluates configured layer-scale cells and selects an operating point with `forget.refuse.calibration.select_optimal_config`.
- Refusal, retention, and fluency are means of binary judge decisions.
- No model-family comparison is used to select a run's operating point.

## Legends

- None; this entry documents the experiment system rather than one figure.

## Interpretation

- The unit is the preserved record of the original multi-model, multi-dataset exploration.

## Notes

- `cache/` is the protected 361 GB result vault.
- Do not delete, overwrite, regenerate, rename, clean, or move any cache artifact without explicit approval for that exact artifact.
- Existing tracked CSV, PNG, YAML, and log files remain tracked; `.pt` files remain ignored.

## References

- Framework: `forget/`.
- Historical one-off commands: `ref/model_matrix_command_history/experiments.md`.

# model_data

## Method

- Read judged confusion and bar evaluations for Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B across four datasets.
- Arrange refusal matrices beside retention matrices or grouped metric bars.
- Write `plots/model_data.png`.

## Variables

- Rows: Inhouse, MMLU, RWKU, and ConceptVectors.
- Columns: the three main model checkpoints.
- Matrix values: mean judge rates for queried-concept/steering-target cells.
- Bar groups: matched and non-matched concept pairs.

## Statistics

- Bars and cells are descriptive means.
- Error bars in grouped bar panels are 95% bootstrap confidence intervals.

## Legends

- Darker matrix cells indicate higher rates.
- Red bars identify matched concept pairs; black bars identify non-matched pairs.

## Interpretation

- Compare selective refusal and retained response behavior across models and datasets.

## Notes

- Missing inputs are displayed as crossed panels.

## References

- Code: `summary/model_data.py`.

# calib_optimal

## Method

- Read each available judged calibration sweep.
- Select the operating layer and scale and plot metric trajectories across scale at that layer.
- Write `plots/calib_optimal.png`.

## Variables

- Rows: datasets.
- Columns: model checkpoints grouped by family.
- Metrics: refusal, retention, and fluency.

## Statistics

- Lines are mean binary judge rates with 95% bootstrap confidence intervals.

## Legends

- Metric colors follow `STYLE.md`.
- Star marks the selected scale.
- Cross marks a missing calibration.

## Interpretation

- Compare selected operating regions across the matrix.

## Notes

- Scale ranges follow each model configuration.

## References

- Code: `summary/calib_optimal.py`.

# calib_scale_layers

## Method

- Read selected Inhouse calibration results for the main model checkpoints.
- Plot selected-layer metric trajectories and layer-wise refusal/retention trajectories across scale.
- Write `plots/calib_scale_layers.png`.

## Variables

- Models: Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B.
- X axis: steering scale.
- Layer color: configured sequential layer colormap.

## Statistics

- Selected-layer lines show mean rates with 95% bootstrap confidence intervals.
- Layer trajectories are descriptive cell means.

## Legends

- Metric color identifies refusal, retention, and fluency.
- Layer color identifies decoder depth.
- Star marks the selected scale.

## Interpretation

- Compare scale sensitivity at the selected layer with the complete layer sweep.

## Notes

- This is a summary diagnostic; the paper unit is `figs/fig5_layer_scale_sweeps/`.

## References

- Code: `summary/calib_scale_layers.py`.

# calib_full

## Method

- Read every available judged calibration sweep.
- Plot one trajectory per layer for refusal, retention, and fluency in separate outputs.
- Write `plots/calib_full_refuse.png`, `plots/calib_full_retain.png`, and `plots/calib_full_fluency.png`.

## Variables

- Rows: model checkpoints.
- Columns: Inhouse, MMLU, RWKU, and ConceptVectors.
- X axis: steering scale.
- Y axis: selected judge metric.

## Statistics

- None; each trajectory is a descriptive layer-wise rate.

## Legends

- Sequential color identifies layer depth.
- Cross marks a missing calibration.

## Interpretation

- Inspect the full layer-scale response surface before reducing a run to one operating point.

## Notes

- None yet.

## References

- Code: `summary/calib_full.py`.
