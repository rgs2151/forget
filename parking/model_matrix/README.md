# Model matrix

## Method

- Run the configured model-dataset rows through baseline generation, activation
  collection, vector fitting, layer-scale calibration, judge scoring, and
  optional bar/confusion evaluations.
- Keep model, dataset, judge, intervention timing, artifact cache, and result
  variant explicit in `experiments.yml`.
- Build summary figures from existing judged CSVs without rerunning models.

## Variables

- Config: `experiments.yml`.
- Datasets: Inhouse, MMLU, RWKU, Concept10, Concept500, and ConceptVectors.
- Models: Llama, Mistral, Qwen, and Phi checkpoints listed in the config.
- Result vault: `cache/`.
- Summary code: `summary/`.
- Summary outputs: `plots/`.

## Statistics

- Calibration selects a layer and scale from the configured sweep using the
  existing refusal/fluency selection score.
- Bar and confusion summaries report judge-derived refusal, retention, and
  fluency rates.
- Figure-specific confidence intervals and regressions are implemented in the
  corresponding summary script.

## Legends

- Exact axes, colors, model ordering, dataset ordering, and missing-result marks
  are defined by each script in `summary/` and the project `STYLE.md`.
- A large cross marks a model-dataset result that is absent.

## Interpretation

- The unit is the complete experiment record for comparing steering strength,
  specificity, and response quality across models and datasets.

## Notes

- `cache/` is protected. Do not delete, overwrite, regenerate, move, or clean
  any artifact without explicit approval for that exact artifact.
- Existing tracked CSV, PNG, YAML, and log files stay tracked. `.pt` files stay
  ignored.
- Active, completed, skipped, and planned config rows remain visible in
  `experiments.yml`.

## References

- Framework: `forget/`.
- Final figures: `figs/main_results/` and `figs/supplementary_results/`.
