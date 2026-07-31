# Decisions

This file records only project-wide choices that must remain consistent across
analyses.

## Protected Result Vault

- Decision: Preserve all existing model-matrix data and results under `parking/model_matrix/cache/`.
- Why: The cache contains expensive activations, vectors, generations, judged CSVs, and evaluation outputs.
- Use this when: Reading, plotting, extending, or reorganizing model-matrix experiments.
- Do not use this for: Disposable caches owned by other compact units.

## Experiment Outputs

- Decision: Shared activations and vectors live in `artifacts/<artifact_cache>/`; CSV and plot variants live in `results/<result_variant>/`.
- Why: Experiments can share expensive model artifacts while keeping evaluation conditions identifiable.
- Use this when: Adding a judge, intervention timing, or other result variant.
- Do not use this for: Independent analyses outside the model matrix.

## Configuration

- Decision: Keep every completed, skipped, and planned run visible in `parking/model_matrix/experiments.yml`; inactive rows remain commented.
- Why: The config is the auditable record of the experiment matrix.
- Use this when: Editing model, dataset, or run coverage.
- Do not use this for: One-off unit-local frozen configurations.

## Scale Windows

- Decision: `small = 0..1`, `mid = 0..10`, `large = 0..100`, and `xlarge = 0..300`, with 10 configured scale values unless explicitly changed.
- Why: Named windows must mean the same range across experiments.
- Use this when: Configuring calibration sweeps.
- Do not use this for: Explicit numeric windows in isolated analyses.

## Intervention Timing

- Decision: Assistant-only intervention remains the framework default; prefill intervention must be explicit.
- Why: Timing changes the intervention and must be recorded in the result configuration.
- Use this when: Running or comparing steering experiments.
- Do not use this for: Analyses that only read completed results.

## Figure Style

- Decision: Preserve the established project figure style in `STYLE.md` during this reorganization.
- Why: Existing publication figures should not change appearance as a side effect of moving code.
- Use this when: Regenerating or extending current figures.
- Do not use this for: A later user-approved restyling pass.
