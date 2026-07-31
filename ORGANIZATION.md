# Organization

## Root Folders

- `data/`: inventory for future independent analysis-ready data.
- `forget/`: installable shared package.
- `parking/`: compact research units that are complete or still being interpreted.
- `figs/`: graduated compact units for final figure outputs.
- `debug/`: unresolved, isolated investigations.
- `ref/`: historical notebooks, diagnostics, and project notes.
- `skills/`: repo-local compact-unit workflows.
- `tmp/`: disposable local files.

## Python Package

`forget/` contains shared framework code:

```text
forget/
  api/
  judge/
  llm/
  plot/
  refuse/
  steering/
```

Keep analysis-specific calculations and figure layouts in their owning units.
Import shared helpers through `forget`, not through sibling units.

## Compact Units

A compact unit owns the code, cache, plots, and documentation for one analysis
question or one tightly grouped set of outputs:

```text
parking/descriptive_unit/
  analysis.py
  cache/
  plots/
  README.md
  caption.md

figs/final_output/
  figure.py
  cache/
  plots/
  README.md
  caption.md
```

Multiple scripts may share one unit when they answer the same question or
produce parallel views of the same results.

## Cache Rules

- A unit reads existing inputs and writes only to its own `cache/` and `plots/`.
- Reuse existing cache files unless recomputation is explicitly requested.
- Ordinary unit caches are ignored by Git.
- Plot and table outputs are tracked.
- `parking/model_matrix/cache/` is a deliberate exception. It is the preserved
  361 GB result vault from the original repository and contains tracked result
  artifacts alongside ignored `.pt` files.
- Nothing inside the model-matrix cache may be changed without explicit
  artifact-level approval.

## Model Matrix Unit

`parking/model_matrix/` owns the original multi-model experiment system:

```text
parking/model_matrix/
  experiments.yml
  experiments.md
  notebooks/
  summary/
  cache/
  plots/
  README.md
```

Within each model-dataset run:

```text
cache/<model>_<dataset>/
  artifacts/<artifact_cache>/
  results/<result_variant>/
```

The old result hierarchy remains intact inside this unit.

## README Structure

Unit READMEs use these sections:

- `Method`
- `Variables`
- `Statistics`
- `Legends`
- `Interpretation`
- `Notes`
- `References`

For grouped units, document each distinct script or output clearly. Keep
cache paths, subsets, thresholds, tests, and panel mappings in the unit README,
not in `DECISIONS.md`.

## Captions

Caption drafts live in `caption.md` inside the owning figure unit. Captions are
checked against the unit code, README, plots, tables, and project decisions.

## History

Move completed investigations into `parking/` when they remain part of the
research record. Move obsolete but informative material into `ref/`. Delete
only material the user explicitly identifies as disposable.
