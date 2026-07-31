# Forget

Research framework and analysis repository for concept-specific refusal steering
in instruction-tuned language models.

## Setup

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate forget
```

The package is installed in editable mode by `environment.yml`. The Hugging Face
token is read from the root `.env`.

## Repository

| Path | Contents |
| --- | --- |
| `forget/` | Shared steering, model, judge, pipeline, and plotting package |
| `parking/model_matrix/` | Full model-dataset experiment matrix, protected results, and summary plots |
| `parking/` | Other complete or actively interpreted research units |
| `figs/` | Final main-text and supplementary figure units |
| `debug/` | New unresolved investigations only |
| `ref/` | Historical diagnostics, notebooks, and project notes |
| `skills/` | Repo-local README and caption workflows |
| `tmp/` | Disposable local files |

See `ORGANIZATION.md` for the compact-unit contract and `DECISIONS.md` for
project-wide experiment rules.

## Model Matrix

The complete explicit experiment matrix is:

```text
parking/model_matrix/experiments.yml
```

List its active rows:

```bash
python -m forget.refuse \
  --config parking/model_matrix/experiments.yml \
  --list
```

The 361 GB result vault is `parking/model_matrix/cache/`. It contains the
original datasets, activations, vectors, calibration results, evaluations,
plots, and logs. Existing artifacts are protected and must not be deleted,
overwritten, or regenerated without explicit approval for the exact artifact.

## Figures

Render the cross-run summary figures:

```bash
python -m parking.model_matrix.summary
```

Render the main and supplementary publication units:

```bash
python -m figs.main_results
python -m figs.supplementary_results
```

Each command reads existing result CSVs and writes only to its owning `plots/`
folder.

Figure appearance follows `STYLE.md`.
