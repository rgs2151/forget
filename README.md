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
| `data/` | Inventory and future independent analysis-ready datasets |
| `forget/` | Shared steering, model, judge, pipeline, and plotting package |
| `parking/model_matrix/` | Original model-dataset exploration and protected result vault |
| `parking/` | Active and completed research units |
| `figs/` | Final paper figure and supplementary units |
| `debug/` | Isolated diagnostics and compatibility investigations |
| `ref/` | Superseded notebooks, visual variants, and project history |
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

Render a publication unit:

```bash
python -m figs.fig4_selective_refusal
python -m figs.fig5_layer_scale_sweeps
python -m figs.fig7_model_size_scaling
python -m figs.supp_bars
python -m figs.supp_confusion
python -m figs.supp_calibration_optimal
python -m figs.supp_calibration_layers
```

Each command reads existing result CSVs and writes only to its owning `plots/`
folder.

Figure appearance follows `STYLE.md`.

## Data

The original experiment datasets remain inside
`parking/model_matrix/cache/` so the protected experiment tree is not split.
New datasets for orthogonal research questions belong in `data/`.

See `data/README.md` for columns, split rules, and the dataset inventory.
