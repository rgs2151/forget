# Inference Analyses

This folder contains notebook-first analyses that sit on top of existing judged artifacts in `parking/model_matrix/cache/`.

## Current notebook

- `semantic_similarity_analysis.ipynb`

This notebook tests the relation between targeted-refusal anomaly and semantic similarity between contexts using `confusion_judged.csv` files.

## Inputs

- `parking/model_matrix/cache/*_inhouse/confusion_judged.csv`

You can change the glob in the notebook to include other datasets.

## Outputs

The notebook outputs are stored in `plots/`:

- `targeted_refusal_similarity_bins.csv`
- `targeted_refusal_similarity_coefficients.csv`
- `targeted_refusal_similarity_anomaly.png`
- `targeted_refusal_similarity_anomaly.pdf`
