# Inference Analyses

This folder contains notebook-first analyses that sit on top of existing judged artifacts in `store/`.

## Current notebook

- `semantic_similarity_analysis.ipynb`

This notebook tests the relation between targeted-refusal anomaly and semantic similarity between contexts using `confusion_judged.csv` files.

## Inputs

- `store/*_inhouse/confusion_judged.csv`

You can change the glob in the notebook to include other datasets.

## Outputs

The notebook writes to `inference/results/`:

- `targeted_refusal_similarity_bins.csv`
- `targeted_refusal_similarity_coefficients.csv`
- `targeted_refusal_similarity_anomaly.png`
- `targeted_refusal_similarity_anomaly.pdf`
