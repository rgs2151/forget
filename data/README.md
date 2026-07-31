# Data

New datasets for research questions outside the original model matrix belong
under `data/`. The six existing datasets remain inside
`parking/model_matrix/cache/` as the explicit legacy exception required to keep
the protected 361 GB experiment intact.

## Structure

Each dataset has `train.csv` and `test.csv` with the same columns:

| column | meaning |
| --- | --- |
| `concept` | concept label used to fit detectors and define steering targets |
| `question` | model input question or prompt |
| `answer` | reference answer when supplied by the source dataset |

The Inhouse directory also contains `raw.csv`, the pre-split source table.

## Alignment

- A row is the atomic example; `concept`, `question`, and `answer` from one row remain aligned.
- Model outputs retain the source row's `concept` and `question`.
- Bar and confusion evaluations compare the row concept with a separate steering-target column.
- Except for RWKU, dataset construction uses an 80/20 split with seed 42.
- MMLU and ConceptVectors use concept-stratified splits.
- Inhouse uses an unstratified split.
- RWKU retains the source refusal-tuning and forget-evaluation subsets, so per-concept counts are not balanced.

## Missing Values

- ConceptVectors supplies no reference answers; its `answer` field is empty by design.
- Other missing fields must not be imputed silently.
- Rows are identified by their retained CSV position and source columns; there is no independent record ID.

## Inventory

| dataset | path | concepts | train rows | test rows | notes |
| --- | --- | ---: | ---: | ---: | --- |
| Inhouse | `parking/model_matrix/cache/inhouse/` | 10 | 4,752 | 1,188 | Project concept-question dataset |
| MMLU | `parking/model_matrix/cache/mmlu/` | 10 | 928 | 232 | Selected subject areas; correct option text stored as the answer |
| RWKU | `parking/model_matrix/cache/rwku/` | 200 | 57,459 | 12,455 | Source subsets retained; unbalanced by concept |
| Concept10 | `parking/model_matrix/cache/concept10/` | 10 | 720 | 360 | AxBench Concept10 |
| Concept500 | `parking/model_matrix/cache/concept500/` | 500 | 36,000 | 18,000 | AxBench Concept500 |
| ConceptVectors | `parking/model_matrix/cache/conceptvectors/` | 91 | 7,644 | 1,911 | Concepts with at least 100 flattened prompts |

## Source Notebooks

Dataset-construction notebooks are preserved under
`parking/model_matrix/notebooks/`. They write only to the owning
model-matrix cache and must not be rerun without explicit approval for the
affected dataset artifacts.
