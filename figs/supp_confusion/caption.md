# Supplementary confusion matrices

## Caption

**Concept-level evaluation matrices.** Columns show model checkpoints grouped by family. Rows are grouped by refusal, retention, and fluency, with In-house, MMLU, RWKU, and ConceptVectors repeated within each metric block. Each matrix uses queried concepts as rows and steering targets as columns; darker cells indicate higher rates. Crossed panels indicate model-dataset pairs that were not run.

## Panel Notes

- Output: `plots/supp_confusion.png`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`.
- Matrix orientation checked against: `supp_confusion.py`.
- Statistics: descriptive cell means only.
