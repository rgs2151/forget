# Supplementary selected calibration

## Caption

**Selected steering layer and scale.** Rows show In-house, MMLU, RWKU, and ConceptVectors; columns show model checkpoints grouped by family. Each panel shows calibration curves at the selected layer, and stars mark the selected scale. Crossed panels indicate model-dataset pairs that were not run.

## Panel Notes

- Output: `plots/supp_optimal.png`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`.
- Selection and encodings checked against: `supp_optimal.py`.
- Intervals: 95% bootstrap confidence intervals.
