# Figure 7

## Caption

**Dataset-resolved scaling of refusal feasibility.** Each panel plots targeted refusal rate against model size using each run's selected layer and scale. Dashed lines show dataset-specific linear fits, with \(R^2\) and \(p\) values shown inside each panel. The fitted trends are positive in all four datasets but vary in strength.

## Panel Notes

- Figure output: `plots/score_size_refusal.png`.
- Model-level values and fit inputs: `plots/score_size_refusal_summary.csv`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`, Figure 7.
- Fit checked against: `scipy.stats.linregress` in `model_size_scaling.py`.
- Remaining limitation: the regression is descriptive.

# Model-size table

## Caption

**Selected in-house settings and evaluation rates for the model-size analysis.** Targeted and untargeted rates correspond to matched and non-matched concept pairs. Layers are zero-indexed.

## Panel Notes

- Table artifact: `plots/publish_table_res.csv`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`.
- Values are generated from selected calibration cells and bar evaluations.
