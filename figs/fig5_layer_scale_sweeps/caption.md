# Figure 5

## Caption

**Layer--scale sweeps for refusal steering.** Rows show Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B on the in-house dataset. The left panels show refusal, retention, and fluency across scale at the selected layer; stars mark the selected scale. The right panels show the same metrics across all layers, with color indicating layer depth. Scale ranges follow the configured sweep for each model.

## Panel Notes

- Full output: `plots/publish_params.png`.
- Reduced output: `plots/publish_params_min.png`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`, Figure 5.
- Encodings checked against: `publish_params.py`.
- Statistics checked against: Seaborn bootstrap confidence intervals in the selected-layer panels.
