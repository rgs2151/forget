# Figure 6

## Caption

**Semantic structure of spillover.** (A) Concept manifold built from question embeddings, with examples of nearby and distant concept pairs. (B) Off-target refusal plotted against semantic similarity for Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B; each point is a non-matched concept pair, contours show pair density, and black lines show fitted trends. (C) Near-versus-far comparison after a median split by semantic similarity. Bars show mean off-target refusal and error bars show standard errors.

## Panel Notes

- Output: `plots/figure1_paper.png` and `plots/figure1_paper.pdf`.
- Analysis tables: `parking/semantic_spillover/plots/`.

## Checks

- Caption source: `paper/latex/acl_new_latex.tex`, Figure 6.
- Encodings and statistics checked against: `semantic_spillover_figure.ipynb`.
- Remaining limitation: the analysis is associational.
