# figure1_paper

## Method

- Pool distinct dataset-concept-question triples from the main bar evaluations.
- Encode questions with `sentence-transformers/all-MiniLM-L6-v2`, average normalized embeddings by concept, and project questions and centroids jointly into three dimensions with UMAP.
- Estimate a three-dimensional Gaussian KDE over the projected questions and render isosurfaces with marching cubes.
- Load question-level and concept-pair spillover tables from `parking/semantic_spillover/plots/`.
- Compare question-level off-target refusal above and below the global median semantic similarity and fit pair-level refusal against semantic similarity for each model.
- Write `plots/figure1_paper.png` and `plots/figure1_paper.pdf`.

## Variables

- Panel A observations: 5,171 unique questions and 311 dataset-concept centroids from six datasets.
- Embeddings: 384-dimensional normalized MiniLM sentence embeddings.
- UMAP: 3 components, 15 neighbors, cosine distance, `min_dist=0.6`, `spread=1.5`, seed 0.
- Panels B and C: off-target judge refusal and cosine similarity between question or concept representations.
- Models: Llama-3.1-8B, Mistral-7B-v0.3, and Qwen2.5-7B.

## Statistics

- Panel A uses Gaussian KDE for display and k-means with \(k=2\) to select illustrative nearby and distant concept pairs.
- Panel B splits question-target pairs at the global median similarity and reports mean off-target refusal with standard error.
- Panel C shows an ordinary least-squares line and Spearman rank correlation for each model.
- The regression null is zero linear slope; the rank-correlation null is zero monotonic association.
- These analyses describe association and do not establish that semantic similarity causes off-target refusal.

## Legends

- Panel A: gray points are questions, black points are concept centroids, and translucent shells show KDE density.
- Panel B: bars compare low- and high-similarity question-target pairs.
- Panel C: points are non-matched concept pairs, contours show pair density, and black lines show fitted trends.

## Interpretation

- The figure tests whether off-target refusal is larger for semantically closer concepts.

## Notes

- The notebook reads fixed result tables and does not modify the protected model-matrix cache.
- Its established visual appearance is preserved during repository reorganization.

## References

- Figure notebook: `semantic_spillover_figure.ipynb`.
- Analysis unit: `parking/semantic_spillover/`.
