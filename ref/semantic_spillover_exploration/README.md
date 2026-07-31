# similarity_bins_and_pair_spillover

## Method

- Historical notebook that compared concept similarity with targeted refusal and baseline-subtracted off-target refusal.
- It summarized similarity bins and computed model-level Spearman associations.

## Variables

- Historical input: judged confusion tables from the former root store.
- Similarity: token-overlap similarity between concept names.
- Outputs: similarity-bin tables, pair-level spillover tables, and model-specific scatter plots.

## Statistics

- Spearman rank correlation tested for a monotonic association between concept similarity and spillover.
- The notebook is superseded, so its reported values are not current paper results.

## Legends

- X axis: concept-name similarity.
- Y axis: refusal or baseline-subtracted off-target refusal.
- Panels: model checkpoints.

## Interpretation

- This was the first semantic-spillover analysis draft.

## Notes

- Historical paths are intentionally not executable in the current layout.

## References

- Notebook: `similarity_bins_and_pair_spillover.ipynb`.
- Active analysis: `parking/semantic_spillover/semantic_similarity_analysis.ipynb`.

# question_pair_embedding_exploration

## Method

- Extended the first draft with sentence-transformer embeddings of questions and concept descriptors.
- Compared question-level and concept-pair similarity with refusal and spillover.
- Projected concept embeddings into two dimensions with PCA.

## Variables

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Historical inputs: judged confusion tables.
- Outputs: question-level tables, pair-level tables, scatter plots, and a PCA concept map.

## Statistics

- Spearman rank correlation tested monotonic associations between embedding similarity and refusal or spillover.
- Standard errors were computed for binned question-level summaries.

## Legends

- Scatter positions encode semantic similarity and refusal-derived outcomes.
- PCA positions encode the first two principal components of concept embeddings.
- Colors and labels identify models or selected concept pairs.

## Interpretation

- This draft tested whether sentence-level embeddings were more informative than concept-name overlap.

## Notes

- It is retained for provenance and should not be used as the current analysis.

## References

- Notebook: `question_pair_embedding_exploration.ipynb`.
- Active analysis: `parking/semantic_spillover/semantic_similarity_analysis.ipynb`.

# concept_manifold_exploration

## Method

- Extended the sentence-embedding draft with a pooled three-dimensional UMAP projection.
- Embedded questions and concept centroids in a shared low-dimensional space.

## Variables

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Projection: three-dimensional UMAP over pooled question vectors and concept centroids.
- Outputs: concept-manifold plots and nearest-pair tables.

## Statistics

- UMAP is a descriptive nonlinear projection; it does not test a null hypothesis.
- Spearman analyses inherited from the earlier notebook remain exploratory.

## Legends

- Coordinates: UMAP dimensions 1--3.
- Points: embedded questions and concept centroids.
- Labels: selected concepts and concept pairs.

## Interpretation

- This draft explored whether semantic neighborhoods could provide an intuitive visual account of off-target refusal.

## Notes

- The projection was not selected as the final paper figure.

## References

- Notebook: `concept_manifold_exploration.ipynb`.
- Final paper figure: `figs/fig6_semantic_spillover/`.

# bar_sample_manifold_variants

## Method

- Built two- and three-dimensional UMAP views from questions present in judged bar evaluations.
- Generated several halo and color variants of the same manifold composition.

## Variables

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Projection: UMAP over question embeddings and concept centroids.
- Outputs: `manifold_bars_3d`, `manifold_bars_halo`, and the retained 3D halo variants under `plots/`.

## Statistics

- UMAP and the visual variants are descriptive.
- A k-means split was used only to construct one visual variant, not for inference.

## Legends

- Coordinates: UMAP dimensions.
- Point position: embedded question or concept centroid.
- Halo and color changes are alternate visual treatments of the same exploratory geometry.

## Interpretation

- These variants document the visual search that preceded the final semantic-spillover figure.

## Notes

- None of these variants is an active paper result.

## References

- Notebook: `bar_sample_manifold_variants.ipynb`.
- Final paper figure: `figs/fig6_semantic_spillover/`.
