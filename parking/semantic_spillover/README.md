# targeted_refusal_similarity_anomaly

## Method

- Read Inhouse confusion evaluations for the selected model set.
- Represent each question and steering-target descriptor with normalized MiniLM sentence embeddings.
- Bin question-target pairs by semantic similarity and summarize targeted-refusal anomaly within each bin.
- Write the binned table, coefficient table, PNG, and PDF under `plots/`.

## Variables

- Inputs: `parking/model_matrix/cache/*_inhouse/results/main/confusion_judged.csv`.
- Measure: judge refusal for non-matched queried-concept and steering-target pairs.
- Similarity: cosine similarity between normalized sentence embeddings.
- Outputs: `targeted_refusal_similarity_bins.csv`, `targeted_refusal_similarity_coefficients.csv`, and `targeted_refusal_similarity_anomaly.{png,pdf}`.

## Statistics

- The notebook fits the documented similarity coefficients and reports descriptive binned means.
- The null for a fitted similarity coefficient is zero association; the alternative is nonzero association.

## Legends

- X axis: semantic-similarity bin.
- Y axis: mean off-target refusal.
- Grouping: model.

## Interpretation

- Test whether off-target refusal changes with semantic proximity.

## Notes

- This is an analysis unit; the final paper composition is in `figs/fig6_semantic_spillover/`.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# spillover_vs_similarity

## Method

- Aggregate non-matched confusion rows to queried-concept/steering-target pairs.
- Compute concept-pair semantic similarity and pair-level mean judge refusal.
- Write the pooled table and per-model scatter plots.

## Variables

- Outputs: `spillover_vs_similarity.csv` and `spillover_vs_similarity_<model>.{png,pdf}`.
- Points: non-matched concept pairs.
- Models: available Inhouse model runs.

## Statistics

- Pair plots show fitted association between semantic similarity and mean off-target refusal.
- The null is no association between the two quantities.

## Legends

- X axis: concept-pair cosine similarity.
- Y axis: mean off-target refusal.
- Panels/files: one model per output.

## Interpretation

- Compare the strength and direction of semantic association across models.

## Notes

- None beyond the fixed notebook configuration.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# spillover_vs_similarity_conceptvectors

## Method

- Repeat the concept-pair analysis for ConceptVectors evaluations.
- Write the pooled table and available per-model plots.

## Variables

- Outputs: `spillover_vs_similarity_conceptvectors.csv` and model-specific PNG/PDF files.
- Dataset: ConceptVectors.
- Measure: pair-level mean off-target judge refusal.

## Statistics

- Fitted association and null interpretation match `spillover_vs_similarity`.

## Legends

- X axis: concept-pair cosine similarity.
- Y axis: mean off-target refusal.
- Files: one available model per output.

## Interpretation

- Check whether the semantic association appears in the larger ConceptVectors inventory.

## Notes

- Missing model outputs reflect unavailable evaluated inputs.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# spillover_question_level_paperfig

## Method

- Retain individual non-matched question-target rows.
- Encode each question and target descriptor and compute cosine similarity.
- Split rows at the global median similarity and summarize judge refusal.
- Write the question-level table, summary table, PNG, and PDF.

## Variables

- Outputs: `spillover_question_level.csv`, `spillover_question_level_summary.csv`, and `spillover_question_level_paperfig.{png,pdf}`.
- Observation: one question under one non-matched steering target.
- Measure: binary judge refusal.

## Statistics

- Bars are mean refusal below and above the global median similarity.
- Error bars are standard errors.
- The comparison is descriptive; no thresholded hypothesis decision is used.

## Legends

- X axis: low- and high-similarity groups.
- Y axis: mean off-target refusal.
- Grouping: model.

## Interpretation

- Compare off-target refusal for semantically distant and nearby question-target pairs.

## Notes

- The question-level CSV is an input to the final Figure 6 notebook.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# spillover_pair_scatter

## Method

- Average question-level refusal within each non-matched concept pair.
- Pair the mean with concept-level semantic similarity.
- Write `spillover_pair_level_sentence_emb.csv` and `spillover_pair_scatter.{png,pdf}`.

## Variables

- Observation: one queried-concept/steering-target pair.
- X axis: semantic similarity.
- Y axis: mean off-target refusal.

## Statistics

- The scatter view fits pair-level association by model.
- The null is zero association; the analysis is descriptive and associational.

## Legends

- Color/group identifies model.
- Points are concept pairs.
- Fitted lines summarize model-specific trends.

## Interpretation

- Show how pair-level off-target refusal varies with semantic similarity.

## Notes

- The pair-level CSV is an input to the final Figure 6 notebook.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# concept_embedding_pairs

## Method

- Average normalized question embeddings within each dataset-concept group.
- Identify illustrative high- and low-similarity concept pairs.
- Write the selected-pair table and two-dimensional pair visualization.

## Variables

- Outputs: `concept_manifold_top_pairs.csv` and `concept_embedding_pairs.{png,pdf}`.
- Representations: normalized MiniLM question embeddings and concept centroids.

## Statistics

- None; pair selection and visualization are descriptive.

## Legends

- Points represent concept centroids.
- Highlighting distinguishes the selected nearby and distant pairs.

## Interpretation

- Provide concrete examples of semantic proximity used in the spillover analysis.

## Notes

- None yet.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.

# concept_manifold_3d

## Method

- Project pooled question embeddings and concept centroids into three dimensions with UMAP.
- Estimate display density and render the concept manifold.
- Write `concept_manifold_3d.{png,pdf}`.

## Variables

- Observations: pooled questions and dataset-concept centroids.
- Representation: normalized MiniLM embeddings.
- Transform: three-dimensional UMAP.

## Statistics

- UMAP and density estimation are descriptive transforms, not hypothesis tests.

## Legends

- Gray points: questions.
- Black points: concept centroids.
- Surfaces: estimated question density.

## Interpretation

- Display the semantic neighborhood structure used to motivate nearby and distant concept comparisons.

## Notes

- The final paper rendering is owned by `figs/fig6_semantic_spillover/`.

## References

- Notebook: `semantic_similarity_analysis.ipynb`.
