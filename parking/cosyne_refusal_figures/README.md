# layer_metrics

## Method

- Read the saved synthetic-concept calibration responses for Llama-3.1-8B,
  Mistral-7B-v0.3, and Qwen2.5-7B from the `prefill_logit` result variant.
- Keep intervention rows with matching queried concept and steering target.
  Average each binary judge measure within each layer–strength cell.
- Plot one line per layer across the saved strengths. Arrange refusal above
  retention, with the three models across columns. Write
  `plots/layer_metrics.{png,pdf,csv}`. Fluency is omitted from this figure.

## Variables

- Script: `replot.py`; run from the repository root with
  `/home/dev/miniconda3/envs/forget/bin/python -m parking.cosyne_refusal_figures.replot`.
- Inputs: `parking/model_matrix/cache/{llama8b,mistral7b,qwen7b}_inhouse/results/prefill_logit/calibration_judged.csv`.
- Measures: `judge_refusal`, `judge_retention`; both binary.
- Points contain 100 responses: ten questions for each of ten targeted concepts.
- Layers: 0–31 for Llama/Mistral; 0–27 for Qwen. Ten strengths: 1–10 for
  Llama/Mistral and 10–100 in steps of ten for Qwen. No synthetic zero-strength
  observations are added.

## Statistics

- Descriptive arithmetic means; no hypothesis test, null/alternative,
  significance threshold, or smoothing is used. Straight line segments connect
  the observed means; no intermediate samples are generated.
- A point is the fraction of its 100 responses receiving the named judge label.
- All layers/strengths are shown; none are selected by the plotting script.
- Retention concerns the targeted answers in this calibration grid. It does
  not estimate preservation of other concepts.

## Legends

- Columns: Llama, Mistral, Qwen. Rows: refusal, retention.
- Horizontal position: `Scale s`, the steering strength. Vertical position:
  refusal or retention rate, with endpoints 0 and 1.
- Every layer has a line. The exact original `custom_cmap` in
  `forget/plot/plot.py` maps the layer index to color: purple/blue through
  cyan/green/yellow to orange/red. Each column has its own layer color bar.
- All lines use alpha 1.0 and equal width 0.7. No layers are faded, recolored,
  hidden, or preferentially emphasized in this restoration pass.

## Interpretation

- Compare where the intervention induces refusal and changes the baseline
  answer. These outcomes are scored separately.

## Notes

- Inputs are read-only. The script asserts full layer/strength coverage,
  100 unique question–concept rows per cell, and no missing metric labels.
- The user requested restoration of the paper's line plot and palette, keeping
  the two-by-three layout. Increasing middle-layer color contrast is a possible
  later palette revision; this pass preserves the exact original colors.

## References

- `figs/fig5_layer_scale_sweeps/publish_params.py` and its README.
- Saved per-run `prefill_logit/config.yml`; `forget/refuse/calibration.py`.

# selectivity

## Method

- Read the four models' synthetic-concept confusion evaluations. Average
  refusal within each queried-concept/steering-target pair and retain the
  recorded operating point. Write four matrices and their numeric table.
- Place the paper's original exclusive/unrestricted refusal schematic panels
  A/B on the left, with four measured model matrices in a two-by-two grid on
  the right (panel C). The schematic pixels are retained without redrawing.
- Separately compute target refusal, off-target refusal, and off-target
  retention from each model–dataset bar evaluation. Weight the 16 combinations
  equally for the abstract's aggregate rates.

## Variables

- Inputs: `parking/model_matrix/cache/{llama8b,mistral7b,qwen7b,phi4}_inhouse/results/prefill_logit/confusion_judged.csv`.
- Aggregate inputs: the same four model keys crossed with
  `{inhouse,mmlu,rwku,conceptvectors}/results/prefill_logit/bars_judged.csv`.
- Ten queried concepts by ten targets; ten questions per matrix cell.
- Schematic input: `inputs/Figure4_original.png`, a byte-identical copy of
  `/home/dev/forget/paper_main/figures/Figure4.png` and the supplied paper asset.
  Crop rectangles (x0,y0,x1,y1): A=(40,0,680,420), B=(700,0,1290,420).
- Outputs: `plots/selectivity.{png,pdf,csv}` and `plots/evaluation_summary.csv`.

## Statistics

- Descriptive binary-label means; no hypothesis test or decision threshold.
- Equal-weight means over the 16 evaluation rows are 0.8019921 target refusal,
  0.1990440 off-target refusal, and 0.7453853 off-target retention. The rounded
  percentages are 80.2%, 19.9%, and 74.5%.
- This explicit aggregation corrects the source manuscript's 19.8% off-target
  refusal value. These means are distinct from averages of the displayed
  synthetic-concept confusion matrices.

## Legends

- A/B: illustrative exclusive versus unrestricted refusal. Red bars summarize
  targeted refusal; black bars summarize untargeted refusal. They are diagrams,
  not new observed data.
- C top row: Llama-3.1-8B, Mistral-7B-v0.3. Bottom row: Qwen2.5-7B, Phi-4.
- Paper axis wording is restored: `Targeted Concepts` vertically and `Steered
  Concepts` horizontally. Rows represent queried concepts; columns represent
  steering targets. Row labels follow `bacteria (c₁)` with the index after the
  name; column labels are c₁ … c₁₀. Names/capitalization match the original.
- Black is refusal rate one; white is zero.

## Interpretation

- The diagonal measures refusal on the targeted concept; off-diagonal cells
  locate refusal on other concepts under the same intervention.

## Notes

- Model labels are bound directly to each source filename. Existing composite
  image placement is not used to assign the model identities.
- Each confusion file must have a single layer/strength setting and ten unique
  questions per concept–target pair. No data are filled or imputed.

## References

- `figs/fig4_selective_refusal/publish_confusion.py` and `publish_bar.py`.
- Primary manuscript reference, as designated by the user:
  `/home/dev/cosine2027/ref/mine/refusal_steering/latex/acl_new_latex_backup.tex`.
  Its Figure 4 A/B descriptions define exclusive versus unrestricted refusal.

# semantic_cloud

## Method

- Preserve the original Figure 6 turquoise-shell cloud as a raster panel from
  `figs/fig6_semantic_spillover/plots/figure1_paper.png`. The crop retains its
  points, shells, axes, Germany/Austria and Cannabis/Jason Momoa annotations.
- Reuse cached MiniLM question-to-target cosine similarities. Join them to
  `prefill_logit` refusal labels using exact `(concept, target, question)` keys,
  separately for each model. Require one-to-one matches for all 900 rows/model.
- Split question–target pairs at the original global median similarity.
  Compute mean refusal and row-wise standard error within each group/model.
- Average current refusal by `(target, queried concept)`, join the cached
  concept-centroid cosine similarities on their named keys, and calculate
  Spearman correlations across the 90 non-matched pairs per model.

## Variables

- Cached similarities: `parking/semantic_spillover/plots/spillover_question_level.csv`
  and `spillover_pair_level_sentence_emb.csv`.
- Updated rates: `parking/model_matrix/cache/{llama8b,mistral7b,qwen7b}_conceptvectors/results/prefill_logit/confusion_judged.csv`.
- Audit inputs: corresponding `results/main/confusion_judged.csv` files.
- Ten ConceptVectors concepts, 900 off-target question–target observations per
  model. Near: cosine ≥ 0.237964905; far: cosine below that median. Each group
  has 450 observations per model.
- Cloud: externally encoded MiniLM questions and concept centroids pooled
  across six datasets; original joint 3D UMAP and Gaussian-density shells.
  No encoder, UMAP, or model inference is rerun. Crop coordinates and source
  hash are recorded in `plots/semantic_provenance.json` and `source_hashes.json`.
- Outputs: `plots/semantic_cloud.{png,pdf}`, `semantic_near_far.csv`,
  `semantic_pairs.csv`, `semantic_correlations.csv`, `semantic_provenance.json`.

## Statistics

- Bars are arithmetic means of binary refusal labels. Error bars are sample
  standard deviation divided by √n, matching the original descriptive SEM.
  Repeated questions and shared concepts are not independent; these are not
  cluster-adjusted confidence intervals and support no thresholded decision.
- Spearman rho measures monotonic association between centroid similarity and
  pair-mean refusal. Rho: 0.2775264, 0.4154070, 0.2018184 for Llama, Mistral,
  Qwen, respectively. The CSV also records conventional two-sided p-values
  (null: zero rank association; alternative: nonzero). Pair dependence makes
  these exploratory; the draft reports rho descriptively and makes no claim
  that all three associations are statistically significant.
- Original cloud UMAP and density surfaces are descriptive transforms, not
  evidence that the LLM's internal states form this geometry or that semantic
  proximity causes spillover.

## Legends

- A: gray question points, darker concept centroids, turquoise projected-density
  shells. Red annotates Germany/Austria; gray annotates Cannabis/Jason Momoa.
- B: gray bars are far pairs; dark red bars are near pairs. The vertical axis
  is off-target refusal rate; horizontal groups identify the three models.
- The near/far rate axis begins at zero. The cloud's axes are UMAP coordinates;
  annotated similarities were calculated in the original embedding space.

## Interpretation

- The original point cloud illustrates semantic proximity. The adjacent bars
  test how off-target refusal differs by proximity using the same run variant
  as the other new panels. These are different observation pools: the cloud
  covers six datasets, while the refusal comparison uses ConceptVectors.

## Notes

- The old cached semantic labels match `main` exactly (900/900 per model).
  They differ from `prefill_logit` in 189 Llama, 175 Mistral, and 441 Qwen rows.
  The old correlations 0.40/0.52/0.21 therefore must not accompany the new bars.
- High-dimensional similarities are reused only after exact question/target
  matching. The historical cloud is preserved; its coordinate cache was not
  needed for the crop and its original projection was not recomputed.
- All input SHA-256 hashes are checked before/after plotting. The unit writes
  only its own `plots/` and `cache/`. No protected result or existing plot is
  modified. The three new figures embed as complete images in Word.

## References

- `figs/fig6_semantic_spillover/semantic_spillover_figure.ipynb`, cells 3–8.
- `parking/semantic_spillover/semantic_similarity_analysis.ipynb`, cells 16/19.
