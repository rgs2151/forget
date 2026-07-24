# Paper Revision Checklist

Scope: title, abstract, introduction, and figure images are not part of this revision.

## Number Audit

- [ ] Make a number-audit table with one row per reported number.
- [ ] Use `plot/figures/publish_table_res.csv` for selected layers.
- [ ] Use `plot/figures/publish_table_res.csv` for selected scales.
- [ ] Use `plot/figures/publish_table_res.csv` for targeted refusal rates.
- [ ] Use `plot/figures/publish_table_res.csv` for targeted retention rates.
- [ ] Use `plot/figures/publish_table_res.csv` for untargeted refusal rates.
- [ ] Use `plot/figures/publish_table_res.csv` for untargeted retention rates.
- [ ] Use `plot/figures/score_size_refusal_summary.csv` for model-size regression values.
- [ ] Add the spillover figure source file to the number-audit table.
- [ ] Remove body numbers outside the fixed opening that cannot be traced to the number-audit table.
- [ ] Remove appendix numbers that cannot be traced to the number-audit table.
- [ ] Replace body numbers after the Introduction that came from older runs.
- [ ] Replace appendix numbers that came from older runs.

## Problem Setup

- [ ] State that each experiment selects one steering target.
- [ ] State that each prompt has one queried concept.
- [ ] State that the model is evaluated under each steering target.
- [ ] Define matrix rows as queried concepts.
- [ ] Define matrix columns as steering targets.
- [ ] Define diagonal cells as targeted conditions.
- [ ] Define off-diagonal cells as untargeted conditions.
- [ ] Define refusal as explicit refusal rate.
- [ ] Define retention as agreement with the unsteered baseline answer.
- [ ] Define fluency as readable-output rate.
- [ ] Remove neuroscience analogy language from the problem setup.

## Method

- [ ] Describe baseline response activation examples.
- [ ] Describe refusal response activation examples.
- [ ] State which tokens are pooled for activation vectors.
- [ ] State how the concept detector is trained.
- [ ] State how the detector gate is computed at inference time.
- [ ] State how the shared refusal direction is computed.
- [ ] State where the gated activation update is applied.
- [ ] State how calibration examples are sampled.
- [ ] State how the selected layer is chosen.
- [ ] State how the selected scale is chosen.
- [ ] State that evaluation uses the calibrated layer.
- [ ] State that evaluation uses the calibrated scale.
- [ ] Remove code file names from the main method text.
- [ ] Put batch sizes in the appendix.

## Experimental Setup

- [ ] List the models shown in the main figures.
- [ ] List the datasets shown in the main figures.
- [ ] State the number of concepts in each matrix.
- [ ] State the number of questions in each matrix cell.
- [ ] State the number of questions in each bar summary.
- [ ] State which reported results use logit scoring.
- [ ] State which reported results use reasoning judge scoring.
- [ ] State what the error bars show in the bar figures.
- [ ] State what uncertainty values are shown in regression figures.
- [ ] Remove full concept lists from the main text.
- [ ] Verify that full concept lists are present in the appendix.
- [ ] Remove full train/test counts from the main text.
- [ ] Verify that full train/test counts are present in the appendix.

## Main Results

- [ ] Start the results section with the cross-model figure.
- [ ] Explain the refusal heatmaps.
- [ ] Explain the retention heatmaps.
- [ ] Explain targeted bars.
- [ ] Explain untargeted bars.
- [ ] Report targeted refusal as the main steering effect.
- [ ] Report untargeted refusal as off-target refusal.
- [ ] Use the word spillover only for the baseline-subtracted measure.
- [ ] Report untargeted retention against the baseline answer.
- [ ] Explain what the layer calibration plot shows.
- [ ] Explain what the scale calibration plot shows.
- [ ] Report model-size trends only from the plotted regression.
- [ ] Remove sentences that call missing runs failed results.
- [ ] Replace blanket Qwen-incomplete statements with exact missing-run notes.
- [ ] Remove Results-section sentences that say only Llama and Mistral are complete.
- [ ] Remove Appendix sentences that say only Llama and Mistral are complete.

## Captions

- [ ] Rewrite the cross-model figure caption to match its panels.
- [ ] Rewrite the qualitative figure caption to match its examples.
- [ ] Rewrite the calibration figure caption to match its panels.
- [ ] Rewrite the spillover figure caption to match its panels.
- [ ] Rewrite the model-size figure caption to match its panels.
- [ ] Remove code file names from main-text captions.
- [ ] Remove calibration file names from main-text captions.
- [ ] Replace table captions that restate column names.
- [ ] Check that each caption states the plotted comparison.
- [ ] Check that each caption states the plotted metric.

## Figure and Table References

- [ ] Give each main-text figure a unique label.
- [ ] Give each appendix figure a unique label.
- [ ] Replace stale Figure S references in the main text.
- [ ] Check each `Figure~\ref{...}` target.
- [ ] Check each `Table~\ref{...}` target.
- [ ] Check that figures appear after first mention.
- [ ] Check that tables appear after first mention.

## Spillover Text

- [ ] State what the spillover figure measures.
- [ ] State the semantic similarity measure used in the spillover figure.
- [ ] State the fitted relationship shown in the spillover figure.
- [ ] Remove text that describes spillover as neural collateral recruitment.
- [ ] State that geometry claims are interpretation rather than mechanistic proof.
- [ ] Tie the main spillover claim to the plotted result.

## Discussion

- [ ] Restate the targeted-refusal result.
- [ ] Restate the untargeted-retention result.
- [ ] Restate the off-target-refusal result.
- [ ] Restate the model-size result.
- [ ] Remove result claims not shown earlier.
- [ ] Remove discussion claims that compare LLM results to specific neural-stimulation outcomes.

## Limitations

- [ ] State that refusal does not prove knowledge deletion.
- [ ] State that judge scoring can introduce measurement error.
- [ ] State that prompt wording can affect results.
- [ ] State that layer choice can affect results.
- [ ] State that scale choice can affect results.
- [ ] State that related concepts can still be affected.
- [ ] State that adversarial robustness is not fully tested.

## Appendix

- [ ] Update the model table with the final reported models.
- [ ] Update the dataset table with the final dataset counts.
- [ ] Update the steering table with final selected layers.
- [ ] Update the steering table with final selected scales.
- [ ] Update the judge section for logit scoring.
- [ ] Update the judge section for reasoning judge scoring.
- [ ] Remove repeated activation-pooling explanations from vector construction.
- [ ] Remove repeated detector-fitting explanations from vector construction.
- [ ] Include one prompt-template example that explains activation pairs.
- [ ] Delete prompt-template examples unrelated to activation pairs.
- [ ] Format the concept inventory as a reference table.
- [ ] Remove notebook-style wording.
- [ ] Remove run-log wording.

## Final Checks

- [ ] Search for old `completed` claims outside the fixed opening.
- [ ] Search for old `incomplete` claims outside the fixed opening.
- [ ] Search for `placeholder`.
- [ ] Search for judge claims that say one mode produced all results.
- [ ] Search for old `15 scales` sweep text.
- [ ] Search for old `6.67--100` sweep text.
- [ ] Check that reported rates match `plot/figures/publish_table_res.csv`.
- [ ] Check that regression values match `plot/figures/score_size_refusal_summary.csv`.
- [ ] Check that every included figure exists in `/home/dev/forget/paper/figures`.
- [ ] Check that the PDF compiles without duplicate-label warnings.
