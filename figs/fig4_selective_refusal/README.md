# publish_bar

## Method

- Read `bars_judged.csv` for the four main models on Inhouse, MMLU, RWKU, and ConceptVectors.
- Keep intervention rows and average `judge_refusal` separately for matched concept-target pairs and non-matched pairs.
- Write `plots/publish_bar.png`.

## Variables

- Models: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B, and Phi-4.
- Groups: targeted (`concept == target`) and untargeted (`concept != target`).
- Measure: binary judge refusal averaged as a rate.
- Result resolution: `prefill_logit`, falling back to `main` when unavailable.

## Statistics

- None; each bar is a descriptive mean over the corresponding evaluation rows.

## Legends

- X axis: Inhouse, MMLU, RWKU, and CV.
- Y axis: refusal rate from 0 to 1.
- Dark red: targeted pairs.
- Black: untargeted pairs.
- Panels: one model per panel in the order listed above.

## Interpretation

- Compare intended target-concept refusal with refusal on other concepts.

## Notes

- This output supplies the dataset-by-model refusal summary used in Figure 4.

## References

- Code: `publish_bar.py`.
- Inputs: `parking/model_matrix/cache/<model>_<dataset>/results/<result>/bars_judged.csv`.

# publish_confusion

## Method

- Read Inhouse `confusion_judged.csv` for the four main models.
- Average `judge_refusal` for every queried-concept and steering-target pair.
- Arrange the means as square matrices and write `plots/publish_confusion.png`.

## Variables

- Rows: queried concepts.
- Columns: steering targets.
- Measure: binary judge refusal averaged as a rate.
- Concept labels: lowercase names on the first matrix and indexed labels on all matrices.

## Statistics

- None; every cell is a descriptive mean.

## Legends

- Value scale: white is 0 and black is 1.
- Diagonal: queried concept matches the steering target.
- Off diagonal: queried concept and steering target differ.

## Interpretation

- Diagonal concentration indicates selective target-concept refusal; dark off-diagonal cells indicate refusal on other concepts.

## Notes

- Matrices have square cells and no colorbar.

## References

- Code: `publish_confusion.py`.

# publish_confusion_ret

## Method

- Apply the same matrix construction as `publish_confusion` to `judge_retention`.
- Write `plots/publish_confusion_ret.png`.

## Variables

- Rows: queried concepts.
- Columns: steering targets.
- Measure: binary judge retention averaged as a rate.

## Statistics

- None; every cell is a descriptive mean.

## Legends

- Value scale: white is 0 and black is 1.
- Diagonal and off-diagonal positions use the same concept ordering as the refusal matrices.

## Interpretation

- Compare answer retention for matched and non-matched concept pairs.

## Notes

- Layout and dimensions exactly match `publish_confusion.png`.

## References

- Code: `publish_confusion.py`.

# publish_disruption

## Method

- Read `prefill_logit` bar evaluations for the four main models.
- Compute targeted and untargeted mean retention within each model and dataset, then average those rates across available models.
- Connect the two dataset-level means and write `plots/publish_disruption.png`.

## Variables

- Datasets: Inhouse, MMLU, RWKU, and ConceptVectors.
- X axis: targeted and untargeted pair groups.
- Measure: mean `judge_retention`.

## Statistics

- None; points are descriptive means across available model-level rates.

## Legends

- Line color identifies the dataset.
- Dark-red marker: targeted mean.
- Black marker: untargeted mean.

## Interpretation

- Each line shows how retention changes between matched and non-matched concept pairs for one dataset.

## Notes

- Missing model-dataset results are omitted from that dataset mean.

## References

- Code: `publish_disruption.py`.

# publish_fluency

## Method

- Repeat the paired dataset summary using `judge_fluency` from the `main` result variant.
- Write `plots/publish_fluency.png`.

## Variables

- Datasets and pair groups match `publish_disruption`.
- Measure: mean binary judge fluency.

## Statistics

- None; points are descriptive means across available model-level rates.

## Legends

- Line color identifies the dataset.
- Dark-red marker: targeted mean.
- Black marker: untargeted mean.

## Interpretation

- Each line compares fluency for matched and non-matched concept pairs.

## Notes

- This is the only Figure 4 component that intentionally reads the `main` result variant.

## References

- Code: `publish_disruption.py`.

# publish_refusal

## Method

- Repeat the paired dataset summary using `judge_refusal` from `prefill_logit`.
- Write `plots/publish_refusal.png`.

## Variables

- Datasets and pair groups match `publish_disruption`.
- Measure: mean binary judge refusal.

## Statistics

- None; points are descriptive means across available model-level rates.

## Legends

- Line color identifies the dataset.
- Dark-red marker: targeted mean.
- Black marker: untargeted mean.

## Interpretation

- Each line compares refusal for matched and non-matched concept pairs.

## Notes

- The dataset legend is placed at the lower left.

## References

- Code: `publish_disruption.py`.
