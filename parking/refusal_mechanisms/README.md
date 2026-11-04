# refusal_mechanisms

Planned study; implementation and experiments have not started.

## Method

Study two questions using conditional steering with ReFT-r1-trained directions:

1. Does steering produce targeted refusal while preserving answers about other concepts?
2. How do classification errors, semantic similarity, representational overlap, and sparsity explain spillover?

Compare exactly two conditions on the same held-out questions:

| Condition | Procedure |
| --- | --- |
| No intervention | Generate the model's baseline answer. |
| Intervention | Apply concept-gated refusal steering. |

| Investigation | What We Measure | Figure Axes |
| --- | --- | --- |
| Does steering work? | Targeted/untargeted refusal, retention, fluency | Strength × rate; one curve per layer |
| Does classification work? | AUROC, balanced accuracy, false-positive and false-negative rates | Layer × classification performance |
| Where does spillover occur? | Refusal and detector activation for every concept pair | Queried concept × steering target; aligned matrices |
| Does semantics explain errors? | Semantic similarity versus false-positive detection and off-target refusal | Similarity × error/refusal rate |
| Does geometry explain errors? | Alignment between concept directions; overlap of their responses to prompts | Direction/response overlap × error/refusal rate |
| Does sparsity explain errors? | Concentration across hidden coordinates and across concepts | Layer × sparsity, alongside classification and spillover |

Learn concept-detection directions from ordinary concept-related examples and
a shared refusal direction from matched answering/refusal examples. Record
detector activity during prefill and generation alongside the resulting answers.
Use the same detector and steering procedure throughout the investigation.

## Variables

- Model: `meta-llama/Llama-3.2-1B-Instruct`; base-model weights remain frozen.
- Dataset: Inhouse only, all ten concepts. No cross-dataset or model-size study.
- Inputs: existing Inhouse splits under `parking/model_matrix/cache/inhouse/`.
- Preserve each question's concept label and source row when constructing pairs and splits.
- Layers: all 16; learn and apply directions at the same layer.
- Proposed sampling: 64 positive and 64 balanced negative training examples per concept, 20 separate validation examples per concept, and 20 separate test questions per concept.
- Evaluation: ten queried concepts by ten steering targets, giving 2,000 intervention generations per operating point. Reuse each question's baseline across steering targets.
- Repeats: three training seeds.
- Intervention timing: prefill and generation; preserve concept-conditioned gating.
- Judge: existing logit mode and refusal, retention, and fluency rubrics.
- Strengths: published AxBench steering-factor grid; select operating settings using validation data.
- Store new artifacts in this unit's `cache/` and figures/tables in `plots/`.

## Statistics

- Refusal, retention, and fluency are means of the corresponding binary judge decisions. Report targeted and untargeted rates separately.
- Classify each prompt as belonging or not belonging to the selected concept. Report AUROC and balanced accuracy, both with chance reference 0.5, plus false-positive and false-negative rates. Set classification thresholds on validation data.
- Measure semantic similarity with sentence-embedding cosine similarity and direction alignment with cosine similarity between learned concept vectors.
- Measure coordinate concentration with Hoyer sparsity on absolute direction weights. Measure concept selectivity from detector responses across the ten concepts. These describe different properties; rank one does not imply sparse coordinates.
- Relate classification errors and off-target refusal to semantic similarity, direction overlap, and sparsity. Keep repeated observations of a question grouped when estimating uncertainty or evaluating explanatory predictions.
- These are planned analyses; no results or significance claims are available.

## Legends

- Concept matrices: rows are queried concepts; columns are steering targets. Diagonal cells are targeted conditions; off-diagonal cells are untargeted conditions.
- Refusal matrices show mean binary judge refusal. Detector matrices show the fraction of prompts classified as the steering target.
- Layer curves compare all 16 layers. Strength curves show the intervention factor on the x axis and the named rate on the y axis.
- Sparsity panels identify whether they show coordinate concentration or selectivity across concepts.
- Figure appearance follows `STYLE.md`.

## Interpretation

- Assess selective steering using high targeted refusal together with low untargeted refusal and preserved untargeted retention.
- Compare the locations of classification mistakes and off-target refusals across concepts and layers.
- Determine whether geometry and sparsity add explanatory information beyond semantic similarity. Associations alone do not establish a causal explanation.

## Notes

- The only experimental conditions are no intervention and intervention.
- This unit investigates refusal mechanisms using an established direction-learning method.
- ReFT-r1 direction learning and conditional application are separate choices; document their implementation explicitly.
- Existing model-matrix results and paper files remain unchanged.

## References

- [AxBench paper](https://arxiv.org/abs/2501.17148).
- ReFT-r1 source: `/home/dev/forget/ref/repos/axbench/axbench/models/lsreft.py` (`LsReFT`).
- External paper and repository collection: `/home/dev/forget/ref/README.md`.
- Existing framework: `forget/refuse/`.
