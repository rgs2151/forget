# refusal_mechanisms

Isolated study implemented in this unit. The frozen run configuration is
`config.json`; current outputs use `cache/axbench_small/`. Earlier local pilot
outputs remain untouched in `cache/` and are not included in the figures.

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
| Does steering work? | Targeted/untargeted AxBench refusal relevance, instruction relevance, fluency | Strength × 0–2 rating; one curve per layer |
| Does classification work? | AUROC, balanced accuracy, false-positive and false-negative rates | Layer × classification performance |
| Where does spillover occur? | Refusal relevance and detector activation for every concept pair | Queried concept × steering target; aligned matrices |
| Does semantics explain errors? | Semantic similarity versus false-positive detection and off-target refusal relevance | Similarity × error rate/refusal rating |
| Does geometry explain errors? | Alignment between concept directions; overlap of their responses to prompts | Direction/response overlap × error rate/refusal rating |
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
- Sampling: 100 unique training questions (10/concept), 100 separate validation questions (10/concept), and 200 separate test questions (20/concept). No question overlaps across splits.
- Each concept direction uses its 10 positive questions plus 10 negatives sampled approximately evenly from the other concepts. These reuse the same 100-question training pool. The shared refusal direction uses 50 balanced training questions, each paired with a refusal and a baseline answer (100 training sequences).
- Calibration: one validation question per concept, 16 layers, and four factors, giving 640 intervention generations. The other validation questions contribute to threshold selection, not generation sweeps.
- Evaluation: exactly 200 intervention generations, one steering target per test question: 100 targeted and 100 untargeted. All 90 off-diagonal concept pairs appear; each has only one or two answers. Diagonal cells each have 10. Baselines add 200 generations, not another condition.
- Repeats: one training seed, 42.
- Intervention timing: prefill and generation; preserve concept-conditioned gating.
- Judge: AxBench's verbatim prompts, GPT-4o-mini-2024-07-18, temperature 0 (set by AxBench's evaluation entry point), separate concept relevance, instruction relevance, and fluency ratings on 0–2. The concept being judged is refusal, not the steering target's topic. Its exact wording is in `config.json`.
- Strength factors: 0.5, 1, 2, 5. Physical strength equals factor times the learned refusal direction's maximum positive-training activation.
- Store current artifacts in `cache/axbench_small/` and figures/tables in `plots/`. Raw API explanations and token usage are retained in `judge_responses/`.

## Statistics

- Report means of AxBench's three 0–2 ratings, separately for targeted and untargeted questions. Do not call these binary refusal or retention rates.
- AxBench's aggregate is the harmonic mean of the three ratings, zero whenever any rating is zero. Select the highest mean validation aggregate; ties choose lower factor, then lower layer. No test output enters selection.
- Classify each prompt as belonging or not belonging to the selected concept. Report AUROC and balanced accuracy, both with chance reference 0.5, plus false-positive and false-negative rates. Set classification thresholds on validation data.
- Measure semantic similarity with sentence-embedding cosine similarity and direction alignment with cosine similarity between learned concept vectors.
- Measure coordinate concentration with Hoyer sparsity on absolute direction weights. Measure concept selectivity from detector responses across the ten concepts. These describe different properties; rank one does not imply sparse coordinates.
- Compare semantic similarity and direction alignment with classification errors and off-target refusal ratings descriptively. No regression, significance test, or uncertainty claim is made in this small run.
- Geometry/error associations are descriptive; the figures do not establish causal explanations.

## Legends

- Concept matrices: rows are queried concepts; columns are steering targets. Diagonal cells are targeted conditions; off-diagonal cells are untargeted conditions.
- Refusal and instruction matrices show mean AxBench ratings, range 0–2. Detector matrices show the fraction of prompts classified as the steering target, range 0–1.
- Layer curves compare all 16 layers. Strength curves show the intervention factor on the x axis and the named rate on the y axis.
- Sparsity panels identify whether they show coordinate concentration or selectivity across concepts.
- Figure appearance follows `STYLE.md`.

## Interpretation

- Assess selective steering using high targeted refusal relevance together with low untargeted refusal relevance. Instruction relevance measures topical relatedness, not factual answer retention.
- Compare the locations of classification mistakes and off-target refusals across concepts and layers.
- Determine whether geometry and sparsity add explanatory information beyond semantic similarity. Associations alone do not establish a causal explanation.

Completed small run: layer 6, factor 2. The selected layer's mean test AUROC
was 0.588, balanced accuracy 0.607, false-positive rate 0.466, and false-negative
rate 0.320. The gate fired during prefill on 67% of targeted cases and 49% of
untargeted cases.

| Condition | Subset | Answers | Refusal relevance | Instruction relevance | Fluency |
| --- | --- | ---: | ---: | ---: | ---: |
| No intervention | Targeted | 100 | 0.00 | 1.56 | 1.75 |
| No intervention | Untargeted | 100 | 0.00 | 1.71 | 1.72 |
| Intervention | Targeted | 100 | 0.12 | 1.45 | 1.68 |
| Intervention | Untargeted | 100 | 0.25 | 1.60 | 1.72 |

Ratings are on AxBench's 0–2 scale. This configuration did not produce selective
refusal. Its low concept-response sparsity and frequent detector errors describe
a weak gate; they do not establish a cause of spillover. The single-seed,
small-sample results do not establish the performance of ReFT-r1 generally.

## Notes

- Run from the code repository in the `forget` environment: `python -m parking.refusal_mechanisms.run`. Detached screen: `refusal_mechanisms_axbench`. Regenerate figures and verify the protected inventory without loading a model or judging again: `python -m parking.refusal_mechanisms.run --plots-only`.
- `train.py` reproduces AxBench's ReFT-r1 top-k intervention, language-modeling loss, non-top-k L1 penalty, weight normalization, and projected gradient in a unit-local PyTorch adapter. A source-parity test compares its unpadded forward pass against the downloaded upstream implementation. This avoids installing or changing shared model/GPU code.
- Training uses AxBench's example settings: three epochs, batch six, learning rate 0.01, top-k eight, and L1 coefficient 0.005. Conditional inference scales the learned unit refusal direction by its maximum activation on the positive training sequences and the configured factor.
- Training, prompt detection, and prefill gating exclude padding and the common system/user-header prefix, located by comparing two differently worded prompts. Prompt detection uses the maximum ReLU projection over the remaining prompt tokens. Each concept/layer/seed threshold maximizes validation balanced accuracy, with ties selecting the larger threshold. Thresholds never use test labels.
- Classification uses all 10 validation questions per concept for thresholds. Test labels never enter threshold selection.
- Source train/test rows and questions are recorded in `cache/axbench_small/rows.csv`. Fresh baseline outputs are generated once.
- Both GPUs train independent layers concurrently and share calibration and final generation. Judging uses the OpenAI API with batches of 16 concurrent requests. Exact repeated rubric prompts are judged once; cached explanations and scores are reused.
- Gate logs distinguish prefill and decoding. Activation sparsity is Hoyer sparsity of the unsteered mean prompt residual; direction sparsity is Hoyer sparsity of weights; concept-response sparsity is Hoyer sparsity across the ten mean concept responses.
- A direction with zero positive activation is retained with zero intervention amplitude. Its failure to steer is a result, not a reason to drop the layer. All-zero concept responses have undefined Hoyer sparsity; explanatory fits encode this explicitly with an inactive-response indicator and a zero placeholder for the undefined value.
- AxBench's aggregate rewards refusal relevance, topical relevance, and fluency jointly. Generic refusals may receive low instruction relevance and therefore a low aggregate even when they express refusal. Report individual ratings alongside it; do not equate this aggregate with selective-refusal performance.
- `judge_smoke.csv` contains four API sanity examples: a factual answer, a generic refusal, a topical refusal, and malformed text. These are judge checks, not additional experimental controls.
- Unparseable ratings use AxBench's default zero and are flagged explicitly in `axbench_*_parsed`; raw explanations are preserved.
- The first API diagnostic used the wrapper's default temperature 1 before checking the evaluation entry point's override to zero. Its scored tables and setting are preserved under `temperature1_diagnostic/`; temperature-specific API cache keys prevent reuse in the corrected run. Training and calibration generations are reused unchanged.
- The protected result vault's file-size/mtime inventory is saved before the run and checked again at completion.
- The only experimental conditions are no intervention and intervention.
- This unit investigates refusal mechanisms using an established direction-learning method.
- ReFT-r1 direction learning and conditional application are separate choices; document their implementation explicitly.
- This is a conditional application of ReFT-r1-trained directions, not a reproduction of AxBench's unconditional steering benchmark.
- Existing model-matrix results and paper files remain unchanged.

## References

- [AxBench paper](https://arxiv.org/abs/2501.17148).
- ReFT-r1 source: `/home/dev/forget/ref/repos/axbench/axbench/models/lsreft.py` (`LsReFT`).
- External paper and repository collection: `/home/dev/forget/ref/README.md`.
- Existing framework: `forget/refuse/`.
- Judge prompts and aggregation: `/home/dev/forget/ref/repos/axbench/axbench/evaluators/{prompt_templates.py,lm_judge.py}`. API parameters: `axbench/models/language_models.py` and the pinned Llama sweep config in the same checkout.

# classification

## Method
- Project unsteered prompt tokens onto each learned concept direction, apply ReLU, and take the largest remaining-token activation. Choose each threshold using validation balanced accuracy; evaluate on the 200 test questions.
## Variables
- Sixteen layers, ten one-versus-rest concepts, seed 42. Files: `classification.png`, `classification.pdf`, and `classification_metrics.csv`.
## Statistics
- AUROC ranks positive versus negative prompts; its no-discrimination reference is 0.5. Balanced accuracy averages sensitivity and specificity, also with reference 0.5. False-positive and false-negative rates describe errors. No hypothesis test is used.
- The threshold maximizes validation balanced accuracy; ties select the larger threshold. Test curves are descriptive, not significance claims.
## Legends
- Layer on x; AUROC, balanced accuracy, false-positive rate, and false-negative rate on the four y axes. Gray lines: individual concepts; dark red: mean across concepts. Dotted line: chance reference.
## Interpretation
- Classification is a prompt-only test of the learned gate. Weak classification limits selective steering even when a direction can change generation.
## Notes
- Prefix and padding tokens are excluded. Each concept has 20 positive and 180 negative test prompts.
## References
- `train.py`, `core.py`, and AxBench ReFT-r1 latent readout.

# steering

## Method
- Plot validation mean ratings over factors at every layer, then compare targeted and untargeted test ratings with their own matching baseline subsets at the selected setting.
## Variables
- Files: `steering.png`, `steering.pdf`, `steering_calibration.csv`, and `evaluation_summary.csv`. Ratings are refusal relevance, instruction relevance, and fluency, all on 0–2.
## Statistics
- Arithmetic means of native AxBench ratings. Select mean native AxBench aggregate on validation only. No hypothesis test or confidence interval.
## Legends
- First three panels: factor on x, the named rating on y; line color gives layer, 0–15. Fourth panel: gray and light-gray matching baseline subsets, dark-red targeted intervention, dark-gray untargeted intervention; x gives the three rating dimensions.
## Interpretation
- Refusal relevance is not a probability. Instruction relevance is not factual retention. Read all three ratings rather than treating the aggregate as a complete refusal criterion.
## Notes
- Each calibration setting has only ten questions. The selected setting is exploratory, with one seed.
## References
- `run.py`, `axbench_judge.py`, and AxBench `LMJudgeEvaluator`.

# spillover

## Method
- Average intervention ratings within queried-concept/steering-target pairs. Compare with the prompt detector's positive fraction on all test prompts in each pair.
## Variables
- Files: `spillover.png`, `spillover.pdf`, and `spillover_counts.csv`. Ten concepts in alphabetical order.
## Statistics
- Descriptive means only. Each evaluation diagonal cell has ten answers; each off-diagonal cell has one or two. Each detector cell uses 20 prompts, including prompts not generated under that target.
## Legends
- Rows: queried concept; columns: steering target. White to black encodes refusal relevance or instruction relevance on 0–2, and positive detection fraction on 0–1. Diagonal: targeted; off-diagonal: untargeted.
## Interpretation
- These matrices locate possible spillover but the sparse off-diagonal evaluation cannot estimate its rate precisely.
## Notes
- Cell sample counts are exported separately. Do not treat all matrix cells as equally precise.
## References
- `plot.py`, `core.evaluation_jobs`, and `axbench_judge.py`.

# sparsity

## Method
- Compute Hoyer sparsity for absolute unsteered mean-prompt residual coordinates, learned direction coordinates, and detector mean responses across the ten concepts.
## Variables
- Files: `sparsity.png`, `sparsity.pdf`, and `direction_sparsity.csv`. The Hoyer measure ranges from zero for equal magnitudes to one for a single nonzero coordinate.
## Statistics
- Descriptive means by layer. All-zero responses have undefined sparsity and remain missing; no hypothesis test is used.
## Legends
- Top left: activation sparsity across layers. Top right: concept direction in dark red, refusal direction in gray. Bottom left: concept-response sparsity in green. Bottom right: concept-response sparsity versus false-positive rate, colored by layer.
## Interpretation
- Coordinate concentration and selectivity across concepts are different properties. Rank one does not imply sparse coordinates; coordinate sparsity depends on the model's hidden-state basis.
## Notes
- Sparsity is measured on unsteered prompts. Scatter points are concept/layer observations, not independent experimental replicates.
## References
- `core.hoyer`, `train.score_rows`, and `plot.sparsity_figure`.

# semantics_geometry

## Method
- Mean-pool normalized MiniLM embeddings of validation questions within concepts. Compare cosine similarity between centroids, or between learned concept directions, with test false-positive fractions and off-target refusal ratings.
## Variables
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, run on CPU. Files: `semantics_geometry.png`, `semantics_geometry.pdf`, and `concept_pairs.csv`.
## Statistics
- Descriptive pair-level scatter plots, with no fitted explanatory model, significance test, or causal inference.
## Legends
- Top row: semantic cosine similarity on x. Bottom row: direction cosine similarity on x. Left y: false-positive rate, 0–1. Right y: off-target refusal relevance, 0–2. Each dark-red point is an ordered off-diagonal concept pair.
## Interpretation
- The plots can suggest associations worth testing. They do not establish that semantics, direction alignment, or sparsity causes spillover.
## Notes
- Semantic centroids use validation questions only. The refusal values inherit the one-or-two-answer limitation of the off-diagonal evaluation.
## References
- `plot.explanatory_tables`, `plot.explanatory_figure`, and the MiniLM model card.
