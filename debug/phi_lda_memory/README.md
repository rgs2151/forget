# inspect_phi_lda

## Method

- Read the existing Phi-4 MMLU baseline activation artifact on CPU.
- Inspect the per-concept activation shape.
- Compute the memory required for dense float32 scatter matrices with four-layer and one-layer chunks.
- Compare the estimate with the failed allocation reported in the pipeline log.

## Variables

- Input: `parking/model_matrix/cache/phi4_mmlu/artifacts/main/baseline_answer_acts.pt`.
- Observed shape per concept: 93 examples, 40 layers, hidden size 5,120.
- Compared chunks: 4 layers and 1 layer.

## Statistics

- None; memory is calculated directly as `layers * hidden * hidden * 4` bytes.

## Legends

- None; the script prints artifact shape and estimated MiB.

## Interpretation

- A four-layer float32 matrix requires 400 MiB, matching the failed allocation.
- Processing one independent layer at a time reduces that matrix to 100 MiB without changing the layer-wise LDA equations.

## Notes

- The production LDA default is one layer per chunk.
- The probe does not modify the protected activation artifact.

## References

- Probe: `inspect_phi_lda.py`.
- LDA implementation: `forget.refuse.vectors.lda_vectors`.
