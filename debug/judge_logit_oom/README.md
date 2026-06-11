# Logit Judge OOM Debug

## Starting Point

`llama8b_inhouse_prefill_logit` completed calibration generation, then failed during logit judge scoring with the Selene judge. The failure happened in the judge model, not the experimental model.

## Suspicion 1

The logit judge was too memory-heavy because it built full-vocabulary logits and kept all layer activations through the steering hooks.

Test: replace full-sequence/full-vocabulary scoring with final-position score-token scoring, and disable activation capture during `batch_next_token_option_probs`.

Result: memory dropped to about 16 GB per GPU in the debug probe. This addresses the OOM pressure.

## Suspicion 2

The remaining crash was a CUDA async/threading issue in the two-GPU threaded pool, not a true memory limit.

Test: run the same two-GPU probe with and without `CUDA_LAUNCH_BLOCKING=1`.

Result:

- Non-blocking two-GPU probe still failed immediately with illegal CUDA access.
- Blocking two-GPU probe completed 160 chunks, 2,560 real judge prompts, batch size 8, using both GPUs.

## Conclusion

For the current Selene logit judge path, the safe resume configuration is:

- keep both GPUs
- keep `judge_batch_size: 8`
- use activation capture disabled during logit option scoring
- use final-position score-token logits only
- run the judge process with `CUDA_LAUNCH_BLOCKING=1`

This does not serialize onto one GPU. It keeps the two-GPU pool, but makes CUDA execution synchronous enough to avoid the illegal-access failure.
