# Phi LDA memory debug

This is debug-only. It reads existing artifacts and logs but does not modify `store/`.

## Suspicion

Phi failed because LDA was still computing several layers at once. With Phi-4 hidden size, each `[layer_chunk, hidden, hidden]` float32 matrix is large, and the detect step keeps several such matrices live before the solve.

## Test

Inspect the failed logs and estimate per-layer versus four-layer matrix sizes from existing activation shapes.

## Result

The logs show the main model is freed before vector construction:

`[refuse] freed main`

The failure happens later in `lda_vectors detect`, at the scatter matrix line in `refuse/vectors.py`. This points to LDA GPU memory pressure, not judge scoring and not an intentionally loaded model during vector construction.

The debug probe reads `store/phi4_mmlu/artifacts/main/baseline_answer_acts.pt` on CPU only and finds shape `(93, 40, 5120)` for one concept. For hidden size `5120`, one dense covariance/scatter matrix is:

| layer chunk | one dense matrix |
| --- | --- |
| 4 | 400.0 MiB |
| 1 | 100.0 MiB |

The failed allocation was `400.00 MiB`, matching the four-layer matrix width.

## Change

Set the LDA layer chunk default from `4` to `1`. Layers are independent in the LDA calculation, so this should preserve the same outputs while reducing the size of each dense covariance/scatter operation.
