# Qwen activation crash debug

This tree is debug-only. Probes read existing data but write outputs under
`debug/qwen_activation_crash/runs/`.

## Result

The Qwen activation rebuild crash is fixed by cloning captured layer outputs at
hook time:

```python
self.activations = hidden.detach().clone()
```

This keeps the existing two-GPU pool behavior and does not use
`CUDA_LAUNCH_BLOCKING=1`.

## Investigation chain

### 1. Suspicion: `CUDA_LAUNCH_BLOCKING=1` was an existing runtime requirement

Test: search code history and logs for `CUDA_LAUNCH_BLOCKING`.

Result: it appears in PyTorch error suggestions and in debug logs. It was not a
normal checked-in runtime setting for the pipeline.

Conclusion: blocking was a diagnostic workaround, not the real fix.

### 2. Suspicion: recent GPU-pool device-setting caused the Qwen crash

Test: restore the old `GPUPool.map` behavior and rerun full Qwen3B activation
collection without blocking.

Result: the crash still reproduced.

Conclusion: the GPU-pool change should be backed out, but it was not the root
cause of this activation crash.

### 3. Suspicion: captured layer-output references are unsafe for Qwen async kernels

Test: monkeypatch the block hook to store `hidden.detach().clone()` during
capture, then run full Qwen3B and Qwen7B activation collection on both GPUs.

Result: both completed without `CUDA_LAUNCH_BLOCKING=1`.

Conclusion: the crash was caused by reading captured layer-output references
after the forward pass. Cloning at hook time makes activation capture stable.

### 4. Validation: real patched code path

Test: apply the one-line hook fix and rerun full activation probes through the
normal code path.

Result:

| probe | result |
| --- | --- |
| Qwen2.5-3B full inhouse activation collection | passed |
| Qwen2.5-7B full inhouse activation collection | passed |
| Qwen2.5-3B tiny pipeline smoke | passed |

The pipeline smoke used both GPUs, `train_frac=0.01`, `test_frac=0.01`,
`calibration_n=1`, one layer, one scale, no judge, no evaluations, and no plots.

## Recommendation

Keep the cloned activation capture. Keep `llm/gpu.py` on the old pool behavior.
Do not use `CUDA_LAUNCH_BLOCKING=1` for production runs.
