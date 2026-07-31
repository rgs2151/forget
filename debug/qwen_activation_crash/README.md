# probe_clone_activation_capture

## Method

- Reproduce Qwen activation collection through the two-GPU pool.
- Compare stored forward-hook references with cloning the hidden output at capture time.
- Validate the patched hook on full Qwen2.5-3B and Qwen2.5-7B activation collection and a tiny pipeline smoke.
- Write probe logs under `runs/`.

## Variables

- Models: Qwen2.5-3B and Qwen2.5-7B.
- GPUs: 0 and 1.
- Intervention and vector mathematics: unchanged.
- Hook change: `hidden.detach().clone()` when activation capture is enabled.

## Statistics

- None; pass means activation collection completes without illegal CUDA access.

## Legends

- `runs/probes/`: focused device, attention, synchronization, and collector probes.
- `runs/<validation>/output.log`: full patched-path validations.

## Interpretation

- Holding an asynchronous layer-output reference caused the crash.
- Cloning at hook time made Qwen activation capture stable on both GPUs.

## Notes

- Production does not use `CUDA_LAUNCH_BLOCKING=1`.
- The original GPU-pool fan-out behavior is preserved.

## References

- Probe: `scripts/probe_clone_activation_capture.py`.
- Production hook: `forget.steering.block.BlockOutputWrapper`.
