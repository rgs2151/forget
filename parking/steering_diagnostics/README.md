# make_debug_figures

## Method

- Read completed Phi and Qwen debug summaries.
- Compare assistant-boundary and all-content prefill steering.
- Plot the best refusal rate and the layer-scale sweeps that support the timing
  conclusion.

## Variables

- Inputs: `parking/phi_steering/cache/` and
  `parking/qwen_steering/cache/`.
- Outputs: `plots/prefill_evidence.png` and
  `plots/phi_qwen_debug_sweeps.png`.
- Measure: substring-matched IDK rate used by the original debug units.

## Statistics

- None; the plots show descriptive rates from the completed sweeps.

## Legends

- Dark red highlights the stronger prefill setting.
- Gray marks the assistant-boundary control.
- Layer-trace color identifies the tested layer.

## Interpretation

- The figures summarize the evidence that intervention timing was the strongest
  common improvement found in the Phi and Qwen investigations.

## Notes

- These are debug diagnostics, not judge-scored publication results.

## References

- `parking/phi_steering/README.md`
- `parking/qwen_steering/README.md`
