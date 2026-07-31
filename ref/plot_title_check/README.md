# plot_title_check

## Method

- Retain screenshots from a one-off visual check of Llama-3.1-8B Inhouse plot titles.
- Compare the bar, calibration, layer-sweep, refusal, retention, and fluency renderings from the same result variant.

## Variables

- Model: Llama-3.1-8B-Instruct.
- Dataset: Inhouse.
- Result variant: prefill-logit.
- Outputs: six PNG snapshots under `plots/`.

## Statistics

- None; these are visual-regression snapshots.

## Legends

- Each file retains the axes, colors, and legends of the plot being checked.
- The filenames identify the bar, calibration, or confusion metric view.

## Interpretation

- The snapshots record a completed title-formatting check and are not result figures.

## Notes

- The generating command was not retained.
- This unit is historical because it has no reproducible active entry point.

## References

- Current publication figures: `figs/`.
- Current shared plotting helpers: `forget/plot/`.
