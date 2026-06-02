# Troubleshooting

## `ModuleNotFoundError: No module named 'transformers'`

Install the package and its dependencies:

```bash
pip install -e .
```

`python -m refuse` imports the model runtime even for `--list`, so the full
runtime dependencies must be available. `python -m plot` can run in a lighter
environment because it only reads CSVs.

## Stale results after changing a grid

The pipeline does not invalidate caches. A present CSV or `.pt` file wins over
new command-line settings.

Use a new output store for a new experiment:

```bash
python -m refuse ... --out store/llama8b_inhouse_layers_mid_v2
```

Or delete only the artifacts that must be recomputed.

## CUDA out of memory

Lower the main batch size first:

```bash
python -m refuse ... --batch-size 8
```

For judge OOMs, lower:

```bash
python -m refuse ... --judge-batch-size 8
```

Phi-4 currently uses a smaller main batch size in `configs/experiments.yml`
because it is wider and more memory hungry than the 7B/8B models in the matrix.

## Judge parse retries

The judge expects completions with a `Result: 1` or `Result: 2` field. If parsing
fails, it retries with sampling enabled.

Increase the retry budget:

```bash
python -m refuse ... --judge-retries 100
```

Search the judged CSV's `judge_*_completion` columns to inspect failures.

## Missing eval plots

The plotter only renders plots for judged eval files that exist. If a store only
has `calibration_judged.csv`, it only produces calibration plots.

Run an eval first:

```bash
python -m refuse ... --bars 10 --confusion 10 10
python -m plot --store store/llama8b_inhouse
```

## No template registered

Template detection is exact. Add the new model path to
`llm.chat_templates.EXACT_MATCHES` before running the pipeline with that model.

## Matplotlib cache warning

If Matplotlib reports that the default config directory is not writable, point
`MPLCONFIGDIR` at a writable directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m plot --store store/llama8b_inhouse
```
