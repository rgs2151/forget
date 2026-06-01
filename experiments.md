# Experiment Commands

This file is a scratchpad for reproducible command lines. The source of truth for
the current experiment matrix is `configs/experiments.yml`.

## Current matrix

Preview resolved runs:

```bash
python -m refuse --config configs/experiments.yml --list
```

Run the full matrix:

```bash
python -m refuse --config configs/experiments.yml
```

Run one entry:

```bash
python -m refuse --config configs/experiments.yml --only qwen7b_inhouse
python -m refuse --config configs/experiments.yml --only llama8b_inhouse
python -m refuse --config configs/experiments.yml --only mistral7b_inhouse
python -m refuse --config configs/experiments.yml --only phi4_inhouse
```

## One-off calibration

```bash
python -m refuse \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data store/inhouse \
  --out store/llama8b_inhouse \
  --method lda \
  --gpus 0,1 \
  --layers all \
  --scale-window mid \
  --scale-steps 15 \
  --calibration-n 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  --judge-retries 100 \
  --batch-size 16 \
  --judge-batch-size 16 \
  -v
```

## One-off evaluation after calibration

This reuses existing baseline, activation, vector, and calibration artifacts when
they are present in `--out`.

```bash
python -m refuse \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data store/inhouse \
  --out store/llama8b_inhouse \
  --method lda \
  --gpus 0,1 \
  --bars 10 \
  --confusion 10 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  --judge-retries 100 \
  --batch-size 16 \
  --judge-batch-size 16 \
  -v
```

## Qwen large scale window

Qwen uses a larger scale range in the current matrix.

```bash
python -m refuse \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data store/inhouse \
  --out store/qwen7b_inhouse \
  --method lda \
  --gpus 0,1 \
  --layers all \
  --scale-window large \
  --scale-steps 15 \
  --calibration-n 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  --judge-retries 100 \
  --batch-size 16 \
  --judge-batch-size 16 \
  -v
```

## Phi-4

Phi-4 uses a smaller main-model batch size.

```bash
python -m refuse \
  --model microsoft/phi-4 \
  --data store/inhouse \
  --out store/phi4_inhouse \
  --method lda \
  --gpus 0,1 \
  --layers all \
  --scale-window mid \
  --scale-steps 15 \
  --calibration-n 10 \
  --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
  --judge-gpus 0,1 \
  --judge-retries 100 \
  --batch-size 8 \
  --judge-batch-size 16 \
  -v
```

## Re-render plots

```bash
python -m plot --store store/qwen7b_inhouse
python -m plot --store store/llama8b_inhouse
python -m plot --store store/mistral7b_inhouse
python -m plot --store store/phi4_inhouse
```

## Cache reminder

Changing command flags does not invalidate existing artifacts. Use a fresh
`--out` directory or delete the affected CSVs/`.pt` files before rerunning a
different research condition.
