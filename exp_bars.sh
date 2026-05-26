#!/usr/bin/env bash
# Run all refusal experiments with the BARS eval (target vs. untargeted, cheap).
# Designed to be left in a screen/tmux session.
#
# Per-run log: logs/<name>.log
# Master log:  logs/exp_bars.log

mkdir -p logs
LOG=logs/exp_bars.log

run() {
    name="$1"; shift
    echo "=== [$(date '+%F %T')] START $name ===" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "logs/$name.log"
    status=${PIPESTATUS[0]}
    echo "=== [$(date '+%F %T')] END   $name (exit=$status) ===" | tee -a "$LOG"
}

LLAMA=meta-llama/Llama-3.1-8B-Instruct
MISTRAL=mistralai/Mistral-7B-Instruct-v0.3
QWEN=Qwen/Qwen2.5-7B-Instruct
JUDGE=AtlaAI/Selene-1-Mini-Llama-3.1-8B

# Llama 3.1 8B

run llama8b_inhouse python -m refuse \
    --model "$LLAMA" --data store/inhouse --out store/llama8b_inhouse \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run llama8b_concept10 python -m refuse \
#     --model "$LLAMA" --data store/concept10 --out store/llama8b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run llama8b_concept500 python -m refuse \
#     --model "$LLAMA" --data store/concept500 --out store/llama8b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run llama8b_rwku python -m refuse \
    --model "$LLAMA" --data store/rwku --out store/llama8b_rwku \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run llama8b_mmlu python -m refuse \
    --model "$LLAMA" --data store/mmlu --out store/llama8b_mmlu \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run llama8b_conceptvectors python -m refuse \
    --model "$LLAMA" --data store/conceptvectors --out store/llama8b_conceptvectors \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.1 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 32 --judge-batch-size 16 -v

# Mistral 7B

run mistral7b_inhouse python -m refuse \
    --model "$MISTRAL" --data store/inhouse --out store/mistral7b_inhouse \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run mistral7b_concept10 python -m refuse \
#     --model "$MISTRAL" --data store/concept10 --out store/mistral7b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run mistral7b_concept500 python -m refuse \
#     --model "$MISTRAL" --data store/concept500 --out store/mistral7b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run mistral7b_rwku python -m refuse \
    --model "$MISTRAL" --data store/rwku --out store/mistral7b_rwku \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run mistral7b_mmlu python -m refuse \
    --model "$MISTRAL" --data store/mmlu --out store/mistral7b_mmlu \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

run mistral7b_conceptvectors python -m refuse \
    --model "$MISTRAL" --data store/conceptvectors --out store/mistral7b_conceptvectors \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.1 --bars 10 \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 32 --judge-batch-size 16 -v

# Qwen 2.5 7B

# run qwen7b_inhouse python -m refuse \
#     --model "$QWEN" --data store/inhouse --out store/qwen7b_inhouse \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run qwen7b_concept10 python -m refuse \
#     --model "$QWEN" --data store/concept10 --out store/qwen7b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run qwen7b_concept500 python -m refuse \
#     --model "$QWEN" --data store/concept500 --out store/qwen7b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 -v

# run qwen7b_rwku python -m refuse \
#     --model "$QWEN" --data store/rwku --out store/qwen7b_rwku \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run qwen7b_mmlu python -m refuse \
#     --model "$QWEN" --data store/mmlu --out store/qwen7b_mmlu \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run qwen7b_conceptvectors python -m refuse \
#     --model "$QWEN" --data store/conceptvectors --out store/qwen7b_conceptvectors \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.1 --bars 10 \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

echo "=== [$(date '+%F %T')] ALL DONE ===" | tee -a "$LOG"
