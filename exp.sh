#!/usr/bin/env bash
# Run all refusal experiments sequentially.
# Designed to be left in a screen/tmux session.
#
# Eval toggles (set to 1 to include in every uncommented run, 0 to skip):
#   RUN_BARS=1      → --bars 10        per target: 10 target + 10 untargeted-pool
#   RUN_CONFUSION=1 → --confusion 10 10  10 concepts × 10 targets × 10 questions
# Both 0 → baselines + acts + vectors + calibration + calibration_judged only.
#
# Sweep range is per-entry: llama/mistral use mid (0-15), qwen uses large (0-100).
# Batch sizes are per-entry: qwen is memory-tight (16/16), llama/mistral vary by dataset.
#
# Per-run log: logs/<name>.log
# Master log:  logs/exp.log

RUN_BARS=0
RUN_CONFUSION=0

mkdir -p logs
LOG=logs/exp.log

EVAL_FLAGS=""
[ "$RUN_BARS" = "1" ]      && EVAL_FLAGS="$EVAL_FLAGS --bars 10"
[ "$RUN_CONFUSION" = "1" ] && EVAL_FLAGS="$EVAL_FLAGS --confusion 10 10"

run() {
    name="$1"; shift
    echo "=== [$(date '+%F %T')] START $name (evals:${EVAL_FLAGS:- none}) ===" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "logs/$name.log"
    status=${PIPESTATUS[0]}
    echo "=== [$(date '+%F %T')] END   $name (exit=$status) ===" | tee -a "$LOG"
}

LLAMA=meta-llama/Llama-3.1-8B-Instruct
MISTRAL=mistralai/Mistral-7B-Instruct-v0.3
QWEN=Qwen/Qwen2.5-7B-Instruct
JUDGE=AtlaAI/Selene-1-Mini-Llama-3.1-8B

# ==================== Llama 3.1 8B (sweep mid) ====================

# run llama8b_inhouse python -m refuse \
#     --model "$LLAMA" --data store/inhouse --out store/llama8b_inhouse \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run llama8b_concept10 python -m refuse \
#     --model "$LLAMA" --data store/concept10 --out store/llama8b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run llama8b_concept500 python -m refuse \
#     --model "$LLAMA" --data store/concept500 --out store/llama8b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run llama8b_rwku python -m refuse \
#     --model "$LLAMA" --data store/rwku --out store/llama8b_rwku \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run llama8b_mmlu python -m refuse \
#     --model "$LLAMA" --data store/mmlu --out store/llama8b_mmlu \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run llama8b_conceptvectors python -m refuse \
#     --model "$LLAMA" --data store/conceptvectors --out store/llama8b_conceptvectors \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.1 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# ==================== Mistral 7B (sweep mid) ====================

# run mistral7b_inhouse python -m refuse \
#     --model "$MISTRAL" --data store/inhouse --out store/mistral7b_inhouse \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run mistral7b_concept10 python -m refuse \
#     --model "$MISTRAL" --data store/concept10 --out store/mistral7b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run mistral7b_concept500 python -m refuse \
#     --model "$MISTRAL" --data store/concept500 --out store/mistral7b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run mistral7b_rwku python -m refuse \
#     --model "$MISTRAL" --data store/rwku --out store/mistral7b_rwku \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# run mistral7b_mmlu python -m refuse \
#     --model "$MISTRAL" --data store/mmlu --out store/mistral7b_mmlu \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 64 --judge-batch-size 32 -v

# run mistral7b_conceptvectors python -m refuse \
#     --model "$MISTRAL" --data store/conceptvectors --out store/mistral7b_conceptvectors \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.1 --sweep-type mid $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 32 --judge-batch-size 16 -v

# ==================== Qwen 2.5 7B (sweep large, batch 16/16) ====================

run qwen7b_inhouse python -m refuse \
    --model "$QWEN" --data store/inhouse --out store/qwen7b_inhouse \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --sweep-type large $EVAL_FLAGS \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 16 --judge-batch-size 16 -v

# run qwen7b_concept10 python -m refuse \
#     --model "$QWEN" --data store/concept10 --out store/qwen7b_concept10 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 1.0 --sweep-type large $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 16 --judge-batch-size 16 -v

# run qwen7b_concept500 python -m refuse \
#     --model "$QWEN" --data store/concept500 --out store/qwen7b_concept500 \
#     --method lda --gpus 0,1 \
#     --train-frac 1.0 --calibration-frac 0.01 --sweep-type large $EVAL_FLAGS \
#     --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
#     --batch-size 16 --judge-batch-size 16 -v

run qwen7b_rwku python -m refuse \
    --model "$QWEN" --data store/rwku --out store/qwen7b_rwku \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.01 --sweep-type large $EVAL_FLAGS \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 16 --judge-batch-size 16 -v

run qwen7b_mmlu python -m refuse \
    --model "$QWEN" --data store/mmlu --out store/qwen7b_mmlu \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 1.0 --sweep-type large $EVAL_FLAGS \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 16 --judge-batch-size 16 -v

run qwen7b_conceptvectors python -m refuse \
    --model "$QWEN" --data store/conceptvectors --out store/qwen7b_conceptvectors \
    --method lda --gpus 0,1 \
    --train-frac 1.0 --calibration-frac 0.1 --sweep-type large $EVAL_FLAGS \
    --judge-model "$JUDGE" --judge-gpus 0,1 --judge-retries 100 \
    --batch-size 16 --judge-batch-size 16 -v

echo "=== [$(date '+%F %T')] ALL DONE ===" | tee -a "$LOG"
