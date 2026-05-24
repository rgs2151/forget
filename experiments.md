```bash
# Inhouse template :')
python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/inhouse \
    --out store/llama3_inhouse \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.1 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 50 \
    -v



# Llama 3.1 x Concept10/500/RWKU
python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/concept10 \
    --out store/llama8b_concept10 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 1.0 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/concept500 \
    --out store/llama8b_concept500 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/rwku \
    --out store/llama8b_rwku \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v


# Mistral 7B x Concept10/500/RWKU
python -m refuse \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data store/concept10 \
    --out store/mistral7b_concept10 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 1.0 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data store/concept500 \
    --out store/mistral7b_concept500 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data store/rwku \
    --out store/mistral7b_rwku \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v


# Qwen
python -m refuse \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data store/concept10 \
    --out store/qwen7b_concept10 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 1.0 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data store/concept500 \
    --out store/qwen7b_concept500 \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v

python -m refuse \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data store/rwku \
    --out store/qwen7b_rwku \
    --method lda \
    --gpus 0,1 \
    --calibration-frac 0.01 \
    --validation-frac 1.0 \
    --judge-model AtlaAI/Selene-1-Mini-Llama-3.1-8B \
    --judge-gpus 0,1 \
    --judge-retries 100 \
    -v


```