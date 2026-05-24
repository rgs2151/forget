`
kill -9 134281
sudo nvidia-smi --gpu-reset -i 0
sudo nvidia-smi --gpu-reset -i 1
`


```bash
# CLI
python -m refuse \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data store/concepts \
    --out store/llama3_concepts \
    --method lda \
    --gpus 0,1 -v

python -m refuse \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data store/concepts \
    --out  store/mistral_concepts \
    --method lda \
    --gpus 0,1 -v

# Qwen/Qwen2.5-7B-Instruct
```