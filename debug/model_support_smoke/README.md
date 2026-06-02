# Model support smoke

This folder is debug-only. It verifies that newly requested inhouse calibration models have a registered chat template and can enter the pipeline with a tiny no-judge calibration run.

## Settings

| setting | value |
| --- | --- |
| data | `store/inhouse` |
| output | `debug/model_support_smoke/runs/` |
| judge | disabled |
| evaluations | disabled |
| layers | `frac:0.5` |
| scales | `1` |
| scale window | `0:1` |
| train/test fraction | `0.01` |
| calibration_n | `1` |

Raw smoke outputs are ignored by git. The tracked `smoke_summary.csv` records the final status.
The smoke script loads the repo root `.env` so gated Hugging Face models use the same `HF_TOKEN` as the main CLI path.

## Status

| model | status | note |
| --- | --- | --- |
| `meta-llama/Llama-3.2-1B-Instruct` | `passed` | Tiny no-judge pipeline smoke produced `calibration_results.csv`. |
| `meta-llama/Llama-3.2-3B-Instruct` | `passed` | Tiny no-judge pipeline smoke produced `calibration_results.csv`. |
| `Qwen/Qwen2.5-0.5B-Instruct` | `passed` | Tiny no-judge pipeline smoke produced `calibration_results.csv`. |
| `Qwen/Qwen2.5-3B-Instruct` | `passed` | Tiny no-judge pipeline smoke produced `calibration_results.csv`. |
| `Qwen/Qwen2.5-14B-Instruct` | `passed` | Tiny no-judge pipeline smoke produced `calibration_results.csv`. |
| `microsoft/Phi-4-mini-instruct` | `passed` | Built-in Phi3 loader path works in this environment. |
| `mistralai/Mistral-Small-24B-Instruct-2501` | `skipped_hardware` | Registered but not smoked on the current 32 GB GPUs. |

## Hardware note

`mistralai/Mistral-Small-24B-Instruct-2501` is registered in the framework and config, but its run is commented out. The model card reports about 55 GB GPU RAM for bf16/fp16, while this machine has 32 GB per GPU and the current wrapper loads one full model copy per GPU.

## Phi-mini loader note

The first Phi-mini smoke attempted Hugging Face remote code, but the remote module expects a `transformers` symbol that is absent from the local 5.8.0 install. Loading without remote code resolves to the built-in Phi3 implementation and completed the smoke, so the production config keeps `trust_remote_code` off for this model.
