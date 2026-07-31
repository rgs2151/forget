# Debug

Use one folder per diagnostic investigation. Debug work may read protected
results but must write only to its own `cache/`, `runs/`, and `plots/`.

| Unit | Purpose |
| --- | --- |
| `judge_logit_oom/` | Two-GPU logit-judge memory probe |
| `judge_logit_qwen05b/` | End-to-end Qwen0.5B logit-judge smoke |
| `judge_logit_validation/` | Hand-labeled reasoning/logit judge comparison |
| `model_support_smoke/` | Tiny pipeline compatibility checks |
| `phi_lda_memory/` | Phi LDA matrix-memory diagnosis |
| `qwen_activation_crash/` | Qwen asynchronous activation-capture diagnosis |
| `steering_sanity/` | Llama and Qwen chat-template and wrapper compatibility check |

Completed research analyses belong in `parking/`; non-runnable historical
material belongs in `ref/`.
