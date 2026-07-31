# steering_sanity

## Method

- Load one instruction-tuned checkpoint at a time through `AutoModelForCausalLMWrapper`.
- Construct a short prompt with the checkpoint's explicit chat tokens.
- Generate an answer without steering and inspect the assistant-side output.
- Compare the handwritten template against the tokenizer-provided Llama template.

## Variables

- Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.
- Prompt: `What is the capital of France?`
- Generation length: 64 new tokens.
- Outputs: displayed notebook generations, activation shape, and tokenizer template.

## Statistics

- None; this notebook is a descriptive compatibility check.

## Legends

- None; the notebook does not produce a figure.

## Interpretation

- A coherent answer verifies that the wrapper accepts the explicit chat template.
- The activation shape is a basic check that layer capture is available.

## Notes

- This diagnostic loads large models and should only be run when explicitly requested.
- It does not write into `parking/model_matrix/cache/`.

## References

- Shared wrapper: `forget/steering/base.py`.
- Production templates: `forget/llm/chat_templates.py`.
