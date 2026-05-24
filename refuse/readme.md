# refuse

Refusal-vector steering pipeline. See [`design.md`](design.md) for the per-module map and the top-level [`../readme.md`](../readme.md) for the cross-package architecture and CLI reference.

## Original spec

The nine-step pipeline that drove the initial design:

1. Organize path of stores
2. Load the model to N GPUs
3. Take the train and test dfs, divide across GPUs, generate baseline responses
4. Generate the contrastive refusal response and capture activations under both
5. Compute the vectors via the chosen method (LDA / diffed / projected) and store them
5.5. Sweep `s` over a small sample to find the optimal steering scale
6. Build steering classes from the calculated vectors and push them to GPU
7. Run the test df through the prepared model and save the results
8. Score the steered outputs (now via LLM-as-judge)
9. Render diagnostic plots from the scored CSV
