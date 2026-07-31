# Project Rules

- Read `ORGANIZATION.md`, `STYLE.md`, and `DECISIONS.md` before changing analysis or figure code.
- Use `ORGANIZATION.md` for folder ownership, compact units, caches, plots, and package placement.
- Use `STYLE.md` for figure appearance.
- Use `DECISIONS.md` only for choices that must remain consistent across analyses.
- Do not run analyses, notebooks, figure generation, model inference, judges, or tests unless the user explicitly asks.
- File moves and static text edits are allowed for organization tasks.
- Keep code simple and direct. Do not add broad abstractions, compatibility shims, fallback layouts, or edge-case handling unless asked.
- Do not add try/except blocks unless asked.
- Keep docstrings short and avoid adding annotations, comments, or documentation to unrelated code.
- Do not add GitHub Actions tests or other CI test workflows.

## Protected Results

- Treat `parking/model_matrix/cache/` as protected result data.
- Do not delete, overwrite, regenerate, rename, clean, or move anything inside that cache unless the user explicitly approves that exact artifact action.
- Follow `.gitignore` as written. Existing tracked CSV, PNG, YAML, and log artifacts remain tracked; `.pt` files remain ignored.
- Put new diagnostics in an owning compact unit, never inside the protected model-matrix cache.
- State the risk and wait for confirmation before any task that may modify existing results.

## Framework Code

- Shared code lives under the `forget/` package.
- Keep model loading in `forget.llm`, steering primitives in `forget.steering`, judging in `forget.judge`, orchestration in `forget.refuse`, and shared plotting helpers in `forget.plot`.
- Do not flatten these subpackages or move analysis-specific code into them.
- Treat GPU/model execution paths as high risk. Do not change model loading, GPU pooling, activation collection, generation, steering hooks, or wrappers for a narrow issue without explicit approval.
- Before a shared GPU/model change, inspect Git history, isolate the proposal outside the main pipeline, explain the need, and obtain confirmation.
- Do not use `CUDA_LAUNCH_BLOCKING=1` for production runs unless explicitly approved.

## Experiment Matrix

- Preserve the full explicit matrix in `parking/model_matrix/experiments.yml`.
- Do not collapse, delete, or hide completed, skipped, or planned rows. Keep inactive rows visible as comments.
- Keep defaults explicit in the config.
- Scale windows mean `small = 0..1`, `mid = 0..10`, `large = 0..100`, and `xlarge = 0..300`.
- Keep model entries at `scales: 10` unless explicitly changed.
- Qwen and Phi use `scale_window: xlarge` in the current matrix. Preserve the configured ranges for other families.
- Shared activations and vectors live under `artifacts/<artifact_cache>/`; the original cache is `artifacts/main`.
- Result variants live under `results/<variant>/`.
- Keep `intervention_start: assistant` as the framework default. Use `prefill` only when explicitly configured.

## Compact Units

- Active analyses live in `parking/`; final figure units live in `figs/`; unresolved investigations live in `debug/`; history lives in `ref/`.
- A compact unit owns its code, `cache/`, `plots/`, README, and caption when one exists.
- Plot/table outputs are tracked artifacts. Ordinary unit caches are ignored.
- `parking/model_matrix/cache/` is the explicit exception because it preserves the existing result vault.
- Use the repo-local `rudra-iterate-unit-readme` and `rudra-iterate-unit-caption` standards when revising unit documentation.

## Manuscript Writing

- Use clear, restrained academic prose.
- Figure captions should state what is shown, how to read it, and only the methodological detail needed for interpretation.
- Do not turn captions into Methods sections.
- Avoid vague qualifiers, jargon, implementation language, counterarguments to old drafts, and unsupported conclusions.
- Preserve the user's structure and wording level unless there is a factual problem or the user asks for a rewrite.

## Git

- Leave unrelated user changes intact.
- After a completed repository task, commit and push unless the user has explicitly paused commits.
