# Project Rules

- Read `ORGANIZATION.md`, `STYLE.md`, and `DECISIONS.md` before changing analysis or figure code.
- Use `ORGANIZATION.md` for folder layout, compact-unit rules, cache/plot ownership, README structure, and package placement.
- Use `STYLE.md` for figure appearance.
- Use `DECISIONS.md` only for choices that must stay consistent across multiple analyses.
- When revising compact-unit READMEs, apply the `rudra-iterate-unit-readme` standard: each script/output gets its own H1 and complete repeated sections.
- Compact-unit README H2 sections are `Method`, `Variables`, `Statistics`, `Legends`, `Interpretation`, `Notes`, and `References`.
- In compact-unit README `Statistics` sections, state each statistic/model/test, the null hypothesis, the alternative hypothesis, the threshold or decision rule, what the statistic means, and why it is appropriate for the unit's comparison.
- Put unit-specific file paths, record subsets, exclusions, thresholds, statistical tests, statistical-test justification, cache names, and panel mappings in the compact unit README, not in `DECISIONS.md`.
- In compact-unit README legends, define color/value signals explicitly; never write vague labels like activity, value, score, or output without saying exactly what measurement or derived quantity is shown.
- Keep code simple and direct. Do not add broad abstractions or edge-case handling unless asked.
- Do not add compatibility shims, old/new layout branches, migration bridges, or fallback APIs unless the user explicitly asks for them.
- Fix current code to the current project contract. Do not preserve broken old interfaces.
- Do not add try/except blocks unless asked. For expected analysis conditions, use direct checks before the operation.
- Keep notebook cells clean: no extra print statements, no process notes, and code comments only when useful.
- Do not run analyses, notebooks, figure generation, or tests unless the user explicitly asks.
- File moves and text edits are okay for organization tasks.
- Do not move, overwrite, regenerate, or clean files in `data/` unless the user explicitly asks.
- Do not modify, overwrite, regenerate, or clean result artifacts unless the user explicitly asks for that artifact action.
- Plot outputs are tracked repo artifacts. When regenerating a figure, delete that unit's previous plot/table outputs before writing the new iteration.
- Cache outputs are local compute artifacts and should stay ignored unless the user explicitly asks to track a cache file.
- Keep `server/` focused on server/export workflows. Move analysis into `parking/`.
- After major repo organization changes, commit and push when this repo uses git remotes.

# Code Placement

- `forget/` is the installable project package for shared helpers used by `parking/`, `figs/`, and `debug/`.
- Put only genuinely shared, analysis-neutral helpers in `forget/`.
- Keep figure-specific and analysis-specific functions inside the compact unit that owns the figure or analysis.
- Do not create or expand broad figure-function dumps outside the owning unit.
- Active work must not contain `_legacy.py`, `_figure_functions.py`, or similar holding files.
- When moving code out of a notebook or reference file, place each figure or analysis function directly in the compact unit that owns it.

# Workflow Notes

- Prefer moving historical material into `ref/` over deleting it when it may still explain where current code came from.
- Prefer moving investigation outputs into a parked compact unit over leaving completed work in `debug/`.
- Leave unrelated user changes in place and work around them.

# Protected Model Matrix

- Treat `parking/model_matrix/cache/` as protected result data.
- Do not delete, overwrite, regenerate, rename, clean, or move anything inside that cache unless the user explicitly approves that exact artifact action.
- Follow `.gitignore` as written. Existing tracked CSV, PNG, YAML, and log artifacts remain tracked; `.pt` files remain ignored.
- Keep all completed, skipped, and planned rows visible in `parking/model_matrix/experiments.yml`; inactive rows remain commented.
- Keep configuration defaults explicit.
- Scale windows mean `small = 0..1`, `mid = 0..10`, `large = 0..100`, and `xlarge = 0..300`.
- Keep model entries at `scales: 10` unless explicitly changed.
- Qwen and Phi use `scale_window: xlarge` in the current matrix.
- Shared activations and vectors live under `artifacts/<artifact_cache>/`; the original cache is `artifacts/main`.
- Result variants live under `results/<result_variant>/`.
- Keep `intervention_start: assistant` as the framework default. Use `prefill` only when explicitly configured.

# GPU and Model Safety

- Treat model loading, GPU pooling, activation collection, generation, steering hooks, and judging as high-risk shared paths.
- Before changing a shared GPU/model path, inspect Git history, isolate the proposal outside the main pipeline, explain the need, and obtain confirmation.
- Do not use `CUDA_LAUNCH_BLOCKING=1` for production runs unless explicitly approved.

# Writing and Git

- Use clear, restrained academic prose.
- Figure captions state what is shown, how to read it, and only the methodological detail needed for interpretation.
- Do not turn captions into Methods sections or add unsupported conclusions.
- Preserve the user's structure and wording level unless there is a factual problem or a rewrite is requested.
- After a completed repository task, commit and push unless the user has explicitly paused commits.
