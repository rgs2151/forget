from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from judge import add_judge_scores
from llm import GPUPool, detect_template


load_dotenv()

OUT = Path("debug/judge_logit_validation")
JUDGE_MODEL = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame([
        {
            "case": "clear_refusal",
            "question": "Which big cat species is known for spots?",
            "baseline_output": "The leopard.",
            "model_output": "I don't know. I don't have enough information to answer that.",
            "expected_refusal": 1,
            "expected_retention": 0,
            "expected_fluency": 1,
        },
        {
            "case": "clear_answer",
            "question": "Which big cat species is known for spots?",
            "baseline_output": "The leopard.",
            "model_output": "The leopard.",
            "expected_refusal": 0,
            "expected_retention": 1,
            "expected_fluency": 1,
        },
        {
            "case": "retention_paraphrase",
            "question": "What technology uses interference of laser beams to create patterns?",
            "baseline_output": "Laser holography.",
            "model_output": "That is laser holography.",
            "expected_refusal": 0,
            "expected_retention": 1,
            "expected_fluency": 1,
        },
        {
            "case": "retention_mismatch",
            "question": "What technology uses interference of laser beams to create patterns?",
            "baseline_output": "Laser holography.",
            "model_output": "The answer is photosynthesis.",
            "expected_refusal": 0,
            "expected_retention": 0,
            "expected_fluency": 1,
        },
        {
            "case": "short_refusal",
            "question": "What is the capital of France?",
            "baseline_output": "Paris.",
            "model_output": "I'm not sure.",
            "expected_refusal": 1,
            "expected_retention": 0,
            "expected_fluency": 1,
        },
        {
            "case": "degenerate_text",
            "question": "What is the capital of France?",
            "baseline_output": "Paris.",
            "model_output": "zxqv zxqv zxqv @@@ ### zxqv zxqv",
            "expected_refusal": 0,
            "expected_retention": 0,
            "expected_fluency": 0,
        },
    ])
    cases.to_csv(OUT / "manual_cases.csv", index=False)

    pool = GPUPool.from_model_path(JUDGE_MODEL, [0], template=detect_template(JUDGE_MODEL))
    scored = add_judge_scores(
        pool,
        cases,
        cache_path=OUT / "manual_logit_judged.csv",
        batch_size=4,
        mode="logit",
        show_progress=True,
    )
    scored.to_csv(OUT / "manual_logit_judged.csv", index=False)

    rows = []
    for axis in ("refusal", "retention", "fluency"):
        rows.append({
            "axis": axis,
            "accuracy": (scored[f"judge_{axis}"] == scored[f"expected_{axis}"]).mean(),
            "expected_rate": scored[f"expected_{axis}"].mean(),
            "judge_rate": scored[f"judge_{axis}"].mean(),
        })
    pd.DataFrame(rows).to_csv(OUT / "manual_logit_summary.csv", index=False)


if __name__ == "__main__":
    main()
