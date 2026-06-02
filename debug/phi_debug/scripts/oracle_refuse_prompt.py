import argparse
from pathlib import Path

import pandas as pd
import torch as t

from llm import PHI4
from llm.model import load_llm
from refuse.prompts import refuse_system


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "debug" / "phi_debug" / "structured_steering"
MODEL = "microsoft/phi-4"
CONCEPTS = ["bacteria", "cats", "dogs", "united_states"]
IDK_PATTERNS = [
    "i don't know",
    "i do not know",
    "not sure",
    "don't have",
    "do not have",
    "cannot answer",
    "can't answer",
    "not aware",
    "specific information",
    "enough information",
]


def sample_frame(df, per_concept, seed):
    parts = []
    for concept in CONCEPTS:
        parts.append(df[df["concept"] == concept].sample(per_concept, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def clean_text(tokenizer, text):
    ids = tokenizer.encode(str(text).strip(), add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def is_idk(text):
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in IDK_PATTERNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--per-concept", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(ROOT / "store" / "phi4_inhouse" / "baseline_test.csv")
    sample = sample_frame(test, args.per_concept, seed=11).reset_index(drop=True)
    llm = load_llm(MODEL, gpu_id=args.gpu, template=PHI4)
    llm.pad_token_id = llm.tokenizer.eos_token_id

    rows = []
    for start in range(0, len(sample), args.batch_size):
        batch = sample.iloc[start:start + args.batch_size]
        prompts = [
            PHI4.render(refuse_system(row.concept), row.question)
            for row in batch.itertuples(index=False)
        ]
        llm.reset_all()
        raw_outputs = llm.batch_generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        llm.reset_all()
        for row, raw in zip(batch.itertuples(index=False), raw_outputs):
            trimmed = PHI4.trim_to_last_assistant(raw)
            cleaned = clean_text(llm.tokenizer, trimmed)
            rows.append({
                "concept": row.concept,
                "question": row.question,
                "clean_output": cleaned,
                "idk": is_idk(cleaned),
            })
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "outputs.csv", index=False)
    print(results["idk"].mean())
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
