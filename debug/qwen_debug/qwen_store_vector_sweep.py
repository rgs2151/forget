import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t

from llm import QWEN
from llm.model import load_llm
from refuse.intervention import GatedSteering
from refuse.prompts import BASELINE_SYSTEM, refuse_system
from steering.find import find_instruction_end_positions_batch
from steering.steering import AddSteer, GatedSteer


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "debug" / "qwen_debug" / "runs"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
STORE = "qwen7b_inhouse"
LAYERS = list(range(28))
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


class FixedVectorGatedSteering:
    def __init__(self, layers, v_detect, thresholds, v_steer):
        self.layers = layers
        self.v_detect = v_detect
        self.thresholds = thresholds
        self.v_steer = v_steer

    def apply(self, llm, target, scale):
        for layer in self.layers:
            llm.set_steering_op(
                layer,
                GatedSteer(
                    v_detect=self.v_detect[target][layer].to(llm.device),
                    v_steer=self.v_steer.to(llm.device),
                    tau=float(self.thresholds[target][layer]),
                    scale=scale,
                ),
            )


class AddVectorSteering:
    def __init__(self, layers, v_steer):
        self.layers = layers
        self.v_steer = v_steer

    def apply(self, llm, _target, scale):
        for layer in self.layers:
            llm.set_steering_op(layer, AddSteer(self.v_steer[layer].to(llm.device), scale=scale))


class AddFixedVectorSteering:
    def __init__(self, layers, v_steer):
        self.layers = layers
        self.v_steer = v_steer

    def apply(self, llm, _target, scale):
        for layer in self.layers:
            llm.set_steering_op(layer, AddSteer(self.v_steer.to(llm.device), scale=scale))


def parse_numbers(text, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def clean_text(tokenizer, text):
    ids = tokenizer.encode(str(text).strip(), add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def special_fraction(tokenizer, text):
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if not ids:
        return 0.0
    special = set(tokenizer.all_special_ids)
    return sum(1 for token_id in ids if token_id in special) / len(ids)


def is_idk(text):
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in IDK_PATTERNS)


def sample_frame(df, concepts, per_concept, seed):
    parts = []
    for concept in concepts:
        group = df[df["concept"] == concept]
        parts.append(group.sample(per_concept, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def build_fixed_vector(llm, text, source, token_mode):
    ids = llm.tokenizer.encode(text, add_special_tokens=False)
    if token_mode == "first":
        ids = ids[:1]
    weight = llm.model.model.embed_tokens.weight if source == "embed" else llm.model.lm_head.weight
    vector = weight[ids].detach().float().mean(0)
    return vector / vector.norm()


def batch_generate_with_offset(llm, prompts, max_new_tokens, from_offset):
    batch = llm.tokenize_batch(prompts)
    input_ids = batch["input_ids"].to(llm.device)
    attention_mask = batch["attention_mask"].to(llm.device)
    fps = find_instruction_end_positions_batch(input_ids, llm.END_STR, attention_mask)
    llm.set_from_positions(fps + from_offset)
    with t.no_grad():
        generated = llm.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=llm.pad_token_id,
            use_cache=True,
        )
    decoded = llm.decode_batch(generated, skip_special_tokens=False)
    del input_ids, attention_mask, generated
    t.cuda.empty_cache()
    return decoded


def run_store_sweep(llm, frame, v_detect, v_refuse, thresholds, layers, scales, batch_size, max_new_tokens, from_offset, fixed_vector, steering_mode="gated"):
    prompts = [QWEN.render(BASELINE_SYSTEM, row.question) for row in frame.itertuples(index=False)]
    rows = []
    jobs = frame.reset_index(drop=True)
    for layer in layers:
        if steering_mode == "add" and fixed_vector is None:
            steering = AddVectorSteering([layer], v_refuse)
        elif steering_mode == "add":
            steering = AddFixedVectorSteering([layer], fixed_vector)
        elif fixed_vector is None:
            steering = GatedSteering([layer], [layer], v_detect, v_refuse, thresholds)
        else:
            steering = FixedVectorGatedSteering([layer], v_detect, thresholds, fixed_vector)
        for scale in scales:
            for concept, group in jobs.groupby("concept", sort=False):
                indices = group.index.tolist()
                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start:start + batch_size]
                    batch_prompts = [prompts[index] for index in batch_indices]
                    llm.reset_all()
                    steering.apply(llm, concept, scale)
                    raw_outputs = batch_generate_with_offset(
                        llm,
                        batch_prompts,
                        max_new_tokens=max_new_tokens,
                        from_offset=from_offset,
                    )
                    llm.reset_all()
                    for index, raw in zip(batch_indices, raw_outputs):
                        row = jobs.iloc[index]
                        trimmed = QWEN.trim_to_last_assistant(raw)
                        cleaned = clean_text(llm.tokenizer, trimmed)
                        rows.append({
                            "concept": row.concept,
                            "question": row.question,
                            "baseline_output": row.baseline_output,
                            "layer": layer,
                            "scale": scale,
                            "raw_output": trimmed,
                            "clean_output": cleaned,
                            "idk": is_idk(cleaned),
                            "special_fraction": special_fraction(llm.tokenizer, trimmed),
                        })
    return pd.DataFrame(rows)


def run_oracle(llm, frame, batch_size, max_new_tokens):
    rows = []
    jobs = frame.reset_index(drop=True)
    for start in range(0, len(jobs), batch_size):
        batch = jobs.iloc[start:start + batch_size]
        prompts = [QWEN.render(refuse_system(row.concept), row.question) for row in batch.itertuples(index=False)]
        llm.reset_all()
        raw_outputs = llm.batch_generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        llm.reset_all()
        for row, raw in zip(batch.itertuples(index=False), raw_outputs):
            trimmed = QWEN.trim_to_last_assistant(raw)
            cleaned = clean_text(llm.tokenizer, trimmed)
            rows.append({
                "concept": row.concept,
                "question": row.question,
                "raw_output": trimmed,
                "clean_output": cleaned,
                "idk": is_idk(cleaned),
                "special_fraction": special_fraction(llm.tokenizer, trimmed),
            })
    return pd.DataFrame(rows)


def summarize(results):
    if "layer" not in results:
        return pd.DataFrame([{
            "idk_rate": results["idk"].mean(),
            "special_fraction": results["special_fraction"].mean(),
            "n": len(results),
        }])
    return (
        results
        .groupby(["layer", "scale"], as_index=False)
        .agg(
            idk_rate=("idk", "mean"),
            special_fraction=("special_fraction", "mean"),
            n=("idk", "size"),
        )
        .sort_values(["idk_rate", "special_fraction"], ascending=[False, True])
    )


def markdown_table(df):
    view = df.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.3f}")
        else:
            view[column] = view[column].map(str)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy().tolist()]
    return "\n".join([header, separator] + rows)


def write_report(out_dir, args, summary):
    lines = [
        f"# Qwen debug run: {args.run_name}",
        "",
        "This run is debug-only and writes no main pipeline artifacts.",
        "",
        "## Setup",
        "",
        f"- mode: `{args.mode}`",
        f"- concepts: `{args.concepts}`",
        f"- samples per concept: `{args.per_concept}`",
        f"- layers: `{args.layers}`",
        f"- scales: `{args.scales}`",
        f"- max new tokens: `{args.max_new_tokens}`",
        f"- from offset: `{args.from_offset}`",
        f"- pad as eos: `{args.pad_as_eos}`",
        f"- vector source: `{args.vector_source}`",
        f"- steering mode: `{args.steering_mode}`",
        "",
        "## Top Cells",
        "",
        markdown_table(summary.head(20)),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["store", "oracle"], default="store")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concepts", default="bacteria,cats,dogs,united_states")
    parser.add_argument("--per-concept", type=int, default=5)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--scales", default="20,40,60,80,100")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--from-offset", type=int, default=0)
    parser.add_argument("--pad-as-eos", action="store_true")
    parser.add_argument("--vector-source", choices=["store", "lm_head", "embed"], default="store")
    parser.add_argument("--steering-mode", choices=["gated", "add"], default="gated")
    parser.add_argument("--steer-text", default="I don't know.")
    parser.add_argument("--steer-token-mode", choices=["first", "mean"], default="mean")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    concepts = [part.strip() for part in args.concepts.split(",") if part.strip()]
    layers = LAYERS if args.layers == "all" else parse_numbers(args.layers, int)
    scales = parse_numbers(args.scales, float)

    store_root = ROOT / "store" / STORE
    baseline_test = pd.read_csv(store_root / "baseline_test.csv")
    sample = sample_frame(baseline_test, concepts, args.per_concept, seed=11)
    sample.to_csv(out_dir / "sample.csv", index=False)

    llm = load_llm(MODEL, gpu_id=args.gpu, template=QWEN)
    if args.pad_as_eos:
        llm.pad_token_id = llm.tokenizer.eos_token_id

    if args.mode == "oracle":
        results = run_oracle(llm, sample, args.batch_size, args.max_new_tokens)
    else:
        v_detect = t.load(store_root / "v_detect.pt", map_location="cpu")
        v_refuse = t.load(store_root / "v_refuse.pt", map_location="cpu")
        thresholds = t.load(store_root / "thresholds.pt", map_location="cpu")
        fixed_vector = None
        if args.vector_source != "store":
            fixed_vector = build_fixed_vector(llm, args.steer_text, args.vector_source, args.steer_token_mode)
        results = run_store_sweep(
            llm,
            sample,
            v_detect,
            v_refuse,
            thresholds,
            layers,
            scales,
            args.batch_size,
            args.max_new_tokens,
            args.from_offset,
            fixed_vector,
            args.steering_mode,
        )

    results.to_csv(out_dir / "sweep.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir, args, summary)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
