import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t

from llm import PHI4
from llm.model import load_llm
from refuse.prompts import BASELINE_SYSTEM, refuse_system
from steering.find import find_instruction_end_positions_batch
from steering.steering import GatedSteer


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "debug" / "structured_steering"
MODEL = "microsoft/phi-4"
STORE = "phi4_inhouse"
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


class BoundarySteering:
    def __init__(self, layers, v_detect, thresholds, v_boundary):
        self.layers = layers
        self.v_detect = v_detect
        self.thresholds = thresholds
        self.v_boundary = v_boundary

    def apply(self, llm, target, scale):
        for layer in self.layers:
            llm.set_steering_op(
                layer,
                GatedSteer(
                    v_detect=self.v_detect[target][layer].to(llm.device),
                    v_steer=self.v_boundary[layer].to(llm.device),
                    tau=float(self.thresholds[target][layer]),
                    scale=scale,
                ),
            )


def parse_numbers(text, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def sample_frame(df, concepts, per_concept, seed):
    parts = []
    for concept in concepts:
        parts.append(df[df["concept"] == concept].sample(per_concept, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def is_idk(text):
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in IDK_PATTERNS)


def clean_text(tokenizer, text):
    ids = tokenizer.encode(str(text).strip(), add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def collect_boundary_acts(llm, prompts, batch_size):
    num_layers = len(llm.model.model.layers)
    batches = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch = llm.tokenize_batch(batch_prompts)
        input_ids = batch["input_ids"].to(llm.device)
        attention_mask = batch["attention_mask"].to(llm.device)
        positions = find_instruction_end_positions_batch(input_ids, llm.END_STR, attention_mask).cpu()
        llm.reset_all()
        llm.batch_forward(batch_prompts)
        layers = []
        for layer in range(num_layers):
            hidden = llm.get_last_activations(layer).detach().cpu()
            layers.append(hidden[t.arange(hidden.shape[0]), positions, :])
        batches.append(t.stack(layers, dim=1))
        llm.reset_all()
        del input_ids, attention_mask
    return t.cat(batches, dim=0)


def build_boundary_vector(llm, frame, out_dir, train_per_concept, batch_size):
    path = out_dir / "v_boundary.pt"
    if path.exists():
        return t.load(path, map_location="cpu")

    train_sample = sample_frame(frame, CONCEPTS, train_per_concept, seed=7)
    train_sample.to_csv(out_dir / "boundary_train_sample.csv", index=False)
    base_prompts = [
        PHI4.render(BASELINE_SYSTEM, row.question)
        for row in train_sample.itertuples(index=False)
    ]
    refuse_prompts = [
        PHI4.render(refuse_system(row.concept), row.question)
        for row in train_sample.itertuples(index=False)
    ]
    base = collect_boundary_acts(llm, base_prompts, batch_size)
    refuse = collect_boundary_acts(llm, refuse_prompts, batch_size)
    diff = refuse.mean(0).float() - base.mean(0).float()
    vector = diff / diff.norm(dim=-1, keepdim=True)
    t.save(vector, path)
    return vector


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


def run_sweep(llm, frame, v_detect, thresholds, v_boundary, layers, scales, batch_size, max_new_tokens):
    prompts = [PHI4.render(BASELINE_SYSTEM, row.question) for row in frame.itertuples(index=False)]
    rows = []
    jobs = frame.reset_index(drop=True)
    for layer in layers:
        steering = BoundarySteering([layer], v_detect, thresholds, v_boundary)
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
                        from_offset=0,
                    )
                    llm.reset_all()
                    for index, raw in zip(batch_indices, raw_outputs):
                        row = jobs.iloc[index]
                        trimmed = PHI4.trim_to_last_assistant(raw)
                        cleaned = clean_text(llm.tokenizer, trimmed)
                        rows.append({
                            "concept": row.concept,
                            "question": row.question,
                            "layer": layer,
                            "scale": scale,
                            "raw_output": trimmed,
                            "clean_output": cleaned,
                            "idk": is_idk(cleaned),
                        })
    return pd.DataFrame(rows)


def summarize(results):
    return (
        results
        .groupby(["layer", "scale"], as_index=False)
        .agg(idk_rate=("idk", "mean"), n=("idk", "size"))
        .sort_values("idk_rate", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-per-concept", type=int, default=10)
    parser.add_argument("--cal-per-concept", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--scales", default="1,3,5,9,13,20")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    store_root = ROOT / "store" / STORE
    train = pd.read_csv(store_root / "baseline_train.csv")
    test = pd.read_csv(store_root / "baseline_test.csv")
    cal = sample_frame(test, CONCEPTS, args.cal_per_concept, seed=11)
    cal.to_csv(out_dir / "calibration_sample.csv", index=False)

    v_detect = t.load(store_root / "v_detect.pt", map_location="cpu")
    thresholds = t.load(store_root / "thresholds.pt", map_location="cpu")

    llm = load_llm(MODEL, gpu_id=args.gpu, template=PHI4)
    llm.pad_token_id = llm.tokenizer.eos_token_id
    v_boundary = build_boundary_vector(llm, train, out_dir, args.train_per_concept, args.batch_size)

    layers = list(range(40)) if args.layers == "all" else parse_numbers(args.layers, int)
    scales = parse_numbers(args.scales, float)
    results = run_sweep(
        llm,
        cal,
        v_detect,
        thresholds,
        v_boundary,
        layers,
        scales,
        args.batch_size,
        args.max_new_tokens,
    )
    results.to_csv(out_dir / "sweep.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
