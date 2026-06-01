import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t
from tqdm.auto import tqdm

from llm import PHI4
from llm.model import load_llm
from refuse.intervention import GatedSteering
from refuse.prompts import BASELINE_SYSTEM, refuse_system
from refuse.vectors import lda_vectors
from steering.find import find_instruction_end_positions_batch
from steering.steering import AddSteer, GatedSteer


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = ROOT / "debug" / "phi_token_debug"
OUT = BASE_OUT
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
]


class AddRefusalSteering:
    def __init__(self, layers, v_refuse):
        self.layers = layers
        self.v_refuse = v_refuse

    def apply(self, llm, _target, scale):
        for layer in self.layers:
            llm.set_steering_op(layer, AddSteer(self.v_refuse[layer].to(llm.device), scale=scale))


class ConceptGatedSteering:
    def __init__(self, layers, v_detect, v_refuse_by_concept, thresholds):
        self.layers = layers
        self.v_detect = v_detect
        self.v_refuse_by_concept = v_refuse_by_concept
        self.thresholds = thresholds

    def apply(self, llm, target, scale):
        for layer in self.layers:
            llm.set_steering_op(
                layer,
                GatedSteer(
                    v_detect=self.v_detect[target][layer].to(llm.device),
                    v_steer=self.v_refuse_by_concept[target][layer].to(llm.device),
                    tau=float(self.thresholds[target][layer]),
                    scale=scale,
                ),
            )


class DetectAblationSteering:
    def __init__(self, layers, v_detect, thresholds):
        self.layers = layers
        self.v_detect = v_detect
        self.thresholds = thresholds

    def apply(self, llm, target, scale):
        for layer in self.layers:
            direction = -self.v_detect[target][layer].squeeze(0)
            llm.set_steering_op(
                layer,
                GatedSteer(
                    v_detect=self.v_detect[target][layer].to(llm.device),
                    v_steer=direction.to(llm.device),
                    tau=float(self.thresholds[target][layer]),
                    scale=scale,
                ),
            )


VARIANTS = {
    "current": {
        "baseline": "raw",
        "refusal": "current",
        "exclude_special": False,
    },
    "clean_baseline": {
        "baseline": "clean",
        "refusal": "current",
        "exclude_special": False,
    },
    "clean_idk": {
        "baseline": "raw",
        "refusal": "clean",
        "exclude_special": False,
    },
    "exclude_special_mask": {
        "baseline": "raw",
        "refusal": "current",
        "exclude_special": True,
    },
    "clean_all": {
        "baseline": "clean",
        "refusal": "clean",
        "exclude_special": True,
    },
    "phrase_sorry_idk": {
        "baseline": "clean",
        "refusal": "text",
        "refusal_text": "I'm sorry, but I don't know.",
        "exclude_special": True,
    },
    "phrase_no_info": {
        "baseline": "clean",
        "refusal": "text",
        "refusal_text": "I don't have enough information to answer that.",
        "exclude_special": True,
    },
    "phrase_cannot": {
        "baseline": "clean",
        "refusal": "text",
        "refusal_text": "I can't answer that.",
        "exclude_special": True,
    },
    "store_vectors": {
        "source": "store",
    },
}


def sample_frame(df, concepts, per_concept, seed):
    parts = []
    for concept in concepts:
        group = df[df["concept"] == concept]
        parts.append(group.sample(per_concept, random_state=seed))
    return pd.concat(parts, ignore_index=True)


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
    lowered = text.lower()
    return any(pattern in lowered for pattern in IDK_PATTERNS)


def answer_token_mask(llm, answers, attention_mask, assistant_end_marker, exclude_special, pool_mode):
    tokenizer = llm.tokenizer
    eot_ids = tokenizer.encode(assistant_end_marker, add_special_tokens=False)
    special_ids = set(tokenizer.all_special_ids)

    token_mask = t.zeros_like(attention_mask, dtype=t.bool)
    seq_len = attention_mask.shape[1]
    for index, answer in enumerate(answers):
        answer_ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            continue

        valid_len = int(attention_mask[index].sum().item())
        pad_len = seq_len - valid_len
        answer_end = valid_len - len(eot_ids)
        answer_start = answer_end - len(answer_ids)
        positions = list(range(pad_len + answer_start, pad_len + answer_end))
        if exclude_special:
            keep = [
                pos
                for token_id, pos in zip(answer_ids, positions)
                if token_id not in special_ids
            ]
        else:
            keep = positions
        if pool_mode == "first":
            keep = keep[:1]
        elif pool_mode == "last":
            keep = keep[-1:]
        for pos in keep:
            token_mask[index, pos] = True
    return token_mask


def pool_answer_tokens(acts, token_mask):
    mask = token_mask[:, None, :, None].to(acts.device, dtype=acts.dtype)
    denom = mask.sum(dim=2).clamp_min(1)
    return (acts * mask).sum(dim=2) / denom


def collect_concept_acts(llm, frame, prompt_fn, answer_fn, exclude_special, pool_mode, batch_size):
    concepts = frame["concept"].unique().tolist()
    num_layers = len(llm.model.model.layers)
    result = {}
    for concept in tqdm(concepts, desc="acts", leave=False):
        group = frame[frame["concept"] == concept].reset_index(drop=True)
        prompts = []
        answers = []
        for row in group.itertuples(index=False):
            answer = answer_fn(row)
            prompts.append(prompt_fn(row, answer))
            answers.append(answer)

        batches = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_answers = answers[start:start + batch_size]
            batch = llm.tokenize_batch(batch_prompts)
            token_mask = answer_token_mask(
                llm,
                batch_answers,
                batch["attention_mask"],
                PHI4.assistant_end_marker,
                exclude_special,
                pool_mode,
            )
            llm.reset_all()
            llm.batch_forward(batch_prompts)
            layer_acts = [
                llm.get_last_activations(layer_index).detach().cpu()
                for layer_index in range(num_layers)
            ]
            acts = t.stack(layer_acts, dim=1)
            batches.append(pool_answer_tokens(acts, token_mask))
            llm.reset_all()
        result[concept] = t.cat(batches, dim=0)
    return result


def load_or_collect_variant(llm, frame, variant_name, variant, batch_size, vector_device, pool_mode):
    variant_dir = OUT / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    if variant.get("source") == "store":
        store_root = ROOT / "store" / "phi4_inhouse"
        return (
            t.load(store_root / "v_detect.pt", map_location="cpu"),
            t.load(store_root / "v_refuse.pt", map_location="cpu"),
            t.load(store_root / "thresholds.pt", map_location="cpu"),
            None,
        )
    know_path = variant_dir / "know_acts.pt"
    refuse_path = variant_dir / "refuse_acts.pt"
    detect_path = variant_dir / "v_detect.pt"
    refuse_vec_path = variant_dir / "v_refuse.pt"
    refuse_concept_path = variant_dir / "v_refuse_by_concept.pt"
    tau_path = variant_dir / "thresholds.pt"

    if know_path.exists() and refuse_path.exists():
        know_acts = t.load(know_path, map_location="cpu")
        refuse_acts = t.load(refuse_path, map_location="cpu")
    else:
        tokenizer = llm.tokenizer

        def baseline_answer(row):
            if variant["baseline"] == "clean":
                return clean_text(tokenizer, row.baseline_output)
            return str(row.baseline_output)

        def refusal_answer(_row):
            if variant["refusal"] == "text":
                return variant["refusal_text"]
            if variant["refusal"] == "clean":
                return "I don't know."
            return PHI4.idk_answer

        know_acts = collect_concept_acts(
            llm,
            frame,
            lambda row, ans: PHI4.render(BASELINE_SYSTEM, row.question, ans),
            baseline_answer,
            variant["exclude_special"],
            pool_mode,
            batch_size,
        )
        refuse_acts = collect_concept_acts(
            llm,
            frame,
            lambda row, ans: PHI4.render(refuse_system(row.concept), row.question, ans),
            refusal_answer,
            variant["exclude_special"],
            pool_mode,
            batch_size,
        )
        t.save(know_acts, know_path)
        t.save(refuse_acts, refuse_path)

    if detect_path.exists() and refuse_vec_path.exists() and tau_path.exists():
        v_detect = t.load(detect_path, map_location="cpu")
        v_refuse = t.load(refuse_vec_path, map_location="cpu")
        thresholds = t.load(tau_path, map_location="cpu")
    else:
        v_detect, v_refuse, thresholds = lda_vectors(
            know_acts,
            refuse_acts,
            frame["concept"].unique().tolist(),
            show_progress=False,
            device=vector_device,
            layer_chunk=1,
        )
        t.save(v_detect, detect_path)
        t.save(v_refuse, refuse_vec_path)
        t.save(thresholds, tau_path)

    if refuse_concept_path.exists():
        v_refuse_by_concept = t.load(refuse_concept_path, map_location="cpu")
    else:
        v_refuse_by_concept = {}
        for concept in frame["concept"].unique().tolist():
            diff = (refuse_acts[concept].mean(0) - know_acts[concept].mean(0)).float()
            v_refuse_by_concept[concept] = diff / diff.norm(dim=-1, keepdim=True)
        t.save(v_refuse_by_concept, refuse_concept_path)

    return v_detect, v_refuse, thresholds, v_refuse_by_concept


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


def generate_outputs(llm, rows, steering, layer, scale, batch_size, max_new_tokens, from_offset):
    out_rows = []
    prompts = [PHI4.render(BASELINE_SYSTEM, row.question) for row in rows.itertuples(index=False)]
    jobs = rows.reset_index(drop=True)
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
                trimmed = PHI4.trim_to_last_assistant(raw)
                cleaned = clean_text(llm.tokenizer, trimmed)
                out_rows.append({
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
    return out_rows


def run_sweep(llm, frame, variant_name, v_detect, v_refuse, thresholds, v_refuse_by_concept, layers, scales, batch_size, max_new_tokens, steering_mode, from_offset):
    rows = []
    for layer in layers:
        if steering_mode == "add":
            steering = AddRefusalSteering([layer], v_refuse)
        elif steering_mode == "concept_gated":
            steering = ConceptGatedSteering([layer], v_detect, v_refuse_by_concept, thresholds)
        elif steering_mode == "detect_ablate":
            steering = DetectAblationSteering([layer], v_detect, thresholds)
        else:
            steering = GatedSteering([layer], [layer], v_detect, v_refuse, thresholds)
        for scale in scales:
            rows.extend(generate_outputs(
                llm,
                frame,
                steering,
                layer,
                scale,
                batch_size,
                max_new_tokens,
                from_offset,
            ))
    result = pd.DataFrame(rows)
    result.insert(0, "variant", variant_name)
    result.to_csv(OUT / variant_name / "sweep.csv", index=False)
    return result


def summarize(results):
    summary = (
        results
        .groupby(["variant", "layer", "scale"], as_index=False)
        .agg(
            idk_rate=("idk", "mean"),
            special_fraction=("special_fraction", "mean"),
            n=("idk", "size"),
        )
        .sort_values(["idk_rate", "special_fraction"], ascending=[False, True])
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    return summary


def markdown_table(df):
    view = df.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.3f}")
        else:
            view[column] = view[column].map(str)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in view.to_numpy().tolist()
    ]
    return "\n".join([header, separator] + rows)


def write_report(summary, args, token_report):
    best = summary.groupby("variant", as_index=False).first()
    lines = [
        "# Phi Token Debug",
        "",
        "This directory is isolated from the main pipeline and `store/` artifacts.",
        "",
        "## Setup",
        "",
        f"- model: `{MODEL}`",
        f"- concepts: `{', '.join(CONCEPTS)}`",
        f"- train samples per concept: `{args.train_per_concept}`",
        f"- calibration samples per concept: `{args.cal_per_concept}`",
        f"- layers: `{args.layers}`",
        f"- scales: `{args.scales}`",
        f"- max new tokens: `{args.max_new_tokens}`",
        f"- steering mode: `{args.steering_mode}`",
        f"- from position offset: `{args.from_offset}`",
        f"- pad as eos: `{args.pad_as_eos}`",
        f"- activation pool mode: `{args.pool_mode}`",
        "",
        "## Baseline Token Pollution",
        "",
        markdown_table(token_report),
        "",
        "## Best Cell Per Variant",
        "",
        markdown_table(best),
        "",
        "## Full Summary",
        "",
        markdown_table(summary),
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


def build_token_report(tokenizer):
    rows = []
    for split in ["baseline_train.csv", "baseline_test.csv"]:
        df = pd.read_csv(ROOT / "store" / "phi4_inhouse" / split)
        for concept in CONCEPTS:
            group = df[df["concept"] == concept]
            fractions = group["baseline_output"].fillna("").map(lambda text: special_fraction(tokenizer, text))
            rows.append({
                "split": split,
                "concept": concept,
                "rows": len(group),
                "mean_special_fraction": fractions.mean(),
                "median_special_fraction": fractions.median(),
            })
    return pd.DataFrame(rows)


def parse_numbers(text, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-per-concept", type=int, default=6)
    parser.add_argument("--cal-per-concept", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--vector-device", default="cpu")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--steering-mode", choices=["gated", "add", "concept_gated", "detect_ablate"], default="gated")
    parser.add_argument("--from-offset", type=int, default=0)
    parser.add_argument("--pad-as-eos", action="store_true")
    parser.add_argument("--pool-mode", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--layers", default="0,2,3")
    parser.add_argument("--scales", default="5,9,13")
    parser.add_argument("--variants", default="current,clean_baseline,clean_idk,exclude_special_mask,clean_all")
    args = parser.parse_args()

    OUT = BASE_OUT / args.run_name
    OUT.mkdir(parents=True, exist_ok=True)
    layers = parse_numbers(args.layers, int)
    scales = parse_numbers(args.scales, float)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    (OUT / "config.json").write_text(json.dumps(vars(args), indent=2))

    train = pd.read_csv(ROOT / "store" / "phi4_inhouse" / "baseline_train.csv")
    test = pd.read_csv(ROOT / "store" / "phi4_inhouse" / "baseline_test.csv")
    train_sample = sample_frame(train, CONCEPTS, args.train_per_concept, seed=7)
    cal_sample = sample_frame(test, CONCEPTS, args.cal_per_concept, seed=11)
    train_sample.to_csv(OUT / "train_sample.csv", index=False)
    cal_sample.to_csv(OUT / "calibration_sample.csv", index=False)

    llm = load_llm(MODEL, gpu_id=args.gpu, template=PHI4)
    if args.pad_as_eos:
        llm.pad_token_id = llm.tokenizer.eos_token_id
    token_report = build_token_report(llm.tokenizer)
    token_report.to_csv(OUT / "token_report.csv", index=False)

    all_results = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        v_detect, v_refuse, thresholds, v_refuse_by_concept = load_or_collect_variant(
            llm,
            train_sample,
            variant_name,
            variant,
            args.batch_size,
            args.vector_device,
            args.pool_mode,
        )
        result = run_sweep(
            llm,
            cal_sample,
            variant_name,
            v_detect,
            v_refuse,
            thresholds,
            v_refuse_by_concept,
            layers,
            scales,
            args.batch_size,
            args.max_new_tokens,
            args.steering_mode,
            args.from_offset,
        )
        all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT / "all_sweeps.csv", index=False)
    summary = summarize(results)
    write_report(summary, args, token_report)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
