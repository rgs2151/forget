import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t

import qwen_store_vector_sweep as sweep
from refuse.activations import collect_answer_activations_batched
from refuse.prompts import BASELINE_SYSTEM, refuse_system
from refuse.vectors import lda_vectors


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "debug" / "qwen_debug" / "runs"


def collect_by_concept(llm, frame, concepts, batch_size, system_fn, answer_fn):
    acts = {}
    for concept in concepts:
        group = frame[frame["concept"] == concept].reset_index(drop=True)
        prompts = []
        answers = []
        for row in group.itertuples(index=False):
            answer = answer_fn(row)
            prompts.append(sweep.QWEN.render(system_fn(row), row.question, answer))
            answers.append(answer)
        acts[concept] = collect_answer_activations_batched(
            llm,
            prompts,
            answers,
            sweep.QWEN.assistant_end_marker,
            batch_size=batch_size,
            show_progress=True,
            progress_desc=f"{concept} acts",
        )
    return acts


def write_report(out_dir, args, summary):
    lines = [
        f"# Qwen clean-vector run: {args.run_name}",
        "",
        "This run rebuilds vectors in debug from cleaned Qwen answers.",
        "",
        "## Setup",
        "",
        f"- concepts: `{args.concepts}`",
        f"- train per concept: `{args.train_per_concept}`",
        f"- samples per concept: `{args.per_concept}`",
        f"- layers: `{args.layers}`",
        f"- scales: `{args.scales}`",
        f"- from offset: `{args.from_offset}`",
        f"- pad as eos: `{args.pad_as_eos}`",
        f"- steering mode: `{args.steering_mode}`",
        "",
        "## Top Cells",
        "",
        sweep.markdown_table(summary.head(20)),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concepts", default="bacteria,cats,dogs,united_states")
    parser.add_argument("--train-per-concept", type=int, default=40)
    parser.add_argument("--per-concept", type=int, default=5)
    parser.add_argument("--layers", default="13,14,15,16,17")
    parser.add_argument("--scales", default="20,40,60,80,100,140,200")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--from-offset", type=int, default=-10000)
    parser.add_argument("--pad-as-eos", action="store_true")
    parser.add_argument("--steering-mode", choices=["gated", "add"], default="gated")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    concepts = [part.strip() for part in args.concepts.split(",") if part.strip()]
    layers = sweep.parse_numbers(args.layers, int)
    scales = sweep.parse_numbers(args.scales, float)

    store_root = ROOT / "store" / sweep.STORE
    train = pd.read_csv(store_root / "baseline_train.csv")
    test = pd.read_csv(store_root / "baseline_test.csv")
    train_sample = sweep.sample_frame(train, concepts, args.train_per_concept, seed=17)
    eval_sample = sweep.sample_frame(test, concepts, args.per_concept, seed=11)
    train_sample.to_csv(out_dir / "train_sample.csv", index=False)
    eval_sample.to_csv(out_dir / "sample.csv", index=False)

    llm = sweep.load_llm(sweep.MODEL, gpu_id=args.gpu, template=sweep.QWEN)
    if args.pad_as_eos:
        llm.pad_token_id = llm.tokenizer.eos_token_id

    def clean_baseline(row):
        cleaned = sweep.clean_text(llm.tokenizer, row.baseline_output)
        return cleaned or row.answer

    know_acts = collect_by_concept(
        llm,
        train_sample,
        concepts,
        args.batch_size,
        lambda _row: BASELINE_SYSTEM,
        clean_baseline,
    )
    refuse_acts = collect_by_concept(
        llm,
        train_sample,
        concepts,
        args.batch_size,
        lambda row: refuse_system(row.concept),
        lambda _row: "I don't know.",
    )

    t.save(know_acts, out_dir / "clean_baseline_acts.pt")
    t.save(refuse_acts, out_dir / "clean_refuse_acts.pt")
    v_detect, v_refuse, thresholds = lda_vectors(know_acts, refuse_acts, concepts, device=llm.device)
    t.save(v_detect, out_dir / "v_detect.pt")
    t.save(v_refuse, out_dir / "v_refuse.pt")
    t.save(thresholds, out_dir / "thresholds.pt")

    results = sweep.run_store_sweep(
        llm,
        eval_sample,
        v_detect,
        v_refuse,
        thresholds,
        layers,
        scales,
        args.batch_size,
        args.max_new_tokens,
        args.from_offset,
        None,
        args.steering_mode,
    )
    results.to_csv(out_dir / "sweep.csv", index=False)
    summary = sweep.summarize(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir, args, summary)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
