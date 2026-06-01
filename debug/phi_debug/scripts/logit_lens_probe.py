import argparse
from pathlib import Path

import pandas as pd
import torch as t

from llm import LLAMA3, PHI4
from llm.model import load_llm


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "debug" / "phi_debug" / "structured_steering"

MODEL_CONFIGS = {
    "llama": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "store": "llama8b_inhouse",
        "template": LLAMA3,
    },
    "phi": {
        "model": "microsoft/phi-4",
        "store": "phi4_inhouse",
        "template": PHI4,
    },
}

PROBE_STRINGS = [
    "I",
    " I",
    "I'm",
    " I'm",
    "don't",
    " don't",
    "know",
    " know",
    "not",
    " not",
    "sure",
    " sure",
]


def token_label(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return repr(text)


def collect_hidden_mean(acts, concepts, layer):
    parts = [acts[concept][:, layer, :].float() for concept in concepts]
    return t.cat(parts, dim=0).mean(0)


def probe_string_rows(tokenizer, delta_logits):
    rows = []
    for text in PROBE_STRINGS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_id = ids[0]
        value = float(delta_logits[token_id].item())
        rank = int((delta_logits > delta_logits[token_id]).sum().item()) + 1
        rows.append({
            "text": text,
            "token_id": token_id,
            "token": token_label(tokenizer, token_id),
            "delta": value,
            "rank": rank,
        })
    return pd.DataFrame(rows).sort_values("rank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concepts", default="bacteria,cats,dogs,united_states")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_key]
    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = [part.strip() for part in args.concepts.split(",") if part.strip()]
    store_root = ROOT / "store" / config["store"]
    acts = t.load(store_root / "baseline_answer_acts_test.pt", map_location="cpu")
    v_refuse = t.load(store_root / "v_refuse.pt", map_location="cpu").float()

    llm = load_llm(config["model"], gpu_id=args.gpu, template=config["template"])
    hidden = collect_hidden_mean(acts, concepts, args.layer).to(llm.device, dtype=llm.model.dtype)
    vector = v_refuse[args.layer].to(llm.device, dtype=llm.model.dtype)

    with t.no_grad():
        base_logits = llm.model.lm_head(llm.model.model.norm(hidden.unsqueeze(0))).squeeze(0)
        steered = hidden + args.scale * vector
        steered_logits = llm.model.lm_head(llm.model.model.norm(steered.unsqueeze(0))).squeeze(0)
    delta = (steered_logits - base_logits).detach().float().cpu()

    top = t.topk(delta, k=40)
    top_rows = pd.DataFrame([
        {
            "rank": i + 1,
            "token_id": int(token_id),
            "token": token_label(llm.tokenizer, token_id),
            "delta": float(value),
        }
        for i, (value, token_id) in enumerate(zip(top.values, top.indices))
    ])
    probes = probe_string_rows(llm.tokenizer, delta)
    top_rows.to_csv(out_dir / "top_delta_tokens.csv", index=False)
    probes.to_csv(out_dir / "probe_string_deltas.csv", index=False)

    print("top token deltas")
    print(top_rows.head(20).to_string(index=False))
    print("\nprobe string deltas")
    print(probes.to_string(index=False))


if __name__ == "__main__":
    main()
