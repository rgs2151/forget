import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from judge.judge import AXIS_TEMPLATE, _score_token_options
from judge.prompts import LOGIT_JUDGE_SYSTEM
from llm import GPUPool


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def gpu_memory():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return "; ".join(line.strip() for line in out.strip().splitlines())


def render_prompts(pool, rows, axis):
    template = pool.template
    axis_template = AXIS_TEMPLATE[axis]
    prompts = []
    for row in rows.itertuples(index=False):
        user = axis_template.format(
            question=row.question,
            reference=row.baseline_output,
            answer=row.model_output,
        )
        prompts.append(template.render(LOGIT_JUDGE_SYSTEM, f"{user}\n\nScore:"))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="store/llama8b_inhouse/results/prefill_logit/calibration_results.csv")
    parser.add_argument("--judge-model", default="AtlaAI/Selene-1-Mini-Llama-3.1-8B")
    parser.add_argument("--axis", default="refusal")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=160)
    parser.add_argument("--out", default="debug/judge_logit_oom/runs/parallel_pool_capture_off_v1")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    report_path = out_dir / "report.md"

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    chunk_size = args.batch_size * len(gpu_ids)
    rows = pd.read_csv(args.calibration).head(args.chunks * chunk_size)

    start_time = time.time()
    pool = GPUPool.from_model_path(
        args.judge_model,
        gpu_ids,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    first_llm = next(iter(pool.llms.values()))
    option_token_ids = _score_token_options(first_llm.tokenizer)
    prompts = render_prompts(pool, rows, args.axis)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk", "rows", "elapsed_s", "gpu_memory", "p1_min", "p1_max", "p2_min", "p2_max"],
        )
        writer.writeheader()
        for chunk in range(args.chunks):
            start = chunk * chunk_size
            batch_prompts = prompts[start:start + chunk_size]
            scores = pool.score_next_token_options(
                batch_prompts,
                option_token_ids,
                batch_size=args.batch_size,
                show_progress=False,
            )
            p1 = [score["1"] for score in scores]
            p2 = [score["2"] for score in scores]
            writer.writerow({
                "chunk": chunk + 1,
                "rows": len(scores),
                "elapsed_s": round(time.time() - start_time, 3),
                "gpu_memory": gpu_memory(),
                "p1_min": min(p1),
                "p1_max": max(p1),
                "p2_min": min(p2),
                "p2_max": max(p2),
            })
            f.flush()

    report_path.write_text(
        "\n".join([
            "# Parallel Logit Judge OOM Probe",
            "",
            "Suspicion: logit scoring failed because the judge wrapper retained full layer activations while scoring.",
            f"Test: score {len(prompts)} real calibration judge prompts with {args.judge_model} on GPUs {gpu_ids}, batch size {args.batch_size}.",
            "Result: completed all chunks without OOM.",
            f"Summary: `{summary_path}`",
            "",
        ])
    )


if __name__ == "__main__":
    main()
