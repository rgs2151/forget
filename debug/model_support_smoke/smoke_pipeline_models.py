from pathlib import Path
import sys

import pandas as pd

from llm.chat_templates import detect_template
from refuse.pipeline import run


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "debug" / "model_support_smoke" / "runs"
SUMMARY = ROOT / "debug" / "model_support_smoke" / "smoke_summary.csv"
DATA_ROOT = ROOT / "store" / "inhouse"

FEASIBLE_MODELS = [
    {
        "name": "llama32_1b",
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "template_family": "llama3",
        "batch_size": 16,
        "trust_remote_code": False,
    },
    {
        "name": "llama32_3b",
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "template_family": "llama3",
        "batch_size": 16,
        "trust_remote_code": False,
    },
    {
        "name": "qwen05b",
        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "template_family": "qwen_chatml",
        "batch_size": 16,
        "trust_remote_code": False,
    },
    {
        "name": "qwen3b",
        "model_path": "Qwen/Qwen2.5-3B-Instruct",
        "template_family": "qwen_chatml",
        "batch_size": 16,
        "trust_remote_code": False,
    },
    {
        "name": "qwen14b",
        "model_path": "Qwen/Qwen2.5-14B-Instruct",
        "template_family": "qwen_chatml",
        "batch_size": 8,
        "trust_remote_code": False,
    },
    {
        "name": "phi4mini",
        "model_path": "microsoft/Phi-4-mini-instruct",
        "template_family": "phi4_mini",
        "batch_size": 16,
        "trust_remote_code": False,
    },
]

SKIPPED_MODELS = [
    {
        "name": "mistral_small24b",
        "model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "template_family": "mistral_small_v7_tekken",
        "batch_size": 8,
        "trust_remote_code": False,
        "status": "skipped_hardware",
        "calibration_output": False,
        "note": "Model card reports ~55 GB GPU RAM; current wrapper loads one full copy per GPU.",
    }
]

MODEL_ORDER = [spec["name"] for spec in FEASIBLE_MODELS] + [spec["name"] for spec in SKIPPED_MODELS]


def write_summary(rows):
    existing = {}
    if SUMMARY.exists():
        existing = {row["name"]: row for row in pd.read_csv(SUMMARY).to_dict("records")}
    for row in rows:
        existing[row["name"]] = row
    ordered = [existing[name] for name in MODEL_ORDER if name in existing]
    pd.DataFrame(ordered).to_csv(SUMMARY, index=False)


def smoke_model(spec):
    out = RUN_ROOT / spec["name"]
    run(
        model_path=spec["model_path"],
        data_root=DATA_ROOT,
        result_root=out,
        method="lda",
        gpu_ids=[0],
        layers="frac:0.5",
        scales=1,
        scale_window="0:1",
        train_frac=0.01,
        test_frac=0.01,
        calibration_n=1,
        evaluations=[],
        judge_model=None,
        batch_size=spec["batch_size"],
        judge_batch_size=16,
        plot=False,
        verbose=True,
        trust_remote_code=spec["trust_remote_code"],
    )
    calibration_output = (out / "calibration_results.csv").exists()
    return {
        **spec,
        "status": "passed",
        "calibration_output": calibration_output,
        "note": "tiny no-judge pipeline smoke completed",
    }


def main():
    rows = []
    selected = set(sys.argv[1:])
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    active_specs = [spec for spec in FEASIBLE_MODELS if not selected or spec["name"] in selected]
    for spec in active_specs:
        detect_template(spec["model_path"])
        rows.append(smoke_model(spec))
        write_summary(rows + SKIPPED_MODELS)
    write_summary(rows + SKIPPED_MODELS)


if __name__ == "__main__":
    main()
