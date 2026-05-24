import argparse
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import run

load_dotenv()


def main():
    p = argparse.ArgumentParser(
        prog="refuse",
        description="refusal-vector steering pipeline",
    )
    p.add_argument("--model", required=True,
                   help="HF model path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data", required=True, type=Path,
                   help="data folder with train.csv and test.csv")
    p.add_argument("--out", required=True, type=Path,
                   help="result store folder for cached artifacts")
    p.add_argument("--method", default="lda",
                   choices=["lda", "diffed", "projected"])
    p.add_argument("--gpus", default="0",
                   help="comma-separated GPU ids, e.g. 0,1")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="fraction of train set to use for vectors/activations (default 1.0)")
    p.add_argument("--test-frac", type=float, default=1.0,
                   help="fraction of test set to keep for baseline/calibration/validation (default 1.0)")
    p.add_argument("--calibration-frac", type=float, default=0.10,
                   help="fraction of (kept) test set to sweep over in calibration (default 0.10 = 10%%)")
    p.add_argument("--validation-frac", type=float, default=0.10,
                   help="fraction of (kept) test set to evaluate at the selected scale (default 0.10 = 10%%)")
    p.add_argument("--judge-model", default=None,
                   help="HF model path to use as LLM-judge for refusal/retention/fluency")
    p.add_argument("--judge-gpus", default=None,
                   help="comma-separated GPU ids for the judge (defaults to --gpus)")
    p.add_argument("--judge-retries", type=int, default=25,
                   help="retry attempts for judge rows that fail to parse (default: 25)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip diagnostic plots at the end of the pipeline")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print which stage is running and which artifacts are cache hits")
    args = p.parse_args()

    run(
        model_path=args.model,
        data_root=args.data,
        result_root=args.out,
        method=args.method,
        gpu_ids=[int(g) for g in args.gpus.split(",")],
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        calibration_frac=args.calibration_frac,
        validation_frac=args.validation_frac,
        plot=not args.no_plot,
        verbose=args.verbose,
        judge_model=args.judge_model,
        judge_gpu_ids=[int(g) for g in args.judge_gpus.split(",")] if args.judge_gpus else None,
        judge_max_retries=args.judge_retries,
    )


if __name__ == "__main__":
    main()
