import argparse
from pathlib import Path

from .chat_templates import TEMPLATES
from .pipeline import run


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
    p.add_argument("--template", default=None,
                   choices=sorted(TEMPLATES.keys()),
                   help="chat template (auto-detected from --model if omitted)")
    p.add_argument("--gpus", default="0",
                   help="comma-separated GPU ids, e.g. 0,1")
    p.add_argument("--n-per-concept", type=int, default=25)
    p.add_argument("--no-plot", action="store_true",
                   help="skip diagnostic plots at the end of the pipeline")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print which stage is running and which artifacts are cache hits")
    args = p.parse_args()

    run(
        model_path=args.model,
        data_root=args.data,
        result_root=args.out,
        template=TEMPLATES[args.template] if args.template else None,
        method=args.method,
        gpu_ids=[int(g) for g in args.gpus.split(",")],
        n_per_concept=args.n_per_concept,
        plot=not args.no_plot,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
