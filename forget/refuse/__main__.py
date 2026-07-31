import argparse
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import run

load_dotenv()


def _add_single_run_flags(p):
    p.add_argument("--model", help="HF model path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data", type=Path, help="data folder with train.csv and test.csv")
    p.add_argument("--out", type=Path, help="result store folder for cached artifacts")
    p.add_argument("--method", default="lda", choices=["lda", "diffed", "projected"])
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids, e.g. 0,1")
    p.add_argument("--layers", default="default",
                   help="calibration layer spec: 'default' | 'all' | 'frac: 0,.5,1' | '3 7 15,18,21,24'")
    p.add_argument("--scale-window", default="mid",
                   help="calibration scale range: small|mid|large|xlarge or 'lo:hi'")
    p.add_argument("--scale-steps", type=int, default=15,
                   help="number of scale steps within the window (default 15)")
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--test-frac", type=float, default=1.0)
    p.add_argument("--calibration-n", type=lambda v: v if v == "all" else int(v), default=10,
                   help="calibration samples; meaning depends on --calibration-concepts")
    p.add_argument("--calibration-concepts", default="all", choices=["all", "random"],
                   help="'all' = samples per concept; 'random' = total random validation samples")
    p.add_argument("--confusion", nargs=2, type=int, metavar=("C", "N"), default=None,
                   help="run confusion eval: C concepts × C targets × N questions each")
    p.add_argument("--bars", type=int, default=None, metavar="N",
                   help="run bars eval: per target, N target + N untargeted-pool questions")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-gpus", default=None, help="GPU ids for judge (defaults to --gpus)")
    p.add_argument("--judge-retries", type=int, default=25)
    p.add_argument("--judge-mode", default="reasoning", choices=["reasoning", "logit"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--judge-batch-size", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--result", default=None, help="result variant folder under <out>/results")
    p.add_argument("--artifact-cache", default="main", help="artifact cache folder under <out>/artifacts")
    p.add_argument("--no-clean-activation-answers", action="store_false", dest="clean_activation_answers")
    p.add_argument("--intervention-start", default="assistant", choices=["assistant", "prefill"])
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")


def _evaluations(args):
    evaluations = []
    if args.confusion is not None:
        c, n = args.confusion
        evaluations.append(("confusion", {"c": c, "n": n}))
    if args.bars is not None:
        evaluations.append(("bars", {"n": args.bars}))
    return evaluations


def main():
    p = argparse.ArgumentParser(prog="refuse", description="refusal-vector steering pipeline")
    p.add_argument("--config", type=Path, default=None,
                   help="experiments yml; runs its matrix (one subprocess per run)")
    p.add_argument("--only", nargs="*", default=None,
                   help="with --config: run only these experiment names")
    p.add_argument("--list", action="store_true",
                   help="with --config: print resolved experiments and exit")
    p.add_argument("--exec", dest="exec_one", default=None, help=argparse.SUPPRESS)
    _add_single_run_flags(p)
    args = p.parse_args()

    if args.config is not None:
        from .config import load_experiments, run_experiments, to_run_kwargs
        if args.exec_one is not None:
            run(**to_run_kwargs(load_experiments(args.config)[args.exec_one]))
        elif args.list:
            for name, cfg in load_experiments(args.config).items():
                if args.only and name not in args.only:
                    continue
                kw = to_run_kwargs(cfg)
                print(f"{name}: model={kw['model_path']} data={kw['data_root']} out={kw['result_root']} "
                      f"layers={kw['layers']} scales={kw['scales']} window={kw['scale_window']} "
                      f"calibration_n={kw['calibration_n']} concept={kw['calibration_concepts']} "
                      f"result={kw['result_name']} judge_mode={kw['judge_mode']} "
                      f"start={kw['intervention_start']} cache={kw['artifact_cache']} "
                      f"evals={[e[0] for e in kw['evaluations']]}")
        else:
            run_experiments(args.config, only=args.only)
        return

    if not (args.model and args.data and args.out):
        p.error("--model, --data, --out are required (or use --config)")

    run(
        model_path=args.model,
        data_root=args.data,
        result_root=args.out,
        method=args.method,
        gpu_ids=[int(g) for g in args.gpus.split(",")],
        layers=args.layers,
        scales=args.scale_steps,
        scale_window=args.scale_window,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        calibration_n=args.calibration_n,
        calibration_concepts=args.calibration_concepts,
        evaluations=_evaluations(args),
        plot=not args.no_plot,
        verbose=args.verbose,
        judge_model=args.judge_model,
        judge_gpu_ids=[int(g) for g in args.judge_gpus.split(",")] if args.judge_gpus else None,
        judge_max_retries=args.judge_retries,
        judge_mode=args.judge_mode,
        batch_size=args.batch_size,
        judge_batch_size=args.judge_batch_size,
        trust_remote_code=args.trust_remote_code,
        result_name=args.result,
        artifact_cache=args.artifact_cache,
        clean_activation_answers=args.clean_activation_answers,
        intervention_start=args.intervention_start,
    )


if __name__ == "__main__":
    main()
