import argparse
from pathlib import Path

from .calib_full import write_calib_full, write_calib_full_metric
from .calib_optimal import write_calib_optimal
from .model_data import write_model_data
from .score_size import write_score_size, write_score_size_refusal
from .util import FULL_METRICS, OUT, STORE


def main():
    parser = argparse.ArgumentParser(description="render summary figures across result stores")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--figure",
        choices=(
            "all",
            "model_data",
            "calib_optimal",
            "calib_full",
            "calib_full_refuse",
            "calib_full_retain",
            "calib_full_fluency",
            "score_size",
            "score_size_refusal",
        ),
        default="all",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    if args.figure in ("all", "model_data"):
        written.append(write_model_data(args.store, args.out))
    if args.figure in ("all", "calib_optimal"):
        written.append(write_calib_optimal(args.store, args.out))
    if args.figure in ("all", "calib_full"):
        written.extend(write_calib_full(args.store, args.out))
    for title, metric, ylabel in FULL_METRICS:
        if args.figure == f"calib_full_{title}":
            written.append(write_calib_full_metric(args.store, args.out, title, metric, ylabel))
    if args.figure in ("all", "score_size"):
        written.append(write_score_size(args.store, args.out))
    if args.figure in ("all", "score_size_refusal"):
        written.append(write_score_size_refusal(args.store, args.out))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
