import argparse
from pathlib import Path

from .publish_bar import write_publish_bar
from .publish_confusion import write_publish_confusion, write_publish_confusion_ret
from .publish_disruption import write_publish_disruption, write_publish_fluency, write_publish_refusal
from .publish_params import write_publish_params, write_publish_params_min
from .supp_bars import write_supp_bars
from .supp_confusion import write_supp_confusion
from .supp_optimal import write_supp_optimal
from .supp_refuse import write_supp_fluency, write_supp_refuse, write_supp_retain
from ..summary.util import OUT, STORE


def main():
    parser = argparse.ArgumentParser(description="render publication figures")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--figure",
        choices=(
            "all",
            "publish_confusion",
            "publish_confusion_ret",
            "publish_bar",
            "publish_disruption",
            "publish_fluency",
            "publish_refusal",
            "publish_params",
            "publish_params_min",
            "supp_bars",
            "supp_confusion",
            "supp_optimal",
            "supp_refuse",
            "supp_retain",
            "supp_fluency",
        ),
        default="all",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    if args.figure in ("all", "publish_confusion"):
        written.append(write_publish_confusion(args.store, args.out))
    if args.figure in ("all", "publish_confusion_ret"):
        written.append(write_publish_confusion_ret(args.store, args.out))
    if args.figure in ("all", "publish_bar"):
        written.append(write_publish_bar(args.store, args.out))
    if args.figure in ("all", "publish_disruption"):
        written.append(write_publish_disruption(args.store, args.out))
    if args.figure in ("all", "publish_fluency"):
        written.append(write_publish_fluency(args.store, args.out))
    if args.figure in ("all", "publish_refusal"):
        written.append(write_publish_refusal(args.store, args.out))
    if args.figure in ("all", "publish_params"):
        written.append(write_publish_params(args.store, args.out))
    if args.figure in ("all", "publish_params_min"):
        written.append(write_publish_params_min(args.store, args.out))
    if args.figure in ("all", "supp_bars"):
        written.append(write_supp_bars(args.store, args.out))
    if args.figure in ("all", "supp_confusion"):
        written.append(write_supp_confusion(args.store, args.out))
    if args.figure in ("all", "supp_optimal"):
        written.append(write_supp_optimal(args.store, args.out))
    if args.figure in ("all", "supp_refuse"):
        written.append(write_supp_refuse(args.store, args.out))
    if args.figure in ("all", "supp_retain"):
        written.append(write_supp_retain(args.store, args.out))
    if args.figure in ("all", "supp_fluency"):
        written.append(write_supp_fluency(args.store, args.out))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
