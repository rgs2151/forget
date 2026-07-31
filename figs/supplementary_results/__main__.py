import argparse
from pathlib import Path

from .paths import OUT, STORE
from .supp_bars import write_supp_bars
from .supp_confusion import write_supp_confusion
from .supp_optimal import write_supp_optimal
from .supp_refuse import write_supp_fluency, write_supp_refuse, write_supp_retain


def main():
    parser = argparse.ArgumentParser(description="render supplementary figures")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--figure",
        choices=(
            "all",
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
