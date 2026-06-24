import argparse
from pathlib import Path

from .publish_bar import write_publish_bar
from .publish_confusion import write_publish_confusion, write_publish_confusion_ret
from .publish_disruption import write_publish_disruption, write_publish_fluency, write_publish_refusal
from .publish_params import write_publish_params, write_publish_params_min
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

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
