import argparse
from pathlib import Path

from forget.plot.results import STORE

from .model_size_scaling import (
    write_publish_table_res,
    write_score_size,
    write_score_size_refusal,
)


OUT = Path(__file__).resolve().parent / "plots"


def main():
    parser = argparse.ArgumentParser(description="render model-size scaling outputs")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--output",
        choices=("all", "score_size", "score_size_refusal", "publish_table_res"),
        default="all",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    if args.output in ("all", "score_size"):
        written.append(write_score_size(args.store, args.out))
    if args.output in ("all", "score_size_refusal"):
        written.append(write_score_size_refusal(args.store, args.out))
    if args.output in ("all", "publish_table_res"):
        written.append(write_publish_table_res(args.store, args.out))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
