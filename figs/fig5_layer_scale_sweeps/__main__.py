import argparse
from pathlib import Path

from .paths import OUT, STORE
from .publish_params import write_publish_params, write_publish_params_min


def main():
    parser = argparse.ArgumentParser(description="render layer-scale figure panels")
    parser.add_argument("--store", default=STORE, type=Path)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument(
        "--figure",
        choices=("all", "publish_params", "publish_params_min"),
        default="all",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    if args.figure in ("all", "publish_params"):
        written.append(write_publish_params(args.store, args.out))
    if args.figure in ("all", "publish_params_min"):
        written.append(write_publish_params_min(args.store, args.out))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
