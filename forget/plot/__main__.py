import argparse
from pathlib import Path

from .plot import make_all


def main():
    p = argparse.ArgumentParser(
        prog="plot",
        description="re-render diagnostic plots from a refuse store directory",
    )
    p.add_argument("--store", required=True, type=Path,
                   help="result store folder (the one with calibration_judged.csv and *_judged.csv eval files)")
    p.add_argument("--out", default=None, type=Path,
                   help="where to write plots (default: <store>/plots)")
    args = p.parse_args()

    written = make_all(args.store, save_dir=args.out)
    for name in written:
        print(name)


if __name__ == "__main__":
    main()
