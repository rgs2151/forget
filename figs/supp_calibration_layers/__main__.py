import argparse

from .paths import OUT, STORE
from .supp_refuse import write_supp_fluency, write_supp_refuse, write_supp_retain


def main():
    parser = argparse.ArgumentParser(description="render full layer-scale supplements")
    parser.add_argument(
        "--metric",
        choices=("all", "refusal", "retention", "fluency"),
        default="all",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    written = []
    if args.metric in ("all", "refusal"):
        written.append(write_supp_refuse(STORE, OUT))
    if args.metric in ("all", "retention"):
        written.append(write_supp_retain(STORE, OUT))
    if args.metric in ("all", "fluency"):
        written.append(write_supp_fluency(STORE, OUT))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
