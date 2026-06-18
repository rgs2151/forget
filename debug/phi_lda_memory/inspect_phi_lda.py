from pathlib import Path

import torch as t


ROOT = Path(__file__).resolve().parents[2]
ACTS_PATH = ROOT / "store/phi4_mmlu/artifacts/main/baseline_answer_acts.pt"


def matrix_mib(hidden, layer_chunk):
    return layer_chunk * hidden * hidden * 4 / 1024**2


def main():
    acts = t.load(ACTS_PATH, map_location="cpu")
    concept = next(iter(acts))
    n_examples, n_layers, hidden = acts[concept].shape

    print(f"artifact={ACTS_PATH}")
    print(f"example_concept={concept}")
    print(f"shape_per_concept=({n_examples}, {n_layers}, {hidden})")
    for chunk in (4, 1):
        print(f"chunk={chunk} dense_matrix_mib={matrix_mib(hidden, chunk):.1f}")


if __name__ == "__main__":
    main()
