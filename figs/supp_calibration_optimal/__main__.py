from .paths import OUT, STORE
from .supp_optimal import write_supp_optimal


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(write_supp_optimal(STORE, OUT))
