from .paths import OUT, STORE
from .supp_bars import write_supp_bars


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(write_supp_bars(STORE, OUT))
