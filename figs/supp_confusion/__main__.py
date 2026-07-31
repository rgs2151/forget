from .paths import OUT, STORE
from .supp_confusion import write_supp_confusion


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(write_supp_confusion(STORE, OUT))
