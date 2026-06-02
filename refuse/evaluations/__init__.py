from .bars import run_bars
from .confusion import run_confusion

EVALUATIONS = {
    "confusion": run_confusion,
    "bars": run_bars,
}
