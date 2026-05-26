from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch as t
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Paths:
    root: Path
    data_root: Path

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def train(self): return self.data_root / "train.csv"

    @property
    def test(self): return self.data_root / "test.csv"

    @property
    def baseline_train(self): return self.root / "baseline_train.csv"

    @property
    def baseline_test(self): return self.root / "baseline_test.csv"

    @property
    def baseline_acts(self): return self.root / "baseline_answer_acts.pt"

    @property
    def refuse_acts(self): return self.root / "refuse_answer_acts.pt"

    @property
    def baseline_test_acts(self): return self.root / "baseline_answer_acts_test.pt"

    @property
    def v_detect(self): return self.root / "v_detect.pt"

    @property
    def v_refuse(self): return self.root / "v_refuse.pt"

    @property
    def thresholds(self): return self.root / "thresholds.pt"

    @property
    def calibration(self): return self.root / "calibration_results.csv"

    @property
    def calibration_judged(self): return self.root / "calibration_judged.csv"

    def eval_path(self, name): return self.root / f"{name}.csv"

    def eval_judged_path(self, name): return self.root / f"{name}_judged.csv"


def cached_pt(paths_dict, compute_fn):
    if all(p.exists() for p in paths_dict.values()):
        return {key: t.load(path) for key, path in paths_dict.items()}
    computed = compute_fn()
    for key, path in paths_dict.items():
        t.save(computed[key], path)
    return computed


def cached_csv_rows(path, df, compute_missing_fn, key_col, batch_size=64,
                    save_every_batch=True, desc=None):
    df = df.copy()
    if key_col not in df.columns:
        df[key_col] = pd.NA
    if path.exists():
        cached = pd.read_csv(path)
        if key_col in cached.columns and len(cached) == len(df):
            df[key_col] = cached[key_col].to_numpy()

    missing_indices = df.index[df[key_col].isna()].tolist()
    starts = list(range(0, len(missing_indices), batch_size))
    iterator = tqdm(starts, desc=desc) if desc and starts else starts
    for start in iterator:
        batch_idx = missing_indices[start:start + batch_size]
        outputs = compute_missing_fn(df.loc[batch_idx])
        df.loc[batch_idx, key_col] = outputs
        if save_every_batch:
            df.to_csv(path, index=False)

    if not save_every_batch:
        df.to_csv(path, index=False)
    return df
