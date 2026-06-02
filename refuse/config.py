import subprocess
import sys
import time
from pathlib import Path

import yaml


def _merge(*layers):
    out = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if value is not None:
                out[key] = value
    return out


def load_experiments(path):
    """Parse an experiments yml into {name: resolved_config} via layered defaults."""
    spec = yaml.safe_load(Path(path).read_text())
    defaults = spec.get("defaults", {})
    models = spec.get("models", {})
    datasets = spec.get("datasets", {})
    data_root = spec.get("data_root", "store")
    store_root = spec.get("store_root", "store")

    resolved = {}
    for entry in spec.get("runs", []):
        model_key, data_key = entry["model"], entry["data"]
        model_cfg = dict(models[model_key])
        model_path = model_cfg.pop("path")
        override = {k: v for k, v in entry.items() if k not in ("model", "data", "name")}
        cfg = _merge(defaults, model_cfg, datasets.get(data_key, {}), override)
        name = entry.get("name", f"{model_key}_{data_key}")
        cfg["model"] = model_path
        cfg["data"] = f"{data_root}/{data_key}"
        cfg["out"] = cfg.get("out", f"{store_root}/{name}")
        cfg["name"] = name
        resolved[name] = cfg
    return resolved


def to_run_kwargs(cfg):
    """Translate a resolved yml config into refuse.pipeline.run kwargs."""
    evaluations = []
    if "confusion" in cfg:
        c, n = cfg["confusion"]
        evaluations.append(("confusion", {"c": c, "n": n}))
    if "bars" in cfg:
        evaluations.append(("bars", {"n": cfg["bars"]}))
    return dict(
        model_path=cfg["model"],
        data_root=cfg["data"],
        result_root=cfg["out"],
        method=cfg.get("method", "lda"),
        gpu_ids=cfg.get("gpus", [0]),
        layers=cfg.get("layers", "default"),
        scales=cfg.get("scales", 15),
        scale_window=cfg.get("scale_window", "mid"),
        train_frac=cfg.get("train_frac", 1.0),
        test_frac=cfg.get("test_frac", 1.0),
        calibration_n=cfg.get("calibration_n", 10),
        evaluations=evaluations,
        judge_model=cfg.get("judge_model"),
        judge_gpu_ids=cfg.get("judge_gpus"),
        judge_max_retries=cfg.get("judge_retries", 25),
        batch_size=cfg.get("batch_size", 64),
        judge_batch_size=cfg.get("judge_batch_size", 32),
        plot=cfg.get("plot", True),
        verbose=cfg.get("verbose", True),
    )


def run_experiments(config_path, only=None):
    """Run each experiment as its own subprocess (process isolation + crash resilience)."""
    experiments = load_experiments(config_path)
    names = [n for n in experiments if not only or n in only]
    Path("logs").mkdir(exist_ok=True)
    master = open("logs/experiments.log", "a", buffering=1)
    for name in names:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        head = f"=== [{stamp}] START {name} ==="
        print(head, flush=True)
        master.write(head + "\n")
        cmd = [sys.executable, "-m", "refuse", "--config", str(config_path), "--exec", name]
        code = subprocess.run(cmd).returncode
        tail = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] END {name} (exit={code}) ==="
        print(tail, flush=True)
        master.write(tail + "\n")
