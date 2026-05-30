import pandas as pd

from .intervention import GatedSteering, make_generation_jobs, run_jobs, sample_per_concept


SCALE_WINDOWS = {"small": (0.0, 5.0), "mid": (0.0, 15.0), "large": (0.0, 100.0)}


def default_intervention_layers(num_layers):
    fractions = [15 / 32, 18 / 32, 21 / 32, 24 / 32]
    return sorted({round(f * num_layers) for f in fractions})


def scale_grid(window="mid", steps=15):
    """Concrete scale values across a window. window: name, 'lo:hi', or (lo, hi)."""
    if isinstance(window, str):
        if window in SCALE_WINDOWS:
            lo, hi = SCALE_WINDOWS[window]
        elif ":" in window:
            lo, hi = (float(x) for x in window.split(":"))
        else:
            raise ValueError(f"unknown scale window {window!r}; use {list(SCALE_WINDOWS)} or 'lo:hi'")
    else:
        lo, hi = window
    return [round(lo + (hi - lo) * i / steps, 2) for i in range(1, steps + 1)]


def resolve_layers(spec, num_layers):
    """Resolve a model-agnostic layer spec into a list of layer-sets.

    'all'   -> every single layer [[0],[1],...,[L-1]]
    'default' -> the fractional canonical set, as one layer-set
    'frac: 0,0.5,1' -> single layers at depth fractions
    '3 7 15,18,21,24' -> explicit (space = new set, comma = layers within a set)
    """
    if spec == "all":
        return [[i] for i in range(num_layers)]
    if spec == "default":
        return [default_intervention_layers(num_layers)]
    if isinstance(spec, str) and spec.startswith("frac:"):
        fracs = [float(x) for x in spec[len("frac:"):].replace(",", " ").split()]
        seen, sets = set(), []
        for f in fracs:
            layer = min(num_layers - 1, max(0, round(f * (num_layers - 1))))
            if layer not in seen:
                seen.add(layer)
                sets.append([layer])
        return sets
    if isinstance(spec, str):
        sets = [[int(x) for x in group.split(",")] for group in spec.split()]
    else:
        sets = [list(s) if isinstance(s, (list, tuple)) else [int(s)] for s in spec]
    for s in sets:
        for layer in s:
            if not 0 <= layer < num_layers:
                raise ValueError(f"layer {layer} out of range for model with {num_layers} layers")
    return sets


def build_grid(num_layers, layers="default", scales=15, scale_window="mid"):
    """The calibration grid: one steering config per (layer_set, scale).

    layers/scales/scale_window are model-level. To add a future axis, add another loop.
    """
    layer_sets = resolve_layers(layers, num_layers)
    scale_vals = scale_grid(scale_window, scales)
    return [
        {"source_layers": ls, "target_layers": ls, "scale": s}
        for ls in layer_sets
        for s in scale_vals
    ]


def select_refusal_scale(results, score_col="judge_refusal", label="intervention"):
    df = results.copy()
    if "label" in df:
        df = df[df["label"] == label]
    if df.empty:
        raise ValueError("No rows available for scale selection.")
    rates = df.groupby("scale", as_index=False)[score_col].mean()
    rates = rates.sort_values([score_col, "scale"], ascending=[False, True])
    return rates.iloc[0]["scale"]


def calibration_sweep(
    pool,
    df,
    grid,
    v_detect,
    v_refuse,
    thresholds,
    system_prompt,
    template,
    *,
    sample_n=10,
    cache_path=None,
    batch_size=128,
    max_new_tokens=64,
    target_col="concept",
    random_state=42,
    log=lambda _msg: None,
):
    """Fill the calibration grid: for every (layer_set, scale) generate the diagonal
    (target == concept) responses over the sampled questions → one flat dataframe.

    Loops layer-outer (one GatedSteering vector-bank per layer) and scales-inner; the
    same sampled questions are reused at every grid point. Resumes by skipping
    (source_layer, scale) pairs already present in cache_path.
    """
    n_per = None if sample_n == "all" else sample_n
    sample = sample_per_concept(df, n_per_concept=n_per, random_state=random_state)
    prompts = [template.render(system_prompt, row.question) for row in sample.itertuples(index=False)]

    layer_to_scales = {}
    for point in grid:
        key = tuple(point["source_layers"])
        layer_to_scales.setdefault(key, [])
        if point["scale"] not in layer_to_scales[key]:
            layer_to_scales[key].append(point["scale"])

    done_pairs, results = set(), []
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        done_pairs = set(zip(cached["source_layer"].astype(str), cached["scale"].astype(float)))
        results.append(cached)

    for layer, scales in layer_to_scales.items():
        layer_list = list(layer)
        todo = [s for s in scales if (str(layer_list), float(s)) not in done_pairs]
        if not todo:
            log(f"  layer {layer_list}: cached")
            continue
        log(f"  layer {layer_list}: {len(todo)} scales × {len(sample)} questions")
        steering = GatedSteering(layer_list, layer_list, v_detect, v_refuse, thresholds)
        jobs = make_generation_jobs(sample, prompts, scales=todo, target_col=target_col)
        res = run_jobs(
            pool, jobs, steering,
            generation_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 1.0},
            batch_size=batch_size,
            trim_fn=template.trim_to_last_assistant,
            result_metadata={"source_layer": layer_list, "target_layer": layer_list},
        )
        results.append(res)
        if cache_path is not None:
            pd.concat(results, ignore_index=True).to_csv(cache_path, index=False)

    return pd.concat(results, ignore_index=True)
