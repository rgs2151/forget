from contextlib import contextmanager

import numpy as np
import optuna
import pandas as pd

from rouge_score import rouge_scorer
from tqdm.auto import tqdm


@contextmanager
def _optuna_logging(enabled):
    previous_verbosity = optuna.logging.get_verbosity()
    try:
        if not enabled:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        yield
    finally:
        optuna.logging.set_verbosity(previous_verbosity)


def _normalize_layers(layer):
    if isinstance(layer, int):
        return [layer]
    return list(layer)


def sample_forget_retain_sets(df, target, forget_n=24, retain_n_per_concept=2, random_state=42):
    forget_df = df[df["concept"] == target].sample(
        min(forget_n, (df["concept"] == target).sum()),
        random_state=random_state,
    ).reset_index(drop=True)

    retain_parts = []
    for _, group in df[df["concept"] != target].groupby("concept"):
        retain_parts.append(
            group.sample(min(len(group), retain_n_per_concept), random_state=random_state)
        )

    retain_df = pd.concat(retain_parts, ignore_index=True) if retain_parts else df.iloc[0:0].copy()
    return forget_df, retain_df


def generate_responses_batched(
    llm,
    chats,
    steer_factory,
    scale,
    source_layer,
    target_layer,
    batch_size=32,
    trim_output_fn=None,
    show_progress=False,
    progress_desc="Generation batches",
    **generation_kwargs,
):
    src_layers = _normalize_layers(source_layer)
    tgt_layers = _normalize_layers(target_layer)
    assert len(src_layers) == len(tgt_layers)

    if trim_output_fn is None:
        trim_output_fn = lambda text: text

    generation_kwargs = dict(generation_kwargs)
    generation_kwargs.setdefault("max_new_tokens", 64)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("temperature", 1.0)

    outputs = []
    for start in tqdm(
        range(0, len(chats), batch_size),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    ):
        batch_chats = chats[start:start + batch_size]
        llm.reset_all()
        for src_layer_item, tgt_layer_item in zip(src_layers, tgt_layers):
            llm.set_steering_op(tgt_layer_item, steer_factory(src_layer_item, scale))
        batch_outputs = llm.generate_from_chats(batch_chats, **generation_kwargs)
        outputs.extend(trim_output_fn(text) for text in batch_outputs)
        llm.reset_all()

    return outputs


def rouge_reference_mean(outputs, reference="I don't know.", scorer=None):
    scorer = scorer or rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if not outputs:
        return 0.0
    return float(np.mean([
        scorer.score(reference, output)["rougeL"].fmeasure
        for output in outputs
    ]))


def make_scale_objective(
    llm,
    forget_chats,
    retain_chats,
    steer_factory,
    source_layer,
    target_layer,
    low=0.25,
    high=32.0,
    lambda_retain=1.0,
    batch_size=32,
    scorer=None,
    log=True,
    trim_output_fn=None,
    reference="I don't know.",
    generation_kwargs=None,
    show_progress=False,
):
    scorer = scorer or rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    generation_kwargs = generation_kwargs or {}

    def objective(trial):
        scale = trial.suggest_float("scale", low, high, log=log)

        forget_outputs = generate_responses_batched(
            llm,
            forget_chats,
            steer_factory=steer_factory,
            scale=scale,
            source_layer=source_layer,
            target_layer=target_layer,
            batch_size=batch_size,
            trim_output_fn=trim_output_fn,
            show_progress=show_progress,
            **generation_kwargs,
        )
        retain_outputs = generate_responses_batched(
            llm,
            retain_chats,
            steer_factory=steer_factory,
            scale=scale,
            source_layer=source_layer,
            target_layer=target_layer,
            batch_size=batch_size,
            trim_output_fn=trim_output_fn,
            show_progress=show_progress,
            **generation_kwargs,
        )

        forget_idk = rouge_reference_mean(forget_outputs, reference=reference, scorer=scorer)
        retain_idk = rouge_reference_mean(retain_outputs, reference=reference, scorer=scorer)
        objective_value = forget_idk - lambda_retain * retain_idk

        trial.set_user_attr("forget_idk", forget_idk)
        trial.set_user_attr("retain_idk", retain_idk)
        trial.set_user_attr("objective", objective_value)
        return -objective_value

    return objective


def optimize_scale_for_target(
    llm,
    target,
    forget_chats,
    retain_chats,
    steer_factory,
    source_layer,
    target_layer,
    n_trials=15,
    low=0.25,
    high=32.0,
    log=True,
    lambda_retain=1.0,
    batch_size=32,
    random_state=42,
    trim_output_fn=None,
    reference="I don't know.",
    generation_kwargs=None,
    show_progress=False,
    show_optuna_logs=False,
):
    objective = make_scale_objective(
        llm=llm,
        forget_chats=forget_chats,
        retain_chats=retain_chats,
        steer_factory=steer_factory,
        source_layer=source_layer,
        target_layer=target_layer,
        low=low,
        high=high,
        log=log,
        lambda_retain=lambda_retain,
        batch_size=batch_size,
        trim_output_fn=trim_output_fn,
        reference=reference,
        generation_kwargs=generation_kwargs,
        show_progress=show_progress,
    )

    with _optuna_logging(show_optuna_logs):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    return study, pd.Series({
        "target": target,
        "best_scale": best_trial.params["scale"],
        "forget_idk": best_trial.user_attrs["forget_idk"],
        "retain_idk": best_trial.user_attrs["retain_idk"],
        "objective": best_trial.user_attrs["objective"],
        "forget_n": len(forget_chats),
        "retain_n": len(retain_chats),
        "source_layer": source_layer,
        "target_layer": target_layer,
    })
 