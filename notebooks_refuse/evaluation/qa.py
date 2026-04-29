from dataclasses import dataclass

import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm

from vectors.activations import build_question_chats
from optimize.objective import normalize_layers


IDK_PHRASES = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i cannot answer",
    "i can't answer",
    "i don't",
    "i dont",
    "i do not",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    target: str
    source_layer: object
    target_layer: object
    scale: float


def sample_per_concept(df, n_per_concept=None, random_state=42):
    if n_per_concept is None:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby("concept"):
        parts.append(group.sample(min(len(group), n_per_concept), random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def load_or_empty_results(csv_path, text_columns=None):
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for column in text_columns or []:
        if column in df:
            df[column] = df[column].fillna("")
    return df


def make_run_specs(objective_df, baseline_scale=0.0, steered_label="steered", baseline_label="baseline"):
    specs = []
    for row in objective_df.itertuples(index=False):
        specs.append(RunSpec(
            label=baseline_label,
            target=row.target,
            source_layer=row.source_layer,
            target_layer=row.target_layer,
            scale=baseline_scale,
        ))
        specs.append(RunSpec(
            label=steered_label,
            target=row.target,
            source_layer=row.source_layer,
            target_layer=row.target_layer,
            scale=row.best_scale,
        ))
    return specs


def _run_done(df, spec):
    if df.empty:
        return False
    mask = (
        (df["label"] == spec.label)
        & (df["target"] == spec.target)
        & (df["source_layer"].astype(str) == str(spec.source_layer))
        & (df["target_layer"].astype(str) == str(spec.target_layer))
        & (df["scale"] == spec.scale)
    )
    return mask.any()


def evaluate_qa_generation_batched(
    llm,
    df,
    chats,
    target,
    steer_factory,
    source_layer,
    target_layer,
    scale,
    batch_size=32,
    generation_kwargs=None,
    trim_output_fn=None,
    show_progress=False,
    progress_desc="QA batches",
):
    src_layers = normalize_layers(source_layer)
    tgt_layers = normalize_layers(target_layer)
    assert len(src_layers) == len(tgt_layers)

    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("max_new_tokens", 128)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("temperature", 1.0)

    all_outputs = []
    for start in tqdm(
        range(0, len(chats), batch_size),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    ):
        batch_chats = chats[start:start + batch_size]
        llm.reset_all()
        for source_layer_item, target_layer_item in zip(src_layers, tgt_layers):
            llm.set_steering_op(target_layer_item, steer_factory(source_layer_item, scale))
        outputs = llm.generate_from_chats(batch_chats, **generation_kwargs)
        all_outputs.extend(outputs)
        llm.reset_all()

    rows = []
    for (_, row), raw in zip(df.iterrows(), all_outputs):
        response = trim_output_fn(raw) if trim_output_fn is not None else raw
        rows.append({
            "concept": row["concept"],
            "target": target,
            "question": row["question"],
            "correct_answer": row["answer"],
            "scale": scale,
            "source_layer": src_layers if len(src_layers) > 1 else src_layers[0],
            "target_layer": tgt_layers if len(tgt_layers) > 1 else tgt_layers[0],
            "label": None,
            "model_output": response,
        })
    return pd.DataFrame(rows)


def run_qa_benchmark(
    llm,
    df,
    system_prompt,
    chat_cls,
    run_specs,
    steer_factory_fn,
    csv_path,
    generation_kwargs=None,
    trim_output_fn=None,
    batch_size=64,
    question_col="question",
):
    results_df = load_or_empty_results(csv_path, text_columns=["model_output"])
    chats = build_question_chats(df, lambda: chat_cls(system_prompt=system_prompt), question_col=question_col)
    pending_specs = [spec for spec in run_specs if not _run_done(results_df, spec)]

    for spec in tqdm(pending_specs, desc="QA benchmark runs"):
        factory = steer_factory_fn(spec.target)
        run_df = evaluate_qa_generation_batched(
            llm,
            df,
            chats,
            target=spec.target,
            steer_factory=factory,
            source_layer=spec.source_layer,
            target_layer=spec.target_layer,
            scale=spec.scale,
            batch_size=batch_size,
            generation_kwargs=generation_kwargs,
            trim_output_fn=trim_output_fn,
        )
        run_df["label"] = spec.label
        results_df = pd.concat([results_df, run_df], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    return results_df


def add_bertscore_columns(df, prediction_col="model_output", correct_col="correct_answer", idk_reference="I don't know."):
    df = df.copy()
    predictions = df[prediction_col].fillna("").tolist()
    correct_refs = df[correct_col].fillna("").tolist()
    _, _, f_correct = bert_score_fn(predictions, correct_refs, lang="en", verbose=True)
    _, _, f_idk = bert_score_fn(predictions, [idk_reference] * len(df), lang="en", verbose=True)
    df["bert_sim_correct"] = f_correct.numpy()
    df["bert_sim_idk"] = f_idk.numpy()
    df["answer_score_gap"] = df["bert_sim_correct"] - df["bert_sim_idk"]
    return df


def idk_ratio(text):
    text_lower = text.lower()
    for phrase in IDK_PHRASES:
        if phrase in text_lower:
            return len(phrase.split()) / max(len(text_lower.split()), 1)
    return 0.0


def add_idk_ratio_column(df, prediction_col="model_output"):
    df = df.copy()
    df["idk_ratio"] = df[prediction_col].fillna("").apply(idk_ratio)
    return df


def load_perplexity_model(device="cuda"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    return tokenizer, model


def perplexity(text, tokenizer, model, device="cuda", max_len=512):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return loss.exp().item()


def add_perplexity_column(df, tokenizer, model, prediction_col="model_output", device="cuda"):
    df = df.copy()
    df["perplexity"] = df[prediction_col].fillna("").apply(
        lambda text: perplexity(text, tokenizer, model, device=device)
    )
    return df
