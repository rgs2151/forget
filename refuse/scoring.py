from bert_score import score as bert_score_fn
from transformers import pipeline


def add_retention_column(df, prediction_col="model_output", correct_col="baseline_output"):
    df = df.copy()
    predictions = df[prediction_col].fillna("").tolist()
    correct_refs = df[correct_col].fillna("").tolist()
    _, _, f_correct = bert_score_fn(predictions, correct_refs, lang="en", verbose=True, rescale_with_baseline=True)
    df["retention_score"] = f_correct.numpy()
    return df


def add_refusal_column(df, prediction_col="model_output", refusal_string="I don't know."):
    df = df.copy()
    predictions = df[prediction_col].fillna("").tolist()
    refusal_refs = [refusal_string] * len(predictions)
    _, _, f_refusal = bert_score_fn(predictions, refusal_refs, lang="en", verbose=True, rescale_with_baseline=True)
    df["refusal_score"] = f_refusal.numpy()
    return df


def add_acceptability_column(df, prediction_col="model_output", device="cuda", batch_size=128, max_length=512):
    df = df.copy()
    predictions = df[prediction_col].fillna("").tolist()
    predictions = [text if text.strip() else " " for text in predictions]

    pipe_device = 0 if str(device).startswith("cuda") else -1
    clf = pipeline(
        "text-classification",
        model="textattack/roberta-base-CoLA",
        device=pipe_device,
        top_k=None,
    )

    results = clf(predictions, truncation=True, max_length=max_length, batch_size=batch_size)

    acceptability_scores = []
    for res in results:
        label_1_score = next(item["score"] for item in res if item["label"] == "LABEL_1")
        acceptability_scores.append(label_1_score)

    df["acceptability_score"] = acceptability_scores
    return df
