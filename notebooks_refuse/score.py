from bert_score import score as bert_score_fn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


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


def add_acceptability_column(df, prediction_col="model_output", device="cuda"):
    df = df.copy()
    predictions = df[prediction_col].fillna("").tolist()
    
    # Clean up empty strings or pipeline throws an error
    predictions = [text if text.strip() else " " for text in predictions]

    # Initialize the pipeline inside
    # Using device=0 or "cuda:0" for the first GPU if device='cuda'
    # textattack/roberta-base-CoLA outputs LABEL_1 for acceptable and LABEL_0 for unacceptable
    pipe_device = 0 if str(device).startswith("cuda") else -1
    clf = pipeline(
        "text-classification", 
        model="textattack/roberta-base-CoLA", 
        device=pipe_device,
        top_k=None, # This guarantees we get scores for all classes instead of just the top 1
    )
    
    # Process in batches
    results = clf(predictions, truncation=True, max_length=512, batch_size=16)
    
    acceptability_scores = []
    for res in results:
        # `res` is a list of dicts: [{'label': 'LABEL_1', 'score': ...}, {'label': 'LABEL_0', 'score': ...}]
        label_1_score = next(item["score"] for item in res if item["label"] == "LABEL_1")
        acceptability_scores.append(label_1_score)
        
    df["acceptability_score"] = acceptability_scores
    return df