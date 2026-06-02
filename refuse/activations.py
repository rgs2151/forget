import torch as t
from tqdm.auto import tqdm


def build_answered_prompts(df, prompt_factory, answer_fn, question_col="question"):
    prompts = []
    answers = []
    for row in df.itertuples(index=False):
        question = getattr(row, question_col)
        answer = answer_fn(row)
        prompts.append(prompt_factory(question, answer))
        answers.append(answer)
    return prompts, answers


def build_question_prompts(df, prompt_factory, question_col="question"):
    prompts = []
    for row in df.itertuples(index=False):
        prompts.append(prompt_factory(getattr(row, question_col)))
    return prompts


def answer_token_mask(llm, answers, attention_mask, assistant_end_marker):
    tokenizer = llm.tokenizer
    eot_ids = tokenizer.encode(assistant_end_marker, add_special_tokens=False)

    token_mask = t.zeros_like(attention_mask, dtype=t.bool)
    seq_len = attention_mask.shape[1]
    for index, answer in enumerate(answers):
        answer_ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            continue

        valid_len = int(attention_mask[index].sum().item())
        pad_len = seq_len - valid_len
        answer_end = valid_len - len(eot_ids)
        answer_start = answer_end - len(answer_ids)
        if answer_start < 0:
            raise ValueError("Could not locate assistant answer tokens in the batch.")
        token_mask[index, pad_len + answer_start : pad_len + answer_end] = True
    return token_mask


def pool_answer_tokens(acts, token_mask):
    """Mean over the answer tokens, returning [N, L, H]."""
    mask = token_mask[:, None, :, None].to(acts.device, dtype=acts.dtype)
    denom = mask.sum(dim=2).clamp_min(1)
    return (acts * mask).sum(dim=2) / denom


def collect_answer_activations_batched(
    llm,
    prompts,
    answers,
    assistant_end_marker,
    batch_size=32,
    show_progress=False,
    progress_desc="Activation batches",
):
    """Pool answer-token activations per example. Returns [N_examples, num_layers, hidden]."""
    num_layers = len(llm.model.model.layers)
    all_acts = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    ):
        batch_prompts = prompts[start:start + batch_size]
        batch_answers = answers[start:start + batch_size]
        batch = llm.tokenize_batch(batch_prompts)
        token_mask = answer_token_mask(
            llm,
            batch_answers,
            batch["attention_mask"],
            assistant_end_marker=assistant_end_marker,
        )

        llm.reset_all()
        llm.batch_forward(batch_prompts)
        layer_acts = [
            llm.get_last_activations(layer_index).detach().cpu()
            for layer_index in range(num_layers)
        ]
        batch_acts = t.stack(layer_acts, dim=1)
        all_acts.append(pool_answer_tokens(batch_acts, token_mask))
        llm.reset_all()

    return t.cat(all_acts, dim=0)


def collect_activations(pool, concept_to_prompts_answers, batch_size=128, show_progress=True):
    concepts = list(concept_to_prompts_answers.keys())
    if not concepts:
        return {}

    n_shards = min(len(pool.gpu_ids), len(concepts))
    shards = [concepts[i::n_shards] for i in range(n_shards)]

    def run(llm, concept_shard):
        acts = {}
        iterator = tqdm(concept_shard, desc="activations") if show_progress else concept_shard
        for concept in iterator:
            prompts, answers = concept_to_prompts_answers[concept]
            acts[concept] = collect_answer_activations_batched(
                llm,
                prompts,
                answers,
                batch_size=batch_size,
                assistant_end_marker=pool.template.assistant_end_marker,
                show_progress=False,
            )
        return acts

    merged = {}
    for shard_acts in pool.map(run, shards):
        merged.update(shard_acts)
    return merged


def cached_concept_activations(
    pool,
    df,
    prompt_fn,
    answer_fn,
    acts_path,
    concept_col="concept",
    batch_size=64,
    show_progress=True,
):
    if acts_path.exists():
        return t.load(acts_path)

    concept_to_prompts_answers = {}
    for concept, frame in df.groupby(concept_col, sort=False):
        frame = frame.reset_index(drop=True)
        prompts, answers = [], []
        for row in frame.itertuples(index=False):
            answer = answer_fn(row)
            prompts.append(prompt_fn(row, answer))
            answers.append(answer)
        concept_to_prompts_answers[concept] = (prompts, answers)

    acts = collect_activations(
        pool,
        concept_to_prompts_answers,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    t.save(acts, acts_path)
    return acts
