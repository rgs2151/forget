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


def answer_token_mask(llm, answers, attention_mask, assistant_end_marker="<|eot_id|>"):
    tokenizer = llm.tokenizer
    assistant_header_ids = tokenizer.encode(llm.instruction_end_marker, add_special_tokens=False)
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


def pack_selected_tokens(acts, token_mask, max_tokens):
    batch_size, num_layers, _, hidden_dim = acts.shape
    packed = acts.new_zeros((batch_size, num_layers, max_tokens, hidden_dim))
    packed_mask = t.zeros((batch_size, max_tokens), dtype=t.bool)
    for index in range(batch_size):
        selected = acts[index, :, token_mask[index], :]
        n_tokens = selected.shape[1]
        if n_tokens == 0:
            continue
        packed[index, :, :n_tokens, :] = selected
        packed_mask[index, :n_tokens] = True
    return packed, packed_mask


def masked_mean_acts(acts, token_mask=None):
    if token_mask is None:
        if acts.shape[2] == 1:
            return acts
        return acts.mean(dim=2, keepdim=True)

    mask = token_mask[:, None, :, None].to(acts.device, dtype=acts.dtype)
    denom = mask.sum(dim=2, keepdim=True).clamp_min(1)
    return (acts * mask).sum(dim=2, keepdim=True) / denom


def flatten_token_rows(acts, token_mask=None):
    if token_mask is None:
        if acts.shape[2] == 1:
            return acts[:, :, 0, :]
        n_items, n_layers, seq_len, hidden = acts.shape
        return acts.permute(0, 2, 1, 3).reshape(n_items * seq_len, n_layers, hidden)

    token_mask = token_mask.to(acts.device).bool()
    token_first = acts.permute(0, 2, 1, 3)
    return token_first[token_mask]


def pool_activation_dict(act_dict, mask_dict):
    return {key: masked_mean_acts(act_dict[key], mask_dict[key]) for key in act_dict}


def collect_answer_activations_batched(
    llm,
    prompts,
    answers,
    batch_size=32,
    assistant_end_marker="<|eot_id|>",
    pool_tokens=False,
    return_token_mask=False,
    show_progress=False,
    progress_desc="Activation batches",
):
    tokenizer = llm.tokenizer
    answer_token_counts = [
        len(tokenizer.encode(answer.strip(), add_special_tokens=False))
        for answer in answers
    ]

    num_layers = len(llm.model.model.layers)
    max_answer_tokens = max(answer_token_counts) if answer_token_counts else 0
    all_acts = []
    all_masks = []
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
        layer_acts = []
        for layer_index in range(num_layers):
            layer_acts.append(llm.get_last_activations(layer_index).detach().cpu())
        batch_acts = t.stack(layer_acts, dim=1)
        batch_acts, token_mask = pack_selected_tokens(batch_acts, token_mask, max_answer_tokens)
        if pool_tokens:
            batch_acts = masked_mean_acts(batch_acts, token_mask)
        all_acts.append(batch_acts)
        all_masks.append(token_mask)
        llm.reset_all()

    acts = t.cat(all_acts, dim=0)
    token_mask = t.cat(all_masks, dim=0)
    if return_token_mask:
        return acts, token_mask
    return acts


def collect_grouped_activations(
    llm,
    frames_by_key,
    prompt_factory_fn,
    answer_fn,
    batch_size=128,
    assistant_end_marker="<|eot_id|>",
    show_progress=True,
    progress_desc="Grouped activations",
):
    prompts_by_key = {}
    answers_by_key = {}
    keys = list(frames_by_key)
    for key in tqdm(keys, desc=progress_desc, disable=not show_progress):
        frame = frames_by_key[key].reset_index(drop=True)
        prompts_by_key[key], answers_by_key[key] = build_answered_prompts(
            frame,
            lambda question, answer: prompt_factory_fn(key, question, answer),
            answer_fn,
        )
    return collect_grouped_activations_from_prompts(
        llm,
        prompts_by_key,
        answers_by_key,
        batch_size=batch_size,
        assistant_end_marker=assistant_end_marker,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def collect_concept_activations(
    llm,
    df,
    prompt_fn,
    answer_fn,
    concept_col="concept",
    batch_size=128,
    assistant_end_marker="<|eot_id|>",
    show_progress=True,
    progress_desc="Grouped activations",
):
    activations = {}
    masks = {}
    grouped = list(df.groupby(concept_col, sort=False))

    for concept, frame in tqdm(grouped, desc=progress_desc, disable=not show_progress):
        frame = frame.reset_index(drop=True)
        prompts = []
        answers = []
        for row in frame.itertuples(index=False):
            answer = answer_fn(row)
            prompts.append(prompt_fn(row, answer))
            answers.append(answer)
        activations[concept], masks[concept] = collect_answer_activations_batched(
            llm,
            prompts,
            answers,
            batch_size=batch_size,
            assistant_end_marker=assistant_end_marker,
            return_token_mask=True,
        )
    return activations, masks


def collect_grouped_activations_from_prompts(
    llm,
    prompts_by_key,
    answers_by_key,
    batch_size=128,
    assistant_end_marker="<|eot_id|>",
    show_progress=True,
    progress_desc="Grouped activations",
):
    activations = {}
    masks = {}
    keys = list(prompts_by_key)
    if set(keys) != set(answers_by_key):
        raise ValueError("prompts_by_key and answers_by_key must have the same keys.")
    for key in keys:
        if len(prompts_by_key[key]) != len(answers_by_key[key]):
            raise ValueError(f"prompts and answers must have the same length for key {key!r}.")

    for key in tqdm(keys, desc=progress_desc, disable=not show_progress):
        activations[key], masks[key] = collect_answer_activations_batched(
            llm,
            prompts_by_key[key],
            answers_by_key[key],
            batch_size=batch_size,
            assistant_end_marker=assistant_end_marker,
            return_token_mask=True,
        )
    return activations, masks
