import torch as t
from tqdm.auto import tqdm


def build_answered_chats(df, chat_factory, answer_fn, question_col="question"):
    chats = []
    answers = []
    for row in df.itertuples(index=False):
        chat = chat_factory()
        chat.add_user_message(getattr(row, question_col))
        answer = answer_fn(row)
        chat.add_assistant_message(answer)
        chats.append(chat)
        answers.append(answer)
    return chats, answers


def build_question_chats(df, chat_factory, question_col="question"):
    chats = []
    for row in df.itertuples(index=False):
        chat = chat_factory()
        chat.add_user_message(getattr(row, question_col))
        chats.append(chat)
    return chats


def answer_token_mask(llm, answers, attention_mask):
    tokenizer = llm.tokenizer.tokenizer
    assistant_header_ids = tokenizer.encode(llm.tokenizer.B_ASSISTANT, add_special_tokens=False)
    eot_ids = tokenizer.encode(llm.tokenizer.E_ASSISTANT, add_special_tokens=False)

    token_mask = t.zeros_like(attention_mask, dtype=t.bool)
    seq_len = attention_mask.shape[1]
    for index, answer in enumerate(answers):
        answer_ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            continue

        valid_len = int(attention_mask[index].sum().item())
        pad_len = seq_len - valid_len
        answer_end = valid_len - len(assistant_header_ids) - len(eot_ids)
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
    chats,
    answers,
    batch_size=32,
    pool_tokens=False,
    return_token_mask=False,
    show_progress=False,
    progress_desc="Activation batches",
):
    tokenizer = llm.tokenizer.tokenizer
    answer_token_counts = [
        len(tokenizer.encode(answer.strip(), add_special_tokens=False))
        for answer in answers
    ]

    num_layers = len(llm.model.model.layers)
    max_answer_tokens = max(answer_token_counts) if answer_token_counts else 0
    all_acts = []
    all_masks = []
    for start in tqdm(
        range(0, len(chats), batch_size),
        desc=progress_desc,
        leave=False,
        disable=not show_progress,
    ):
        batch_chats = chats[start:start + batch_size]
        batch_answers = answers[start:start + batch_size]
        batch = llm.tokenizer.tokenize_batch(batch_chats)
        token_mask = answer_token_mask(llm, batch_answers, batch["attention_mask"])

        llm.reset_all()
        llm.forward_from_chats(batch_chats)
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
    chat_factory_fn,
    answer_fn,
    batch_size=128,
    show_progress=True,
    progress_desc="Grouped activations",
):
    activations = {}
    masks = {}
    keys = list(frames_by_key)
    for key in tqdm(keys, desc=progress_desc, disable=not show_progress):
        frame = frames_by_key[key].reset_index(drop=True)
        chats, answers = build_answered_chats(frame, lambda: chat_factory_fn(key), answer_fn)
        activations[key], masks[key] = collect_answer_activations_batched(
            llm,
            chats,
            answers,
            batch_size=batch_size,
            return_token_mask=True,
        )
    return activations, masks
