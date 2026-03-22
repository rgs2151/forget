import torch as t

def find_last_subtensor_position(
    tensor: t.Tensor,
    sub_tensor: t.Tensor,
) -> int:
    """
    Find the last position of a sub-tensor in a tensor
    """
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(
    tokens: t.Tensor,
    end_str: t.Tensor,
) -> int:
    """
    Find the last position of a sub-tensor in a tensor
    """
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def find_instruction_end_positions_batch(
    batch_tokens: t.Tensor,
    end_str: t.Tensor,
    attention_mask: t.Tensor,
) -> t.Tensor:
    """Find instruction end position for each sample in a left-padded batch.

    Returns (batch,) tensor of column indices in the padded sequence.
    """
    results = []
    for i in range(batch_tokens.size(0)):
        valid_mask = attention_mask[i].bool()
        valid_tokens = batch_tokens[i][valid_mask]
        pos_in_valid = find_instruction_end_postion(valid_tokens, end_str)
        pad_len = (~valid_mask).sum().item()
        results.append((pos_in_valid if pos_in_valid != -1 else 0) + pad_len)
    return t.tensor(results, device=batch_tokens.device)

