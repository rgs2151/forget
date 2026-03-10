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

