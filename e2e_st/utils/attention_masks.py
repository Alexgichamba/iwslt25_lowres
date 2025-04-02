import torch
from typing import Optional

def key_padding_mask(padded_input: torch.Tensor,
                    input_lengths: Optional[torch.Tensor] = None,
                    pad_idx: Optional[int] = 2) -> torch.Tensor:

    """ Create a mask to identify non-padding positions.

    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: (Optional) the actual lengths of each sequence before padding, shape (N,).
        pad_idx: (Optional) the index used for padding tokens, to be ignored during attention.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """

    # if input is a 2D tensor, (N, T), add an extra dimension
    if padded_input.dim() == 2:
        padded_input = padded_input.unsqueeze(-1)

    if input_lengths is not None:
        # If lengths are provided, use them to create the mask
        N, T, _ = padded_input.shape
        mask = torch.ones((N, T), dtype=torch.bool)  # Initialize mask to all True (padding)
        for i in range(N):
            mask[i, :input_lengths[i]] = False       # Set non-padding positions to False
    else:
        mask = (padded_input.squeeze(-1) == pad_idx)

    return mask


def causal_mask(input: torch.Tensor) -> torch.Tensor:
    """
    Create an attention mask for causal self-attention based on input lengths.

    Args:
        input (torch.Tensor): The input tensor of shape (N, T, *).

    Returns:
        attn_mask (torch.Tensor): The causal self-attention mask of shape (T, T)
    """
    T = input.shape[1]  # seq_len
    attn_mask   = torch.zeros(T, T, dtype=torch.bool)  # Shape (T, T)
    causal_mask = ~torch.tril(torch.ones(T, T)).bool() # Lower triangular matrix
    attn_mask   = attn_mask | causal_mask
    # Return single-head attention mask without expanding for multi-head
    return attn_mask