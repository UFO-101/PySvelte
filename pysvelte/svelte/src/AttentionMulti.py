from typing import List, Union
from torchtyping import TensorType as TT

import numpy as np
import torch
import einops
import torchvision.transforms

Tensor = Union[np.ndarray, torch.Tensor]


def init(
    input: Union[str, List[str], TT["batch", "pos"]],
    easy_transformer: torch.nn.Module,
    info_weighted: Tensor = None,
    head_labels=None,
):
    """Visualize the attention patterns for multiple attention heads.

    This component is used to visualize attention patterns from a
    Transformer self-attention module. A version of this component was
    used to generate the attention explorer seen here:
    https://transformer-circuits.pub/2021/framework/2L_HP_normal.html
    and linked from our paper:
    https://transformer-circuits.pub/2021/framework/index.html

    Args:
      tokens: a list of of strings representing tokens
      attention: A [N, N, H] array representing attention probabilities,
        where N is the number of tokens and H is the number of heads
        (or analogous value like number of NMF factors).

        Attention weights are expected to be in [0, 1].

      info_weighted: (optional) A [N, N, H] array represented
        re-weighted attention patterns. If provided, the component
        will allow toggling between this pattern and the standard
        pattern.

      head_labels: human readable labels for heads. Optional.

    """
    # assert (
    #     len(tokens) == attention.shape[0]
    # ), "tokens and activations must be same length"
    # assert (
    #     attention.shape[0] == attention.shape[1]
    # ), "first two dimensions of attention must be equal"
    # assert attention.ndim == 3, "attention must be 3D"
    # if head_labels is not None:
    #     assert (
    #         len(head_labels) == attention.shape[-1]
    #     ), "head_labels must correspond to number of attention heads"
    # if info_weighted is not None:
    #     assert (
    #         attention.shape == info_weighted.shape
    #     ), "info_weighted must be the same shape as attention"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = len(input)
    print("input", input, "seq_len", seq_len)
    # output, model_cache = easy_transformer.run_with_cache(input)
    # easy_transformer.reset_hooks()
    # for i in range(len(easy_transformer.cfg.n_layers)):
    # attention = model_cache[f"blocks.0.attn.hook_attn"][0].permute(1, 2, 0)

    # Create a new input sequence which consists of the first half of the original repeated twice
    # This is to make the attention patterns more visible
    induction_input = torch.cat([input[: seq_len // 2]] * 2, dim=0)
    _, induction_cache = easy_transformer.run_with_cache(induction_input)
    induction_attention = induction_cache[f"blocks.0.attn.hook_attn"][0].permute(
        1, 2, 0
    )
    print("induction_attention", induction_attention.shape)

    # induction_scores_array = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    def calc_prefix_matching_score(attn_pattern):
        # Pattern has shape [head_index, query_pos, key_pos]
        print("attn_pattern.shape", attn_pattern.shape)
        prefix_matching_diag = attn_pattern.diagonal(
            1 - (seq_len // 2), dim1=-2, dim2=-1
        )
        print("prefix_matching_diag.shape", prefix_matching_diag.shape)
        prefix_matching_score = einops.reduce(
            prefix_matching_diag, "head_index pos -> head_index", "mean"
        )
        print("prefix_matching_score.shape", prefix_matching_score.shape)
        print("prefix_matching_score", prefix_matching_score)
        return prefix_matching_score

    prefix_matching_scores = calc_prefix_matching_score(
        induction_attention.permute(2, 0, 1)
    )
    attn_out = induction_cache["blocks.0.attn.hook_result"][0]
    unembedded_logits = easy_transformer.unembed(attn_out.permute(1, 0, 2))
    print("unembedded_logits.shape", unembedded_logits.shape)
    print("input.shape", input.shape)
    input_logits = unembedded_logits.index_select(-1, input)

    norize = torchvision.transforms.Normalize(mean=0, std=1)
    normalized_induction_attention = normalize(induction_attention.permute(2, 0, 1))
    normalized_induction_input_logits = normalize(input_logits)
    # Compute the sum of element-wise difference between the normalized induction attention and the normalized induction input logits
    unembedding_score = torch.sum(
        torch.abs(normalized_induction_attention - normalized_induction_input_logits),
        dim=(1, 2),
    )
    # Calculate the maximum possible score
    max_unembedding_score = torch.sum(
        torch.ones(normalized_induction_attention.shape, device=device), dim=(1, 2)
    )
    copying_scores = 1 - unembedding_score / max_unembedding_score

    induction_scores = prefix_matching_scores + copying_scores / 2

    # Foreach token in the input get the unembedded logits for that token

    # Return a dictionary of only the arguments that were not None
    return_args = {
        k: v
        for k, v in [
            ("tokens", [str(x) for x in induction_input.tolist()]),
            ("attention", induction_attention),
            # ("tokens", [str(x) for x in input.tolist()]),
            # ("attention", attention),
            ("info_weighted", info_weighted),
            ("head_labels", head_labels),
        ]
        if v is not None
    }
    return_args["prefix_matching_scores"] = (
        prefix_matching_scores.detach().cpu().tolist()
    )
    return_args["copying_scores"] = copying_scores.detach().cpu().tolist()
    return_args["induction_scores"] = induction_scores.detach().cpu().tolist()
    return return_args
