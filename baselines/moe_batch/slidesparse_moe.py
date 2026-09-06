"""SlideSparse routed-expert execution for serving integration.

The caller supplies expert-padded activations and owns routing/combination.
Weight pruning is initialization work; activation expansion is always online.
"""

import torch

from .moe_batch_baselines import (
    SlideSparseBatch,
    prune_2_of_8,
    slide_activation_2_of_8,
    slide_weight_2_of_8,
)


class SlideSparseProjection:
    def __init__(self, weight: torch.Tensor):
        if weight.dtype != torch.bfloat16 or weight.ndim != 3 or not weight.is_cuda:
            raise ValueError("SlideSparse serving requires CUDA BF16 [experts,N,K]")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SlideSparse weights must be prepared before capture")
        self.weight = prune_2_of_8(weight.detach())
        self.plans = {}

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("SlideSparse requires expert-padded [experts,M,K] input")
        if activation.shape[0] != self.weight.shape[0] or activation.shape[2] != self.weight.shape[2]:
            raise ValueError("SlideSparse activation and weight dimensions disagree")
        m = activation.shape[1]
        if m not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"Warm up SlideSparse expert capacity M={m} before capture")
            expanded = slide_weight_2_of_8(self.weight)
            self.plans[m] = SlideSparseBatch(expanded, m, allocate_output=False)
        return self.plans[m](slide_activation_2_of_8(activation))

    def close(self):
        for plan in self.plans.values():
            plan.close()
        self.plans.clear()
