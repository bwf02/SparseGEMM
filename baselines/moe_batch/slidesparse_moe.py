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
    def __init__(self, weight: torch.Tensor, *, offload_source: bool = False):
        if weight.dtype != torch.bfloat16 or weight.ndim != 3 or not weight.is_cuda:
            raise ValueError("SlideSparse serving requires CUDA BF16 [experts,N,K]")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SlideSparse weights must be prepared before capture")
        self.weight = prune_2_of_8(weight.detach())
        self.device = weight.device
        self.plans = {
            256: SlideSparseBatch(slide_weight_2_of_8(self.weight), 256, allocate_output=False)
        }
        self.compressed = self.plans[256].compressed
        self.workspace = self.plans[256].workspace
        if offload_source:
            self.weight = self.weight.cpu()

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("SlideSparse requires expert-padded [experts,M,K] input")
        if activation.shape[0] != self.weight.shape[0] or activation.shape[2] != self.weight.shape[2]:
            raise ValueError("SlideSparse activation and weight dimensions disagree")
        valid_m = activation.shape[1]
        m = (valid_m + 15) // 16 * 16
        if m != valid_m:
            activation = torch.nn.functional.pad(activation, (0, 0, 0, m - valid_m))
        if m not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"Warm up SlideSparse expert capacity M={m} before capture")
            expanded = slide_weight_2_of_8(self.weight.to(self.device))
            self.plans[m] = SlideSparseBatch(
                expanded, m, allocate_output=False, compressed_reference=self.compressed,
                workspace_reference=self.workspace,
            )
        output = self.plans[m](slide_activation_2_of_8(activation))
        return output if m == valid_m else output[:, :valid_m]

    def close(self):
        for plan in self.plans.values():
            plan.close()
        self.plans.clear()
