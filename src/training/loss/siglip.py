import torch
import torch.nn.functional as F
from torch import nn

from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
from torch import Tensor
from typing import Union

class SiglipLoss(nn.Module):
    """
    SigLIP-style pairwise sigmoid loss for vision encoder distillation.
    Student and teacher image features are compared in a batch.
    """
    def __init__(self, init_logit_scale=torch.log(torch.tensor(10.0)), init_logit_bias=-10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(init_logit_scale.clone())
        self.logit_bias  = nn.Parameter(torch.tensor(init_logit_bias, dtype=torch.float32))

    @staticmethod
    def _l2_normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        student_feats: Union[Float[Tensor, "batch 1 embedding"]], 
        teacher_feats: Union[Float[Tensor, "batch 1 embedding"]]
    ) -> Float[Tensor, ""]: 
        # squeeze the singleton dimension
        student = self._l2_normalize(student_feats).view(student_feats.shape[0], -1)
        teacher = self._l2_normalize(teacher_feats).view(teacher_feats.shape[0], -1)

        t = self.logit_scale.exp()
        logits = t * (student @ teacher.t()) + self.logit_bias  # (N, N)

        N = student.size(0)
        # +1 for matching pairs (diagonal), -1 for non-matching pairs
        labels = -torch.ones_like(logits)
        labels.fill_(-1.0)
        labels.diagonal().fill_(1.0)

        # SigLIP loss: -log(sigmoid(labels * logits)), averaged over batch
        loss = F.softplus(-labels * logits).sum() / N
        return loss

