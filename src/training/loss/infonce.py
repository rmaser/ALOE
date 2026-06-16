import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch import nn


class InfoNCE(nn.Module):
    """
    Computes the InfoNCE loss for contrastive learning.
    
    Args:
        features1 (torch.Tensor): Embeddings from the first view 
                                 (e.g., query), shape [batch_size, embed_dim].
        features2 (torch.Tensor): Embeddings from the second view
                                 (e.g., key), shape [batch_size, embed_dim].
        temperature (float): Temperature parameter for scaling logits.
    """
                                 
    def __init__(self, temperature=0.1):
        '''
        Temperature parameter to scale the logits. Set to 0.1 by default (shown to work well in the SimCLR paper).
        '''
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        features1: Float[Tensor, "batch embedding"],
        features2: Float[Tensor, "batch embedding"],
    ) -> Float[Tensor, ""]:
        # 1. Normalize the features so dot product is cosine similarity
        f1 = F.normalize(features1, dim=1)
        f2 = F.normalize(features2, dim=1)
    
        # 2. Calculate the similarity matrix (logits)
        # The dot product of normalized vectors is the cosine similarity.
        # We want to pull f1[i] and f2[i] together (positive pairs).
        # We want to push f1[i] and f2[j] (where i != j) apart (negative pairs).
        # Shape: [batch_size, batch_size]
        logits = f1 @ f2.T / self.temperature
        
        # 3. Create the ground-truth labels.
        # The positive pair for f1[i] is f2[i], which is at index `i` in the logits matrix.
        # So the labels are just [0, 1, 2, ..., batch_size-1].
        batch_size = f1.shape[0]
        labels = torch.arange(batch_size, device=features1.device)
        
        # 4. Calculate the cross-entropy loss in both directions
        # Loss for f1 predicting f2
        loss_i_j = F.cross_entropy(logits, labels)
        
        # Loss for f2 predicting f1 (just transpose the logits)
        loss_j_i = F.cross_entropy(logits.T, labels)
        
        # 5. Average the two losses
        loss = (loss_i_j + loss_j_i) / 2
        
        return loss
