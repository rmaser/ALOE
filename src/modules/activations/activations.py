from bcos.modules import DetachableModule
import torch

class DetachableGELU(DetachableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Ignore extra args like detach_output
        pass

    def forward(self, x):
        gate = 0.5 * (1 + torch.erf(x/torch.sqrt(torch.tensor(2.0))))
        if self.detach:
            # print(f"DEBUG: DetachableGELU detaching! {self.detach}")
            gate = gate.detach()
        return gate * x