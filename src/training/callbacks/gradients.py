from pytorch_lightning.callbacks import Callback

import torch

class RecordGradients(Callback):
    '''
    Callback to record gradients of the model for each layer independently
    '''
    def __init__(self):
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = torch.linalg.norm(param.grad)
                pl_module.log(f"gradients/{name}_norm", grad_norm)
