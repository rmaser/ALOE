from pytorch_lightning.callbacks import Callback
import torch

class RecordWeights(Callback):
    '''
    Callback to record weight norms of the model for each layer independently
    '''
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # We log every step, but the logger might aggregate or sample
        for name, param in pl_module.named_parameters():
            weight_norm = torch.linalg.norm(param.data)
            weight_mean = param.data.mean()
            weight_std = param.data.std()
            pl_module.log(f"weights/{name}_norm", weight_norm)
            pl_module.log(f"weights/{name}_mean", weight_mean)
            pl_module.log(f"weights/{name}_std", weight_std)
