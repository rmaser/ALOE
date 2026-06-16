from pytorch_lightning.callbacks import Callback

class SetLR(Callback):
    """
    Callback to override learning rate after checkpoint loading.
    This ensures the LR is set even when resuming from a checkpoint.
    """
    def __init__(self, lr): 
        self.lr = lr
    
    def on_fit_start(self, trainer, pl_module):
        # set all optimizers' param group LRs
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = self.lr
        # sync all schedulers' base_lrs and optionally restart epoch count
        for sch in trainer.lr_scheduler_configs:  # PL >=1.7
            s = sch.scheduler
            if hasattr(s, "base_lrs"):
                s.base_lrs = [self.lr for _ in s.base_lrs]
    
    # def on_train_epoch_start(self, trainer, pl_module):
    #     """
    #     Also set LR at the start of each epoch to ensure it persists.
    #     This is a safety measure in case schedulers try to modify it.
    #     """
    #     for opt in trainer.optimizers:
    #         for pg in opt.param_groups:
    #             pg["lr"] = self.lr
