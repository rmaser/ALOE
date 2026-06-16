import time
from pytorch_lightning.callbacks import Callback
from typing import Optional

class TimerCallback(Callback):
    def __init__(self):
        self.epoch_start_time: Optional[float] = None
        self.train_start_time: Optional[float] = None
        self.validation_start_time: Optional[float] = None
        self.train_time: float = 0.0
        self.step_start_time: Optional[float] = None
        self.step_time_total: float = 0.0
        self.step_count: int = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.train_start_time = time.time()
        self.train_time = 0.0
        self.step_start_time = None
        self.step_time_total = 0.0
        self.step_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            pl_module.log("train_step_time", step_time, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.step_time_total += step_time
            self.step_count += 1
            self.step_start_time = None

    def on_validation_start(self, trainer, pl_module):
        # Pause training timer when validation starts
        if self.train_start_time is not None:
            self.train_time += time.time() - self.train_start_time
            self.train_start_time = None
        self.step_start_time = None
        
        self.validation_start_time = time.time()

    def on_validation_end(self, trainer, pl_module):
        # Log validation time using direct logger access
        if self.validation_start_time is not None:
            validation_time = time.time() - self.validation_start_time

            if trainer.logger:
                trainer.logger.log_metrics({
                    "validation_time": validation_time
                }, step=trainer.global_step)
        
        # Resume training timer after validation
        self.train_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Capture any remaining training time
        if self.train_start_time is not None:
            self.train_time += time.time() - self.train_start_time
        
        # These hooks are safe for pl_module.log()
        pl_module.log("train_time", self.train_time, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.step_count > 0:
            average_step_time = self.step_time_total / self.step_count
            pl_module.log("train_step_time_avg", average_step_time, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.epoch_start_time is not None:
            total_epoch_time = time.time() - self.epoch_start_time
            pl_module.log("epoch_total_time", total_epoch_time, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


class StepTimerCallback(Callback):
    def __init__(self):
        self.step_start_time: Optional[float] = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            pl_module.log("train/step_time", step_time, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.step_start_time = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.step_start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            pl_module.log("val/step_time", step_time, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.step_start_time = None
