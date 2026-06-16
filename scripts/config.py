from pydantic import BaseModel
from typing import Optional

class TrainerConfig(BaseModel):
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    log_every_n_steps: int = 10
    callbacks: Optional[list] = None
