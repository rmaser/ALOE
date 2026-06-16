from typing import Optional
from pydantic import BaseModel

class TransformConfig(BaseModel):
    model_source: str
    model_name: str
    override_resolution: Optional[int] = None
    force_val_transforms_in_train: bool = False
    is_bcos: bool = False
    #: Pass through to ``AutoImageProcessor.from_pretrained`` (required for Hub :class:`AloeImageProcessor`).
    image_processor_trust_remote_code: bool = False
    #: For ALOE processors (no HF fast class), prefer torchvision tensor ops in HFTransform.
    prefer_torchvision_fast_for_aloe: bool = True