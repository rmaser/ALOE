"""
ALOE image processor — standard ViT-style preprocessing with B-cos channel expansion.

B-cos models expect 6-channel input ``[x, 1-x]`` (on the **normalised** 3-channel
tensor ``x``), matching ``HFTransform`` in ``src/data/transforms/transforms.py`` when
``is_bcos=True``.  Using ``-x`` for the second half was incorrect for training parity
and broke native-Hub k-NN / linear-probe eval.  This processor wraps resize →
normalise → to-tensor and appends the second half, so the pipeline matches
ModelFactory + dataloader code paths.

The processor is serialisable and loadable with
``AutoImageProcessor.from_pretrained``.

Typical usage
-------------
::

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("your-org/aloe-siglip2-base", trust_remote_code=True)
    inputs = proc(images=pil_image, return_tensors="pt")
    # inputs.pixel_values.shape == (1, 6, H, W)

    out = model(**inputs)

Building from a backbone processor
---------------------------------
::

    from src.models.hf.aloe import AloeImageProcessor
    backbone_proc = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    aloe_proc = AloeImageProcessor.from_backbone_processor(backbone_proc)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from transformers import ViTImageProcessor
from transformers.image_processing_utils import BatchFeature


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class AloeImageProcessor(ViTImageProcessor):
    """
    Image processor for all ALOE vision backbones.

    Extends :class:`~transformers.ViTImageProcessor` with an optional
    B-cos 6-channel expansion: the normalised 3-channel tensor ``x`` is expanded to
    ``torch.cat([x, 1.0 - x], dim=channel)``, same as ``HFTransform`` for B-cos training.

    Native ALOE checkpoints should save ``preprocessor_config.json`` with
    ``image_processor_type`` / ``auto_map["AutoImageProcessor"]`` so
    :class:`transformers.AutoImageProcessor` resolves the class when using
    ``trust_remote_code=True``.

    Parameters
    ----------
    expand_to_6ch : bool
        If ``True`` (default), concatenate ``[pixel_values, 1 - pixel_values]``
        along the channel dimension after preprocessing (on the normalised tensor).
        Set to ``False`` for standard 3-channel output (e.g. visualisation).
    All other parameters are forwarded to :class:`~transformers.ViTImageProcessor`
    (``do_resize``, ``size``, ``do_normalize``, ``image_mean``, ``image_std``, …).
    """

    model_input_names = ["pixel_values"]

    def to_dict(self) -> dict[str, Any]:
        """Serialise with explicit type and B-cos flag (see :meth:`~transformers.image_processing_utils.ImageProcessingMixin.to_dict`)."""
        out = super().to_dict()
        out["image_processor_type"] = self.__class__.__name__
        out["expand_to_6ch"] = self.expand_to_6ch
        return out

    def __init__(
        self,
        expand_to_6ch: bool = True,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs: Any,
    ) -> None:
        self.expand_to_6ch = expand_to_6ch
        super().__init__(
            do_resize=do_resize,
            size=size or {"height": 224, "width": 224},
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean if image_mean is not None else _IMAGENET_MEAN,
            image_std=image_std if image_std is not None else _IMAGENET_STD,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )

    def preprocess(
        self,
        images: Union[Any, List[Any]],
        *,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """
        Preprocess one or more images.

        Delegates to :class:`~transformers.ViTImageProcessor` for all
        standard steps, then — if ``expand_to_6ch=True`` — appends ``(1 - pv)``
        (same convention as ``HFTransform`` with ``is_bcos=True``).
        """
        result = super().preprocess(images, return_tensors=return_tensors, **kwargs)

        if self.expand_to_6ch and "pixel_values" in result:
            pv = result["pixel_values"]
            if isinstance(pv, np.ndarray):
                result["pixel_values"] = np.concatenate([pv, 1.0 - pv], axis=1)
            else:
                import torch

                result["pixel_values"] = torch.cat([pv, 1.0 - pv], dim=1)

        return result

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_backbone_processor(
        cls,
        backbone_processor: Any,
        *,
        expand_to_6ch: bool = True,
    ) -> "AloeImageProcessor":
        """
        Build an :class:`AloeImageProcessor` by copying preprocessing
        parameters from an existing backbone processor (e.g. the one loaded
        by ``AutoImageProcessor.from_pretrained("google/siglip2-base-…")``).

        Parameters
        ----------
        backbone_processor :
            Any HF image processor with ``image_mean``, ``image_std``, and
            (optionally) a ``size`` or ``crop_size`` attribute.
        expand_to_6ch :
            Whether to append the 6-channel expansion step (default ``True``).
        """
        mean = getattr(backbone_processor, "image_mean", _IMAGENET_MEAN)
        std  = getattr(backbone_processor, "image_std",  _IMAGENET_STD)

        # Resolve image size from various naming conventions used across HF processors.
        size: Dict[str, int] = {"height": 224, "width": 224}
        for attr in ("size", "crop_size"):
            raw = getattr(backbone_processor, attr, None)
            if isinstance(raw, dict) and "height" in raw and "width" in raw:
                size = {"height": raw["height"], "width": raw["width"]}
                break
            if isinstance(raw, int):
                size = {"height": raw, "width": raw}
                break

        do_rescale = getattr(backbone_processor, "do_rescale", True)
        rescale_factor = getattr(backbone_processor, "rescale_factor", 1 / 255)
        do_normalize = getattr(backbone_processor, "do_normalize", True)
        do_resize = getattr(backbone_processor, "do_resize", True)
        do_convert_rgb = getattr(backbone_processor, "do_convert_rgb", True)

        return cls(
            expand_to_6ch=expand_to_6ch,
            do_resize=do_resize,
            size=size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=list(mean),
            image_std=list(std),
            do_convert_rgb=do_convert_rgb,
        )
