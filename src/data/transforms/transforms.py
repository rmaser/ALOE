import time
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor
from torchvision.transforms import InterpolationMode
from loguru import logger
from src.data.transforms.config import TransformConfig

class HFTransform:
    def __init__(self, transform_config: TransformConfig):
        self.transform_config = transform_config

    def _processor_trust_kw(self) -> dict:
        if self.transform_config.image_processor_trust_remote_code:
            return {"trust_remote_code": True}
        return {}
    
    def _get_transforms(self):
        override_resolution = self.transform_config.override_resolution
        tr = self._processor_trust_kw()
        
        if override_resolution:
            replacement_resolution = {"height": override_resolution, "width": override_resolution}
            train_processor = self._load_processor_with_retry(
                self.transform_config.model_name, 
                use_fast=True, 
                do_resize=False, 
                size=replacement_resolution,
                crop_size=replacement_resolution,
                do_center_crop=False,
                **tr,
            )
            val_processor = self._load_processor_with_retry(
                self.transform_config.model_name, 
                use_fast=True, 
                do_resize=True, 
                size=replacement_resolution,
                crop_size=replacement_resolution,
                do_center_crop=True,
                **tr,
            )
        else:
            train_processor = self._load_processor_with_retry(
                self.transform_config.model_name, use_fast=True, do_resize=False, **tr
            )
            val_processor = self._load_processor_with_retry(
                self.transform_config.model_name, use_fast=True, do_resize=True, **tr
            )
        
        logger.info(f"Using HuggingFace processor: {self.transform_config.model_name}")

        train_tfms = self._build_modern_augmentations(train_processor)
        
        if self.transform_config.force_val_transforms_in_train:
            logger.info("Forcing val transforms in train: returning validation processor instead.")
            return (
                {"tfms": None, "processor": val_processor},
                {"tfms": None, "processor": val_processor}
            )
            
        return (
            {"tfms": train_tfms, "processor": train_processor},
            {"tfms": None, "processor": val_processor}
        )

    @staticmethod
    def _is_aloe_processor(processor) -> bool:
        return processor.__class__.__name__ == "AloeImageProcessor"

    @staticmethod
    def _resolve_size_conf(conf, fallback_side: int = 224) -> tuple[int, int]:
        if isinstance(conf, dict):
            if "height" in conf and "width" in conf:
                return int(conf["height"]), int(conf["width"])
            if "shortest_edge" in conf:
                side = int(conf["shortest_edge"])
                return side, side
        if isinstance(conf, int):
            return int(conf), int(conf)
        return fallback_side, fallback_side

    def _fast_preprocess_single(self, im, processor):
        if hasattr(im, "convert") and im.mode != "RGB":
            im = im.convert("RGB")

        if isinstance(im, torch.Tensor):
            x = im
            if x.ndim == 4:
                x = x.squeeze(0)
            if x.dtype == torch.uint8 or x.max().item() > 1.0:
                x = x.float() / 255.0
            else:
                x = x.float()
        else:
            x = TF.pil_to_tensor(im).float() / 255.0

        if bool(getattr(processor, "do_resize", True)):
            h, w = self._resolve_size_conf(getattr(processor, "size", {"height": 224, "width": 224}))
            x = TF.resize(x, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)

        if bool(getattr(processor, "do_center_crop", False)):
            ch, cw = self._resolve_size_conf(
                getattr(processor, "crop_size", getattr(processor, "size", {"height": 224, "width": 224}))
            )
            x = TF.center_crop(x, [ch, cw])

        if bool(getattr(processor, "do_normalize", True)):
            mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
            std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
            x = TF.normalize(x, mean=mean, std=std)

        return x

    def _process_images(self, processed_imgs, processor):
        use_torchvision_fast = (
            self.transform_config.prefer_torchvision_fast_for_aloe
            and self._is_aloe_processor(processor)
        )
        if use_torchvision_fast:
            tensors = [self._fast_preprocess_single(im, processor) for im in processed_imgs]
            return torch.stack(tensors, dim=0)
        return processor(images=processed_imgs, return_tensors="pt")["pixel_values"]

    def get_transform_wrappers(self):
        train_conf, val_conf = self._get_transforms()
        train_wrapper = self._make_transform_wrapper(train_conf["processor"], train_conf["tfms"])
        val_wrapper = self._make_transform_wrapper(val_conf["processor"], val_conf["tfms"])
        return train_wrapper, val_wrapper

    def _build_modern_augmentations(self, processor):
        """
        Builds a standard ViT augmentation pipeline (RRC + Flip).
        Extracts target resolution directly from the processor config.
        """
        
        size_conf = processor.size
        if "height" in size_conf:
            target_size = size_conf["height"]
        elif "shortest_edge" in size_conf:
            target_size = size_conf["shortest_edge"]
        else:
            raise Exception(f"Could not determine size from {size_conf}")
        
        logger.info(f"Target size: {target_size}")

        return transforms.Compose([
            transforms.RandomResizedCrop(
                target_size, 
                scale=(0.08, 1.0), # Standard ImageNet scale
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(),
        ])

    def _load_processor_with_retry(self, name, max_retries=5, **kwargs):
        # ALOE custom processor has no HF fast implementation; forcing use_fast=False
        # avoids the fallback warning and redundant dispatch.
        if "use_fast" in kwargs and str(name).startswith("rmaser/aloe-"):
            kwargs["use_fast"] = False
        for attempt in range(max_retries):
            try:
                return AutoImageProcessor.from_pretrained(name, **kwargs)
            except (OSError, TypeError) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(random.uniform(1, 3) * (attempt + 1))

    def _make_transform_wrapper(self, processor, train_tfms=None):
        def transform_wrapper(example):
            keys = ["image", "img", "jpg", "image.png"]
            val = next((example.pop(k) for k in keys if k in example), None)
            
            if val is None:
                raise KeyError(f"No image key found in {list(example.keys())}")
            
            # Clean junk
            [example.pop(k) for k in list(example.keys()) if k not in ["label", "variant", "fine_label", "coarse_label"]]

            # Augment
            images = val if isinstance(val, list) else [val]
            processed_imgs = []
            for im in images:
                if hasattr(im, "convert") and im.mode != "RGB":
                    im = im.convert("RGB")
                if train_tfms:
                    im = train_tfms(im)
                processed_imgs.append(im)

            # Process
            pixel_values = self._process_images(processed_imgs, processor)
            if not isinstance(val, list):
                pixel_values = pixel_values.squeeze(0)

            # Concatenate inverse for Bcos models only
            if self.transform_config.is_bcos:
                if pixel_values.shape[0] == 3:
                    pixel_inverse = (1.0 - pixel_values).detach()
                    pixel_values = torch.cat([pixel_values, pixel_inverse], dim=0)
                elif len(pixel_values.shape) > 3 and pixel_values.shape[-3] == 3:
                    pixel_inverse = (1.0 - pixel_values).detach()
                    pixel_values = torch.cat([pixel_values, pixel_inverse], dim=-3)

            
            example['image'] = pixel_values
            
            # Label fix
            if "variant" in example:
                example['label'] = example.pop("variant")
            if "fine_label" in example:
                example['label'] = example.pop("fine_label")
            if "coarse_label" in example and "label" not in example:
                example['label'] = example.pop("coarse_label")
            return example
            
        return transform_wrapper