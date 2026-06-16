# Some parts written by Gemini 3 Pro
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
from open_clip.transformer import VisionTransformer
from open_clip.timm_model import TimmModel
from einops import rearrange

from .utils import hooked_resblock_forward, \
    hooked_resblock_timm_forward, \
    hooked_attentional_pooler_timm_forward, \
    vit_dynamic_size_forward, \
    min_max, \
    hooked_torch_multi_head_attention_forward
from .attention_core import compute_chefer_from_adapter, compute_legrad_from_adapter


def _legrad_vision_root(model: nn.Module) -> nn.Module:
    """Vision tower for HF ``*ForImageClassification`` (otherwise ``model``)."""
    return getattr(model, "vision_model", model)


def _legrad_model_config(model: nn.Module):
    cfg = getattr(model, "config", None)
    if cfg is not None:
        return cfg
    vision = getattr(model, "vision_model", None)
    return getattr(vision, "config", None)


def _legrad_is_aloe_tower(module: nn.Module) -> bool:
    return type(module).__name__ == "AloeVisionTransformer"


def _legrad_is_hf_vit_config(cfg) -> bool:
    return cfg is not None and getattr(cfg, "model_type", "") == "vit"


def _legrad_is_hf_siglip_config(cfg) -> bool:
    return cfg is not None and getattr(cfg, "model_type", "") in {
        "siglip",
        "siglip2",
        "siglip_vision_model",
        "siglip2_vision_model",
    }


def _legrad_is_hf_siglip_module(model: nn.Module, cfg) -> bool:
    vision = _legrad_vision_root(model)
    vision_type = type(vision).__name__
    return "SiglipVision" in vision_type or _legrad_is_hf_siglip_config(cfg)


def _legrad_encoder_layers_seq(vision: nn.Module) -> nn.Module:
    if hasattr(vision, "encoder") and hasattr(vision.encoder, "layers"):
        return vision.encoder.layers
    if hasattr(vision, "layer"):
        return vision.layer
    enc = vision.encoder
    if hasattr(enc, "layer"):
        return enc.layer
    raise ValueError(f"Cannot find encoder layer list on {type(vision).__name__}")


def _legrad_layer_attention(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    raise AttributeError(f"No attention submodule on {type(layer).__name__}")


def _legrad_prefix_tokens_for_attention(vision: nn.Module) -> int:
    cfg = getattr(vision, "config", None)
    if cfg is None:
        return 1
    if _legrad_is_aloe_tower(vision):
        n_reg = int(getattr(cfg, "aloe_num_registers", 0))
        n_cls = 1 if getattr(cfg, "aloe_cls_token", True) else 0
        return n_reg + n_cls
    if _legrad_is_hf_siglip_config(cfg):
        return 0
    return int(getattr(cfg, "num_register_tokens", 0)) + 1


def _legrad_cls_attention_row(vision: nn.Module) -> int:
    """Query-row index for CLS attention maps (HF DINOv3 and native ALOE both use index 0)."""
    return 0


def _legrad_patch_size(vision: nn.Module) -> int:
    cfg = getattr(vision, "config", None)
    if cfg is None:
        raise ValueError("Missing config on vision module")
    ps = getattr(cfg, "aloe_patch_size", None)
    if ps is not None:
        return int(ps)
    p = cfg.patch_size
    return int(p[0] if isinstance(p, (list, tuple)) else p)


def _legrad_target_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t_indices = target
    if t_indices.device != logits.device:
        t_indices = t_indices.to(logits.device)
    if t_indices.ndim == 0 or t_indices.numel() == 1:
        indices = t_indices.view(-1).expand(logits.shape[0])
    else:
        indices = t_indices
    return logits.gather(1, indices.view(-1, 1)).sum()


def _legrad_extract_logits(output) -> torch.Tensor:
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    if isinstance(output, tuple):
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    raise TypeError(f"Cannot extract logits from {type(output)!r}")


class _LeWrapperClassifierAdapter:
    """Adapter from concrete classifier modules to shared LeGrad/Chefer cores."""

    def __init__(self, wrapper: "LeWrapper") -> None:
        self.wrapper = wrapper
        self.root = wrapper.model
        self.vision = _legrad_vision_root(self.root)
        self.outer = self.root if getattr(self.root, "vision_model", None) is not None else self.vision
        self.patch_size = wrapper.patch_size
        self._logits: torch.Tensor | None = None

    def run(self, image: torch.Tensor) -> None:
        self.wrapper.captured_features = {}
        self._clear_attention_maps()
        out = self.root(image, output_attentions=True)
        self._logits = _legrad_extract_logits(out)

    def _clear_attention_maps(self) -> None:
        for layer_idx in self.all_layer_indices():
            attn = _legrad_layer_attention(self.layers()[layer_idx])
            if hasattr(attn, "attention_maps"):
                attn.attention_maps = None
        head_attn = self._pool_attention_module()
        if head_attn is not None and hasattr(head_attn, "attention_maps"):
            head_attn.attention_maps = None

    def layers(self) -> nn.Module:
        return _legrad_encoder_layers_seq(self.vision)

    def layer_indices(self) -> list[int]:
        return list(range(self.wrapper.starting_depth, len(self.layers())))

    def all_layer_indices(self) -> list[int]:
        return list(range(len(self.layers())))

    def attention_map(self, layer_idx: int) -> torch.Tensor:
        attn = _legrad_layer_attention(self.layers()[layer_idx])
        attn_map = getattr(attn, "attention_maps", None)
        if attn_map is None:
            raise ValueError(f"Attention map for layer {layer_idx} was not captured")
        return attn_map

    def hidden_state(self, layer_idx: int) -> torch.Tensor:
        try:
            return self.wrapper.captured_features[layer_idx]
        except KeyError as exc:
            raise ValueError(
                f"Layer {layer_idx} hidden state was not captured. "
                f"Have: {list(self.wrapper.captured_features.keys())}"
            ) from exc

    def uses_attention_pooling(self) -> bool:
        return self._pool_attention_module() is not None and self.wrapper.model_type == "hf_siglip"

    def patch_indices(self, sequence_length: int) -> torch.Tensor:
        num_prefix = _legrad_prefix_tokens_for_attention(self.vision)
        if sequence_length <= num_prefix:
            raise ValueError(f"Sequence length {sequence_length} is not larger than prefix length {num_prefix}")
        return torch.arange(num_prefix, sequence_length, device=self._device())

    def cls_attention_row(self) -> int:
        return _legrad_cls_attention_row(self.vision)

    def target_score(self, target: torch.Tensor) -> torch.Tensor:
        if self._logits is None:
            raise ValueError("Adapter has not run a forward pass yet")
        return _legrad_target_score(self._logits, target)

    def score_from_hidden(self, hidden_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled_from_hidden(hidden_state, output_attentions=False)
        logits = self._logits_from_pooled(pooled)
        return _legrad_target_score(logits, target)

    def score_and_pool_attention_from_hidden(
        self,
        hidden_state: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_attn = self._pool_attention_module()
        if head_attn is None:
            raise ValueError("Model does not expose an attention-pooling head")
        if hasattr(head_attn, "attention_maps"):
            head_attn.attention_maps = None
        pooled = self._pooled_from_hidden(hidden_state, output_attentions=True)
        attn_map = getattr(head_attn, "attention_maps", None)
        if attn_map is None:
            raise ValueError("Pooling-head attention map was not captured")
        logits = self._logits_from_pooled(pooled)
        return _legrad_target_score(logits, target), attn_map

    def final_score_and_pool_attention(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = self.all_layer_indices()[-1]
        return self.score_and_pool_attention_from_hidden(self.hidden_state(layer_idx), target)

    def final_pool_attention(self) -> torch.Tensor:
        head_attn = self._pool_attention_module()
        if head_attn is None:
            raise ValueError("Model does not expose an attention-pooling head")
        attn_map = getattr(head_attn, "attention_maps", None)
        if attn_map is None:
            raise ValueError("Final pooling-head attention map was not captured")
        return attn_map

    def _device(self) -> torch.device:
        if self._logits is not None:
            return self._logits.device
        try:
            return next(self.root.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _pool_attention_module(self) -> nn.Module | None:
        head = getattr(self.vision, "head", None)
        if head is None:
            return None
        attn = getattr(head, "attention", None)
        return attn if isinstance(attn, nn.Module) else None

    def _post_norm(self, hidden_state: torch.Tensor) -> torch.Tensor:
        post_ln = getattr(self.vision, "post_layernorm", None)
        if post_ln is not None:
            return post_ln(hidden_state)
        norm = (
            getattr(self.vision, "norm", None)
            or getattr(self.vision, "layernorm", None)
            or getattr(self.vision, "final_layer_norm", None)
        )
        if norm is not None:
            return norm(hidden_state)
        return hidden_state

    def _pooled_from_hidden(self, hidden_state: torch.Tensor, *, output_attentions: bool) -> torch.Tensor:
        feat = self._post_norm(hidden_state)
        head = getattr(self.vision, "head", None)
        if head is None:
            pooled = feat[:, self.cls_attention_row(), :]
        elif output_attentions and isinstance(getattr(head, "attention", None), nn.MultiheadAttention):
            pooled = self._torch_mha_pool(head, feat)
        else:
            try:
                pooled = head(feat, None, output_attentions=output_attentions)
            except TypeError:
                try:
                    pooled = head(feat, None)
                except TypeError:
                    pooled = head(feat)
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            if pooled.ndim == 3:
                pooled = pooled[:, self.cls_attention_row(), :]
        return pooled

    def _torch_mha_pool(self, head: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
        attn = head.attention
        batch_size = hidden_state.shape[0]
        query = head.probe.repeat(batch_size, 1, 1)
        key = value = hidden_state
        if not getattr(attn, "batch_first", False):
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads
        if attn.in_proj_weight is None:
            q = F.linear(query, attn.q_proj_weight, attn.in_proj_bias[:embed_dim] if attn.in_proj_bias is not None else None)
            k = F.linear(key, attn.k_proj_weight, attn.in_proj_bias[embed_dim:2 * embed_dim] if attn.in_proj_bias is not None else None)
            v = F.linear(value, attn.v_proj_weight, attn.in_proj_bias[2 * embed_dim:] if attn.in_proj_bias is not None else None)
        else:
            q_w, k_w, v_w = attn.in_proj_weight.chunk(3, dim=0)
            if attn.in_proj_bias is None:
                q_b = k_b = v_b = None
            else:
                q_b, k_b, v_b = attn.in_proj_bias.chunk(3, dim=0)
            q = F.linear(query, q_w, q_b)
            k = F.linear(key, k_w, k_b)
            v = F.linear(value, v_w, v_b)

        if not getattr(attn, "batch_first", False):
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        weights = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1)
        pooled = weights @ v
        pooled = pooled.transpose(1, 2).reshape(batch_size, -1, embed_dim)
        pooled = attn.out_proj(pooled)
        attn.attention_maps = weights

        residual = pooled
        pooled = head.layernorm(pooled)
        pooled = residual + head.mlp(pooled)
        return pooled[:, 0]

    def _logits_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        classifier = getattr(self.outer, "classifier", None) or getattr(self.vision, "classifier", None)
        logit_layer = getattr(self.outer, "logit_layer", None) or getattr(self.vision, "logit_layer", None)
        logits = classifier(pooled) if classifier is not None else pooled
        return logit_layer(logits) if logit_layer is not None else logits


class LeWrapper(nn.Module):
    """
    Wrapper around OpenCLIP to add LeGrad to OpenCLIP's model while keep all the functionalities of the original model.
    """

    def __init__(self, model, layer_index=-2):
        super(LeWrapper, self).__init__()

        # Unwrap torch.compile model if present to ensure hooks work with eager autograd
        if hasattr(model, "_orig_mod"):
            print("LeWrapper: Unwrapping compiled model for explanation.")
            self.model = model._orig_mod
        else:
            self.model = model
        
        # Dictionary to capture layer features during forward pass
        
        # Dictionary to capture layer features during forward pass
        # Keys are layer indices, values are the feature tensors
        self.captured_features = {}

        # ------------ copy of model's attributes and methods ------------
        # for attr in dir(model):
        #     if not attr.startswith('__'):
        #         setattr(self, attr, getattr(model, attr))

        # ------------ activate hooks & gradient ------------
        self._activate_hooks(layer_index=layer_index)

    def _activate_hooks(self, layer_index):
        model = self.model
        model_cfg = _legrad_model_config(model)
        # ------------ identify model's type ------------
        if hasattr(model, 'visual') and isinstance(model.visual, VisionTransformer):
            # --- Activate dynamic image size ---
            model.visual.forward = types.MethodType(vit_dynamic_size_forward, model.visual)
            # Get patch size
            self.patch_size = model.visual.patch_size[0]
            # Get starting depth (in case of negative layer_index)
            self.starting_depth = layer_index if layer_index >= 0 else len(
                model.visual.transformer.resblocks) + layer_index

            if model.visual.attn_pool is None:
                model.model_type = 'clip'
                self._activate_self_attention_hooks()
            else:
                model.model_type = 'coca'
                self._activate_att_pool_hooks(layer_index=layer_index)

        elif hasattr(model, 'visual') and isinstance(model.visual, TimmModel):
            # --- Activate dynamic image size ---
            model.visual.trunk.dynamic_img_size = True
            model.visual.trunk.patch_embed.dynamic_img_size = True
            model.visual.trunk.patch_embed.strict_img_size = False
            model.visual.trunk.patch_embed.flatten = False
            model.visual.trunk.patch_embed.output_fmt = 'NHWC'
            model.model_type = 'timm_siglip'
            # --- Get patch size ---
            self.patch_size = model.visual.trunk.patch_embed.patch_size[0]
            # --- Get starting depth (in case of negative layer_index) ---
            self.starting_depth = layer_index if layer_index >= 0 else len(model.visual.trunk.blocks) + layer_index
            self._activate_timm_attn_pool_hooks(layer_index=layer_index)
            
        elif _legrad_is_hf_siglip_module(model, model_cfg):
            # HF SigLIP / SigLIP2
            self.model_type = 'hf_siglip'
            model.model_type = 'hf_siglip'
            vision = _legrad_vision_root(model)
            model.visual = vision # Alias for easier access if needed, but safer to use vision_model
            
            # Get patch size
            # HF config: patch_size is usually in config
            cfg = getattr(model_cfg, "vision_config", model_cfg)
            if cfg is None:
                cfg = getattr(vision, "config", None)
            if cfg is None:
                raise ValueError("Missing SigLIP vision config")
            self.patch_size = cfg.patch_size
            num_layers = cfg.num_hidden_layers

            self.starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
            self._activate_hf_siglip_hooks(layer_index)

        elif hasattr(model, "vision_model") and _legrad_is_aloe_tower(model.vision_model):
            tower = model.vision_model
            model.visual = tower
            self.patch_size = _legrad_patch_size(tower)
            num_layers = tower.config.num_hidden_layers
            self.starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
            if getattr(tower.config, "aloe_pooler_type", "") == "multihead_attention":
                self.model_type = "hf_siglip"
                model.model_type = "hf_siglip"
                self._activate_hf_siglip_hooks(layer_index)
            else:
                self.model_type = "aloe_hf"
                model.model_type = "aloe_hf"
                self._activate_aloe_encoder_attention_hooks(layer_index)

        elif model_cfg is not None and getattr(model_cfg, 'model_type', '') == 'dinov3_vit':
            # DINOv3
            self.model_type = 'dinov3'
            model.model_type = 'dinov3'
            patch_size = model_cfg.patch_size
            self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
            num_layers = model_cfg.num_hidden_layers
            self.starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
            self._activate_dinov3_hooks(layer_index)

        elif _legrad_is_hf_vit_config(model_cfg):
            # Plain HF ViT supervised classifier wrappers use the same CLS-token
            # encoder self-attention path as DINO-style classifiers.
            self.model_type = 'hf_vit'
            model.model_type = 'hf_vit'
            vision = _legrad_vision_root(model)
            self.patch_size = _legrad_patch_size(vision)
            num_layers = int(getattr(model_cfg, "num_hidden_layers"))
            self.starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
            self._activate_dinov3_hooks(layer_index)

        else:
            raise ValueError(
                f"Model type {type(model)} currently not supported, see legrad.list_pretrained() for a list of available models")

    def _activate_self_attention_hooks(self):
        # ---------- Apply Hooks + Activate/Deactivate gradients ----------
        # Necessary steps to get intermediate representations
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.transformer.resblocks'):
                # get the depth
                depth = int(name.split('visual.transformer.resblocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers ---
        for layer in range(self.starting_depth, len(self.model.visual.transformer.resblocks)):
            self.model.visual.transformer.resblocks[layer].attn.forward = types.MethodType(hooked_torch_multi_head_attention_forward,
                                                                                     self.model.visual.transformer.resblocks[
                                                                                         layer].attn)
            self.model.visual.transformer.resblocks[layer].forward = types.MethodType(hooked_resblock_forward,
                                                                                self.model.visual.transformer.resblocks[
                                                                                    layer])

    def _activate_att_pool_hooks(self, layer_index):
        # ---------- Apply Hooks + Activate/Deactivate gradients ----------
        # Necessary steps to get intermediate representations
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.transformer.resblocks'):
                # get the depth
                depth = int(name.split('visual.transformer.resblocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers ---
        for layer in range(self.starting_depth, len(self.model.visual.transformer.resblocks)):
            self.model.visual.transformer.resblocks[layer].forward = types.MethodType(hooked_resblock_forward,
                                                                                self.model.visual.transformer.resblocks[
                                                                                    layer])
        # --- Apply hook on the attentional pooler ---
        self.model.visual.attn_pool.attn.forward = types.MethodType(hooked_torch_multi_head_attention_forward,
                                                              self.model.visual.attn_pool.attn)

    def _activate_timm_attn_pool_hooks(self, layer_index):
        # --- Deactivate gradient for module that don't need it ---

        # --- Deactivate gradient for module that don't need it ---
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.trunk.attn_pool'):
                param.requires_grad = True
            if name.startswith('visual.trunk.blocks'):
                # get the depth
                depth = int(name.split('visual.trunk.blocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers by modifying the block's forward ---
        for layer in range(self.starting_depth, len(self.visual.trunk.blocks)):
            self.visual.trunk.blocks[layer].forward = types.MethodType(hooked_resblock_timm_forward,
                                                                       self.visual.trunk.blocks[layer])

        self.visual.trunk.attn_pool.forward = types.MethodType(hooked_attentional_pooler_timm_forward,
                                                               self.visual.trunk.attn_pool)
                                                               
    def _activate_hf_siglip_hooks(self, layer_index):
        # HF SigLIP: full vision stack lives on ``model`` or under ``model.vision_model`` (HF IC wrappers).
        vision = _legrad_vision_root(self.model)
        root = self.model
        if hasattr(vision, "config"):
            vision.config._attn_implementation = "eager"
        for module in vision.modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = "eager"

        # --- Deactivate gradient for module that don't need it ---
        for _name, param in vision.named_parameters():
            param.requires_grad = False

            # Standalone vision models may carry a linear head named ``classifier``.
            if "head" in _name or "classifier" in _name:
                param.requires_grad = True

        if root is not vision:
            for name, param in root.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True

        encoder = vision.encoder
        
        # Activate gradients for last layers
        for name, param in encoder.named_parameters():
            # heuristic for layer depth
            if 'layers' in name:
                try:
                    parts = name.split('layers.')
                    if len(parts) > 1:
                        depth = int(parts[1].split('.')[0])
                        if depth >= self.starting_depth:
                            param.requires_grad = True
                except (ValueError, IndexError):
                    pass

        def get_attention_hook(module):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    module.attention_maps = output[1]

            return hook

        def get_attention_pre_hook(module, args, kwargs):
            kw = dict(kwargs) if kwargs else {}
            kw["output_attentions"] = True
            return args, kw

        # --- Register Hooks instead of Monkeypatching ---
        def get_activation(layer_id):
            def hook(model, input, output):
                # Handle both tuple output (common in HF) and tensor output
                if isinstance(output, tuple):
                    self.captured_features[layer_id] = output[0]
                else:
                    self.captured_features[layer_id] = output
            return hook

        feature_layers = encoder.layers
        for layer_idx in range(self.starting_depth, len(feature_layers)):
            layer = feature_layers[layer_idx]
            layer.register_forward_hook(get_activation(layer_idx))

        for layer in feature_layers:
            attn_mod = _legrad_layer_attention(layer)
            attn_mod.register_forward_hook(get_attention_hook(attn_mod))
            attn_mod.register_forward_pre_hook(get_attention_pre_hook, with_kwargs=True)

        # Hook pooling head attention if it exists.  HF SigLIP/SigLIP2 uses
        # ``nn.MultiheadAttention`` here, while native ALOE uses a custom module.
        if hasattr(vision, "head") and hasattr(vision.head, "attention"):
            attn_mod = vision.head.attention
            attn_mod.register_forward_hook(get_attention_hook(attn_mod))
            if not isinstance(attn_mod, nn.MultiheadAttention):
                attn_mod.register_forward_pre_hook(get_attention_pre_hook, with_kwargs=True)

    def _activate_aloe_encoder_attention_hooks(self, layer_index):
        """ALOE ViT/DINO-style poolers: hooks on ``encoder.layers.*.self_attn`` (no HF ``layer.attention``)."""
        root = self.model
        vision = _legrad_vision_root(root)
        if hasattr(vision, "config"):
            vision.config._attn_implementation = "eager"
        for module in vision.modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = "eager"

        for name, param in root.named_parameters():
            param.requires_grad = False
        for name, param in vision.named_parameters():
            param.requires_grad = False

        for name, param in vision.encoder.named_parameters():
            if "layers" in name:
                try:
                    depth = int(name.split("layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                if depth >= self.starting_depth:
                    param.requires_grad = True

        for name, param in vision.named_parameters():
            if name.startswith("post_layernorm") or name.startswith("head"):
                param.requires_grad = True

        for name, param in root.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

        layers = _legrad_encoder_layers_seq(vision)

        def get_attention_hook(module):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    module.attention_maps = output[1]

            return hook

        def get_aloe_pre_hook(module, args, kwargs):
            kw = dict(kwargs) if kwargs else {}
            kw["output_attentions"] = True
            return args, kw

        for layer in layers:
            attn = _legrad_layer_attention(layer)
            attn.register_forward_hook(get_attention_hook(attn))
            attn.register_forward_pre_hook(get_aloe_pre_hook, with_kwargs=True)

        def get_activation(layer_id):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.captured_features[layer_id] = output[0]
                else:
                    self.captured_features[layer_id] = output

            return hook

        for layer_idx in range(self.starting_depth, len(layers)):
            layers[layer_idx].register_forward_hook(get_activation(layer_idx))

    def _activate_dinov3_hooks(self, layer_index):
        # DINOv3
        root = self.model
        vision = _legrad_vision_root(root)
        
        # Force eager attention implementation natively
        if hasattr(vision, 'config'):
            vision.config._attn_implementation = "eager"

        # Recursive search for all attention configs
        for module in vision.modules():
             if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
                 module.config._attn_implementation = "eager"

        # --- Deactivate gradient for module that don't need it ---
        for name, param in root.named_parameters():
             param.requires_grad = False
             if 'head' in name or 'classifier' in name: 
                 param.requires_grad = True
        
        # Activate gradients for attention layers (heuristic)
        # We need gradients for attention weights
        for name, module in vision.named_modules():
             if 'attention' in name.lower():
                 for p in module.parameters():
                     p.requires_grad = True
        
        # --- Register Hooks on Attention Modules ---
        def get_attention_hook(module):
            def hook(module, input, output):
                # DINOv3 forward returns (attn_output, attn_weights)
                # verify attn_weights has grad
                if isinstance(output, tuple) and len(output) > 1:
                    weights = output[1]
                    module.attention_maps = weights
                else:
                    pass
            return hook

        def get_attention_pre_hook(module, args, kwargs):
            # Ensure eager mode is active right before forward
            if hasattr(module, 'config') and getattr(module.config, '_attn_implementation', '') != 'eager':
                module.config._attn_implementation = 'eager'
            
            # Inject output_attentions=True
            kwargs['output_attentions'] = True
            return args, kwargs
            
        # Register hooks on ALL attention modules found
        # Robust search
        for name, module in vision.named_modules():
             if 'attention' in name.split('.')[-1] and not isinstance(module, nn.ModuleList):
                 # This is likely an attention module
                 # Ensure config
                 if hasattr(module, 'config'):
                      module.config._attn_implementation = "eager"
                      
                 module.register_forward_hook(get_attention_hook(module))
                 module.register_forward_pre_hook(get_attention_pre_hook, with_kwargs=True)
                 
        # --- Register Hooks on Layer Outputs (for LeGrad) ---
        # We still need captured_features for LeGrad. 
        # Typically 'layers' list.
        layers = _legrad_encoder_layers_seq(vision)
        
        def get_output_hook(layer_id):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.captured_features[layer_id] = output[0]
                else:
                    self.captured_features[layer_id] = output
            return hook        

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= self.starting_depth:
                 layer.register_forward_hook(get_output_hook(layer_idx))

    def compute_legrad(self, text_embedding, image=None, apply_correction=True):
        if 'clip' in self.model_type:
            return self.compute_legrad_clip(text_embedding, image)
        elif 'siglip' in self.model_type:
            return self.compute_legrad_siglip(text_embedding, image, apply_correction=apply_correction)
        elif 'coca' in self.model_type:
            return self.compute_legrad_coca(text_embedding, image)
        elif self.model_type in ("dinov3", "aloe_hf", "hf_vit"):
            # For DINOv3 / ALOE (encoder self-attention path), use the same entry point.
            # DINOv3 is typically used as a backbone for classification or feature extraction
            # Here we assume it's used with a text embedding (like CLIP) OR for classification
            # If text_embedding is actually a class index (int), we should handle it.
            # But the signature says text_embedding.
            # Existing compute_legrad calls assume CLIP-like usage.
            # If user wants classification LeGrad, they typically call compute_legrad_hf_siglip_from_class directly.
            # Let's verify usage pattern. usually LeGradExplainer calls correct method.
            # For now, implementing the method matching this signature.
            return self.compute_legrad_dinov3(text_embedding, image)

    def compute_legrad_hf_siglip_from_class(self, class_idx, image=None, apply_correction=False, correction_threshold=0.8, normalize=True):
        """
        Compute LeGrad for Hugging Face SigLIP2 models.
        
        Args:
            class_idx: Target class index.
            image: Input image tensor.
            apply_correction: Whether to apply empty text correction (default False as this is classification).
            correction_threshold: Threshold for correction.
        """
        del apply_correction, correction_threshold
        if image is None:
            raise ValueError("Classifier LeGrad requires an image tensor")
        adapter = _LeWrapperClassifierAdapter(self)
        return compute_legrad_from_adapter(adapter, image, torch.as_tensor(class_idx), normalize=normalize)

    def compute_legrad_clip(self, text_embedding, image=None):
        num_prompts = text_embedding.shape[0]
        if image is not None:
            image = image.repeat(num_prompts, 1, 1, 1)
            _ = self.encode_image(image)

        blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())

        image_features_list = []

        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
            intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
            intermediate_feat = F.normalize(intermediate_feat, dim=-1)
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        w = h = int(math.sqrt(num_tokens))

        # ----- Get explainability map
        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)): #type:ignore
            self.visual.zero_grad()
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
            one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
            one_hot = torch.sum(one_hot * sim)

            attn_map = blocks_list[self.starting_depth + layer].attn.attention_maps  # [b, num_heads, N, N]

            # -------- Get explainability map --------
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
                0]  # [batch_size * num_heads, N, N]
            grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)  # separate batch and attn heads
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # average attn over [CLS] + patch tokens
            expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
            accum_expl_map += expl_map

        # Min-Max Norm
        accum_expl_map = min_max(accum_expl_map)
        return accum_expl_map

    def compute_legrad_coca(self, text_embedding, image=None):
        if image is not None:
            _ = self.encode_image(image)

        blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())

        image_features_list = []

        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
            intermediate_feat = intermediate_feat.permute(1, 0, 2)  # [batch, num_patch, dim]
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        w = h = int(math.sqrt(num_tokens))

        # ----- Get explainability map
        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
            self.visual.zero_grad()
            # --- Apply attn_pool ---
            image_embedding = self.visual.attn_pool(img_feat)[:,
                              0]  # we keep only the first pooled token as it is only this one trained with the contrastive loss
            image_embedding = image_embedding @ self.visual.proj

            sim = text_embedding @ image_embedding.transpose(-1, -2)  # [1, 1]
            one_hot = torch.sum(sim)

            attn_map = self.visual.attn_pool.attn.attention_maps  # [num_heads, num_latent, num_patch]

            # -------- Get explainability map --------
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
                0]  # [num_heads, num_latent, num_patch]
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=0)[0, 1:]  # average attn over heads + select first latent
            expl_map = rearrange(image_relevance, '(w h) -> 1 1 w h', w=w, h=h)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
            accum_expl_map += expl_map

        # Min-Max Norm
        accum_expl_map = (accum_expl_map - accum_expl_map.min()) / (accum_expl_map.max() - accum_expl_map.min())
        return accum_expl_map

    def _init_empty_embedding(self):
        if not hasattr(self, 'empty_embedding'):
            # For the moment only SigLIP is supported & they all have the same tokenizer
            _tok = open_clip.get_tokenizer(model_name='ViT-B-16-SigLIP')
            empty_text = _tok(['a photo of a']).to(self.logit_scale.data.device)
            empty_embedding = self.encode_text(empty_text)
            empty_embedding = F.normalize(empty_embedding, dim=-1)
            self.empty_embedding = empty_embedding.t()

    def compute_legrad_siglip(
        self, text_embedding, image=None, apply_correction=True, correction_threshold=0.8, normalize=True
    ):
        # --- Forward CLIP ---
        blocks_list = list(dict(self.visual.trunk.blocks.named_children()).values())
        if image is not None:
            _ = self.encode_image(image)  # [bs, num_patch, dim] bs=num_masks

        image_features_list = []
        for blk in blocks_list[self.starting_depth:]:
            intermediate_feat = blk.feat_post_mlp
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[1]
        w = h = int(math.sqrt(num_tokens))

        if apply_correction:
            self._init_empty_embedding()
            accum_expl_map_empty = 0

        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)): # type: ignore
            self.zero_grad()
            pooled_feat = self.visual.trunk.attn_pool(img_feat)
            pooled_feat = F.normalize(pooled_feat, dim=-1)
            # -------- Get explainability map --------
            sim = text_embedding @ pooled_feat.transpose(-1, -2)  # [num_mask, num_mask]
            one_hot = torch.sum(sim)
            grad = torch.autograd.grad(one_hot, [self.visual.trunk.attn_pool.attn_probs], retain_graph=True,
                                       create_graph=True)[0]
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=1)[:, 0]  # average attn over [CLS] + patch tokens
            expl_map = rearrange(image_relevance, 'b (w h) -> b 1 w h', w=w, h=h)
            accum_expl_map += expl_map

            if apply_correction:
                # -------- Get empty explainability map --------
                sim_empty = pooled_feat @ self.empty_embedding
                one_hot_empty = torch.sum(sim_empty)
                grad_empty = \
                    torch.autograd.grad(one_hot_empty, [self.visual.trunk.attn_pool.attn_probs], retain_graph=True,
                                        create_graph=True)[0]
                grad_empty = torch.clamp(grad_empty, min=0.)

                image_relevance_empty = grad_empty.mean(dim=1)[:, 0]  # average attn over heads + select query's row
                expl_map_empty = rearrange(image_relevance_empty, 'b (w h) -> b 1 w h', w=w, h=h)
                accum_expl_map_empty += expl_map_empty

        if apply_correction:
            heatmap_empty = min_max(accum_expl_map_empty)
            accum_expl_map[heatmap_empty > correction_threshold] = 0

        if normalize:
            Res = min_max(accum_expl_map)
        else:
            Res = accum_expl_map
        
        Res = F.interpolate(Res, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]

        return Res

    def compute_legrad_dinov3_from_class(self, class_idx, image=None, normalize=True):
        """
        Compute LeGrad for DINOv3 / ALOE (cls or mean pooler) classification.
        """
        if image is None:
            raise ValueError("Classifier LeGrad requires an image tensor")
        adapter = _LeWrapperClassifierAdapter(self)
        return compute_legrad_from_adapter(adapter, image, torch.as_tensor(class_idx), normalize=normalize)
    
    def compute_chefer_dinov3_from_class(self, class_idx, image=None, normalize=True):
        """
        Compute Chefer CAM (Gradient-weighted Attention Rollout) for DINOv3 / ALOE (cls or mean pooler).
        """
        if image is None:
            raise ValueError("Chefer CAM requires an image tensor")
        adapter = _LeWrapperClassifierAdapter(self)
        return compute_chefer_from_adapter(adapter, image, torch.as_tensor(class_idx), normalize=normalize)


class LePreprocess(nn.Module):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    """

    def __init__(self, preprocess, image_size):
        super(LePreprocess, self).__init__()
        self.transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                preprocess.transforms[-3],
                preprocess.transforms[-2],
                preprocess.transforms[-1],
            ]
        )

    def forward(self, image):
        return self.transform(image)
