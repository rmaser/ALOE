# Self-contained B-cos primitives for the ALOE HF package.
# No dependency on the project's `src.modules.*` or the `bcos` PyPI package.

from __future__ import annotations

import math
import warnings
from functools import wraps
from typing import Any, Callable, Optional, Union, cast

import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention_utils import eager_attention_forward


# ---------------------------------------------------------------------------
# DetachableModule — replaces bcos.modules.DetachableModule
# ---------------------------------------------------------------------------

class DetachableModule(nn.Module):
    """
    Base class for B-cos modules that must be able to detach dynamic weights
    during explanation-mode inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.detach: bool = False

    def set_explanation_mode(self, activate: bool = True) -> None:
        self.detach = activate

    @property
    def is_in_explanation_mode(self) -> bool:
        return self.detach


# ---------------------------------------------------------------------------
# NoBias — replaces bcos.modules.norms.NoBias
# ---------------------------------------------------------------------------

def NoBias(make_layer):  # noqa: N802
    """
    Wraps a layer factory and removes the bias by setting it to ``None``
    after instantiation.  Usage: ``NoBias(nn.LayerNorm)(dim)``.
    """

    @wraps(make_layer)
    def _init(*args, **kwargs):
        """Build the norm layer, then drop its bias tensor so it stays unused."""
        norm = make_layer(*args, **kwargs)
        assert norm.bias is not None, \
            "NoBias: wrapping a layer that already has no bias is a no-op."
        norm.bias = None
        return norm

    return _init


# ---------------------------------------------------------------------------
# BcosLinear / BcosUnnormedLinear
# ---------------------------------------------------------------------------

class _NormedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        """Linear with row-normalised weights (used by :class:`BcosLinear`, not ALOE ViT blocks)."""
        w = self.weight / LA.vector_norm(self.weight, dim=1, keepdim=True)
        return F.linear(x, w, self.bias)


class BcosLinear(DetachableModule):
    """
    B-cos linear layer with unit-normalised weights.

    Parameters
    ----------
    in_features, out_features : int
    b : float
        Exponent for the dynamic scaling (``|cos|^(b-1)``).
    max_out : int
        MaxOut width (1 = standard).
    detach_output : bool
        If True and in explanation mode, the output tensor is detached.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        b: Union[int, float] = 2,
        max_out: int = 1,
        detach_output: bool = False,
    ) -> None:
        if bias:
            raise ValueError("BcosLinear does not support bias.")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        self.b = b
        self.max_out = max_out
        self.detach_output = detach_output

        self.linear = _NormedLinear(
            in_features, out_features * max_out, bias=False, device=device, dtype=dtype
        )
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    @property
    def dtype(self):
        return self.linear.weight.dtype

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward(x)
        if self.detach_output and self.detach:
            out = out.detach()
        return out

    def _forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)

        if self.max_out > 1:
            out = out.unflatten(-1, (self.out_features, self.max_out)).max(dim=-1).values

        if self.b == 1:
            return out

        norm = LA.vector_norm(x, dim=-1, keepdim=True) + 1e-12
        maybe_detached = out.detach() if self.detach else out
        norm_d = norm.detach() if self.detach else norm

        if self.b == 2:
            scale = maybe_detached.abs() / norm_d
        else:
            scale = (maybe_detached / norm_d).abs().clamp(min=1e-6).pow(self.b - 1)

        return scale * out

    def extra_repr(self) -> str:
        s = f"in={self.in_features}, out={self.out_features}, B={self.b}"
        if self.max_out > 1:
            s += f", max_out={self.max_out}"
        if self.detach_output:
            s += ", detach_output=True"
        return s


class BcosUnnormedLinear(BcosLinear):
    """
    B-cos linear with **unnormed** (standard ``nn.Linear``) weights — the
    variant used in ALOE transformers.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        self.linear = nn.Linear(
            self.in_features,
            self.out_features * self.max_out,
            bias=False,
            device=device,
            dtype=dtype,
        )


class BcosUnnormedLinear_v2(BcosUnnormedLinear):
    """Like :class:`BcosUnnormedLinear` but dynamic scale from cosine(|x|,|w|) at ``b=2`` (v2 path)."""

    def _compute_scale(self, x: Tensor, weight: Tensor, detach: bool = False) -> Tensor:
        """Per-token |cos| between L2-normalised activations and weight rows (optional detach for explain)."""
        # F.normalize is faster and handles the 1e-12 epsilon under the hood
        normed_x = F.normalize(x, p=2, dim=-1, eps=1e-12)
        normed_w = F.normalize(weight, p=2, dim=1, eps=1e-12)
        
        # Detach before the matmul if needed to save the backward pass graph overhead
        if detach:
            normed_x = normed_x.detach()
            normed_w = normed_w.detach()
            
        cos_sim = F.linear(normed_x, normed_w)
        
        return cos_sim.abs()

    def _forward(self, x: Tensor) -> Tensor:
        """Unnormed linear output times v2 B-cos scale (max-out and general ``b`` like v1)."""
        out = self.linear(x)

        if self.max_out > 1:
            out = out.unflatten(-1, (self.out_features, self.max_out)).max(dim=-1).values

        if self.b == 1:
            return out

        if self.b == 2:
            scale = self._compute_scale(x, self.linear.weight, self.detach)
        else:
            scale = self._compute_scale(x, self.linear.weight, self.detach).abs().clamp(min=1e-6).pow(self.b - 1)

        return scale * out


def select_bcos_unnormed_linear(config: Any) -> Callable[..., BcosUnnormedLinear]:
    """Return ``BcosUnnormedLinear`` or ``BcosUnnormedLinear_v2`` per ``config.aloe_bcos_impl``."""
    impl = getattr(config, "aloe_bcos_impl", "v1")
    if impl == "v2":
        return cast(Callable[..., BcosUnnormedLinear], BcosUnnormedLinear_v2)
    if impl == "v1":
        return BcosUnnormedLinear
    raise ValueError(f"aloe_bcos_impl must be 'v1' or 'v2', got {impl!r}")


def select_bcos_unnormed_conv2d(config: Any) -> Callable[..., "BcosUnnormedConv2d"]:
    """Return ``BcosUnnormedConv2d`` or ``BcosUnnormedConv2d_v2`` per ``config.aloe_bcos_impl``."""
    impl = getattr(config, "aloe_bcos_impl", "v1")
    if impl == "v2":
        return cast(Callable[..., BcosUnnormedConv2d], BcosUnnormedConv2d_v2)
    if impl == "v1":
        return BcosUnnormedConv2d
    raise ValueError(f"aloe_bcos_impl must be 'v1' or 'v2', got {impl!r}")


# ---------------------------------------------------------------------------
# BcosUnnormedConv2d
# ---------------------------------------------------------------------------

class BcosUnnormedConv2d(DetachableModule):
    """
    B-cos unnormed 2-D convolution — the conv analogue of
    :class:`BcosUnnormedLinear`, used in ALOE conv stems.

    Forward math matches ``BcosConv2d_unnormed`` in ``src/modules/bcos/bcos_conv.py``:
    patch norms via ``calc_patch_norms`` / ``_calc_patch_norms_slow`` (dilation, groups),
    optional ``max_out``, and the same dynamic scaling as the reference implementation.

    Stores weights in ``self.linear`` (an ``nn.Conv2d``) so that weight
    keys match the BcosConverter output format (``*.linear.weight``).
    In explanation mode the dynamic B-cos scale is detached so that
    contribution maps flow back through the un-scaled path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        b: float = 2.0,
        max_out: int = 1,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        assert max_out > 0, f"max_out should be greater than 0, was {max_out}"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.b = b
        self.max_out = max_out
        self.patch_size = math.prod(self.kernel_size)

        if isinstance(dilation, int):
            dilation_check = dilation > 1
        else:
            dilation_check = any(d > 1 for d in dilation)
        if dilation_check:
            warnings.warn("dilation > 1 is much slower!", stacklevel=2)
            self.calc_patch_norms = self._calc_patch_norms_slow  # type: ignore[method-assign]

        self.linear = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * max_out,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, in_tensor: Tensor) -> Tensor:
        out = self.linear(in_tensor)

        if self.max_out > 1:
            m = self.max_out
            o = self.out_channels
            out = out.unflatten(dim=1, sizes=(o, m))
            out = out.max(dim=2, keepdim=False).values

        if self.b == 1:
            return out

        norm_x = self.calc_patch_norms(in_tensor)

        maybe_detached_out = out

        if self.detach:
            maybe_detached_out = out.detach()
            norm_x = norm_x.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / (norm_x)
        else:
            abs_cos = (maybe_detached_out / norm_x).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        return dynamic_scaling * out

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        squares = in_tensor**2
        if self.groups == 1:
            squares = squares.sum(1, keepdim=True)
        else:
            g = self.groups
            c = self.in_channels
            squares = squares.unflatten(1, (g, c // g)).sum(2)

        norms = (
            F.avg_pool2d(
                squares,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )
            * self.patch_size
            + 1e-6
        ).sqrt_()

        if self.groups > 1:
            n, g, h, w = norms.shape
            o = self.out_channels
            norms = torch.repeat_interleave(norms, repeats=o // g, dim=1)

        return norms

    def _calc_patch_norms_slow(self, in_tensor: Tensor) -> Tensor:
        ones_kernel = torch.ones_like(self.linear.weight)
        return (
            F.conv2d(
                in_tensor**2,
                ones_kernel,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            + 1e-6
        ).sqrt_()

    def extra_repr(self) -> str:
        s = f"B={self.b}"
        if self.max_out > 1:
            s += f", max_out={self.max_out}"
        s += ","
        return s + " " + self.linear.extra_repr()


class BcosUnnormedConv2d_v2(BcosUnnormedConv2d):
    """
    B-cos unnormed 2-D convolution — the conv analogue of
    :class:`BcosUnnormedLinear`, used in ALOE conv stems.

    Forward math matches ``BcosConv2d_unnormed`` in ``src/modules/bcos/bcos_conv.py``:
    patch norms via ``calc_patch_norms`` / ``_calc_patch_norms_slow`` (dilation, groups),
    optional ``max_out``, and the same dynamic scaling as the reference implementation.

    Stores weights in ``self.linear`` (an ``nn.Conv2d``) so that weight
    keys match the BcosConverter output format (``*.linear.weight``).
    In explanation mode the dynamic B-cos scale is detached so that
    contribution maps flow back through the un-scaled path.
    """

    def forward(self, in_tensor: Tensor) -> Tensor:
        out = self.linear(in_tensor)

        if self.max_out > 1:
            m = self.max_out
            o = self.out_channels
            out = out.unflatten(dim=1, sizes=(o, m))
            out = out.max(dim=2, keepdim=False).values

        if self.b == 1:
            return out

        norm_x = self.calc_patch_norms(in_tensor)
        norm_w = torch.linalg.vector_norm(self.linear.weight, dim=(1, 2, 3), keepdim=True)
        norm_w = norm_w.view(1, -1, 1, 1) # Match (B, C, H, W) shape

        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm_x = norm_x.detach()
            norm_w = norm_w.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / (norm_x * norm_w)
        else:
            abs_cos = (maybe_detached_out / norm_x * norm_w).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        return dynamic_scaling * out


# ---------------------------------------------------------------------------
# DetachableLayerNorm
# ---------------------------------------------------------------------------

class DetachableLayerNorm(nn.LayerNorm, DetachableModule):
    """
    LayerNorm that detaches the variance estimate in explanation mode so that
    contribution maps flow through the un-normalised path.
    """

    def __init__(self, *args, **kwargs) -> None:
        DetachableModule.__init__(self)
        kwargs.pop("b", None)
        kwargs.pop("detach_output", None)
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if not self.detach:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        d = len(self.normalized_shape)
        var, mean = torch.var_mean(x, dim=tuple(range(-d, 0)), unbiased=False, keepdim=True)
        std = (var.detach() + self.eps).sqrt_()
        x = (x - mean) / std
        if self.weight is not None:
            x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return x


class NoBiasDetachableLayerNorm(DetachableLayerNorm):
    """
    :class:`DetachableLayerNorm` without an additive bias parameter.

    B-cos models strip the bias from every norm layer (the BcosConverter uses
    ``NoBias(DetachableLayerNorm)`` for all replacements).  Using this class in
    the native ALOE model ensures the state dict is bias-free and structurally
    identical to the converted checkpoint — no missing-key warnings, no wasted
    parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Remove the bias parameter registered by nn.LayerNorm.__init__.
        del self.bias
        self.register_parameter("bias", None)


# ---------------------------------------------------------------------------
# Detachable activations
# All follow the same pattern: activation(x) = gate(x) * x.
# In explanation mode the gate is detached so contribution maps trace back
# linearly through the input.
# ---------------------------------------------------------------------------

class DetachableGELU(DetachableModule):
    """Exact GELU via erf."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if not self.detach:
            return F.gelu(x)  # fused CUDA kernel in normal path
        gate = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gate.detach() * x


class DetachableGELUApprox(DetachableModule):
    """Approximate GELU via tanh (``gelu_pytorch_tanh`` / ``gelu_new``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if not self.detach:
            return F.gelu(x, approximate="tanh")  # fused CUDA kernel in normal path
        gate = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
        return gate.detach() * x


class DetachableSiLU(DetachableModule):
    """SiLU / Swish: gate = sigmoid(x)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if not self.detach:
            return F.silu(x)  # fused CUDA kernel in normal path
        gate = torch.sigmoid(x)
        return gate.detach() * x


class DetachableReLU(DetachableModule):
    """ReLU: gate = (x > 0). In explanation mode the indicator is detached."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if not self.detach:
            return F.relu(x)  # fused CUDA kernel in normal path
        gate = (x > 0).to(x.dtype)
        return gate.detach() * x


class DetachableIdentity(DetachableModule):
    """Identity (no non-linearity). Useful for probing pure B-cos linear models."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


# Maps HF / torch activation names to their detachable counterparts.
_ACT2DETACHABLE: dict[str, type[DetachableModule]] = {
    "gelu": DetachableGELU,
    "gelu_new": DetachableGELUApprox,
    # Legacy BcosConverter replaced SigLIP2's PytorchGELUTanh with exact DetachableGELU.
    # Keep that behaviour here so native ALOE matches previous training and exported checkpoints.
    "gelu_pytorch_tanh": DetachableGELU,
    "gelu_fast": DetachableGELUApprox,
    "silu": DetachableSiLU,
    "swish": DetachableSiLU,
    "relu": DetachableReLU,
    "identity": DetachableIdentity,
    "linear": DetachableIdentity,
}


def build_detachable_activation(name: str) -> DetachableModule:
    """Return the detachable activation for the given HF/torch *name*."""
    cls = _ACT2DETACHABLE.get(name)
    if cls is None:
        supported = ", ".join(sorted(_ACT2DETACHABLE))
        raise ValueError(f"Unsupported activation {name!r}. Supported: {supported}.")
    return cls()


# ---------------------------------------------------------------------------
# AloeMultiHeadAttentionPooler  (pooler-only, q/k/v plain, out_proj B-cos)
# ---------------------------------------------------------------------------

class AloeMultiHeadAttentionPooler(DetachableModule):
    """
    Multihead attention used exclusively in the ALOE pooling head.
    Accepts q/k/v tensors separately (not a self-attention module).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        out_b: float = 2.0,
        *,
        out_proj_cls: Callable[..., BcosUnnormedLinear] | None = None,
        attn_implementation: str | None = "sdpa",
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout
        self.is_causal = False

        # Q is separate: probe → Q (different source from K/V).
        # K and V are fused: hidden_states → KV with one (D, 2D) GEMM.
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.kv_proj = nn.Linear(embedding_dim, 2 * embedding_dim, bias=False)
        _Lin = out_proj_cls or BcosUnnormedLinear
        self.out_proj = _Lin(embedding_dim, embedding_dim, b=out_b)

        attn_impl = attn_implementation or "sdpa"
        if attn_impl == "eager":
            self._attention_fn: Callable[..., Any] = eager_attention_forward
        else:
            self._attention_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        del value
        B, T_q, C = query.shape
        T_kv = key.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, T_q, H, D).transpose(1, 2)
        k, v = self.kv_proj(key).split(self.embedding_dim, dim=-1)
        k = k.view(B, T_kv, H, D).transpose(1, 2)
        v = v.view(B, T_kv, H, D).transpose(1, 2)

        if self.detach:
            k = k.detach()
            q = q.detach()

        attention_fn = eager_attention_forward if output_attentions else self._attention_fn
        attn_output, attn_weights = attention_fn(
            self,
            q,
            k,
            v,
            attn_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(B, T_q, C).contiguous()
        return self.out_proj(attn_output), (attn_weights if output_attentions else None)
