"""
Explainer module with various explanation methods.

Provides a unified interface for computing model explanations using different methods:
- B-cos native explanations
- Input x Gradient
- Saliency (Captum)
- Guided Backpropagation (Captum)
- Integrated Gradients (Captum)
- GradientShap (Captum)
- DeepLift (Captum)
- LIME (Captum)
- AttnLRP (LXT)
"""
from jaxtyping import install_import_hook
install_import_hook("src.explanation", "beartype.beartype")

# Base classes
from .abstract import BaseExplainer, ExplanationOutput
from .captum_base import CaptumExplainer

# B-cos and gradient-based
from .bcos import BcosExplainer
from .input_x_gradient import InputXGradientExplainer

# Captum-based explainers
from .saliency import SaliencyExplainer
from .guided_backprop import GuidedBackpropExplainer
from .integrated_gradients import IntegratedGradientsExplainer
from .gradient_shap import GradientShapExplainer
from .deeplift import DeepLiftExplainer
from .lime import LimeExplainer
from .attnlrp import AttnLRPExplainer

__all__ = [
    # Base
    "BaseExplainer",
    "ExplanationOutput",
    "CaptumExplainer",
    # Native
    "BcosExplainer",
    "InputXGradientExplainer",
    # Captum
    "SaliencyExplainer",
    "GuidedBackpropExplainer",
    "IntegratedGradientsExplainer",
    "GradientShapExplainer",
    "DeepLiftExplainer",
    "LimeExplainer",
    # LXT
    "AttnLRPExplainer",
]
