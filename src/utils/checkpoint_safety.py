from __future__ import annotations

import torch


def register_trusted_checkpoint_safe_globals() -> None:
    """Allowlist trusted local classes needed for PyTorch 2.6 checkpoint resume.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True`` in more
    call paths. Lightning resume checkpoints may contain serialized references to
    local training-loss classes in hyperparameters or callback state. Register the
    trusted project-local classes here so ``last.ckpt`` can be resumed safely.
    """

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    from src.training.loss.cosine import CosineDistanceLoss
    from src.training.loss.infonce import InfoNCE
    from src.training.loss.mse import MSELoss
    from src.training.loss.siglip import SiglipLoss

    add_safe_globals(
        [
            CosineDistanceLoss,
            InfoNCE,
            MSELoss,
            SiglipLoss,
        ]
    )
