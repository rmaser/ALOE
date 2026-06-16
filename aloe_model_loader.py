"""
Bridge file for LLaVA-MORE ↔ ALOE integration.

Loaded by LLaVA-MORE's `AloeVisionTower` via `importlib.util.spec_from_file_location`.
Must be placed at `<ALOE_REPO_ROOT>/aloe_model_loader.py`.

This file should stay *very thin*: it only sets up import paths so that ALOE's
`src/` can be imported under LLaVA-MORE's namespace package, then delegates all
actual Hydra composition / model creation logic to `src.models.model_loader`.
"""

from __future__ import annotations

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = str(REPO_ROOT / "configs")

# ---------------------------------------------------------------------------
# Make aloe's src/ a search path on LLaVA-MORE's 'src' namespace package.
# ---------------------------------------------------------------------------
try:
    import src as _src_pkg
    _src_path = getattr(_src_pkg, "__path__", None)
    if _src_path is not None and str(SRC_DIR) not in list(_src_path):
        _src_pkg.__path__ = list(_src_path) + [str(SRC_DIR)]
except ImportError:
    # 'src' not yet imported; SRC_DIR is already in sys.path so src.models etc.
    # will be importable as top-level packages.
    pass


# ---------------------------------------------------------------------------
# Delegate public API to the canonical loader.
# ---------------------------------------------------------------------------
from src.models.model_loader import (  # noqa: E402
    AloeModelBundle,
    build_model_factory_config,
    compose_aloe_model_config,
    get_aloe_model_bundle,
    load_backbone_image_processor,
)


__all__ = [
    "AloeModelBundle",
    "build_model_factory_config",
    "compose_aloe_model_config",
    "get_aloe_model_bundle",
    "load_backbone_image_processor",
]