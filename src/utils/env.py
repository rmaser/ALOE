import os
from pathlib import Path


def _path_from_env_or_scratch(
    env_key: str,
    scratch_suffix: str,
    fallback: str,
) -> Path:
    raw = os.getenv(env_key)
    if raw:
        return Path(raw)

    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / scratch_suffix

    return Path(fallback)


def get_cache_path() -> Path:
    path = _path_from_env_or_scratch("CACHE_PATH", "ALOE/.cache", ".cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_feature_cache_path() -> Path:
    path = get_cache_path() / "features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    path = _path_from_env_or_scratch(
        "DATA_PATH",
        "data/huggingface/datasets",
        ".cache/data",
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_cache_path() -> Path:
    path = _path_from_env_or_scratch(
        "HF_CACHE_PATH",
        "data/huggingface/hub",
        ".cache/hf_cache",
    )
    if "HF_HUB_CACHE" in os.environ and "HF_CACHE_PATH" not in os.environ:
        path = Path(os.environ["HF_HUB_CACHE"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_seed() -> int:
    return int(os.getenv("SEED", 42))
