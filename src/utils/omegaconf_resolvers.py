"""OmegaConf resolvers shared by Hydra entrypoints (no ModelFactory changes)."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

_EQUIVALENT_HF_VISION_INIT_CHECKPOINTS: dict[str, str] = {
    "rmaser/aloe-siglip2-base": "google/siglip2-base-patch16-224",
    "rmaser/aloe-siglip2-large": "google/siglip2-large-patch16-256",
    "rmaser/aloe-siglip2-so400m": "google/siglip2-so400m-patch16-256",
    "rmaser/aloe-siglip2-so400m-432": "google/siglip2-so400m-patch14-384",
    "rmaser/aloe-dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "rmaser/aloe-dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "rmaser/aloe-dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "rmaser/aloe-vit-base": "google/vit-base-patch16-224",
}


def resolve_local_weights_spec(
    spec: Any,
    model_name: Any,
    local_weights_key: Any = None,
) -> Any:
    """
    Pick one checkpoint list for the current backbone.

    * ``spec`` is ``weight_source.student.load_weights_from_local_checkpoints``: either a
      **list** (pass through) or a **dict** keyed by Hub id / short names.
    * ``model_name``: ``backbone.student.name`` (with fallbacks supplied from YAML).
    * ``local_weights_key``: optional override tried first (same idea as a backbone field).

    Returns an OmegaConf list node suitable as ``student_factory.load_weights_from_local_checkpoints``.
    """
    if spec is None or spec == "null":
        return OmegaConf.create([])
    if OmegaConf.is_list(spec):
        return spec
    if isinstance(spec, (list, tuple)):
        return OmegaConf.create(list(spec))

    container = (
        OmegaConf.to_container(spec, resolve=False)
        if OmegaConf.is_config(spec)
        else spec
    )
    if not isinstance(container, dict):
        return spec

    keys_try: list[str] = []
    if local_weights_key not in (None, "null", ""):
        keys_try.append(str(local_weights_key))
    mn = "" if model_name in (None, "null") else str(model_name)
    if mn:
        keys_try.append(mn)
        tail = mn.split("/")[-1]
        keys_try.append(tail)
        keys_try.append(tail.replace("-", "_"))

    seen: set[str] = set()
    ordered: list[str] = []
    for k in keys_try:
        if k and k not in seen:
            seen.add(k)
            ordered.append(k)

    for k in ordered:
        if k in container:
            return OmegaConf.create(container[k])

    if not ordered:
        raise ValueError(
            "local_weights_for_model: empty model_name and local_weights_key — "
            "set backbone.student (or factory target_model_config fallbacks in YAML)."
        )
    raise KeyError(
        "local_weights_for_model: no entry for "
        f"{mn!r} (tried {ordered!r}); keys={sorted(container)!r}"
    )


def _resolver_local_weights_for_model(
    spec: Any,
    model_name: Any,
    local_weights_key: Any = None,
) -> Any:
    return resolve_local_weights_spec(spec, model_name, local_weights_key)


def _resolver_hydra_runtime_choice(key: str, default: str = "") -> str:
    """Resolve ``HydraConfig.get().runtime.choices[key]`` when running under ``@hydra.main``."""
    try:
        from hydra.core.hydra_config import HydraConfig

        val = HydraConfig.get().runtime.choices.get(key)
        if val is None:
            return default
        return str(val)
    except Exception:
        return default




def resolve_equivalent_hf_vision_init_checkpoint(
    model_name: Any,
    default: Any = None,
) -> Any:
    """
    Return the vanilla HF vision checkpoint matching a native ALOE Hub backbone.

    Falls back to ``default`` when ``model_name`` is unknown so configs can still
    defer to an explicitly selected teacher backbone.
    """
    if model_name in (None, "null", ""):
        return default

    key = str(model_name)
    if key in _EQUIVALENT_HF_VISION_INIT_CHECKPOINTS:
        return _EQUIVALENT_HF_VISION_INIT_CHECKPOINTS[key]

    tail = key.split("/")[-1]
    for known, checkpoint in _EQUIVALENT_HF_VISION_INIT_CHECKPOINTS.items():
        if tail == known.split("/")[-1]:
            return checkpoint

    return default


def _resolver_equivalent_hf_vision_init_checkpoint(
    model_name: Any,
    default: Any = None,
) -> Any:
    return resolve_equivalent_hf_vision_init_checkpoint(model_name, default)


def resolve_student_hf_vision_init_checkpoint(
    student: Any,
    teacher_fallback: Any = None,
) -> Any:
    """
    Resolve ``student_factory.hf_vision_init_checkpoint`` from ``backbone.student``.

    The **backbone** is the source of truth for where trunk weights come from:

    * If the student loads a trunk from ``backbone_checkpoint`` or
      ``load_weights_from_local_checkpoints``, return ``None`` (HF vision-init mapping
      must not run — it would overwrite those weights in :class:`HfModelFactory`).
    * Else if ``hf_vision_init_checkpoint`` is set on the student to a non-empty string,
      use it (explicit override for scratch runs).
    * Else return :func:`resolve_equivalent_hf_vision_init_checkpoint` for the
      student ``name`` and ``teacher_fallback``.
    """
    if student in (None, "null", ""):
        return resolve_equivalent_hf_vision_init_checkpoint(None, teacher_fallback)

    if OmegaConf.is_config(student):
        student = OmegaConf.to_container(student, resolve=True)
    if not isinstance(student, dict):
        return resolve_equivalent_hf_vision_init_checkpoint(None, teacher_fallback)

    def _trunk_from_local_weights(lw: Any) -> bool:
        if lw is None or lw == "null":
            return False
        if isinstance(lw, dict):
            return len(lw) > 0
        if isinstance(lw, (list, tuple)):
            return len(lw) > 0
        return False

    bc = student.get("backbone_checkpoint")
    if bc not in (None, "", "null"):
        return None

    if _trunk_from_local_weights(student.get("load_weights_from_local_checkpoints")):
        return None

    explicit = student.get("hf_vision_init_checkpoint")
    if explicit not in (None, "", "null"):
        return explicit

    return resolve_equivalent_hf_vision_init_checkpoint(
        student.get("name"),
        teacher_fallback,
    )


def _resolver_student_hf_vision_init_checkpoint(
    student: Any,
    teacher_fallback: Any = None,
) -> Any:
    return resolve_student_hf_vision_init_checkpoint(student, teacher_fallback)


def register_aloe_omegaconf_resolvers(*, replace: bool = False) -> None:
    """Register all ALOE OmegaConf resolvers (safe to call multiple times)."""
    if replace or not OmegaConf.has_resolver("select_distillation_layers"):
        OmegaConf.register_new_resolver(
            "select_distillation_layers",
            lambda layers: OmegaConf.create(
                [int(layers * 1 / 3) - 1, int(layers * 2 / 3) - 1, layers - 1]
            ),
            replace=True,
        )
    if replace or not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    if replace or not OmegaConf.has_resolver("local_weights_for_model"):
        OmegaConf.register_new_resolver(
            "local_weights_for_model",
            _resolver_local_weights_for_model,
            replace=True,
        )
    if replace or not OmegaConf.has_resolver("hydra_runtime_choice"):
        OmegaConf.register_new_resolver(
            "hydra_runtime_choice",
            _resolver_hydra_runtime_choice,
            replace=True,
        )
    if replace or not OmegaConf.has_resolver("equivalent_hf_vision_init_checkpoint"):
        OmegaConf.register_new_resolver(
            "equivalent_hf_vision_init_checkpoint",
            _resolver_equivalent_hf_vision_init_checkpoint,
            replace=True,
        )
    if replace or not OmegaConf.has_resolver("student_hf_vision_init_checkpoint"):
        OmegaConf.register_new_resolver(
            "student_hf_vision_init_checkpoint",
            _resolver_student_hf_vision_init_checkpoint,
            replace=True,
        )
