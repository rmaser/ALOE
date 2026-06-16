from __future__ import annotations

import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CONFIG_DIR = str(_REPO_ROOT / "configs")

_EXPECTED_DATASETS = (
    "imagenet1k, caltech101, stanford_cars, cifar10, cifar100, dtd, "
    "oxford_flowers102, food101, sun397, fgvc_aircraft"
)
_EXPECTED_HUB_MODELS = (
    "siglip2/aloe_siglip2_base_hub,siglip2/aloe_siglip2_large_hub,"
    "siglip2/aloe_siglip2_so400m_hub,siglip2/aloe_siglip2_so400m_432_hub,"
    "dinov3/aloe_dinov3_small_hub,dinov3/aloe_dinov3_base_hub,"
    "dinov3/aloe_dinov3_large_hub,google_vit/aloe_vit_base_hub"
)
_EXPECTED_CLASSIFIER_DATASETS = "imagenet1k"


class TestPublicSurface(unittest.TestCase):
    def tearDown(self) -> None:
        GlobalHydra.instance().clear()

    def _compose(
        self,
        config_name: str,
        overrides: list[str],
        *,
        return_hydra_config: bool = False,
    ):
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            return compose(
                config_name=config_name,
                overrides=overrides,
                return_hydra_config=return_hydra_config,
            )

    def test_pixi_public_tasks_use_publish_names_and_scratch_paths(self) -> None:
        manifest = tomllib.loads((_REPO_ROOT / "pixi.toml").read_text())
        tasks = manifest["tasks"]

        for task_name in (
            "train_native_distill_siglip2_base",
            "train_native_distill_siglip2_large",
            "train_native_distill_siglip2_so400m",
            "train_native_distill_siglip2_so400m_432",
            "train_native_distill_dinov3_small",
            "train_native_distill_dinov3_base",
            "train_native_distill_dinov3_large",
            "train_native_distill_google_vit_base",
            "eval_native_hf_all_hub_lp",
        ):
            self.assertIn(task_name, tasks)

        self.assertFalse(any("no_reg" in name for name in tasks))
        self.assertIn(
            "experiment=eval/embeddings/native_hf_all_hub_lp",
            tasks["eval_native_hf_all_hub_lp"],
        )

        env = manifest["feature"]["cuda"]["activation"]["env"]
        for key in ("CACHE_PATH", "DATA_PATH", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
            self.assertIn("SCRATCH", env[key])

    def test_public_hf_eval_sweep_covers_all_models_and_datasets(self) -> None:
        cfg = self._compose(
            "eval",
            ["experiment=eval/embeddings/native_hf_all_hub_lp"],
            return_hydra_config=True,
        )

        self.assertEqual(cfg.hydra.sweeper.params["data/dataset"], _EXPECTED_DATASETS)
        self.assertEqual(
            cfg.hydra.sweeper.params["model/backbone@backbone.student"],
            _EXPECTED_HUB_MODELS,
        )

        lp = cfg.evaluators.linear_probe
        self.assertEqual(lp.batch_size, 8192)
        self.assertEqual(list(lp.lr), [0.01, 0.003, 0.001])
        self.assertEqual(lp.bcos_impl, "v2")
        self.assertEqual(lp.feature_device, "cuda")
        self.assertEqual(lp.trainer_config.max_epochs, 50)

    def test_public_classifier_hf_eval_covers_all_published_models(self) -> None:
        cfg = self._compose(
            "eval",
            ["experiment=eval/classifier/native_hf"],
            return_hydra_config=True,
        )

        self.assertEqual(cfg.hydra.sweeper.params["data/dataset"], _EXPECTED_CLASSIFIER_DATASETS)
        self.assertEqual(
            cfg.hydra.sweeper.params["model/backbone@backbone.student"],
            _EXPECTED_HUB_MODELS,
        )

    def test_public_eval_defaults_use_shared_cache_path_resolution(self) -> None:
        from src.eval.evaluator import Evaluator

        for rel_path in (
            "configs/evaluator/linear_probe.yaml",
            "configs/evaluator/knn.yaml",
            "configs/evaluator/zero_shot.yaml",
            "src/eval/evaluator.py",
        ):
            with self.subTest(path=rel_path):
                self.assertNotIn("/dev/shm", (_REPO_ROOT / rel_path).read_text())

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir) / "cache-root"
            env_overrides = {
                "SCRATCH": tmpdir,
                "CACHE_PATH": str(feature_root),
            }
            with patch.dict(os.environ, env_overrides, clear=False):
                evaluator = Evaluator()
            self.assertEqual(Path(evaluator.feature_cache_dir), feature_root / "features")

    def test_public_repo_omits_publish_regularization_and_register_extras(self) -> None:
        missing_paths = (
            "configs/adapter/base.yaml",
            "configs/adapter/lora.yaml",
            "configs/backbone_name/dinov3_base.yaml",
            "configs/backbone_name/dinov3_large.yaml",
            "configs/backbone_name/dinov3_small.yaml",
            "configs/backbone_name/google_vit_base.yaml",
            "configs/backbone_name/siglip2_base.yaml",
            "configs/backbone_name/siglip2_large.yaml",
            "configs/backbone_name/siglip2_so400m.yaml",
            "configs/backbone_name/siglip2_so400m_432px.yaml",
            "src/explainability/explainers/bcos/__init__.py",
            "src/explainability/explainers/bcos/mixin.py",
            "src/explainability/explainers/bcos/utils.py",
            "src/training/loss/regularization.py",
            "configs/regularization/teacher_norm.yaml",
            "configs/evaluator/linear_probe_v2.yaml",
            "configs/experiment/native/distill_aloe_170k_no_reg.yaml",
            "configs/experiment/native/distill_aloe_300k_no_reg.yaml",
            "configs/experiment/native/distill_registers_cosine.yaml",
            "configs/model/backbone/siglip2/aloe_siglip2_base_registers.yaml",
            "configs/model/backbone/siglip2/aloe_siglip2_base_registers_hub.yaml",
            "configs/model/backbone/siglip2/aloe_siglip2_base_registers_local_weights.yaml",
            "configs/model/supervised/siglip2/aloe_siglip2_base_imagenet1k_lp_registers.yaml",
        )
        for rel_path in missing_paths:
            with self.subTest(path=rel_path):
                self.assertFalse((_REPO_ROOT / rel_path).exists())

        for rel_path in (
            "configs/experiment/native/distill_aloe_170k_432_safe.yaml",
            "configs/experiment/native/distill_aloe_170k_drop_register_tokens.yaml",
        ):
            with self.subTest(path=rel_path):
                self.assertTrue((_REPO_ROOT / rel_path).exists())

        hf_init = (_REPO_ROOT / "src/models/hf/__init__.py").read_text()
        aloe_models = (_REPO_ROOT / "src/models/hf/aloe_models.py").read_text()
        model_factory = (_REPO_ROOT / "src/models/model_factory/model_factory.py").read_text()
        model_factory_cfg = (_REPO_ROOT / "src/models/model_factory/config.py").read_text()
        model_factory_utils = (_REPO_ROOT / "src/models/model_factory/utils.py").read_text()
        manifest = tomllib.loads((_REPO_ROOT / "pixi.toml").read_text())
        self.assertNotIn("export_native_model_to_hub", hf_init)
        self.assertNotIn("export_native_model_to_hub", aloe_models)
        self.assertNotIn("push_aloe_code_to_hub", aloe_models)
        self.assertNotIn("get_peft_model", model_factory)
        self.assertNotIn("LoraConfig", model_factory)
        self.assertNotIn("adapter:", model_factory_cfg)
        self.assertNotIn("check_and_fix_lora_keys", model_factory_utils)
        self.assertNotIn("peft", manifest["pypi-dependencies"])


if __name__ == "__main__":
    unittest.main()
