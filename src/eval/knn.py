import time
import torch.nn as nn
import pytorch_lightning as L
from sklearn.neighbors import KNeighborsClassifier
from loguru import logger
from typing import Dict, Optional

from src.eval.evaluator import Evaluator

class KNNEvaluator(Evaluator):
    def __init__(
        self,
        k: int = 20,
        feature_cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        batch_size: int = 256,
    ):
        super().__init__(feature_cache_dir=feature_cache_dir, force_recompute=force_recompute)
        self.k = k
        self.batch_size = batch_size

    def evaluate(
        self,
        model: nn.Module,
        datamodule: L.LightningDataModule,
        trainer: L.Trainer,
    ) -> Dict[str, float]:
        
        device = next(model.parameters()).device
        
        # Extract Features
        train_features, train_labels = self.get_features(
            model, datamodule, "train", device
        )
        val_features, val_labels = self.get_features(
            model, datamodule, "val", device
        )
        
        # Convert to numpy
        train_features_np = train_features.numpy()
        train_labels_np = train_labels.numpy()
        val_features_np = val_features.numpy()
        val_labels_np = val_labels.numpy()
        
        logger.info(f"Fitting k-NN (k={self.k})...")
        start_time = time.time()
        
        knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        knn.fit(train_features_np, train_labels_np)
        
        logger.info("Predicting validation set...")
        preds = knn.predict(val_features_np)
        
        # Calculate accuracy
        acc = (preds == val_labels_np).mean()
        duration = time.time() - start_time
        
        logger.info(f"k-NN finished in {duration:.2f}s. Acc: {acc:.4f}")
        
        return {f"knn_{self.k}_acc": acc, "knn_time": duration}
