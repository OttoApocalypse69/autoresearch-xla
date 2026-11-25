"""
Resumable multi-chip FSDP pretraining engine for scaling autoresearch-xla
architectures up to 1B+ parameters on Kaggle TPU v5e-8 and Colab GPU/TPU.
"""

from scale.config import ModelSpec, TrainConfig, get_model_preset

__all__ = ["ModelSpec", "TrainConfig", "get_model_preset"]
