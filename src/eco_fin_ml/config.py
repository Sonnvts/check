from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    path: str
    target: str
    task: str  # "regression" or "classification"
    time_column: Optional[str] = None
    id_columns: List[str] = field(default_factory=list)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class FeatureConfig:
    enable_feature_engineering: bool = True
    polynomial_degree: int = 2
    max_features: Optional[int] = None
    selection_method: str = "mutual_info"


@dataclass
class TrainingConfig:
    time_series: bool = False
    enable_tuning: bool = True
    n_iter: int = 20
    cv_folds: int = 5
    scoring: Optional[str] = None
    use_shap: bool = True
    use_lime: bool = True
    use_pdp: bool = True
    enable_deep_learning: bool = False


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    experiment_name: str = "default_run"


@dataclass
class ProjectConfig:
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    output: OutputConfig
    models: Dict[str, Any] = field(default_factory=dict)


DEFAULT_METRICS = {
    "regression": ["rmse", "mae", "r2"],
    "classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
}
