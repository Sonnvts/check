from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay


def compute_shap(model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return {"explainer": explainer, "values": shap_values, "feature_names": feature_names}


def plot_shap_summary(shap_values: Any, feature_names: List[str], output_path: str) -> None:
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_lime(model: Any, X: np.ndarray, y: np.ndarray, feature_names: List[str], task: str) -> Any:
    explainer = LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=["class_0", "class_1"] if task == "classification" else ["target"],
        mode="classification" if task == "classification" else "regression",
    )
    return explainer.explain_instance(X[0], model.predict_proba if task == "classification" else model.predict)


def plot_pdp(model: Any, X: np.ndarray, features: List[int], output_path: str) -> None:
    plt.figure()
    PartialDependenceDisplay.from_estimator(model, X, features)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
