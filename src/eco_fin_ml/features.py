from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    polynomial_degree: int,
    enable_feature_engineering: bool,
) -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if enable_feature_engineering and polynomial_degree > 1:
        numeric_steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    max_features: int | None,
    selection_method: str,
) -> Tuple[np.ndarray, SelectKBest | None]:
    if max_features is None:
        return X, None

    if task == "classification":
        scoring = f_classif if selection_method == "f_test" else mutual_info_classif
    else:
        scoring = f_regression if selection_method == "f_test" else mutual_info_regression

    selector = SelectKBest(score_func=scoring, k=max_features)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector
