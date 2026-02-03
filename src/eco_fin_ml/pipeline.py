from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from eco_fin_ml.config import ProjectConfig
from eco_fin_ml.data import load_dataset, split_time_series
from eco_fin_ml.evaluate import (
    classification_confusion,
    classification_curves,
    classification_metrics,
    regression_metrics,
)
from eco_fin_ml.explain import compute_lime, compute_shap, plot_pdp, plot_shap_summary
from eco_fin_ml.features import build_preprocessor, select_features
from eco_fin_ml.models import (
    TorchSequenceConfig,
    TorchSequenceTrainer,
    build_models,
    build_stacking,
    fit_arima,
    fit_garch,
    fit_sarima,
    make_sequence_data,
    evaluate_arima,
    HybridArimaResidual,
    LSTMModel,
    GRUModel,
    TransformerModel,
)
from eco_fin_ml.reporting import ensure_dir, save_json, save_metrics, write_methodology


LOGGER = logging.getLogger(__name__)


def setup_logging(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def prepare_data(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = load_dataset(config.data.path)
    if config.data.time_column:
        df = df.sort_values(config.data.time_column)

    y = df[config.data.target]
    X = df.drop(columns=[config.data.target] + config.data.id_columns)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        config.features.polynomial_degree,
        config.features.enable_feature_engineering,
    )

    return X, y, preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if config.training.time_series and config.data.time_column:
        data = pd.concat([X, y], axis=1)
        train, val, test = split_time_series(data, config.data.test_size, config.data.val_size)
        return (
            train.drop(columns=[config.data.target]),
            val.drop(columns=[config.data.target]),
            test.drop(columns=[config.data.target]),
            train[config.data.target],
            val[config.data.target],
            test[config.data.target],
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config.data.test_size + config.data.val_size,
        random_state=config.data.random_state,
        stratify=y if config.data.task == "classification" else None,
    )
    val_ratio = config.data.val_size / (config.data.test_size + config.data.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_ratio,
        random_state=config.data.random_state,
        stratify=y_temp if config.data.task == "classification" else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_model(model: Any, param_grid: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray, config: ProjectConfig) -> Any:
    if not param_grid:
        return model
    tuner = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=config.training.n_iter,
        cv=config.training.cv_folds,
        scoring=config.training.scoring,
        random_state=config.data.random_state,
        n_jobs=-1,
    )
    tuner.fit(X, y)
    return tuner.best_estimator_


def evaluate_model(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    if task == "classification":
        if y_prob is None:
            y_prob = y_pred
        return classification_metrics(y_true, y_pred, y_prob)
    return regression_metrics(y_true, y_pred)


def run_pipeline(config: ProjectConfig) -> Dict[str, Any]:
    output_dir = ensure_dir(config.output.output_dir) / config.output.experiment_name
    setup_logging(output_dir)

    LOGGER.info("Loading and preparing data")
    X, y, preprocessor = prepare_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    LOGGER.info("Preprocessing data")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed, selector = select_features(
        X_train_processed, y_train.values, config.data.task, config.features.max_features, config.features.selection_method
    )
    if selector is not None:
        X_val_processed = selector.transform(X_val_processed)
        X_test_processed = selector.transform(X_test_processed)

    feature_names = [f"feature_{idx}" for idx in range(X_train_processed.shape[1])]

    metrics_summary: Dict[str, Dict[str, float]] = {}
    models = build_models(config.data.task)

    estimators = list(models.items())[:3]
    models["stacking"] = build_stacking(config.data.task, estimators)

    LOGGER.info("Training models")
    for name, model in models.items():
        LOGGER.info("Fitting model: %s", name)
        model.fit(X_train_processed, y_train)
        if config.data.task == "classification":
            y_prob = model.predict_proba(X_val_processed)[:, 1]
            y_pred = model.predict(X_val_processed)
        else:
            y_pred = model.predict(X_val_processed)
            y_prob = None
        metrics_summary[name] = evaluate_model(config.data.task, y_val.values, y_pred, y_prob)

    LOGGER.info("Evaluating time-series models")
    if config.training.time_series:
        arima_result = fit_arima(y_train, (1, 1, 1))
        arima_forecast = arima_result.forecast(steps=len(y_test))
        metrics_summary["arima"] = {"rmse": evaluate_arima(y_test, arima_forecast.values)}

        sarima_result = fit_sarima(y_train, (1, 1, 1), (1, 1, 1, 12))
        sarima_forecast = sarima_result.forecast(steps=len(y_test))
        metrics_summary["sarima"] = {"rmse": evaluate_arima(y_test, sarima_forecast.values)}

        garch_result = fit_garch(y_train)
        metrics_summary["garch"] = {"loglikelihood": garch_result.loglikelihood}

        hybrid_model = HybridArimaResidual((1, 1, 1), models["random_forest"])
        hybrid_model.fit(y_train, X_train_processed)
        hybrid_forecast = hybrid_model.predict(X_test_processed, steps=len(y_test))
        metrics_summary["hybrid_arima_ml"] = {"rmse": evaluate_arima(y_test, hybrid_forecast)}

    if config.training.enable_deep_learning:
        LOGGER.info("Training deep learning sequence models")
        seq_config = TorchSequenceConfig()
        X_seq, y_seq = make_sequence_data(X_train_processed, y_train.values, seq_config.seq_length)
        if len(X_seq) > 0:
            input_size = X_seq.shape[-1]
            output_size = 1
            model_map = {
                "lstm": LSTMModel(input_size, 32, output_size, config.data.task),
                "gru": GRUModel(input_size, 32, output_size, config.data.task),
                "transformer": TransformerModel(input_size, 32, output_size, config.data.task),
            }
            for name, torch_model in model_map.items():
                trainer = TorchSequenceTrainer(torch_model, config.data.task, seq_config)
                trainer.fit(X_seq, y_seq)
                X_test_seq, y_test_seq = make_sequence_data(X_test_processed, y_test.values, seq_config.seq_length)
                if len(X_test_seq) == 0:
                    continue
                preds = trainer.predict(X_test_seq)
                if config.data.task == "classification":
                    y_pred = (preds >= 0.5).astype(int)
                    metrics_summary[f"{name}_seq"] = evaluate_model(
                        config.data.task, y_test_seq, y_pred, preds
                    )
                else:
                    metrics_summary[f"{name}_seq"] = evaluate_model(
                        config.data.task, y_test_seq, preds
                    )

    LOGGER.info("Saving metrics")
    metrics_df = save_metrics(metrics_summary, output_dir, config.output.experiment_name)

    best_model_name = metrics_df[metrics_df.columns[0]].idxmin() if config.data.task == "regression" else metrics_df["roc_auc"].idxmax()

    LOGGER.info("Best model selected: %s", best_model_name)

    methodology_notes = [
        "Loaded user-supplied dataset with configurable preprocessing steps.",
        "Applied feature engineering and selection based on configuration.",
        "Evaluated econometric, ML, and optional deep learning models.",
        "Selected best model using validation metrics and documented results.",
    ]
    write_methodology(methodology_notes, output_dir, config.output.experiment_name)
    save_json({"best_model": best_model_name, "metrics": metrics_summary}, output_dir, "summary")

    if config.training.use_shap:
        LOGGER.info("Computing SHAP explanations")
        best_model = models.get(best_model_name)
        if best_model is not None:
            shap_result = compute_shap(best_model, X_test_processed, feature_names)
            shap_path = pathlib.Path(output_dir) / "shap_summary.png"
            plot_shap_summary(shap_result["values"], feature_names, str(shap_path))

    if config.training.use_lime:
        LOGGER.info("Computing LIME explanations")
        best_model = models.get(best_model_name)
        if best_model is not None:
            lime_exp = compute_lime(best_model, X_train_processed, y_train.values, feature_names, config.data.task)
            lime_path = pathlib.Path(output_dir) / "lime_explanation.txt"
            lime_path.write_text(str(lime_exp.as_list()), encoding="utf-8")

    if config.training.use_pdp:
        LOGGER.info("Creating PDP plots")
        best_model = models.get(best_model_name)
        if best_model is not None:
            pdp_path = pathlib.Path(output_dir) / "pdp.png"
            plot_pdp(best_model, X_test_processed, [0], str(pdp_path))

    if config.data.task == "classification":
        LOGGER.info("Generating classification plots")
        best_model = models.get(best_model_name)
        if best_model is not None:
            y_prob = best_model.predict_proba(X_test_processed)[:, 1]
            y_pred = best_model.predict(X_test_processed)
            curves = classification_curves(y_test.values, y_prob)
            plt.figure()
            plt.plot(curves["fpr"], curves["tpr"], label="ROC")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(pathlib.Path(output_dir) / "roc_curve.png")
            plt.close()

            conf = classification_confusion(y_test.values, y_pred)
            plt.figure()
            plt.imshow(conf, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(pathlib.Path(output_dir) / "confusion_matrix.png")
            plt.close()
    else:
        LOGGER.info("Generating prediction plot")
        best_model = models.get(best_model_name)
        if best_model is not None:
            y_pred = best_model.predict(X_test_processed)
            plt.figure()
            plt.plot(y_test.values, label="Actual")
            plt.plot(y_pred, label="Predicted")
            plt.legend()
            plt.tight_layout()
            plt.savefig(pathlib.Path(output_dir) / "prediction_vs_actual.png")
            plt.close()

    return {
        "metrics": metrics_summary,
        "best_model": best_model_name,
        "output_dir": str(output_dir),
    }
