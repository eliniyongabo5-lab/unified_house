"""Validation-selected classical baselines for the revised task datasets.

This module expects ``TaskData.x`` and ``TaskData.split`` to have already been
prepared by :mod:`revised_data`.  It deliberately neither fits preprocessing
transforms nor reads or scores the held-out test partitions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def _specifications(task: str, seed: int):
    """Return bounded, deterministic candidate grids for one task."""
    if task == "slope":
        return {
            "dummy": [(DummyClassifier(strategy="prior"), {})],
            "logistic_regression": [
                (LogisticRegression(C=c, class_weight=weight, max_iter=3000,
                                    random_state=seed),
                 {"C": c, "class_weight": weight})
                for c in (0.01, 0.1, 1.0, 10.0)
                for weight in (None, "balanced")
            ],
            "rbf_svc": [
                (SVC(C=c, gamma=gamma, kernel="rbf", probability=True,
                     class_weight="balanced", random_state=seed),
                 {"C": c, "gamma": gamma, "class_weight": "balanced"})
                for c in (0.1, 1.0, 10.0)
                for gamma in ("scale", 0.1, 1.0)
            ],
            "random_forest": [
                (RandomForestClassifier(
                    n_estimators=200, max_depth=depth,
                    min_samples_leaf=leaf, class_weight="balanced",
                    random_state=seed, n_jobs=2),
                 {"n_estimators": 200, "max_depth": depth,
                  "min_samples_leaf": leaf, "class_weight": "balanced"})
                for depth in (None, 6, 12) for leaf in (1, 4)
            ],
            "hist_gradient_boosting": [
                (HistGradientBoostingClassifier(
                    learning_rate=rate, max_leaf_nodes=leaves,
                    l2_regularization=l2, max_iter=200,
                    early_stopping=False, random_state=seed),
                 {"learning_rate": rate, "max_leaf_nodes": leaves,
                  "l2_regularization": l2, "max_iter": 200,
                  "early_stopping": False})
                for rate in (0.05, 0.1) for leaves in (15, 31)
                for l2 in (0.0, 1.0)
            ],
        }
    return {
        "dummy": [(DummyRegressor(strategy="mean"), {})],
        "ridge": [(Ridge(alpha=alpha), {"alpha": alpha})
                  for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)],
        "rbf_svr": [
            (TransformedTargetRegressor(
                regressor=SVR(C=c, gamma=gamma, epsilon=epsilon,
                              kernel="rbf"),
                transformer=StandardScaler()),
             {"C": c, "gamma": gamma, "epsilon": epsilon,
              "target_transform": "StandardScaler"})
            for c in (0.1, 1.0, 10.0) for gamma in ("scale", 0.1)
            for epsilon in (0.01, 0.1)
        ],
        "random_forest": [
            (RandomForestRegressor(
                n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                random_state=seed, n_jobs=2),
             {"n_estimators": 200, "max_depth": depth,
              "min_samples_leaf": leaf})
            for depth in (None, 8, 16) for leaf in (1, 3)
        ],
        "hist_gradient_boosting": [
            (HistGradientBoostingRegressor(
                learning_rate=rate, max_leaf_nodes=leaves,
                l2_regularization=l2, max_iter=200,
                early_stopping=False, random_state=seed),
             {"learning_rate": rate, "max_leaf_nodes": leaves,
              "l2_regularization": l2, "max_iter": 200,
              "early_stopping": False})
            for rate in (0.05, 0.1) for leaves in (15, 31)
            for l2 in (0.0, 1.0)
        ],
    }


def _validation_score(task: str, model: Any, x: np.ndarray,
                      y: np.ndarray) -> float:
    if task == "slope":
        probability = _stable_probability(model, x)
        return float(balanced_accuracy_score(y, probability >= 0.5))
    prediction = np.maximum(np.asarray(model.predict(x), dtype=float), 0.0)
    return float(np.sqrt(mean_squared_error(y, prediction)))


def _stable_probability(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_)
    stable = np.flatnonzero(classes == 1)
    if len(stable) != 1:
        raise ValueError("Saved classifier does not have stable class label 1")
    return probabilities[:, stable[0]]


def _json_value(value: Any) -> Any:
    """Convert numpy values in records without weakening JSON validation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_records(output: Path, tuning: list[dict[str, Any]],
                   selected: list[dict[str, Any]]) -> None:
    """Checkpoint records after each family so long runs expose progress."""
    (output / "tuning_records.json").write_text(
        json.dumps(tuning, indent=2) + "\n")
    (output / "selected_models.json").write_text(
        json.dumps(selected, indent=2) + "\n")


def fit_baselines(data, output, seed: int = 42) -> list[dict[str, Any]]:
    """Tune and save classical models using only fixed train/validation splits.

    Returns one record for the selected model in every task/family pair.  All
    candidate results, including failures and fit durations, are also written
    to ``tuning_records.json``.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tuning: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for task, task_data in data.items():
        if task_data.x is None or task_data.split is None:
            raise ValueError(f"{task}: prepared x and fixed split are required")
        train = np.asarray(task_data.split["train"], dtype=int)
        val = np.asarray(task_data.split["val"], dtype=int)
        x_train, y_train = task_data.x[train], task_data.target[train]
        x_val, y_val = task_data.x[val], task_data.target[val]
        maximize = task == "slope"

        for family, candidates in _specifications(task, seed).items():
            print(f"classical baseline: {task}/{family} "
                  f"({len(candidates)} candidates)", flush=True)
            best_model = None
            best_record = None
            for candidate_index, (model, params) in enumerate(candidates):
                started = time.perf_counter()
                record: dict[str, Any] = {
                    "task": task, "family": family,
                    "candidate": candidate_index, "params": params,
                    "model_params": _json_value(model.get_params(deep=False)),
                }
                try:
                    with threadpool_limits(limits=2):
                        model.fit(x_train, y_train)
                    score = _validation_score(task, model, x_val, y_val)
                    if not np.isfinite(score):
                        raise ValueError("validation score is not finite")
                    record.update(status="ok", val_score=score)
                    if (best_record is None or
                            (maximize and score > best_record["val_score"]) or
                            (not maximize and score < best_record["val_score"])):
                        best_model, best_record = model, record
                except Exception as exc:  # retain failures for auditability
                    record.update(status="failed", val_score=None,
                                  error=f"{type(exc).__name__}: {exc}")
                finally:
                    record["fit_seconds"] = time.perf_counter() - started
                    tuning.append(_json_value(record))

            if best_model is None or best_record is None:
                _write_records(output, tuning, selected)
                raise RuntimeError(f"{task}/{family}: every candidate failed")

            model_dir = output / task
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{family}.joblib"
            joblib.dump(best_model, model_path)
            chosen = {
                "task": task,
                "family": family,
                "params": _json_value(best_record["params"]),
                "val_score": best_record["val_score"],
                "model_path": str(model_path),
                "fit_seconds": best_record["fit_seconds"],
                "model_params": _json_value(best_model.get_params(deep=False)),
            }
            selected.append(chosen)
            _write_records(output, tuning, selected)
            print(f"classical baseline: selected {task}/{family} "
                  f"val_score={best_record['val_score']:.6g}", flush=True)

    return selected


def predict_baseline(model_path, x, task: str | None = None) -> np.ndarray:
    """Load a saved baseline and return stable-class probability or prediction."""
    model = joblib.load(model_path)
    x = np.asarray(x)
    classification = task == "slope" or (
        task is None and hasattr(model, "predict_proba"))
    if classification:
        return _stable_probability(model, x)
    return np.maximum(np.asarray(model.predict(x), dtype=float), 0.0)


# Short spelling retained for callers that use the module alongside
# revised_experiments.predict.
predict = predict_baseline
