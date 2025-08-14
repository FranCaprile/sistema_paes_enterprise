from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import (
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _lin_pipe(estimator):
    """Pequeño helper: escalar antes de modelos lineales."""
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("est", estimator)])


def make_base_models(include: List[str] | None = None, random_state: int = 42):
    """
    Devuelve lista de (nombre, estimador) para usar en el stacking.
    Cambia `include` si quieres otra combinación.
    """
    catalog = {
        "rf":  RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=random_state),
        "gbr": GradientBoostingRegressor(random_state=random_state),
        "ridge": _lin_pipe(Ridge(random_state=random_state)),
        "bayesridge": _lin_pipe(BayesianRidge()),
        # Extras si quieres probar:
        "linreg": _lin_pipe(LinearRegression()),
        "bag": BaggingRegressor(n_estimators=200, n_jobs=-1, random_state=random_state),
        "ada": AdaBoostRegressor(random_state=random_state),
    }
    if include is None:
        include = ["rf", "gbr", "ridge", "bayesridge"]
    return [(name, clone(catalog[name])) for name in include]


def build_stacked_regressor(
    base_models=None,
    final_estimator=None,
    cv: int = 5,
    passthrough: bool = False,
    n_jobs: int = -1,
):
    """
    Crea un StackingRegressor. Por defecto usa BayesianRidge de meta-modelo.
    """
    if base_models is None:
        base_models = make_base_models()
    if final_estimator is None:
        final_estimator = BayesianRidge()
    return StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        n_jobs=n_jobs,
    )


def fit_and_eval(model, X_train, y_train, X_test=None, y_test=None) -> Tuple[object, Dict]:
    """Entrena y (si hay test) devuelve métricas {RMSE, R2}."""
    model.fit(X_train, y_train)
    metrics = {}
    if X_test is not None and y_test is not None and len(X_test) > 0:
        preds = model.predict(X_test)
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "R2": float(r2_score(y_test, preds)),
        }
    return model, metrics


def save_model(model, outdir: str | Path, variable: str, prueba: str) -> str:
    """Guarda el modelo con nombre legible."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = prueba.lower().replace(" ", "_").replace(".", "")
    path = outdir / f"stacked_{variable}_{slug}.joblib"
    dump(model, path)
    return str(path)
