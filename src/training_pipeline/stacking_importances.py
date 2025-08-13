# src/training_pipeline/stacking_importance.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.pipeline import Pipeline
except Exception:
    Pipeline = None  # opcional


def _unwrap_estimator(est):
    """Si viene como Pipeline, devuelve el último paso (o 'est' si existe)."""
    if Pipeline is not None and isinstance(est, Pipeline):
        if "est" in est.named_steps:
            return est.named_steps["est"]
        # último paso como fallback
        return list(est.named_steps.values())[-1]
    return est


def _feature_importances_for_base(est, n_features: int) -> np.ndarray:
    """
    Importancia por feature para un modelo base ya entrenado.
    Árboles: feature_importances_
    Lineales: |coef_|
    Si no existe, devuelve ceros.
    """
    est_u = _unwrap_estimator(est)
    if hasattr(est_u, "feature_importances_") and est_u.feature_importances_ is not None:
        imp = np.asarray(est_u.feature_importances_, dtype=float)
        if imp.shape[0] != n_features:
            imp = np.pad(imp, (0, max(0, n_features - imp.shape[0])))[:n_features]
        return imp

    if hasattr(est_u, "coef_") and est_u.coef_ is not None:
        coef = np.ravel(np.asarray(est_u.coef_, dtype=float))
        if coef.shape[0] != n_features:
            coef = np.pad(coef, (0, max(0, n_features - coef.shape[0])))[:n_features]
        return np.abs(coef)

    return np.zeros(n_features, dtype=float)


def stacked_feature_importance(
    stack_model,
    feature_names: Sequence[str],
) -> np.ndarray:
    """
    Combina importancias de modelos base ponderadas por coeficiente del meta-modelo.
    Si passthrough=True, también suma los coeficientes del meta sobre las features originales.
    Retorna vector normalizado (suma = 1).
    """
    n_feats = len(feature_names)
    base_models = list(stack_model.estimators_)
    meta = _unwrap_estimator(stack_model.final_estimator_)

    # coeficientes del meta (largo = n_base [+ n_feats si passthrough=True])
    if hasattr(meta, "coef_") and meta.coef_ is not None:
        meta_coef = np.ravel(np.asarray(meta.coef_, dtype=float))
    else:
        meta_coef = np.ones(len(base_models), dtype=float)  # fallback

    global_imp = np.zeros(n_feats, dtype=float)

    # contribución de cada base ponderada por el coef del meta
    for i, base in enumerate(base_models):
        base_imp = _feature_importances_for_base(base, n_feats)
        s = base_imp.sum()
        if s > 0:
            base_imp = base_imp / s
        weight = abs(meta_coef[i]) if i < len(meta_coef) else 0.0
        global_imp += weight * base_imp

    # passthrough: coeficientes extra del meta sobre las features originales
    if len(meta_coef) > len(base_models):
        extra = meta_coef[len(base_models):]
        extra = np.abs(extra)
        if extra.shape[0] != n_feats:
            extra = np.pad(extra, (0, max(0, n_feats - extra.shape[0])))[:n_feats]
        global_imp += extra

    total = global_imp.sum()
    if total > 0:
        global_imp = global_imp / total
    return global_imp


def _slug(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "-")
    )


def save_stacking_importances(
    stack_model,
    X_train: Union[pd.DataFrame, np.ndarray],
    reports_dir: Union[str, Path],
    prueba: str,
    prefix: str = "stacking_allvars",
    top_k: int = 20,
) -> Tuple[Path, Path]:
    """
    Calcula importancias globales del stacking y guarda:
      - CSV completo (todas las features, importancia normalizada)
      - PNG Top-k con barras horizontales
    Retorna (csv_path, png_path).
    """
    # nombres de columnas
    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
    else:
        # si es numpy, inventamos nombres genéricos
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    importances = stacked_feature_importance(stack_model, feature_names)

    fi_dir = Path(reports_dir) / "feature_importances"
    fi_dir.mkdir(parents=True, exist_ok=True)

    slug_prueba = _slug(prueba)
    csv_path = fi_dir / f"{prefix}_{slug_prueba}_importances.csv"
    png_path = fi_dir / f"{prefix}_{slug_prueba}_top{min(top_k, len(feature_names))}.png"

    # CSV completo
    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    df_imp.to_csv(csv_path, index=False)

    # Figura Top-k
    k = min(top_k, len(feature_names))
    idx_sorted = np.argsort(importances)[::-1][:k]
    top_feats = [feature_names[i] for i in idx_sorted]
    top_vals = importances[idx_sorted]

    plt.figure(figsize=(10, max(6, int(k * 0.35))))
    plt.barh(range(k), top_vals[::-1])
    plt.yticks(range(k), [f.replace("_", " ") for f in top_feats[::-1]])
    plt.xlabel("Importancia (normalizada)")
    plt.title(f"Top {k} features — Stacking All Vars — {prueba}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, png_path
