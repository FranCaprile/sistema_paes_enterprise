# src/prediction_service/batch_predict.py
import sys
from pathlib import Path
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text

# Asegurar import de utilidades existentes
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from training_pipeline.load_data import cargar_configuracion, cargar_dataset  # ya lo usas
# ---------- Helpers ----------

def slug(s: str) -> str:
    # mismo slug del training: minúsculas, sin espacios ni puntos, sin tildes
    s = s.lower().replace(" ", "_").replace(".", "")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def normalize_rut_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[^0-9A-Za-z]", "", regex=True)
         .str.strip()
    )

def pick_latest_model(ml_dir: Path, prueba_slug: str) -> Path:
    # busca el último *.joblib que termina en _stacking_allvars_<slug>.joblib
    patt = re.compile(rf".*_stacking_allvars_{re.escape(prueba_slug)}\.joblib$")
    candidates = sorted([p for p in ml_dir.glob("*.joblib") if patt.match(p.name)],
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No encontré modelos Stacking para '{prueba_slug}' en {ml_dir}")
    return candidates[0]

def load_rmse_from_metrics(reports_dir: Path, prueba_col: str) -> float:
    m = pd.read_csv(reports_dir / "metrics.csv")
    # buscar fila del Stacking (nombre contiene 'Stacking'); tomar la métrica de esa prueba
    stacking_rows = m[m["Modelo"].str.contains("Stacking", case=False, na=False)]
    if stacking_rows.empty:
        return np.nan
    # En metrics wide, las columnas de pruebas son exactamente los nombres de prueba
    if prueba_col not in stacking_rows.columns:
        return np.nan
    # tomar el primer valor no nulo
    val = stacking_rows[prueba_col].dropna()
    return float(val.iloc[0]) if len(val) else np.nan

def get_ruts_iii_medio_last_year(db_uri: str) -> pd.DataFrame:
    """
    Intenta 2 consultas:
      (A) usando student_sections(year) + level(name='III Medio') para (año actual - 1)
      (B) fallback: alumnos hoy en IV Medio (si no hay student_sections.year)
    Devuelve columnas: student_rut, student_name (name puede venir vacío).
    """
    eng = create_engine(db_uri)
    year_last = datetime.now().year - 1
    with eng.connect() as cx:
        # A) con student_sections.year
        try:
            q = text("""
                SELECT DISTINCT s.rut AS student_rut,
                                COALESCE(s.fullname, s.name, '') AS student_name
                FROM student_sections ss
                JOIN student s ON s.id = ss.student_id
                JOIN level   l ON l.id = ss.level_id
                WHERE l.name ILIKE 'III Medio'
                  AND (ss.year = :yy OR ss.year::text = :yy_txt)
            """)
            df = pd.read_sql(q, cx, params={"yy": year_last, "yy_txt": str(year_last)})
            if not df.empty:
                df["student_rut"] = normalize_rut_series(df["student_rut"])
                return df.drop_duplicates(subset=["student_rut"])
        except Exception:
            pass

        # B) fallback: hoy en IV Medio
        try:
            q2 = text("""
                SELECT DISTINCT s.rut AS student_rut,
                                COALESCE(s.fullname, s.name, '') AS student_name
                FROM student s
                JOIN level l ON l.id = s.level_id
                WHERE l.name ILIKE 'III Medio'
            """)
            df2 = pd.read_sql(q2, cx)
            df2["student_rut"] = normalize_rut_series(df2["student_rut"])
            return df2.drop_duplicates(subset=["student_rut"])
        except Exception as e:
            raise RuntimeError(f"No pude obtener RUTs de alumnos: {e}")

def build_allvars_features(dfs: dict, ruts: pd.Series) -> pd.DataFrame:
    """
    Construye el set de features 'All Vars' para un conjunto de RUTs.
    Usa los mismos archivos de data/processed que el training.
    """
    # Copias locales y normalización de clave
    def norm(df, rut_col_guess=("student_rut","rut")):
        for c in rut_col_guess:
            if c in df.columns:
                df = df.rename(columns={c: "student_rut"})
                break
        if "student_rut" not in df.columns:
            raise KeyError("No se encontró columna RUT en un DataFrame de entrada.")
        df["student_rut"] = normalize_rut_series(df["student_rut"])
        # quitar duplicados en columnas
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    df_sepa  = norm(dfs["df_pruebas_sepa"])
    df_ind   = norm(dfs["df_indicadores"])
    df_tax   = norm(dfs["df_taxonomia"])
    df_pca   = norm(dfs["df_pca"])
    df_diff  = norm(dfs["df_avg_diff"]) if "df_avg_diff" in dfs else pd.DataFrame(columns=["student_rut"])

    # Empezar del padrón de RUTs
    base = pd.DataFrame({"student_rut": normalize_rut_series(ruts)}).drop_duplicates()

    # Merge incremental
    df = (base
          .merge(df_sepa, on="student_rut", how="left")
          .merge(df_ind,  on="student_rut", how="left")
          .merge(df_tax,  on="student_rut", how="left")
          .merge(df_pca,  on="student_rut", how="left")
          .merge(df_diff, on="student_rut", how="left"))

    # Quitar targets si estuvieran, y metadatos
    targets = ["C. Lectora", "Matemática", "Historia", "Ciencias", "M2"]
    drop_cols = ["Nombre", "Admisión"] + [c for c in targets if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Mantener solo numéricas (como en training)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.fillna(0.0)
    # Si el modelo tiene feature_names_in_ lo usaremos para re-alinear columnas luego
    return base.join(num)

def align_features_for_model(X: pd.DataFrame, model) -> pd.DataFrame:
    # Respeta el orden/selección de columnas del modelo si existe atributo
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X.reindex(columns=cols, fill_value=0.0)
    # StackingRegressor a veces guarda en final_estimator_
    if hasattr(model, "final_estimator_") and hasattr(model.final_estimator_, "feature_names_in_"):
        cols = list(model.final_estimator_.feature_names_in_)
        return X.reindex(columns=cols, fill_value=0.0)
    return X  # fallback

# ---------- Main ----------

def main():
    cfg = cargar_configuracion()
    # 1) Cargar datasets procesados
    rutas = cfg["output_paths"].copy()
    dfs = {k: cargar_dataset(Path(v)) for k, v in rutas.items()}
    # 2) Conexión a Neon (si está en config.yml)
    db_uri = None
    if "db" in cfg and "url" in cfg["db"]:
        db_uri = cfg["db"]["url"]

    if not db_uri:
        print("⚠️ No hay db.url en config.yml → solo puedo proyectar si me das manualmente los RUTs.")
        return

    # 3) RUTs objetivo (III Medio año pasado)
    padron = get_ruts_iii_medio_last_year(db_uri)
    if padron.empty:
        print("⚠️ No se encontraron alumnos candidatos.")
        return

    # 4) Features All Vars
    feats = build_allvars_features(dfs, padron["student_rut"])
    # Guardar nombre si venía del DB
    feats = padron[["student_rut","student_name"]].merge(feats, on="student_rut", how="right")

    # 5) Cargar modelos Stacking (Lenguaje / Matemática)
    ml_dir = Path(cfg["paths"]["ml_models_dir"])
    rep_dir = Path(cfg["paths"]["reports_dir"])
    slug_len = slug("C. Lectora")
    slug_mat = slug("Matemática")

    mdl_len_path = pick_latest_model(ml_dir, slug_len)
    mdl_mat_path = pick_latest_model(ml_dir, slug_mat)

    mdl_len = joblib.load(mdl_len_path)
    mdl_mat = joblib.load(mdl_mat_path)

    # Alinear columnas según el modelo
    X_num = feats.drop(columns=["student_name","student_rut"], errors="ignore")
    X_len = align_features_for_model(X_num, mdl_len)
    X_mat = align_features_for_model(X_num, mdl_mat)

    # 6) Predicciones
    preds_len = mdl_len.predict(X_len)
    preds_mat = mdl_mat.predict(X_mat)

    # 7) Error aprox (RMSE) por prueba
    rmse_len = load_rmse_from_metrics(rep_dir, "C. Lectora")
    rmse_mat = load_rmse_from_metrics(rep_dir, "Matemática")

    out = pd.DataFrame({
        "student_rut": feats["student_rut"],
        "student_name": feats.get("student_name", pd.Series([""]*len(feats))),
        "pred_lenguaje": preds_len,
        "err_lenguaje_rmse": rmse_len,
        "pred_matematica": preds_mat,
        "err_matematica_rmse": rmse_mat,
    })

    out = out.sort_values(["student_name","student_rut"]).reset_index(drop=True)

    year_now = datetime.now().year
    out_path = rep_dir / f"projections_{year_now}.csv"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Proyecciones guardadas en: {out_path}")
    print(out.head(10))

if __name__ == "__main__":
    main()
