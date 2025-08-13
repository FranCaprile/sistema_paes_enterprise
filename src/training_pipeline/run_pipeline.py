# src/training_pipeline/run_pipeline.py
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dataset_preparation.all_vars import dataset_allvars
from stacking_importances import save_stacking_importances

from stacking import make_base_models, build_stacked_regressor, fit_and_eval
from sklearn.linear_model import BayesianRidge

# Asegurar que el directorio raíz esté en sys.path para importar config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from load_data import cargar_dataset, cargar_configuracion
from dataset_preparation import preparar_dataset
from models import obtener_modelos
from train_model import entrenar_un_modelo
from evaluate import resultados_a_df

def main():
    cfg = cargar_configuracion()

    """

    CODIGO PARA MODELOS
    # 2️⃣ Cargar todos los datasets en un dict
    rutas = cfg['output_paths'].copy()
    rutas['df_paes'] = cfg['paths']['paes_encrypted_xlsx']
    dfs = {
        nombre: cargar_dataset(Path(path), tipo='excel' if nombre == 'df_paes' else 'csv')
        for nombre, path in rutas.items()
    }

    all_results = []  # aquí juntamos métricas de todos los modelos (incluido stacking)
    for prueba in cfg['pruebas']:
        for variable in cfg['variables']:
            print(f"> Procesando prueba «{prueba}», variable «{variable}»")
            try:
                X_train, X_test, y_train, y_test = preparar_dataset(dfs, prueba, variable)
            except Exception as e:
                print(f"  ⚠️ Omitido por error en preparación: {e}")
                continue

            if X_train.shape[0] == 0 or X_train.shape[1] == 0:
                print(f"  ⚠️ No hay datos para {prueba}/{variable}, saltando.")
                continue

            # 5️⃣ Entrenar modelos base SOLO para métricas (no guardamos)
            modelos = obtener_modelos()
            resultados = {}
            for name, model in modelos.items():
                try:
                    resultados[name] = entrenar_un_modelo(
                        model, X_train, X_test, y_train, y_test,
                        test_size=cfg['training']['test_size'],
                        random_state=cfg['training']['random_state']
                    )
                except Exception as e:
                    print(f"    ⚠️ Error entrenando {name}: {e}")

            # Convertimos a DF y etiquetamos
            df_res = resultados_a_df(resultados)
            df_res['prueba'] = prueba
            df_res['variable'] = variable
    """

    # 1) Cargar datasets
    rutas = cfg['output_paths'].copy()
    rutas['df_paes'] = cfg['paths']['paes_encrypted_xlsx']
    dfs = {
        nombre: cargar_dataset(Path(path), tipo='excel' if nombre == 'df_paes' else 'csv')
        for nombre, path in rutas.items()
    }

    all_results = []

    for prueba in cfg['pruebas']:
        print(f"\n=== Stacking unificado para prueba «{prueba}» (todas las variables) ===")

        # 2) Dataset unificado con todas las variables
        try:
            X_train, X_test, y_train, y_test = dataset_allvars(
                dfs, target_col=prueba,
                test_size=cfg['training']['test_size'],
                random_state=cfg['training']['random_state']
            )
        except Exception as e:
            print(f"  ⚠️ Error preparando dataset unificado: {e}")
            continue

        if X_train.shape[0] == 0 or X_train.shape[1] == 0:
            print(f"  ⚠️ No hay datos para stacking unificado en «{prueba}», saltando.")
            continue

        # 3) Entrenar stacking (único por prueba)
        base = make_base_models(include=["rf", "gbr", "ridge", "bayesridge"])
        stack = build_stacked_regressor(
            base_models=base,
            final_estimator=BayesianRidge(),
            cv=5,
            passthrough=False
        )
        stack, stack_metrics = fit_and_eval(stack, X_train, y_train, X_test, y_test)

        # 4) Guardar SOLO el stacking con fecha
        ml_dir = Path(cfg['paths']['ml_models_dir'])
        ml_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%d-%m")
        slug_prueba = prueba.lower().replace(" ", "_").replace(".", "")
        fname = f"{date_tag}_stacking_allvars_{slug_prueba}.joblib"
        joblib.dump(stack, ml_dir / fname)
        print(f"  ✓ Stacking (todas variables) guardado en: {ml_dir / fname}")

        # guardar importancias (CSV + PNG)
        csv_path, png_path = save_stacking_importances(
            stack_model=stack,
            X_train=X_train,                               # usa el mismo X_train del stacking
            reports_dir=cfg["paths"]["reports_dir"],
            prueba=prueba,
            prefix="stacking_allvars",
            top_k=20
        )
        print(f"  ✓ Feature importances: {csv_path.name}, {png_path.name}")



        # 5) Añadir UNA fila de métricas del stacking (variable='allvars')
        if stack_metrics:
            all_results.append(pd.DataFrame([{
                "modelo": "Stacking-AllVars(rf,gbr,ridge,br)",
                "rmse": round(stack_metrics["RMSE"], 4),
                "r2": round(stack_metrics["R2"], 4),
                "prueba": prueba,
                "variable": "allvars",
            }]))

        # 6) (Opcional) Métricas de modelos base por variable (NO se guardan modelos)
        for variable in cfg['variables']:
            print(f"> Procesando prueba «{prueba}», variable «{variable}»")
            try:
                X_tr, X_te, y_tr, y_te = preparar_dataset(dfs, prueba, variable)
            except Exception as e:
                print(f"  ⚠️ Omitido por error en preparación: {e}")
                continue

            if X_tr.shape[0] == 0 or X_tr.shape[1] == 0:
                print(f"  ⚠️ No hay datos para {prueba}/{variable}, saltando.")
                continue

            modelos = obtener_modelos()
            resultados = {}
            for name, model in modelos.items():
                try:
                    resultados[name] = entrenar_un_modelo(
                        model, X_tr, X_te, y_tr, y_te,
                        test_size=cfg['training']['test_size'],
                        random_state=cfg['training']['random_state']
                    )
                except Exception as e:
                    print(f"    ⚠️ Error entrenando {name}: {e}")

            df_res = resultados_a_df(resultados)
            df_res['prueba'] = prueba
            df_res['variable'] = variable
            all_results.append(df_res)

    # === GUARDAR ÚNICO CSV DE MÉTRICAS EN FORMATO ANCHO (pretty) ===
    rep_dir = Path(cfg['paths']['reports_dir'])
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not all_results:
        print("❌ No se generaron resultados de métricas.")
        return

    # Unir todas las métricas acumuladas (incluye stacking con variable='allvars')
    df_all = pd.concat(all_results, ignore_index=True)

    # Mapeo de nombres bonitos para la columna 'modelo'
    pretty_name = {
        "LinearRegression": "Regresión Lineal",
        "BayesianRidge": "Regresión Bayesiana",
        "Ridge": "Ridge",
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "AdaBoost": "AdaBoost",
        "Bagging": "Bagging",
        "MLP": "MLP (Neural Net)",
        "Stacking-AllVars(rf,gbr,ridge,br)": "Stacking (All Vars)",
    }

    # Orden deseado de columnas (pruebas)
    order_cols = ["C. Lectora", "Ciencias", "Historia", "Matemática"]

    df_all_w = df_all.copy()
    df_all_w["modelo"] = df_all_w["modelo"].map(pretty_name).fillna(df_all_w["modelo"])
    df_all_w["rmse"] = pd.to_numeric(df_all_w["rmse"], errors="coerce")

    # Pivotear a formato ancho: fila=(Modelo, Variable) / columnas=pruebas / valor=RMSE
    wide = (
        df_all_w.pivot_table(
            index=["modelo", "variable"],
            columns="prueba",
            values="rmse",
            aggfunc="first"
        )
        .reindex(columns=order_cols)  # respeta el orden deseado
        .reset_index()
        .rename(columns={"modelo": "Modelo", "variable": "Variable"})
    )

    # Renombrar cabecera para que quede como tu ejemplo
    if "C. Lectora" in wide.columns:
        wide = wide.rename(columns={"C. Lectora": "C._Lectora"})

    # Redondear
    for c in ["C._Lectora", "Ciencias", "Historia", "Matemática"]:
        if c in wide.columns:
            wide[c] = wide[c].round(2)

    # Ordenar filas y columnas
    wide = wide.sort_values(["Variable", "Modelo"], kind="stable").reset_index(drop=True)
    cols_final = ["Modelo", "C._Lectora", "Ciencias", "Historia", "Matemática", "Variable"]
    wide = wide[[c for c in cols_final if c in wide.columns]]

    # Guardar como ÚNICO archivo de métricas (ancho) llamado 'metrics.csv'
    out_csv = rep_dir / "metrics.csv"
    wide.to_csv(out_csv, index=False)
    print(f"   - Métricas: {out_csv}")


    print(f"✅ Pipeline completo. Artefactos generados en:")
    print(f"   - Modelo (solo stacking): {cfg['paths']['ml_models_dir']}")
    print(f"   - Métricas: {rep_dir}")


if __name__ == '__main__':
    main()