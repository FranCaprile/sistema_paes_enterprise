# src/training_pipeline/run_pipeline.py
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Asegurar que el directorio raíz esté en sys.path para importar config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from load_data import cargar_dataset, cargar_configuracion
from dataset_preparation import preparar_dataset
from models import obtener_modelos
from train_model import entrenar_un_modelo
from evaluate import resultados_a_df, guardar_metricas


def main():
    # 1️⃣ Cargar configuración
    cfg = cargar_configuracion()  # carga sin argumentos

    # 2️⃣ Cargar todos los datasets en un dict
    rutas = cfg['output_paths'].copy()
    rutas['df_paes'] = cfg['paths']['paes_encrypted_xlsx']
    dfs = {
        nombre: cargar_dataset(Path(path), tipo='excel' if nombre == 'df_paes' else 'csv')
        for nombre, path in rutas.items()
    }

    all_results = []
    modelos_store = {}

    # 3️⃣ Iterar sobre cada combinación prueba-variable
    for prueba in cfg['pruebas']:
        for variable in cfg['variables']:
            print(f"> Procesando prueba «{prueba}», variable «{variable}»")
            # Preparar dataset
            try:
                X_train, X_test, y_train, y_test = preparar_dataset(dfs, prueba, variable)
            except Exception as e:
                print(f"  ⚠️ Omitido por error en preparación: {e}")
                continue

            # 4️⃣ Saltar si no hay datos válidos
            if X_train.shape[0] == 0 or X_train.shape[1] == 0:
                print(f"  ⚠️ No hay datos para {prueba}/{variable}, saltando.")
                continue

            # 5️⃣ Entrenar cada modelo
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

            # 6️⃣ Almacenar resultados y modelos
            df_res = resultados_a_df(resultados)
            df_res['prueba'] = prueba
            df_res['variable'] = variable
            all_results.append(df_res)
            modelos_store[(prueba, variable)] = modelos

    # 7️⃣ Concatenar y guardar reporte de métricas completo
    rep_dir = Path(cfg['paths']['reports_dir'])
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not all_results:
        print("❌ No se generaron resultados de métricas.")
        return
    
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(rep_dir / 'metrics.csv', index=False)

    
    # 8️⃣ Seleccionar mejor modelo global (menor RMSE)
    best_row = df_all.sort_values('rmse').iloc[0]
    best_prueba, best_var, best_model = (
        best_row['prueba'], best_row['variable'], best_row['modelo']
    )
    best_obj = modelos_store[(best_prueba, best_var)][best_model]

    # 9️⃣ Guardar el mejor modelo
    ml_dir = Path(cfg['paths']['ml_models_dir'])
    ml_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_obj, ml_dir / f"{best_prueba}_{best_var}_{best_model}.joblib")

    # 🔟 Graficar importancia de características si existe
    if hasattr(best_obj, 'feature_importances_'):
        importances = best_obj.feature_importances_
        # Usar X_train de la iteración ganadora
        X_feat = preparar_dataset(dfs, best_prueba, best_var)[0]
        features = X_feat.columns
        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title(f"Feature Importance: {best_model}")
        fig.tight_layout()
        fig.savefig(rep_dir / 'feature_importance.png')

    print(f"✅ Pipeline completo. Artefactos generados en:")
    print(f"   - Modelo:   {ml_dir}")
    print(f"   - Métricas: {rep_dir}")
    
if __name__ == '__main__':
    main()
