import pandas as pd

def resultados_a_df(resultados: dict) -> pd.DataFrame:
    rows = []
    for nombre, mets in resultados.items():
        rows.append({'modelo': nombre, **mets})
    return pd.DataFrame(rows)

def guardar_metricas(df: pd.DataFrame, path):
    df.to_csv(path, index=False)
