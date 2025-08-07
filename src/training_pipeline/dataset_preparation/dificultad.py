import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
import pandas as pd

def dataset_dificultad(dfs: Dict[str, pd.DataFrame], prueba: str):
    df = dfs['df_pruebas_sepa'].merge(dfs['df_avg_diff'], on='student_rut')
    X = df[[c for c in df.columns if c.startswith('diff_')]]
    y = df[prueba]
    return train_test_split(X, y, test_size=0.2, random_state=42)