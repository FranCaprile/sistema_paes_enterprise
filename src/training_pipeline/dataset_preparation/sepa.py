import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict

def dataset_sepa(dfs: Dict[str, pd.DataFrame], prueba: str):
    df = dfs['df_pruebas_sepa']
    X = df[[c for c in df.columns if c not in ['student_rut', prueba]]]
    y = df[prueba]
    return train_test_split(X, y, test_size=0.2, random_state=42)