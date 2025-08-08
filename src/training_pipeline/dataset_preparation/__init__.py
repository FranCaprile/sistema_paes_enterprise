from .bloom import dataset_bloom
from .indicadores import dataset_indicadores
from .sepa import dataset_sepa
from .pca import dataset_pca
from .dificultad import dataset_dificultad
from typing import Dict
import pandas as pd

PREPARERS = {
    "bloom":       dataset_bloom,
    "indicadores": dataset_indicadores,
    "sepa":        dataset_sepa,
    "pca":         dataset_pca,
    "dificultad":  dataset_dificultad,
}


def preparar_dataset(dfs: Dict[str, pd.DataFrame], prueba: str, variable: str):
    """
    Dispatch a la función concreta según 'variable'.
    dfs: dict con todos los dataframes cargados.
    prueba: nombre de la prueba (e.g. 'lenguaje').
    variable: uno de PREPARERS.keys().
    """
    if variable not in PREPARERS:
        raise ValueError(f"Variable desconocida: {variable}")
    return PREPARERS[variable](dfs, prueba)