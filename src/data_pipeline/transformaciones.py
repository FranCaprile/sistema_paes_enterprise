import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def transformar_df_pruebas_sepa(df_avg: pd.DataFrame, n_recientes: int = 12) -> pd.DataFrame:
    """
    Toma un DataFrame con columnas:
      - student_id
      - student_name
      - student_rut
      - test_type       (ej. "Competencia Lectora 2023")
      - average_score   (float)

    Y devuelve un nuevo DataFrame con:
      - id (student_id)
      - mat_1 ... mat_n_recientes  (puntajes promedio de Matemáticas, ordenados de más reciente a más antiguo)
      - leng_1 ... leng_n_recientes (puntajes promedio de Lectura, ordenados de más reciente a más antiguo)
      - student_rut

    Rellena con el promedio de cada tipo si hay menos de n_recientes pruebas.
    """
    # Trabajamos sobre copia para no mutar df_avg original
    df = df_avg.copy()

    # 1) Creamos columnas auxiliares
    df['type'] = df['test_type'].str.contains('Len', case=False).astype(int)
    df['year'] = df['test_type'].str.extract(r'(\d{4})').astype(int)

    # 2) Seleccionamos las n_recientes más recientes por estudiante y tipo
    df_recent = (
        df
        .sort_values(['student_id', 'type', 'year'], ascending=[True, True, False])
        .groupby(['student_id', 'type'], as_index=False)
        .head(n_recientes)
    )

    # 3) Construcción de las listas de puntajes para cada estudiante
    test_dict = {}
    for _, row in df_recent.iterrows():
        sid = row['student_id']
        t   = row['type']
        score = row['average_score']
        rut = row['student_rut']

        if sid not in test_dict:
            test_dict[sid] = {'scores': {0: [], 1: []}, 'rut': rut}

        test_dict[sid]['scores'][t].append(score)

    # 4) Rellenar con promedios y formar filas
    records = []
    for sid, info in test_dict.items():
        mat_scores = info['scores'][0]
        len_scores = info['scores'][1]
        avg_mat = round(np.mean(mat_scores), 2) if mat_scores else 0.0
        avg_len = round(np.mean(len_scores), 2) if len_scores else 0.0

        # Completar hasta n_recientes
        mat_scores += [avg_mat] * (n_recientes - len(mat_scores))
        len_scores += [avg_len] * (n_recientes - len(len_scores))

        row = [sid] + mat_scores + len_scores + [info['rut']]
        records.append(row)

    # 5) Generar DataFrame final
    col_mat = [f"mat_{i}" for i in range(1, n_recientes+1)]
    col_len = [f"leng_{i}" for i in range(1, n_recientes+1)]
    columns = ['student_id'] + col_mat + col_len + ['student_rut']

    df_final = pd.DataFrame(records, columns=columns)

    return df_final


def transformar_df_indicadores(df_indicator: pd.DataFrame, df_answer: pd.DataFrame, df_student: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """
    Construye el DataFrame final de indicadores:
      - Renombra columnas inconsistentes.
      - Filtra los k indicadores más repetidos.
      - Hace los joins con answer y student.
      - Agrupa por rut y descripción, y pivotea por promedio_score.
    """
    # 0. Validaciones mínimas
    if df_indicator is None or df_answer is None or df_student is None:
        raise ValueError("Alguno de los DataFrames es None. Revisa la carga en loaders.py")

    # 1. Normalizar nombres de columnas para que coincidan
    df_indicator = df_indicator.rename(columns={"questionId": "question_id"})


    # 2. Contar repetición por descripción
    counts = df_indicator.groupby("description").size().reset_index(name="count")
    counts = counts[counts["count"] > k]

    # 3. Filtrar indicadores relevantes
    df_indicators = df_indicator[df_indicator["description"].isin(counts["description"])]

    # 4. Megajoin
    df_merged = pd.merge(df_answer, df_indicators, on="question_id", how="inner")
    df_merged = pd.merge(df_merged, df_student, left_on="student_id", right_on="id", how="inner")

    # 5. Agrupar
    df_grouped = df_merged.groupby(["rut", "description"]).agg(
        total_preguntas=("score", "count"),
        promedio_score=("score", "mean")
    ).reset_index()

    # 6. Pivot
    df_pivot = df_grouped.pivot(index="rut", columns="description", values="promedio_score")

    # 7. Limpieza de nombres
    df_pivot.columns.name = None
    df_final = df_pivot.reset_index()

    return df_final


def transformar_df_taxonomia(df_indicadores: pd.DataFrame, taxonomia: dict) -> pd.DataFrame:
    """
    Dado el df de indicadores (con columna 'rut' y luego columnas de indicadores),
    agrupa esos indicadores en las categorías de la taxonomía promediando sus scores.
    `taxonomia` es un dict como el cargado desde taxonomia.yml.
    """
    # Validaciones mínimas
    if df_indicadores is None or df_indicadores.empty:
        raise ValueError("df_indicadores está vacío o es None.")

    if "rut" not in df_indicadores.columns:
        raise KeyError("Se espera que df_indicadores tenga columna 'rut'.")

    # Preprocesar nombres para matching (todo en minúscula)
    indicadores = [col for col in df_indicadores.columns if col != "rut"]
    # Construir un mapping taxonomía -> lista de indicadores que caen ahí
    cols_taxonomia = {tax: [] for tax in taxonomia.keys()}

    indicadores_lower = {ind: ind.lower() for ind in indicadores}
    taxonomia_lower = {
        tax: [v.lower() for v in values]
        for tax, values in taxonomia.items()
    }

    for indicador, indicador_limpio in indicadores_lower.items():
        for tax, keywords in taxonomia_lower.items():
            if any(keyword in indicador_limpio for keyword in keywords):
                cols_taxonomia[tax].append(indicador)

    # Para cada estudiante, promediar por categoría
    dict_taxonomia = {}
    for _, row in df_indicadores.iterrows():
        rut = row["rut"]
        puntajes_taxonomia = {}
        for tax, indicators in cols_taxonomia.items():
            if indicators:
                # Evitar error si columna no existe (aunque debería)
                puntajes_taxonomia[tax] = row[ indicators ].mean()
            else:
                puntajes_taxonomia[tax] = 0.0
        dict_taxonomia[rut] = puntajes_taxonomia

    # Armar DataFrame final
    df_taxonomia = pd.DataFrame.from_dict(dict_taxonomia, orient="index").reset_index()
    df_taxonomia = df_taxonomia.rename(columns={"index": "rut"})

    # Reordenar columnas: rut primero, luego en el orden de taxonomia.keys()
    columnas = ["rut"] + list(taxonomia.keys())
    df_taxonomia = df_taxonomia.loc[:, columnas]

    return df_taxonomia



def transformar_df_pca(df_indicadores: pd.DataFrame, n_components: int = 8) -> pd.DataFrame:
    """
    Aplica PCA sobre df_indicadores (que debe tener 'rut' y luego columnas numéricas de indicadores).
    Devuelve un DataFrame con las componentes principales: columnas PC1..PCn y 'rut'.
    """

    if df_indicadores is None or df_indicadores.empty:
        raise ValueError("df_indicadores está vacío o es None.")

    if "rut" not in df_indicadores.columns:
        raise KeyError("Se espera que df_indicadores tenga columna 'rut'.")

    # Trabajar con copia para no mutar externo
    df_local = df_indicadores.copy()

    # Índice temporal
    df_local = df_local.set_index("rut")

    # Reemplazar NaN
    df_local = df_local.fillna(0)

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df_local)

    # Construir DataFrame de PCs
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(transformed, index=df_local.index, columns=pc_columns).reset_index()

    return df_pca


def transformar_avg_diff_scores(df: pd.DataFrame, fillna_value=0.0) -> pd.DataFrame:
    """
    Toma el DataFrame de average_diff_scores y devuelve un pivot con columnas
    como '0-2012 - Fácil', etc. Si algo falla, imprime info de depuración.
    """
    # 0. Validaciones
    required = ['student_id', 'student_rut', 'test_type', 'score', 'dimension_value']
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas requeridas: {faltantes}")

    if df.empty:
        print("[WARN] El DataFrame de input está vacío.")
        return pd.DataFrame(columns=['id', 'rut'])  # fallback

    # 1. Copia y revisión rápida
    df_local = df.copy()
    print(f"[INFO] transformando_avg_diff_scores: {len(df_local)} filas recibidas")
    print("Valores únicos de test_type (muestra):", df_local['test_type'].dropna().unique()[:10])
    print("Valores únicos de dimension_value (muestra):", df_local['dimension_value'].dropna().unique()[:10])

    # 2. Extraer tipo y año
    df_local['type_flag'] = df_local['test_type'].astype(str).str.contains('Len', case=False).astype(int)
    df_local['year'] = df_local['test_type'].astype(str).str.extract(r'(\d{4})', expand=False)
    # Rellenar años faltantes con cadena vacía para que la concatenación no falle
    df_local['year'] = df_local['year'].fillna('')
    df_local['type'] = df_local['type_flag'].astype(str) + '-' + df_local['year'].astype(str)

    # 3. Agrupar para sacar promedio
    df_avg = (
        df_local
        .groupby(['student_id', 'student_rut', 'dimension_value', 'type'], dropna=False)
        .agg({'score': 'mean'})
        .reset_index()
    )

    if df_avg.empty:
        print("[WARN] Después del groupby, df_avg está vacío.")
        return pd.DataFrame(columns=['id', 'rut'])

    # 4. Construir la columna combinada y pivotear
    df_avg['type_dimension'] = df_avg['type'].astype(str) + ' - ' + df_avg['dimension_value'].astype(str)

    df_pivot = df_avg.pivot_table(
        index=['student_id', 'student_rut'],
        columns='type_dimension',
        values='score'
    ).reset_index()

    # 5. Renombrar y llenar nulos
    df_pivot = df_pivot.rename(columns={'student_rut': 'rut', 'student_id': 'id'})
    df_pivot = df_pivot.fillna(fillna_value)

    print(f"[INFO] Resultado de pivot tiene {df_pivot.shape[0]} filas y {df_pivot.shape[1]} columnas")
    return df_pivot
