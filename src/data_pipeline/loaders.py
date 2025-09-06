import os
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
import traceback


def cargar_configuracion():
    """
    Lee y devuelve todo el contenido de config/config.yml como dict.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def crear_engine():
    cfg = cargar_configuracion()
    uri = cfg["database"]["sepa_uri"]
    return create_engine(uri)


def _get_dimension_id(engine, dim_name: str) -> int:
    """Busca el id de la dimensión por nombre (case-insensitive)."""
    sql = text("""
        SELECT id
        FROM public.dimension
        WHERE LOWER(name) = LOWER(:dim_name)
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"dim_name": dim_name}).fetchone()
    return row[0] if row else None

def cargar_datos_brutos():
    cfg = cargar_configuracion()  # ya lo tienes en tu proyecto
    engine = create_engine(cfg["database"]["sepa_uri"])  # cadena de conexión Neon

    # 1) Cargar PAES Excel como ya haces…
    df_paes = pd.read_excel(cfg["paths"]["paes_encrypted_xlsx"], skiprows=1)

    # 2) Resolver ID de “Dificultad” en tiempo de ejecución
    dim_dificultad = cfg.get("dimensions", {}).get("dificultad", "Dificultad")
    dificultad_id = _get_dimension_id(engine, dim_dificultad)
    if dificultad_id is None:
        print(f"[WARN] No encontré la dimensión '{dim_dificultad}'. "
              f"Revisa la tabla public.dimension. Continuaré sin average_diff_scores.")
    
    # 3) Queries principales (siempre con esquema `public.` para evitar search_path)
    q_average_scores = """
    SELECT 
    s.id   AS student_id,
    s.fullname AS student_name,
    REGEXP_REPLACE(UPPER(CAST(s.rut AS TEXT)), '[^0-9K]', '', 'g') AS student_rut,
    ass.name  AS test_type,
    ROUND(AVG(a.score), 2) AS average_score
    FROM public.answer a
    JOIN public.student   s   ON s.id = a.student_id
    JOIN public.question  q   ON q.id = a.question_id
    JOIN public.assessment ass ON ass.id = q.assessment_id
    GROUP BY s.id, ass.name
    ORDER BY s.id, ass.name;
    """

    q_indicator = "SELECT * FROM public.indicator;"
    q_answer   = "SELECT score, question_id, student_id FROM public.answer;"
    q_student = """
    SELECT 
    id,
    REGEXP_REPLACE(UPPER(CAST(rut AS TEXT)), '[^0-9K]', '', 'g') AS rut
    FROM public.student;
    """
    # 4) Query de dificultad (solo si tenemos el id)
    if dificultad_id is not None:
        q_avg_diff = text("""
            SELECT 
                s.id AS student_id,
                s.fullname AS student_name,
                s.rut AS student_rut,
                ass.name AS test_type,
                q.id AS question_id,
                a.score AS score,
                dv.value AS dimension_value,
                dv.dimension_id AS dimension_type
            FROM public.answer a
            JOIN public.student s ON s.id = a.student_id
            JOIN public.question q ON q.id = a.question_id
            JOIN public.assessment ass ON ass.id = q.assessment_id
            JOIN public.question_dimensions_values qdv ON qdv.question_id = q.id
            JOIN public.dimension_values dv ON dv.id = qdv.dimension_value_id
            JOIN public.dimension d ON d.id = dv.dimension_id
            WHERE d.id = :dificultad_id
        """)
    else:
        q_avg_diff = None

    # 5) Ejecutar y devolver
    with engine.connect() as conn:
        df_avg       = pd.read_sql_query(text(q_average_scores), conn)
        df_indicator = pd.read_sql_query(text(q_indicator), conn)
        df_answer    = pd.read_sql_query(text(q_answer), conn)
        df_student   = pd.read_sql_query(text(q_student), conn)
        if q_avg_diff is not None:
            df_avg_diff = pd.read_sql_query(q_avg_diff, conn, params={"dificultad_id": dificultad_id})
        else:
            df_avg_diff = pd.DataFrame()

    print(f"[OK] Query average_scores → {len(df_avg)} filas")
    print(f"[OK] Query indicator → {len(df_indicator)} filas")
    print(f"[OK] Query answer → {len(df_answer)} filas")
    print(f"[OK] Query student → {len(df_student)} filas")
    print(f"[OK] Query average_diff_scores → {len(df_avg_diff)} filas")

    return {
        "db_queries": {
            "average_scores": df_avg,
            "indicator": df_indicator,
            "answer": df_answer,
            "student": df_student,
            "average_diff_scores": df_avg_diff,
        },
        "df_paes": df_paes,
    }



if __name__ == "__main__":
    data = cargar_datos_brutos()
    # Pequeña verificación
    for i, df in enumerate(data["paes"], start=1):
        print(f"PAES #{i} columnas:", df.columns.tolist())
    for tbl, df in data["db_tables"].items():
        print(f"{tbl}: {len(df) if df is not None else 'ERROR'} filas")
    for tbl, df in data["db_tables"].items():
        print(f"  {tbl}: {len(df) if df is not None else 'ERROR'} rows")
    for name, df in data["db_queries"].items():
        print(f"  Query {name}: {len(df) if df is not None else 'ERROR'} rows")


