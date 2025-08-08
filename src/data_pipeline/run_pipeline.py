
from loaders import cargar_datos_brutos, cargar_configuracion
from transformaciones import (
    transformar_df_pruebas_sepa,
    transformar_df_indicadores,
    transformar_df_taxonomia,
    transformar_avg_diff_scores,
    transformar_avg_diff_scores_agg,   # <--- NUEVA
    transformar_df_pca,
)
def main():
    cfg = cargar_configuracion()
    data = cargar_datos_brutos()

    # Transformar pruebas SEPA
    df_pruebas = transformar_df_pruebas_sepa(data["db_queries"]["average_scores"])

    # Transformar indicadores (tres queries separadas)
    df_indicator = data["db_queries"]["indicator"]
    df_answer = data["db_queries"]["answer"]
    df_student = data["db_queries"]["student"]


    df_indicadores = transformar_df_indicadores(df_indicator, df_answer, df_student)

    # Transformar preguntas por dificultad
    df_avg_diff_scores = transformar_avg_diff_scores(data["db_queries"]["average_diff_scores"])
    df_avg_diff_scores_agg = transformar_avg_diff_scores_agg(data["db_queries"]["average_diff_scores"])  # <--- NUEVO

    # --- guardados intermedios ---
    output_path_pruebas_sepa = cfg["output_paths"]["df_pruebas_sepa"]
    output_path_indicadores = cfg["output_paths"]["df_indicadores"]
    output_path_diff_scores = cfg["output_paths"]["df_avg_diff"]
    output_path_diff_scores_agg = cfg["output_paths"]["df_avg_diff_agg"]      


    df_pruebas.to_csv(output_path_pruebas_sepa, index=False)
    df_indicadores.to_csv(output_path_indicadores, index=False)
    df_avg_diff_scores.to_csv(output_path_diff_scores, index=False)
    df_avg_diff_scores_agg.to_csv(output_path_diff_scores_agg, index=False)   

    print(f"Guardado df_pruebas_sepa en {output_path_pruebas_sepa}")
    print(f"Guardado df_indicadores en {output_path_indicadores}")
    print(f"Guardado df_avg_diff_scores en {output_path_diff_scores}")
    print(f"Guardado df_avg_diff_scores_agg en {output_path_diff_scores_agg}") 

    # --- taxonomía ---
    df_taxonomia = transformar_df_taxonomia(df_indicadores, cfg["taxonomia"])
    df_taxonomia.to_csv(cfg["output_paths"]["df_taxonomia"], index=False)
    print(f"Guardado df_taxonomia en {cfg['output_paths']['df_taxonomia']}")

    # --- PCA ---
    df_pca = transformar_df_pca(df_indicadores, n_components=8)
    df_pca.to_csv(cfg["output_paths"]["df_pca"], index=False)
    print(f"Guardado df_pca en {cfg['output_paths']['df_pca']}")


if __name__ == "__main__":
    main()
