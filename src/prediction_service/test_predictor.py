import sys
from pathlib import Path

# 👉 agrega /src al PYTHONPATH
SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.insert(0, str(SRC_DIR))

from prediction_service.predictor import predecir_paes_por_rut, get_any_student_rut

def main():
    import sys
    rut = sys.argv[1] if len(sys.argv) > 1 else None
    if rut is None:
        rut = get_any_student_rut()
        if rut is None:
            print("⚠️ No encontré RUTs en df_pruebas_sepa y no se entregó uno por CLI.")
            sys.exit(1)
        print(f"(Usando RUT de ejemplo encontrado en datos: {rut})")

    pred = predecir_paes_por_rut(rut)
    if not pred:
        print("⚠️ No se obtuvieron predicciones.")
        sys.exit(2)

    print(f"\n✅ Predicciones PAES para RUT {rut}:")
    for prueba, yhat in pred.items():
        print(f"  - {prueba}: {yhat:.2f}")

if __name__ == "__main__":
    main()
