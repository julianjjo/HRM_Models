# inspect_c4.py
# Este script inspecciona la estructura del dataset allenai/c4 sin descargarlo por completo.
# Utiliza el modo streaming para ser rápido y eficiente en el uso de recursos.

from datasets import load_dataset
import itertools

# --- Configuración ---
DATASET_NAME = "allenai/c4"
CONFIG_NAME = "en.noblocklist"
NUM_EXAMPLES_TO_SHOW = 3

def inspect_dataset_structure():
    """
    Carga la estructura de un dataset en modo streaming, imprime sus características
    y muestra algunos ejemplos de cada split.
    """
    print("="*80)
    print(f"INSPECCIONANDO EL DATASET: '{DATASET_NAME}' (Configuración: '{CONFIG_NAME}')")
    print("="*80)
    
    try:
        # Usar streaming=True es CRUCIAL para no descargar terabytes de datos.
        # Esto solo descarga los metadatos y los ejemplos que pidamos.
        print("\nCargando la estructura del dataset en modo streaming...")
        dataset = load_dataset(DATASET_NAME, CONFIG_NAME, streaming=True)
        print("¡Estructura cargada con éxito!\n")

        # Imprimir la información general del DatasetDict
        print("--- Resumen General del Dataset ---")
        print(dataset)
        print("-" * 35 + "\n")

        # Iterar sobre cada split disponible (ej. 'train', 'validation')
        for split_name, split_dataset in dataset.items():
            print(f"=== Inspeccionando el split: '{split_name}' ===")
            
            # 1. Mostrar las columnas y sus tipos de datos
            print("\n[1] Columnas y Tipos de Datos:")
            print(split_dataset.features)
            
            # 2. Mostrar los primeros N ejemplos
            print(f"\n[2] Mostrando los primeros {NUM_EXAMPLES_TO_SHOW} ejemplos del split '{split_name}':")
            
            # Tomamos los primeros N ejemplos del iterador del dataset en streaming
            examples = list(itertools.islice(split_dataset, NUM_EXAMPLES_TO_SHOW))
            
            for i, example in enumerate(examples):
                print(f"\n--- Ejemplo {i+1} ---")
                for column, value in example.items():
                    # Acortamos el texto para que la salida sea legible
                    if isinstance(value, str) and len(value) > 250:
                        value_to_print = value[:250] + "..."
                    else:
                        value_to_print = value
                    print(f"  - {column}: {value_to_print}")
            print("\n" + "="*40 + "\n")

    except Exception as e:
        print(f"\nERROR: No se pudo inspeccionar el dataset. Causa: {e}")
        print("Asegúrate de tener conexión a internet y de que el nombre del dataset y la configuración son correctos.")
        
    print("="*80)
    print("Inspección finalizada. Copia toda la salida de este script y pégala para que pueda adaptar el código.")
    print("="*80)


if __name__ == "__main__":
    inspect_dataset_structure()