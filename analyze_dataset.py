import datasets

def main():
    # Cargar el dataset
    dataset = datasets.load_dataset("sanjay920/goat-sharegpt")
    print("Splits disponibles y número de ejemplos:")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} ejemplos")

    print("\nCampos y tipos de datos por split:")
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        features = dataset[split].features
        for field, dtype in features.items():
            print(f"  - {field}: {dtype}")

    print("\nEjemplo de datos por split:")
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        print(dataset[split][0])

    print("\nEstadísticas básicas por split:")
    for split in dataset.keys():
        conv_lengths = []
        num_messages = []
        for example in dataset[split]:
            # Asume que cada ejemplo tiene un campo 'conversation' que es una lista de mensajes
            conv = example.get("conversation", [])
            conv_lengths.append(sum(len(str(msg)) for msg in conv))
            num_messages.append(len(conv))
        avg_conv_length = sum(conv_lengths) / len(conv_lengths) if conv_lengths else 0
        avg_num_messages = sum(num_messages) / len(num_messages) if num_messages else 0
        print(f"Split: {split}")
        print(f"  - Longitud promedio de conversación (caracteres): {avg_conv_length:.2f}")
        print(f"  - Número promedio de mensajes por conversación: {avg_num_messages:.2f}")

if __name__ == "__main__":
    main()