from datasets import load_dataset

def main():
    ds = load_dataset("sanjay920/goat-sharegpt")
    print("=" * 40)
    print("Splits disponibles:")
    for split in ds.keys():
        print(f"  - {split}")
    print("=" * 40)
    print("Campos en cada ejemplo por split:")
    for split in ds.keys():
        columns = ds[split].column_names
        print(f"  [{split}]")
        for col in columns:
            print(f"    • {col}")
    print("=" * 40)

if __name__ == "__main__":
    main()