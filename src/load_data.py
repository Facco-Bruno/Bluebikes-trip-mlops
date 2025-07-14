import os
import pandas as pd


def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    print(f"📄 Lendo CSV de: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            on_bad_lines="skip",  # ignora linhas corrompidas
        )
    except UnicodeDecodeError:
        print("⚠️ Problema com UTF-8. Tentando com ISO-8859-1...")
        df = pd.read_csv(
            csv_path,
            encoding="ISO-8859-1",
            on_bad_lines="skip",
        )

    print(f"💾 Salvando Parquet em: {parquet_path}")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print("✅ Conversão concluída com sucesso.")


def main() -> None:
    os.makedirs("data", exist_ok=True)

    csv_path = "data/202307-bluebikes-tripdata.csv"
    parquet_path = "data/202307-bluebikes-tripdata.parquet"

    convert_csv_to_parquet(csv_path, parquet_path)


if __name__ == "__main__":
    main()
