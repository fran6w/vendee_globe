from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

race_2020_path = root_dir / "data" / "race_2020.parquet"
web_2020_path = root_dir / "data" / "web_2020.parquet"

race_2024_path = root_dir / "data" / "race_2024.parquet"
web_2024_path = root_dir / "data" / "web_2024.parquet"
wiki_2024_path = root_dir / "data" / "wiki_2024.parquet"
