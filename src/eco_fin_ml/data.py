from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    file_path = pathlib.Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix in {".parquet"}:
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file format: {suffix}")


def split_time_series(
    data: pd.DataFrame,
    test_size: float,
    val_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_total = len(data)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    train_end = n_total - n_test - n_val
    val_end = n_total - n_test

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    return train, val, test
