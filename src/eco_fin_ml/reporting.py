from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import pandas as pd


def ensure_dir(path: str) -> pathlib.Path:
    output_dir = pathlib.Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metrics(metrics: Dict[str, Dict[str, float]], output_dir: str, name: str) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T
    csv_path = pathlib.Path(output_dir) / f"{name}_metrics.csv"
    latex_path = pathlib.Path(output_dir) / f"{name}_metrics.tex"
    df.to_csv(csv_path, index=True)
    df.to_latex(latex_path, index=True, float_format="%.4f")
    return df


def save_json(data: Dict, output_dir: str, name: str) -> None:
    path = pathlib.Path(output_dir) / f"{name}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_methodology(notes: List[str], output_dir: str, name: str) -> None:
    path = pathlib.Path(output_dir) / f"{name}_methodology.md"
    with path.open("w", encoding="utf-8") as file:
        file.write("# Methodology Notes\n\n")
        for note in notes:
            file.write(f"- {note}\n")
