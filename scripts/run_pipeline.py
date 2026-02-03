from __future__ import annotations

import argparse
import pathlib

import yaml

from eco_fin_ml.config import DataConfig, FeatureConfig, OutputConfig, ProjectConfig, TrainingConfig
from eco_fin_ml.pipeline import run_pipeline


def load_config(path: str) -> ProjectConfig:
    with pathlib.Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return ProjectConfig(
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        training=TrainingConfig(**raw["training"]),
        output=OutputConfig(**raw["output"]),
        models=raw.get("models", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Eco-Fin ML Research Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
