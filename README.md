# Eco-Fin ML Research Pipeline

A reproducible, transparent, and high-performance research pipeline for econometrics, machine learning, and finance/banking applications. The project supports both **regression** and **classification** and is designed for academic publication workflows.

## Highlights
- General data loading (CSV/Excel/Parquet) with no hard-coded datasets
- Econometric + ML + deep learning models
- Time-series aware splitting and modeling
- Hyperparameter tuning and model comparison
- Explainable AI (SHAP, LIME, PDP)
- Publication-ready outputs (CSV + LaTeX tables)

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/default.yaml
```

Outputs (metrics, plots, logs, methodology notes) will be generated in `outputs/`.
