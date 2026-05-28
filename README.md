# SVM Classifier

**Support Vector Machine** with multiple kernels, GridSearchCV tuning, and multi-dataset support.

## Overview

- Three datasets: **Wine**, **Iris**, **Breast Cancer**
- Four kernels: linear, polynomial, RBF, sigmoid
- StandardScaler preprocessing + stratified train/test split
- GridSearchCV for automatic hyperparameter optimization
- **Streamlit dashboard** with kernel comparison charts

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --dataset wine --kernel rbf --tune
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# SVM-Classifier
