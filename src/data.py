import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
from typing import Tuple


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_classification_data(
    dataset_name: str = "wine",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    loader = DATASETS.get(dataset_name, load_wine)
    data = loader()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info(
        f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, "
        f"{len(feature_names)} features, {len(class_names)} classes"
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names
