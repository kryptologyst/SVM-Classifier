import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from loguru import logger
from typing import Optional, Literal
import joblib


class SVMClassifier:
    def __init__(
        self,
        kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        random_state: int = 42,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel, C=C, gamma=gamma,
            probability=True, random_state=random_state,
        )
        self.feature_names_: list = []
        self.class_names_: list = []

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: list = None, class_names: list = None,
    ) -> "SVMClassifier":
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.class_names_ = class_names or [f"c{c}" for c in np.unique(y)]
        self.model.fit(X, y)
        n_sv = len(self.model.support_vectors_)
        logger.info(f"SVM fitted: kernel={self.kernel}, {n_sv} support vectors")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(
                y, y_pred, target_names=self.class_names_, output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "n_support_vectors": int(len(self.model.support_vectors_)),
        }

    def grid_search(
        self, X: np.ndarray, y: np.ndarray,
        param_grid: dict = None, cv: int = 5,
    ) -> dict:
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "kernel": ["rbf", "linear"],
            }
        grid = GridSearchCV(
            SVC(probability=True), param_grid, cv=cv, scoring="accuracy", n_jobs=-1,
        )
        grid.fit(X, y)
        logger.info(f"Best params: {grid.best_params_} — Score: {grid.best_score_:.3f}")
        return {
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_),
            "cv_results": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in grid.cv_results_.items()
                if k in ["mean_test_score", "std_test_score", "params"]
            },
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
