import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger


class SVMVisualizer:
    @staticmethod
    def plot_decision_boundary_2d(
        model, X: np.ndarray, y: np.ndarray,
        feature_names: list, class_names: list,
        save_path: Optional[Path] = None,
    ) -> None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
        scatter = plt.scatter(
            X[:, 0], X[:, 1], c=y, cmap="viridis",
            edgecolor="k", s=50,
        )
        if hasattr(model, "support_vectors_"):
            sv = model.support_vectors_
            plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
                       edgecolors="red", linewidths=1.5, label="Support Vectors")
            plt.legend()
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"SVM ({model.kernel}) Decision Boundary")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: list, class_names: list,
        save_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix — SVM")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
