import typer
import sys
from loguru import logger

from .config import settings
from .data import load_classification_data
from .model import SVMClassifier
from .visualizer import SVMVisualizer

app = typer.Typer(help="SVM Classifier CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(
    dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer"),
    kernel: str = typer.Option("rbf", help="Kernel: linear, poly, rbf, sigmoid"),
    C: float = typer.Option(1.0, help="Regularization parameter"),
    tune: bool = typer.Option(False, help="Run GridSearchCV"),
    visualize: bool = typer.Option(True, help="Generate plots"),
):
    logger.info(f"Training SVM on {dataset} | kernel={kernel} | C={C}")
    X_train, X_test, y_train, y_test, fn, cn = load_classification_data(dataset)
    model = SVMClassifier(kernel=kernel, C=C)
    model.fit(X_train, y_train, fn, cn)
    results = model.evaluate(X_test, y_test)
    logger.info(
        f"Accuracy: {results['accuracy']:.2%} | "
        f"Support Vectors: {results['n_support_vectors']}"
    )
    if tune:
        best = model.grid_search(X_train, y_train)
        logger.info(f"Best: {best['best_params']} → {best['best_score']:.3f}")
    if visualize and X_train.shape[1] >= 2:
        vis = SVMVisualizer()
        vis.plot_confusion_matrix(
            results["confusion_matrix"], cn,
            save_path=settings.plots_dir / "confusion_matrix.png",
        )
    model.save(str(settings.models_dir / "svm_model.joblib"))
    logger.success("Training complete!")


if __name__ == "__main__":
    app()
