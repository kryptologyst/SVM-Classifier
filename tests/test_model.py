import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SVMClassifier
from src.data import load_classification_data


class TestSVMClassifier:
    @pytest.fixture
    def data(self):
        return load_classification_data("wine")

    @pytest.fixture
    def fitted_model(self, data):
        X_train, _, y_train, _, fn, cn = data
        model = SVMClassifier(kernel="rbf")
        model.fit(X_train, y_train, fn, cn)
        return model

    def test_fit(self, fitted_model, data):
        _, X_test, _, y_test, _, _ = data
        preds = fitted_model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_accuracy_above_85(self, fitted_model, data):
        _, X_test, _, y_test, _, _ = data
        results = fitted_model.evaluate(X_test, y_test)
        assert results["accuracy"] > 0.85

    def test_proba(self, fitted_model, data):
        _, X_test, _, _, _, _ = data
        proba = fitted_model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_grid_search(self, data):
        X_train, _, y_train, _, _, _ = data
        model = SVMClassifier()
        best = model.grid_search(X_train, y_train, cv=3)
        assert "best_params" in best
        assert best["best_score"] > 0.5

    def test_different_kernels(self, data):
        X_train, X_test, y_train, y_test, _, _ = data
        for k in ["linear", "rbf", "poly"]:
            model = SVMClassifier(kernel=k)
            model.fit(X_train, y_train)
            acc = model.evaluate(X_test, y_test)["accuracy"]
            assert acc > 0.80

    def test_save_load(self, fitted_model, data, tmp_path):
        _, X_test, _, _, _, _ = data
        p = str(tmp_path / "svm.joblib")
        fitted_model.save(p)
        m2 = SVMClassifier()
        m2.load(p)
        assert np.array_equal(
            fitted_model.predict(X_test[:5]),
            m2.predict(X_test[:5]),
        )
