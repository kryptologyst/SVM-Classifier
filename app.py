import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_classification_data
from src.model import SVMClassifier

st.set_page_config(page_title="SVM Classifier", page_icon="⚡", layout="wide")
st.title("⚡ SVM Classifier")
st.markdown("Support Vector Machine with multiple kernels and GridSearchCV tuning.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X_train, X_test, y_train, y_test, fn, cn = load_classification_data(dataset_name)

tab1, tab2 = st.tabs(["Train & Tune", "Compare Kernels"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    with c2:
        C = st.select_slider("C (Regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
    with c3:
        tune = st.checkbox("Run GridSearchCV", value=False)

    if st.button("Train SVM", type="primary"):
        with st.spinner("Training..."):
            model = SVMClassifier(kernel=kernel, C=C)
            model.fit(X_train, y_train, fn, cn)
            results = model.evaluate(X_test, y_test)
        st.success(f"Accuracy: **{results['accuracy']:.2%}** | Support Vectors: **{results['n_support_vectors']}**")
        if tune:
            with st.spinner("Grid search..."):
                best = model.grid_search(X_train, y_train)
            st.info(f"Best params: {best['best_params']} → {best['best_score']:.3f}")

with tab2:
    st.subheader("Kernel Comparison")
    if st.button("Compare All Kernels", type="primary"):
        results = []
        for k in ["linear", "poly", "rbf", "sigmoid"]:
            m = SVMClassifier(kernel=k)
            m.fit(X_train, y_train)
            r = m.evaluate(X_test, y_test)
            results.append({"Kernel": k, "Accuracy": r["accuracy"], "SV": r["n_support_vectors"]})
        df = pd.DataFrame(results)
        st.dataframe(df.set_index("Kernel"), use_container_width=True)
        st.bar_chart(df.set_index("Kernel")["Accuracy"])
