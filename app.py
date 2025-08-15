# app.py
# Predictive Analytics App (complete) — Streamlit
# - No reset logic
# - Data persists across tabs
# - Profiling enabled (ydata-profiling fallback)
# - Auto-select random index for explainer
# - Robust SHAP contributions logic (no length mismatch)
# - Small compact charts with 10pt fonts
#
# Ensure requirements.txt includes (minimum):
# streamlit
# pandas
# numpy
# scikit-learn
# imbalanced-learn
# explainerdashboard
# ydata-profiling
# shap
# matplotlib
# openpyxl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (5, 3),
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io
import tempfile
from pathlib import Path
import random
import traceback

# Data profiling imports (fallback)
try:
    from ydata_profiling import ProfileReport  # pip: ydata-profiling
except Exception:
    try:
        from pandas_profiling import ProfileReport
    except Exception:
        ProfileReport = None

# ML stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Imbalanced learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# SHAP
import shap

# Optional Explainer Dashboard
try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    EXPLAINERDASH_AVAILABLE = True
except Exception:
    EXPLAINERDASH_AVAILABLE = False

# -----------------------------
# Streamlit page config & session state defaults
# -----------------------------
st.set_page_config(page_title="Predictive Analytics App", layout="wide", page_icon="📊")

# Session state initialization (persistence)
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'profile_generated' not in st.session_state:
    st.session_state.profile_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'profile_html' not in st.session_state:
    st.session_state.profile_html = None
if 'explainer_selected_index' not in st.session_state:
    st.session_state.explainer_selected_index = None

# -----------------------------
# Helper functions
# -----------------------------
def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

@st.cache_data
def load_data_from_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        raise

@st.cache_data
def generate_profile_report_html(data):
    if ProfileReport is None:
        return "<h3>Data profiling library not available</h3><p>Please add 'ydata-profiling' to requirements.txt</p>"
    profile = ProfileReport(data, title="Dataset Profiling Report", explorative=True, minimal=False)
    return profile.to_html()

def safe_shap_explainer(fitted_model, background_data, feature_names=None):
    """
    Return a shap.Explainer instance robustly.
    """
    try:
        if feature_names is not None:
            expl = shap.Explainer(fitted_model, background_data, feature_names=feature_names)
        else:
            expl = shap.Explainer(fitted_model, background_data)
        return expl
    except Exception:
        try:
            return shap.Explainer(fitted_model, background_data)
        except Exception:
            return None

def ensure_shap_array_for_idx(shap_values_obj, idx):
    """
    From a shap.Explanation or numpy array, safely extract contribution vector for row idx.
    Returns a 1D numpy array or None.
    """
    try:
        # If shap_values_obj is numpy array of shape (n_samples, n_features)
        if isinstance(shap_values_obj, np.ndarray):
            arr = shap_values_obj
            if arr.ndim == 2:
                return arr[idx].ravel()
            if arr.ndim == 3:
                # sometimes (n_samples, n_outputs, n_features) -> choose first sample's first output if ambiguous
                if arr.shape[0] == idx + 1:
                    # pick sample idx
                    s = arr[idx]
                    if s.ndim == 2:
                        # choose output with largest absolute mean
                        out_idx = int(np.argmax(np.abs(s).mean(axis=1)))
                        return s[out_idx].ravel()
                    else:
                        return s.ravel()
                else:
                    # fallback to first sample
                    return arr[0].ravel()
            if arr.ndim == 1:
                return arr.ravel()
        else:
            # shap.Explanation object
            try:
                values = shap_values_obj.values  # could be array-like or nested
            except Exception:
                values = shap_values_obj
            # values might be (n_samples, n_features) or (n_samples, n_outputs, n_features)
            vals = np.array(values)
            if vals.ndim == 2:
                return vals[idx].ravel()
            if vals.ndim == 3:
                # pick class with biggest absolute impact for the selected sample (if predict_proba available we could use class idx)
                sample_vals = vals[idx]  # shape (n_outputs, n_features)
                # choose output row with largest absolute mean
                out_idx = int(np.argmax(np.abs(sample_vals).mean(axis=1)))
                return sample_vals[out_idx].ravel()
            if vals.ndim == 1:
                return vals.ravel()
    except Exception:
        return None

def make_pdp_values(clf_pipeline, base_row_df, feature, grid):
    """
    Given a pipeline and a base row as DataFrame, vary `feature` across `grid`,
    return list of predictions / probabilities robustly.
    """
    preds = []
    for v in grid:
        temp = base_row_df.copy()
        temp[feature] = v
        try:
            if hasattr(clf_pipeline, "predict_proba"):
                p = clf_pipeline.predict_proba(temp)
                # choose highest probability value
                preds.append(float(np.max(p)))
            else:
                p = clf_pipeline.predict(temp)
                preds.append(float(np.ravel(p)[0]))
        except Exception:
            preds.append(np.nan)
    return preds

# -----------------------------
# Main layout: Title + Sidebar - Upload
# -----------------------------
st.title("📊 Predictive Analytics Application")
st.markdown("Comprehensive platform for profiling, model-building, evaluation, and predictions.")

with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose your dataset (CSV or XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("App status:")
    st.write(f"Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"Model trained: {st.session_state.model_trained}")

# Handle file upload & persistence (no resets)
if uploaded_file:
    try:
        data = load_data_from_file(uploaded_file)
        st.session_state.data = data
        st.session_state.data_uploaded = True
        st.session_state.current_file = uploaded_file.name
        st.success("✅ Data uploaded and stored in session.")
        st.write(f"**Shape:** {data.shape} | **Columns:** {len(data.columns)}")
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        st.session_state.data_uploaded = False

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Profiling",
    "🚀 Model Development",
    "🔍 Model Evaluation",
    "📈 Predictions"
])

# -----------------------------
# TAB 1: Data Profiling
# -----------------------------
with tab1:
    st.header("📊 Data Profiling")
    st.markdown("Understand your dataset before modeling.")
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset in the sidebar to begin profiling.")
    else:
        data = st.session_state.data
        st.subheader("Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", data.shape[0])
        with c2:
            st.metric("Columns", data.shape[1])
        with c3:
            mem_mb = data.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{mem_mb:.1f} MB")
        with c4:
            st.metric("Missing Values", int(data.isnull().sum().sum()))
        with st.expander("Quick Preview"):
            st.dataframe(data.head(5))
        if st.button("🔍 Generate Comprehensive Profile Report"):
            if ProfileReport is None:
                st.error("Data profiling library not installed. Add 'ydata-profiling' to requirements.txt and redeploy.")
            else:
                with st.spinner("Generating profiling report..."):
                    try:
                        html = generate_profile_report_html(data)
                        st.session_state.profile_html = html
                        st.session_state.profile_generated = True
                        st.success("✅ Profile generated.")
                    except Exception as e:
                        st.error(f"Profile generation failed: {e}")
        if st.session_state.profile_generated and st.session_state.profile_html:
            st.subheader("📋 Profiling Report")
            components.html(st.session_state.profile_html, height=700, scrolling=True)

# -----------------------------
# TAB 2: Model Development
# -----------------------------
with tab2:
    st.header("🚀 Model Development")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload a dataset first.")
    else:
        data = st.session_state.data.copy()
        st.subheader("🎯 Target & Feature Selection")
        all_cols = list(data.columns)
        target_column = st.selectbox("Choose target column", all_cols)
        st.session_state.previous_target = target_column

        available_features = [c for c in all_cols if c != target_column]
        feature_mode = st.radio("Feature selection:", ["Use all features", "Select features"], horizontal=True)
        if feature_mode == "Select features":
            feature_cols = st.multiselect("Pick features", available_features, default=available_features)
            if not feature_cols:
                st.warning("Select at least one feature.")
                st.stop()
        else:
            feature_cols = available_features
        st.session_state.previous_features = feature_cols

        st.info(f"Selected {len(feature_cols)} features.")
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()

        # Encode target if categorical
        if y_raw.dtype == "object" or y_raw.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = [str(c) for c in le_target.classes_]
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = [str(c) for c in np.unique(y)]

        st.subheader("📊 Target Distribution")
        colA, colB = st.columns(2)
        with colA:
            st.write(class_counts(y))
        with colB:
            try:
                dist_df = pd.DataFrame(list(class_counts(y).items()), columns=['class', 'count']).set_index('class')
                st.bar_chart(dist_df)
            except Exception:
                pass

        st.subheader("🤖 Model Selection")
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        model_choice = st.selectbox("Algorithm", list(model_map.keys()), index=3)
        model = model_map[model_choice]

        with st.expander("⚙️ Hyperparameters (optional)"):
            if model_choice in ["Logistic Regression", "SVM"]:
                C_val = st.slider("C", 0.01, 10.0, 1.0)
                model.set_params(C=C_val)
            if model_choice == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)
                model.set_params(kernel=kernel)
            if model_choice == "Random Forest":
                n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
                max_depth = st.slider("max_depth (0 = no limit)", 0, 30, 0)
                model.set_params(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth))

        st.subheader("⚖️ Class Balancing")
        balance = st.selectbox("Balancing", ["None", "SMOTE", "Random Oversample", "Random Undersample"])
        sampler = None
        if balance == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif balance == "Random Oversample":
            sampler = RandomOverSampler(random_state=42)
        elif balance == "Random Undersample":
            sampler = RandomUnderSampler(random_state=42)

        st.subheader("📊 Train/Test Split")
        test_pct = st.slider("Test size (%)", 10, 50, 20)

        # Build preprocessing
        numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
        categorical_feats = X.select_dtypes(exclude=np.number).columns.tolist()
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
        ])

        # Pipeline with optional sampler
        if sampler is None:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        else:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("classifier", model)])

        st.markdown("---")
        if st.button("🚀 Train Model", use_container_width=True):
            # Validate
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                st.error("Need at least 2 classes/values in target to train.")
            else:
                with st.spinner("Training..."):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100.0,
                                                                            random_state=42, stratify=(y if len(unique)>=2 else None))
                        clf_pipeline.fit(X_train, y_train)
                        y_pred = clf_pipeline.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        # Store model and test splits in session state
                        st.session_state.model_trained = True
                        st.session_state.trained_clf = clf_pipeline
                        st.session_state.trained_le_target = le_target
                        st.session_state.trained_feature_cols = feature_cols
                        st.session_state.trained_class_names = class_names
                        st.session_state.test_results = {
                            'y_test': np.array(y_test),
                            'y_pred': np.array(y_pred),
                            'accuracy': float(acc)
                        }
                        # Save test dataframe (reset index)
                        st.session_state.trained_X_test = X_test.reset_index(drop=True)
                        st.session_state.trained_y_test = np.array(y_test)
                        st.success(f"✅ Trained. Accuracy: {acc:.3f}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.text(traceback.format_exc())

# -----------------------------
# TAB 3: Model Evaluation (Explainer + SHAP + PDP)
# -----------------------------
with tab3:
    st.header("🔍 Model Evaluation & Explainer")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Train a model in Model Development tab first.")
    else:
        clf_pipeline = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results
        X_test_df = st.session_state.trained_X_test.copy()
        y_test_arr = st.session_state.trained_y_test

        # Performance summary
        st.subheader("📊 Performance Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
        with c2:
            st.metric("Test samples", len(y_test_arr))
        with c3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        # Detailed classification report (if classification)
        st.subheader("📋 Performance Report")
        try:
            report = classification_report(y_test_arr, test_results['y_pred'], output_dict=True, target_names=class_names)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
        except Exception:
            try:
                report = classification_report(y_test_arr, test_results['y_pred'], output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
            except Exception:
                st.info("Classification report not available for this model/problem type.")

        # Auto-select random index on first view
        if st.session_state.explainer_selected_index is None:
            if len(X_test_df) > 0:
                st.session_state.explainer_selected_index = int(random.randrange(0, len(X_test_df)))
            else:
                st.session_state.explainer_selected_index = None

        if st.session_state.explainer_selected_index is None:
            st.info("Test set empty — cannot show explainer details.")
        else:
            idx = st.slider("Select index for explanation", 0, max(0, len(X_test_df)-1),
                            value=int(st.session_state.explainer_selected_index), step=1)
            st.session_state.explainer_selected_index = int(idx)

            # Columns for selection + prediction display
            sel_col, pred_col = st.columns([1, 2])
            with sel_col:
                st.markdown("### Selected index")
                st.write(f"**{idx}**")
                st.write("")
            with pred_col:
                st.markdown("### Prediction")
                try:
                    single_row = X_test_df.iloc[[idx]]
                    pred = clf_pipeline.predict(single_row)
                    pred_proba = None
                    try:
                        pred_proba = clf_pipeline.predict_proba(single_row)
                    except Exception:
                        pred_proba = None

                    if le_target is not None:
                        try:
                            decoded = le_target.inverse_transform(pred)
                            st.write(f"**Prediction:** {decoded[0]}")
                        except Exception:
                            st.write(f"**Prediction (encoded):** {pred[0]}")
                    else:
                        st.write(f"**Prediction:** {pred[0]}")

                    if pred_proba is not None:
                        if le_target is not None and hasattr(le_target, "classes_"):
                            cols = [str(c) for c in le_target.classes_]
                        else:
                            cols = [f"class_{i}" for i in range(pred_proba.shape[1])]
                        prob_df = pd.DataFrame(pred_proba, columns=cols)
                        st.write("**Probabilities**")
                        st.dataframe(prob_df.T.rename(columns={0: "probability"}))
                    else:
                        st.write("Probability not available for this classifier.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.text(traceback.format_exc())

            st.markdown("---")
            # Contributions Plot (SHAP)
            st.markdown("### Contributions (SHAP-style)")
            try:
                # Build preprocessed background for SHAP
                # Use a small background sample to speed up
                try:
                    preprocessor = clf_pipeline.named_steps.get("preprocessor", None)
                    fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                except Exception:
                    preprocessor = None
                    fitted_clf = clf_pipeline

                # Build background as preprocessor applied to sample of X_test_df
                background_rows = X_test_df.sample(min(len(X_test_df), 200), replace=False)
                # If preprocessor exists, transform background for faster TreeExplainer
                try:
                    if preprocessor is not None:
                        background_trans = preprocessor.transform(background_rows)
                    else:
                        background_trans = background_rows.values
                except Exception:
                    # fallback: use raw background values
                    background_trans = background_rows.values

                # Feature names for SHAP values: try to extract from preprocessor
                try:
                    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                        feat_names = preprocessor.get_feature_names_out()
                    else:
                        feat_names = X_test_df.columns.tolist()
                except Exception:
                    feat_names = X_test_df.columns.tolist()

                # Create explainer robustly
                shap_explainer = safe_shap_explainer(fitted_clf, background_trans, feature_names=feat_names)
                shap_vals_vector = None
                if shap_explainer is not None:
                    try:
                        # We need preprocessed single instance if explainer expects transformed input
                        try:
                            if preprocessor is not None:
                                single_proc = preprocessor.transform(X_test_df.iloc[[idx]])
                            else:
                                single_proc = X_test_df.iloc[[idx]].values
                        except Exception:
                            single_proc = X_test_df.iloc[[idx]].values

                        shap_vals = shap_explainer(single_proc)
                        shap_vals_vector = ensure_shap_array_for_idx(shap_vals, 0)  # single_proc is single sample, idx=0
                    except Exception:
                        shap_vals_vector = None

                # If shap_vals_vector None, attempt fallback: use shap.TreeExplainer on fitted_clf directly with raw X_test_df
                if shap_vals_vector is None:
                    try:
                        try:
                            tree_expl = shap.TreeExplainer(fitted_clf)
                            shap_vals_raw = tree_expl.shap_values(X_test_df)
                            shap_vals_vector = ensure_shap_array_for_idx(shap_vals_raw, idx)
                        except Exception:
                            shap_vals_vector = None
                    except Exception:
                        shap_vals_vector = None

                # Prepare contributions DataFrame safely
                if shap_vals_vector is None:
                    st.info("SHAP values not available for this pipeline/model. Showing feature differences instead.")
                    # fallback: difference from mean (simple contribution proxy)
                    base = X_test_df.mean()
                    diff = X_test_df.iloc[idx] - base
                    contrib_df = pd.DataFrame({
                        "feature": X_test_df.columns.tolist(),
                        "contribution": diff.values
                    })
                else:
                    # Ensure shap length matches candidate feature list; pick the best matching feature list
                    contrib = np.asarray(shap_vals_vector).ravel()
                    # If lengths mismatch, align to one of candidates (prefer feat_names then X_test_df.columns)
                    if len(contrib) == len(feat_names):
                        feat_list = list(feat_names)
                    elif len(contrib) == len(X_test_df.columns):
                        feat_list = list(X_test_df.columns)
                    else:
                        # If mismatch, try to trim or pad contributions
                        # Choose feat_list as X_test_df.columns and either slice or pad contrib
                        feat_list = list(X_test_df.columns)
                        if len(contrib) > len(feat_list):
                            contrib = contrib[:len(feat_list)]
                        else:
                            # pad zeros
                            pad_len = len(feat_list) - len(contrib)
                            contrib = np.concatenate([contrib, np.zeros(pad_len)])
                    contrib_df = pd.DataFrame({
                        "feature": feat_list,
                        "contribution": contrib
                    })

                contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
                contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

                # Show top contributions table and compact bar chart
                st.write("Top contributions (absolute impact)")
                st.dataframe(contrib_df[["feature", "contribution"]].head(20), use_container_width=True)

                topn = min(10, len(contrib_df))
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(contrib_df.head(topn)["feature"][::-1], contrib_df.head(topn)["contribution"][::-1])
                ax.set_xlabel("Contribution (signed)", fontsize=10)
                ax.set_ylabel("Feature", fontsize=10)
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Contributions failed: {e}")
                st.text(traceback.format_exc())

            st.markdown("---")
            # Partial Dependence Plot (approximate)
            st.markdown("### Partial Dependence Plot (approx)")
            try:
                numeric_cols = X_test_df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.info("No numeric features available for PDP.")
                else:
                    pdp_feat = numeric_cols[0]
                    st.write(f"PDP feature: **{pdp_feat}**")
                    base_row = X_test_df.iloc[[idx]].copy().reset_index(drop=True)
                    grid = np.linspace(X_test_df[pdp_feat].min(), X_test_df[pdp_feat].max(), num=40)
                    pdp_preds = make_pdp_values(clf_pipeline, base_row, pdp_feat, grid)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(grid, pdp_preds, linewidth=1.8, marker='o', markersize=3)
                    ax.set_xlabel(pdp_feat, fontsize=10)
                    ax.set_ylabel("Predicted output / prob", fontsize=10)
                    ax.tick_params(axis='x', labelsize=10)
                    ax.tick_params(axis='y', labelsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"PDP failed: {e}")
                st.text(traceback.format_exc())

            # Optionally embed ExplainerDashboard (if available)
            st.markdown("---")
            st.markdown("### Explainer Dashboard (optional)")
            if EXPLAINERDASH_AVAILABLE:
                try:
                    sample_n = min(1000, len(X_test_df))
                    sample_idx = np.random.choice(len(X_test_df), size=sample_n, replace=False)
                    X_dash = X_test_df.iloc[sample_idx]
                    y_dash = y_test_arr[sample_idx]
                    # If pipeline, pass fitted classifier and preprocessed X_dash? ExplainerDashboard handles raw X if given classifier that expects transformed input; but behavior varies.
                    # We'll attempt to create explainer with the classifier (not pipeline) if possible.
                    try:
                        fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                        expl = ClassifierExplainer(
                            fitted_clf,
                            X_dash,
                            y_dash,
                            labels=class_names,
                            model_output="probability",
                        )
                        dashboard = ExplainerDashboard(
                            expl,
                            title="Model Performance Dashboard",
                            shap_interaction=False,
                            whatif=True,
                            importances=True,
                            shap_dependence=True,
                            decision_trees=isinstance(fitted_clf, DecisionTreeClassifier),
                        )
                        try:
                            html_str = dashboard.to_html()
                        except TypeError:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                html_path = Path(tmpdir) / "db.html"
                                dashboard.to_html(filename=str(html_path))
                                html_str = html_path.read_text(encoding="utf-8")
                        components.html(html_str, height=800, scrolling=True)
                    except Exception:
                        st.info("ExplainerDashboard embedding failed for this pipeline/model. The above custom panels provide explanation details.")
                except Exception:
                    st.info("ExplainerDashboard not available or failed to render.")
            else:
                st.info("ExplainerDashboard library not installed. Add 'explainerdashboard' to requirements if you want the full embedded dashboard.")

# -----------------------------
# TAB 4: Predictions (new data)
# -----------------------------
with tab4:
    st.header("📈 Predictions on New Data")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Train a model first.")
    else:
        st.success("Model is ready for predictions.")
        clf_pipeline = st.session_state.trained_clf
        trained_feature_cols = st.session_state.trained_feature_cols
        trained_le_target = st.session_state.trained_le_target

        st.subheader("Upload data for prediction")
        new_file = st.file_uploader("Upload CSV/XLSX with same features", type=["csv", "xlsx"], key="pred_file")
        if new_file:
            try:
                if new_file.name.endswith(".csv"):
                    new_df = pd.read_csv(new_file)
                else:
                    new_df = pd.read_excel(new_file)
                st.dataframe(new_df.head(), use_container_width=True)
                missing = [c for c in trained_feature_cols if c not in new_df.columns]
                extra = [c for c in new_df.columns if c not in trained_feature_cols]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    features_df = new_df[trained_feature_cols]
                    if extra:
                        st.warning(f"Ignoring extra columns: {extra[:5]}{'...' if len(extra)>5 else ''}")
                    if st.button("🔮 Generate Predictions", use_container_width=True):
                        with st.spinner("Predicting..."):
                            try:
                                preds = clf_pipeline.predict(features_df)
                                probs = None
                                try:
                                    probs = clf_pipeline.predict_proba(features_df)
                                except Exception:
                                    probs = None
                                results = new_df.copy()
                                if trained_le_target is not None:
                                    try:
                                        decoded = trained_le_target.inverse_transform(preds)
                                        results["prediction"] = decoded
                                        results["prediction_encoded"] = preds
                                    except Exception:
                                        results["prediction"] = preds
                                else:
                                    results["prediction"] = preds
                                if probs is not None:
                                    # name prob cols
                                    if trained_le_target is not None and hasattr(trained_le_target, "classes_"):
                                        prob_cols = [f"prob_{c}" for c in trained_le_target.classes_]
                                    else:
                                        prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
                                    for i, col in enumerate(prob_cols):
                                        results[col] = probs[:, i]
                                st.success(f"Predicted {len(results)} rows.")
                                st.dataframe(results, use_container_width=True)
                                csv_buf = io.StringIO()
                                results.to_csv(csv_buf, index=False)
                                st.download_button("📥 Download Predictions CSV", data=csv_buf.getvalue(), file_name=f"predictions_{new_file.name.split('.')[0]}.csv")
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                st.text(traceback.format_exc())
            except Exception as e:
                st.error(f"Failed to load prediction file: {e}")
                st.text(traceback.format_exc())

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | © 2025 CE Team Prediction Pro Analytics App")
