# app.py
"""
Predictive Analytics Streamlit App using Shapash for model interpretation
- Data profiling exported to standalone HTML and embedded
- Shapash SmartExplainer used instead of ExplainerDashboard
- Shapash report exported to standalone HTML and embedded
- No separate SHAP visuals (Shapash provides interpretation)
- Defensive: sanitization, fallbacks, friendly messages
"""

import io
import os
import re
import tempfile
import traceback
import random
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 110,
})

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# ---------- Optional / guarded third-party imports ----------
# Profiling libs
try:
    from ydata_profiling import ProfileReport
    PROFILING_LIB = "ydata_profiling"
except Exception:
    try:
        from pandas_profiling import ProfileReport
        PROFILING_LIB = "pandas_profiling"
    except Exception:
        ProfileReport = None
        PROFILING_LIB = None

try:
    from streamlit_pandas_profiling import st_profile_report  # optional nicety
    HAS_ST_PROFILE = True
except Exception:
    HAS_ST_PROFILE = False

# ML stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# imbalanced-learn (optional)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMB_AVAILABLE = True
except Exception:
    ImbPipeline = Pipeline
    SMOTE = RandomOverSampler = RandomUnderSampler = None
    IMB_AVAILABLE = False

# Shapash
try:
    # SmartExplainer interface
    from shapash.explainer.smart_explainer import SmartExplainer
    SHAPASH_AVAILABLE = True
except Exception:
    SmartExplainer = None
    SHAPASH_AVAILABLE = False

# -----------------------------
# Streamlit config & session defaults
# -----------------------------
st.set_page_config(page_title="Predictive Analytics (Shapash)", layout="wide", page_icon="📊")
st.title("📊 Predictive Analytics — Profiling & Shapash (static HTML embeds)")
st.markdown("Upload → Profile → Train → Generate Shapash report (HTML) → Embed → Predict")

defaults = {
    "data_uploaded": False,
    "profile_generated": False,
    "model_trained": False,
    "current_file": None,
    "data": None,
    "profile_html_path": None,
    "shapash_html_path": None,
    "explainer_selected_index": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Helper functions
# -----------------------------
def safe_filename_base(name: str) -> str:
    base = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_\-]", "_", base)[:128]

def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

def load_data_safe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

def is_scalar_value(v):
    return not isinstance(v, (list, dict, set, tuple, np.ndarray, pd.Series))

def sanitize_dataframe_for_profiling(df: pd.DataFrame, check_n: int = 50) -> pd.DataFrame:
    """Sanitize DataFrame before passing to profiling libraries."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # drop all-empty columns
    df = df.dropna(axis=1, how="all")
    # try convert object-like numeric columns
    for col in df.columns:
        if df[col].dtype == object:
            nonnull_sample = df[col].dropna().head(check_n)
            try:
                coerced = pd.to_numeric(nonnull_sample)
                if len(coerced) >= max(1, int(len(nonnull_sample) * 0.9)):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    # drop columns with complex objects
    drop_cols = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(check_n).tolist()
        for v in sample_vals:
            if not is_scalar_value(v):
                drop_cols.append(col)
                break
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    # drop constant columns
    nunique = df.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    df = df.loc[:, keep_cols]
    df = df.reset_index(drop=True)
    return df

def generate_profile_html_file(df: pd.DataFrame, tmp_dir: Optional[str] = None) -> str:
    """Generate a profiling HTML file using ydata_profiling or pandas_profiling and return path."""
    if ProfileReport is None:
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        with open(fd.name, "w", encoding="utf-8") as fh:
            fh.write("<h3>Profiling library not installed</h3><p>Install ydata-profiling or pandas-profiling.</p>")
        return fd.name

    df_s = sanitize_dataframe_for_profiling(df)
    if df_s.shape[1] == 0:
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        with open(fd.name, "w", encoding="utf-8") as fh:
            fh.write("<h3>No columns available to profile after sanitization.</h3>")
        return fd.name

    profile = ProfileReport(df_s, title="Dataset Profiling Report", explorative=True)
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
    fd_path = fd.name
    fd.close()
    try:
        profile.to_file(fd_path)
        return fd_path
    except Exception:
        try:
            html_str = profile.to_html()
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            return fd_path
        except Exception:
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write("<h3>Profiling generation failed</h3><pre>{}</pre>".format(traceback.format_exc()))
            return fd_path

def generate_shapash_html_file(fitted_clf, preprocessor, X_train, y_train, X_test, y_test, tmp_dir: Optional[str] = None) -> str:
    """
    Generate a Shapash HTML report and return its file path.
    Uses SmartExplainer.compile and SmartExplainer.generate_report.
    """
    if not SHAPASH_AVAILABLE:
        raise RuntimeError("Shapash not installed in the environment.")

    # Prepare SmartExplainer instance
    # Pass preprocessing if available so Shapash can inverse-transform features
    kwargs = {}
    if preprocessor is not None:
        kwargs["preprocessing"] = preprocessor

    xpl = SmartExplainer(model=fitted_clf, **kwargs)

    # compile the explainer using a sample (compilation can be heavy on large data)
    # prefer test set if present, otherwise use X_train
    try:
        # Shapash expects a DataFrame for x
        x_compile = X_test if X_test is not None and len(X_test) > 0 else X_train
        xpl.compile(x=x_compile, model=fitted_clf)
    except Exception as e:
        # fallback: try compile with raw X if preprocessor was provided or without model arg
        try:
            xpl.compile(x=x_compile)
        except Exception:
            # compilation failed: raise with context
            raise

    # export report to file (generate_report)
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
    fd_path = fd.name
    fd.close()
    try:
        # generate_report supports several args; include train/test for richer report if available
        gen_kwargs = {"output_file": fd_path}
        if X_train is not None and y_train is not None:
            gen_kwargs["x_train"] = X_train
            gen_kwargs["y_train"] = y_train
        if X_test is not None and y_test is not None:
            gen_kwargs["x_test"] = X_test
            gen_kwargs["y_test"] = y_test

        # generate the HTML report
        xpl.generate_report(**gen_kwargs)
        return fd_path
    except Exception:
        # as a last-ditch attempt, try to generate with minimal args
        try:
            xpl.generate_report(output_file=fd_path)
            return fd_path
        except Exception:
            # write the traceback into the file and raise
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write("<h3>Shapash report generation failed</h3><pre>{}</pre>".format(traceback.format_exc()))
            raise

# -----------------------------
# Sidebar - Upload
# -----------------------------
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("Status:")
    st.write(f"- Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"- Model trained: {st.session_state.model_trained}")

if uploaded_file:
    try:
        df = load_data_safe(uploaded_file)
        st.session_state.data = df
        st.session_state.data_uploaded = True
        st.session_state.current_file = uploaded_file.name
        st.success("✅ Data uploaded successfully")
        st.write(f"**Shape:** {df.shape}")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(df.head(5))
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.session_state.data_uploaded = False
else:
    if st.session_state.data is None:
        st.session_state.data_uploaded = False

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Profiling", "🚀 Model Development", "🔍 Model Evaluation (Shapash)", "📈 Predictions"])

# -----------------------------
# TAB 1: Data Profiling (HTML embed)
# -----------------------------
with tab1:
    st.header("📊 Data Profiling (embedded static HTML)")
    st.markdown("Generates a sanitized profiling report (ydata/pandas profiling) and embeds it as HTML.")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload a dataset in the sidebar to begin profiling.")
    else:
        df = st.session_state.data
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            mem_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{mem_mb:.1f} MB")
        with c4:
            st.metric("Missing values", int(df.isnull().sum().sum()))
        with st.expander("Quick Preview"):
            st.dataframe(df.head())

        if st.button("🔍 Generate & Embed Profile (HTML)"):
            if ProfileReport is None:
                st.error("Profiling library not installed. Add 'ydata-profiling' or 'pandas-profiling' to your environment.")
            else:
                with st.spinner("Generating profiling HTML..."):
                    try:
                        profile_html_path = generate_profile_html_file(df)
                        st.session_state.profile_html_path = profile_html_path
                        st.session_state.profile_generated = True
                        st.success("✅ Profiling HTML generated.")
                    except Exception as e:
                        st.error(f"Profile generation failed: {e}")
                        st.text(traceback.format_exc())

        if st.session_state.profile_generated and st.session_state.profile_html_path:
            st.subheader("📋 Embedded Profiling Report")
            try:
                with open(st.session_state.profile_html_path, "r", encoding="utf-8") as fh:
                    html_content = fh.read()
                components.html(html_content, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Failed to embed profiling report: {e}")
                st.text(traceback.format_exc())

# -----------------------------
# TAB 2: Model Development
# -----------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Select target, features, algorithm, balance classes, and train the model.")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    else:
        data = st.session_state.data.copy()
        all_cols = list(data.columns)
        target_column = st.selectbox("Choose target column", all_cols)
        available_features = [c for c in all_cols if c != target_column]
        feature_mode = st.radio("Feature selection:", ["Use all features", "Select features"], horizontal=True)
        if feature_mode == "Select features":
            feature_cols = st.multiselect("Pick features", available_features, default=available_features)
            if not feature_cols:
                st.warning("Select at least one feature.")
                st.stop()
        else:
            feature_cols = available_features

        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()

        if y_raw.dtype == "object" or y_raw.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = [str(c) for c in le_target.classes_]
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = [str(c) for c in np.unique(y)]

        st.subheader("📊 Target Distribution")
        a1, a2 = st.columns(2)
        with a1:
            st.write(class_counts(y))
        with a2:
            try:
                dist_df = pd.DataFrame(list(class_counts(y).items()), columns=["class", "count"]).set_index("class")
                st.bar_chart(dist_df)
            except Exception:
                pass

        st.subheader("🤖 Model Selection")
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(),
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
        if balance == "SMOTE" and IMB_AVAILABLE:
            sampler = SMOTE(random_state=42)
        elif balance == "Random Oversample" and IMB_AVAILABLE:
            sampler = RandomOverSampler(random_state=42)
        elif balance == "Random Undersample" and IMB_AVAILABLE:
            sampler = RandomUnderSampler(random_state=42)
        elif balance != "None" and not IMB_AVAILABLE:
            st.warning("imbalanced-learn not installed — balancing options are disabled")

        st.subheader("📊 Train/Test Split")
        test_pct = st.slider("Test size (%)", 10, 50, 20)

        numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
        categorical_feats = X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_transformer = Pipeline([("onehot", ohe)])

        preprocessor = ColumnTransformer(
            [("num", numeric_transformer, numeric_feats), ("cat", categorical_transformer, categorical_feats)],
            remainder="drop",
        )

        if sampler is None:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        else:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("classifier", model)])

        st.markdown("---")
        if st.button("🚀 Train Model", use_container_width=True):
            try:
                unique = np.unique(y)
                if len(unique) < 2:
                    st.error("Need at least 2 classes/values in target to train.")
                else:
                    with st.spinner("Training..."):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_pct / 100.0, random_state=42, stratify=(y if len(unique) >= 2 else None)
                        )
                        clf_pipeline.fit(X_train, y_train)
                        y_pred = clf_pipeline.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        st.session_state.model_trained = True
                        st.session_state.trained_clf = clf_pipeline
                        st.session_state.trained_le_target = le_target
                        st.session_state.trained_feature_cols = feature_cols
                        st.session_state.trained_class_names = class_names
                        st.session_state.test_results = {"y_test": np.array(y_test), "y_pred": np.array(y_pred), "accuracy": float(acc)}
                        st.session_state.trained_X_test = X_test.reset_index(drop=True)
                        st.session_state.trained_y_test = np.array(y_test)

                        st.success(f"✅ Trained. Accuracy: {acc:.3f}")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.text(traceback.format_exc())

# -----------------------------
# TAB 3: Model Evaluation & Shapash
# -----------------------------
with tab3:
    st.header("🔍 Model Evaluation & Shapash (HTML embed)")
    st.markdown(
        """
        - Performance metrics (classification report)
        - Shapash interactive report exported to static HTML and embedded below.
        - If Shapash report generation fails, a set of fallback diagnostics are shown.
        """
    )

    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Train a model first.")
    else:
        clf_pipeline = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results

        y_test = test_results["y_test"]
        y_pred = test_results["y_pred"]
        accuracy = test_results["accuracy"]

        st.subheader("📊 Performance Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with c2:
            st.metric("Test samples", len(y_test))
        with c3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        st.subheader("📋 Classification Report")
        try:
            report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
        except Exception:
            report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🖥️ Shapash Report (static HTML embedded)")
        st.markdown("Shapash will be compiled and a standalone HTML report will be generated and embedded below. This may take a moment for larger datasets.")

        if not SHAPASH_AVAILABLE:
            st.info("Shapash not installed. Add `shapash` to your environment to enable the interactive report.")
        else:
            if st.button("📦 Generate & Embed Shapash Report (HTML)"):
                with st.spinner("Compiling Shapash and generating HTML..."):
                    try:
                        # extract fitted preprocessor and classifier from pipeline
                        fitted_pre = None
                        fitted_clf = clf_pipeline
                        try:
                            fitted_pre = clf_pipeline.named_steps.get("preprocessor", None)
                            fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                        except Exception:
                            fitted_pre = None
                            fitted_clf = clf_pipeline

                        # prepare X_train, y_train, X_test, y_test to pass to Shapash generate_report
                        # prefer using the original training/test splits if available in session_state; otherwise use samples
                        X_train = None
                        y_train = None
                        X_test = None
                        y_test_for_report = None

                        # If training results saved, try to reconstruct
                        try:
                            # If we have trained_X_test stored, use it
                            if "trained_X_test" in st.session_state and st.session_state.trained_X_test is not None:
                                X_test = st.session_state.trained_X_test.copy()
                                y_test_for_report = st.session_state.trained_y_test.copy()
                            # As a fallback, sample from original dataset using stored feature cols
                            else:
                                data_all = st.session_state.data
                                if data_all is not None:
                                    X_test = data_all[st.session_state.trained_feature_cols].copy().iloc[: min(300, len(data_all))].reset_index(drop=True)
                                    # try to derive y_test_for_report from previous target if present
                                    prev_target = st.session_state.get("previous_target")
                                    if prev_target and prev_target in data_all.columns:
                                        y_test_for_report = data_all[prev_target].iloc[: min(300, len(data_all))].to_numpy()
                        except Exception:
                            X_test = None
                            y_test_for_report = None

                        # For X_train/y_train, try to use portions of dataset if possible (not strictly required)
                        # Keep them as None if not available; Shapash generate_report accepts partial info.

                        # Attempt to generate Shapash HTML
                        shapash_html_path = generate_shapash_html_file(
                            fitted_clf,
                            fitted_pre,
                            X_train,
                            y_train,
                            X_test,
                            y_test_for_report,
                        )
                        st.session_state.shapash_html_path = shapash_html_path
                        # Read and embed
                        with open(shapash_html_path, "r", encoding="utf-8") as fh:
                            sh_html = fh.read()
                        components.html(sh_html, height=900, scrolling=True)
                        st.success("✅ Shapash HTML generated and embedded.")
                    except Exception as e:
                        st.error(f"Shapash report generation failed: {e}")
                        st.text(traceback.format_exc())

            # If previously generated, show embedded report
            if st.session_state.shapash_html_path:
                try:
                    with open(st.session_state.shapash_html_path, "r", encoding="utf-8") as fh:
                        sh_html = fh.read()
                    components.html(sh_html, height=900, scrolling=True)
                except Exception:
                    pass

        # Provide fallback diagnostics if Shapash not available or fails
        st.markdown("---")
        st.markdown("**Fallback diagnostics (if Shapash not available)**")
        try:
            cm = confusion_matrix(y_test, y_pred)
            st.markdown("**Confusion Matrix**")
            st.dataframe(pd.DataFrame(cm))
        except Exception:
            st.info("Confusion matrix not available.")
        try:
            clf_model = clf_pipeline.named_steps.get("classifier", clf_pipeline) if hasattr(clf_pipeline, "named_steps") else clf_pipeline
            if hasattr(clf_pipeline, "predict_proba"):
                proba_all = clf_pipeline.predict_proba(st.session_state.trained_X_test if "trained_X_test" in st.session_state else st.session_state.data[st.session_state.trained_feature_cols])
                if proba_all.shape[1] == 2 and len(y_test) > 0:
                    fpr, tpr, _ = roc_curve(y_test, proba_all[: len(y_test), 1])
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                    ax.plot([0, 1], [0, 1], "--", linewidth=0.7)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    precision, recall, _ = precision_recall_curve(y_test, proba_all[: len(y_test), 1])
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(recall, precision)
                    ax2.set_xlabel("Recall")
                    ax2.set_ylabel("Precision")
                    plt.tight_layout()
                    st.pyplot(fig2)
        except Exception:
            st.info("ROC/PR generation failed or not applicable.")

# -----------------------------
# TAB 4: Predictions
# -----------------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("Upload new data (same features used in training) to obtain predictions and download results.")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Train a model first.")
    else:
        clf_pipeline = st.session_state.trained_clf
        trained_feature_cols = st.session_state.trained_feature_cols
        trained_le_target = st.session_state.trained_le_target

        new_file = st.file_uploader("Upload CSV/XLSX with same features", type=["csv", "xlsx"], key="pred_file")
        if new_file:
            try:
                name = new_file.name.lower()
                if name.endswith(".csv"):
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
                        st.warning(f"Ignoring extra columns: {extra[:5]}{'...' if len(extra) > 5 else ''}")
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
                                    if trained_le_target is not None and hasattr(trained_le_target, "classes_"):
                                        prob_cols = [f"prob_{c}" for c in trained_le_target.classes_]
                                    else:
                                        prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
                                    for i, col in enumerate(prob_cols):
                                        results[col] = probs[:, i]
                                st.success(f"Predicted {len(results)} rows.")
                                st.dataframe(results, use_container_width=True)
                                base = safe_filename_base(new_file.name)
                                fname = f"predictions_{base}.csv"
                                csv_buf = io.StringIO()
                                results.to_csv(csv_buf, index=False)
                                st.download_button("📥 Download Predictions CSV", data=csv_buf.getvalue(), file_name=fname)
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                st.text(traceback.format_exc())
            except Exception as e:
                st.error(f"Failed to load prediction file: {e}")
                st.text(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Built with ❤️ Streamlit | © 2025 Predictive Analytics Team")
