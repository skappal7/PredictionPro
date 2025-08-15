# app.py
"""
Predictive Analytics Streamlit App
- Profiling: exported to standalone HTML and embedded
- ExplainerDashboard: exported to standalone HTML and embedded (uses preprocessed X_dash)
- No separate SHAP visuals (dashboard provides them)
- Defensive error handling and fallbacks
"""

import io
import os
import re
import html
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

# -------------------------
# Optional third-party libraries (guarded imports)
# -------------------------
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

# Optional nicer embedding helper (not required)
try:
    from streamlit_pandas_profiling import st_profile_report
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

# imbalanced-learn
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMB_AVAILABLE = True
except Exception:
    ImbPipeline = Pipeline
    SMOTE = RandomOverSampler = RandomUnderSampler = None
    IMB_AVAILABLE = False

# ExplainerDashboard
try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    EXPLAINERDASH_AVAILABLE = True
except Exception:
    EXPLAINERDASH_AVAILABLE = False

# -------------------------
# Page config and session defaults
# -------------------------
st.set_page_config(page_title="Predictive Analytics App", layout="wide", page_icon="📊")
st.title("📊 Predictive Analytics — Profiling & Explainer (HTML-embedded)")
st.markdown("Upload dataset → Generate profile → Train model → View ExplainerDashboard (embedded) → Predict")

_session_defaults = {
    "data_uploaded": False,
    "profile_generated": False,
    "model_trained": False,
    "current_file": None,
    "data": None,
    "profile_html_path": None,
    "dashboard_html_path": None,
    "explainer_selected_index": None,
}
for k, v in _session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Helpers
# -------------------------
def safe_filename_base(name: str) -> str:
    base = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_\-]", "_", base)[:128]

def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

def load_data_safe(uploaded_file):
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
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.dropna(axis=1, how="all")
    for col in df.columns:
        if df[col].dtype == "object":
            nonnull_sample = df[col].dropna().head(check_n)
            try:
                coerced = pd.to_numeric(nonnull_sample)
                if len(coerced) >= max(1, int(len(nonnull_sample) * 0.9)):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    drop_cols = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(check_n).tolist()
        for v in sample_vals:
            if not is_scalar_value(v):
                drop_cols.append(col)
                break
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    nunique = df.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    df = df.loc[:, keep_cols]
    df = df.reset_index(drop=True)
    return df

def generate_profile_html_file(df: pd.DataFrame, tmp_dir: Optional[str] = None) -> str:
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
        # prefer to_file for better compatibility
        try:
            profile.to_file(fd_path)
            return fd_path
        except Exception:
            html_str = profile.to_html()
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            return fd_path
    except Exception:
        # fallback minimal html
        fallback = "<html><body><h3>Profiling generation failed</h3><pre>{}</pre></body></html>".format(html.escape(traceback.format_exc()))
        with open(fd_path, "w", encoding="utf-8") as fh:
            fh.write(fallback)
        return fd_path

def generate_explainer_dashboard_html_file(fitted_clf, X_dash: pd.DataFrame, y_dash: np.ndarray, class_names: Optional[list] = None, tmp_dir: Optional[str] = None) -> str:
    if not EXPLAINERDASH_AVAILABLE:
        raise RuntimeError("explainerdashboard not installed.")
    expl = ClassifierExplainer(fitted_clf, X_dash, y_dash, labels=class_names if class_names else None, model_output="probability")
    dashboard = ExplainerDashboard(
        expl,
        title="Model Performance Dashboard",
        shap_interaction=False,
        whatif=True,
        importances=True,
        shap_dependence=True,
        decision_trees=isinstance(fitted_clf, DecisionTreeClassifier),
        hide_poweredby=True,
        fluid=True,
    )
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
    fd_path = fd.name
    fd.close()
    # try multiple export methods
    try:
        # preferred to_file
        dashboard.to_file(fd_path)
        return fd_path
    except Exception:
        try:
            # some versions require to_html
            html_str = dashboard.to_html()
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            return fd_path
        except Exception:
            try:
                dashboard.to_html(filename=fd_path)
                return fd_path
            except Exception:
                raise

def patch_html_titlefont(html_str: str) -> str:
    if not isinstance(html_str, str):
        return html_str
    s = html_str
    s = re.sub(r'("titlefont"\s*:\s*)\{', r'"title_font": {', s)
    s = re.sub(r'(\.titlefont\s*=\s*)\{', r'.title_font = {', s)
    s = re.sub(r'\.titlefont\b', r'.title_font', s)
    s = re.sub(r'(\btitlefont\s*:\s*)\{', r'title_font: {', s)
    return s

# -------------------------
# Sidebar: file upload
# -------------------------
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Profiling", "🚀 Model Development", "🔍 Model Evaluation", "📈 Predictions"])

# -------------------------
# TAB 1: Data Profiling (HTML embed)
# -------------------------
with tab1:
    st.header("📊 Data Profiling (HTML-embedded)")
    st.markdown("Generate a profiling report (sanitized) and embed as standalone HTML.")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data in the sidebar to begin profiling.")
    else:
        df = st.session_state.data
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            st.metric("Memory", f"{mem_mb:.1f} MB")
        with c4:
            st.metric("Missing values", int(df.isnull().sum().sum()))

        with st.expander("Quick Preview"):
            st.dataframe(df.head())

        if st.button("🔍 Generate & Embed Profile (HTML)"):
            if ProfileReport is None:
                st.error("Profiling library not installed. Add 'ydata-profiling' or 'pandas-profiling' to your environment.")
            else:
                with st.spinner("Generating profile (this may take a while for large datasets)..."):
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
                st.error(f"Failed to read/embed profile HTML: {e}")
                st.text(traceback.format_exc())

# -------------------------
# TAB 2: Model Development
# -------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Select target, features, algorithm, balance classes, and train.")
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
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct / 100.0, random_state=42, stratify=(y if len(unique) >= 2 else None))
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

# -------------------------
# TAB 3: Model Evaluation & ExplainerDashboard (HTML-embedded)
# -------------------------
with tab3:
    st.header("🔍 Model Evaluation")
    st.markdown("""
    **Model Evaluation Components:**
    - **Performance Metrics**: Accuracy, precision, recall, F1-score for each class
    - **Interactive Dashboard**: Explore predictions, feature importance, and decision boundaries (embedded HTML)
    - **Model Interpretability**: Use the ExplainerDashboard for SHAP and what-if analysis
    """)

    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Please train a model in the Model Development tab first.")
    else:
        clf = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results

        y_test = test_results["y_test"]
        y_pred = test_results["y_pred"]
        accuracy = test_results["accuracy"]

        st.subheader("📊 Model Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Test Samples", len(y_test))
        with col3:
            st.metric("Features Used", len(st.session_state.trained_feature_cols))

        st.subheader("📋 Detailed Performance Report")
        try:
            report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
        except Exception:
            report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🖥️ Interactive Model Explorer (ExplainerDashboard embedded)")
        st.markdown("""
        The dashboard below is exported to a standalone HTML file and embedded. It provides:
        - Feature importances & SHAP global summary
        - Individual prediction explanations & what-if
        - Model performance diagnostics (ROC, confusion matrix, PR)
        """)

        try:
            # Prepare data for dashboard embedding — follow your working pattern
            data = st.session_state.data
            X_all = data[st.session_state.trained_feature_cols]
            # attempt to get preprocessor and classifier from pipeline
            fitted_pre = None
            fitted_clf = clf
            try:
                fitted_pre = clf.named_steps["preprocessor"]
                fitted_clf = clf.named_steps["classifier"]
            except Exception:
                # pipeline doesn't match expected shape; fallback to entire pipeline as model
                fitted_pre = None
                fitted_clf = clf

            # Build a preprocessed DataFrame (like your working example)
            # We'll sample indices from the test set where available, otherwise sample from X_all
            n_test = len(y_test)
            sample_n = min(300, n_test if n_test > 0 else len(X_all))
            if n_test > 0:
                # choose random indices from test set range
                idx = np.random.choice(range(len(X_all)), size=sample_n, replace=False)
            else:
                idx = np.random.choice(range(len(X_all)), size=sample_n, replace=False)

            # Transform X for dashboard input
            try:
                if fitted_pre is not None:
                    # transform selected samples and build DataFrame of transformed features
                    X_dash_arr = fitted_pre.transform(X_all.iloc[idx])
                    try:
                        feat_names = fitted_pre.get_feature_names_out()
                    except Exception:
                        # fallback feature names
                        feat_names = [f"feature_{i}" for i in range(X_dash_arr.shape[1])]
                    X_dash = pd.DataFrame(X_dash_arr, columns=feat_names)
                else:
                    X_dash = X_all.iloc[idx].reset_index(drop=True)
                    try:
                        feat_names = X_dash.columns.tolist()
                    except Exception:
                        feat_names = [f"feature_{i}" for i in range(X_dash.shape[1])]
                # get matching y_dash (from test results if available)
                # if we trained with a split, use y_test sample; otherwise attempt to sample from full y array
                try:
                    if isinstance(y_test, np.ndarray) and len(y_test) >= sample_n:
                        y_dash = y_test[np.random.choice(len(y_test), size=sample_n, replace=False)]
                    else:
                        # fallback: try to derive from the original target column in data
                        y_raw_col = st.session_state.data[st.session_state.get("previous_target")] if st.session_state.get("previous_target") in st.session_state.data.columns else None
                        if y_raw_col is not None:
                            y_dash = y_raw_col.iloc[idx].to_numpy()
                        else:
                            # if we cannot build y_dash, use the stored test y if exists
                            y_dash = st.session_state.trained_y_test if st.session_state.get("trained_y_test") is not None else np.zeros(len(X_dash), dtype=int)
                except Exception:
                    y_dash = st.session_state.trained_y_test if st.session_state.get("trained_y_test") is not None else np.zeros(len(X_dash), dtype=int)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare X_dash for ExplainerDashboard: {e}")

            # Build the explainer & dashboard and export to HTML
            try:
                dashboard_html_path = generate_explainer_dashboard_html_file(fitted_clf, X_dash, y_dash, class_names=class_names)
                with open(dashboard_html_path, "r", encoding="utf-8") as fh:
                    dashboard_html = fh.read()
                # best-effort patch for titlefont mismatches (string replace)
                dashboard_html = patch_html_titlefont(dashboard_html)
                components.html(dashboard_html, height=900, scrolling=True)
                st.success("✅ ExplainerDashboard embedded.")
                st.session_state.dashboard_html_path = dashboard_html_path
            except Exception as e:
                # Dashboard construction/export failed — show helpful message and fallback diagnostics
                st.error(f"Dashboard generation failed: {e}")
                st.info("ExplainerDashboard could not be generated in this environment. Showing fallback diagnostics (no SHAP visuals).")
                st.text(traceback.format_exc())
                # Fallback diagnostics (confusion matrix, ROC/PR, feature importances if available)
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(pd.DataFrame(cm))
                except Exception:
                    st.info("Confusion matrix unavailable.")
                try:
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X_all)
                        if proba.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y_test, proba[:len(y_test), 1]) if len(y_test) > 0 else ([], [], [])
                            if len(fpr) > 0:
                                roc_auc = auc(fpr, tpr)
                                fig, ax = plt.subplots(figsize=(5, 3))
                                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                                ax.plot([0, 1], [0, 1], "--", linewidth=0.7)
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)
                                precision, recall, _ = precision_recall_curve(y_test, proba[:len(y_test), 1])
                                fig2, ax2 = plt.subplots(figsize=(5, 3))
                                ax2.plot(recall, precision)
                                ax2.set_xlabel("Recall")
                                ax2.set_ylabel("Precision")
                                plt.tight_layout()
                                st.pyplot(fig2)
                    else:
                        st.info("No predict_proba available for ROC/PR.")
                except Exception:
                    st.info("ROC/PR fallback generation failed.")
                try:
                    # feature importances
                    try:
                        clf_model = clf.named_steps.get("classifier", clf) if hasattr(clf, "named_steps") else clf
                    except Exception:
                        clf_model = clf
                    if hasattr(clf_model, "feature_importances_"):
                        fi = np.array(clf_model.feature_importances_)
                        # try to get feature names
                        try:
                            pre = clf.named_steps.get("preprocessor", None) if hasattr(clf, "named_steps") else None
                            if pre is not None and hasattr(pre, "get_feature_names_out"):
                                feat_names = pre.get_feature_names_out()
                            else:
                                feat_names = X_all.columns.tolist()
                        except Exception:
                            feat_names = X_all.columns.tolist()
                        fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(20)
                        st.markdown("**Feature importances (top 20)**")
                        st.dataframe(fi_df, use_container_width=True)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Dashboard generation pre-processing failed: {e}")
            st.text(traceback.format_exc())

# -------------------------
# TAB 4: Predictions
# -------------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("Upload new data (CSV/XLSX) with the same features used in training to get predictions and download results.")
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
st.markdown("Built with ❤️ using Streamlit | © 2025 Predictive Analytics Team")
