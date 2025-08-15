# app.py
"""
Predictive Analytics Streamlit App — Full Version (fixed)
- Robust profiling with sanitization + fallback HTML if ProfileReport.to_html fails
- ExplainerDashboard attempted; if it fails (Plotly titlefont error), show internal fallback explainer visuals
- Compact SHAP visuals and safe pipeline handling
- Defensive programming and user guidance
"""

import io
import re
import html
import tempfile
import traceback
import random
from pathlib import Path

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
# Optional libs (safeguarded)
# -------------------------
# Profiling
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

# ML libraries
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

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

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
st.title("📊 Predictive Analytics — Explorer & Explainer")
st.markdown("Upload dataset → profile → train → explain → predict.")

# session defaults
_default_keys = {
    "data_uploaded": False,
    "profile_generated": False,
    "model_trained": False,
    "current_file": None,
    "data": None,
    "profile_html": None,
    "explainer_selected_index": None,
}
for k, v in _default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Helper functions
# -------------------------
def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

def load_data_safe(uploaded_file):
    """Load CSV or Excel safely, with fallback attempts."""
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

def sanitize_dataframe_for_profiling(df: pd.DataFrame, check_n=50):
    """
    Sanitize DataFrame before profiling:
      - Ensure DataFrame type
      - Drop fully-empty columns
      - Attempt to coerce object columns that are numeric-like
      - Drop columns that contain nested or non-scalar values (lists/dicts/arrays)
      - Drop constant columns (nunique <= 1)
      - Reset index
    """
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.dropna(axis=1, how="all")
    # coerce object cols that look numeric
    for col in df.columns:
        if df[col].dtype == "object":
            nonnull = df[col].dropna().head(check_n)
            try:
                coerced = pd.to_numeric(nonnull)
                if len(coerced) >= len(nonnull) * 0.9:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    # drop nested-type columns
    drop_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(check_n).tolist()
        for v in sample:
            if not is_scalar_value(v):
                drop_cols.append(col)
                break
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    # drop constant columns
    nunique = df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    df = df.loc[:, keep]
    df = df.reset_index(drop=True)
    return df

def generate_profile_report_safe(df: pd.DataFrame):
    """
    Generate profiling HTML:
      - Try ProfileReport.to_html(); if that fails, provide fallback HTML containing
        basic pandas summaries so profiling never raises an exception for the app.
    """
    if ProfileReport is None:
        return "<h3>Profiling library not available</h3><p>Install 'ydata-profiling' or 'pandas-profiling' to enable full reports.</p>"
    try:
        df_s = sanitize_dataframe_for_profiling(df)
        if df_s.shape[1] == 0:
            return "<h3>No columns available to profile after sanitization.</h3>"
        profile = ProfileReport(df_s, title="Dataset Profiling Report", explorative=True)
        try:
            return profile.to_html()
        except Exception:
            # fallback: build HTML from pandas summaries
            try:
                html_parts = []
                html_parts.append("<h2>Fallback Profiling Report</h2>")
                html_parts.append("<p><em>ProfileReport.to_html() failed — presenting a pandas-based fallback summary.</em></p>")
                html_parts.append("<h3>Sample (first 10 rows)</h3>")
                html_parts.append(df_s.head(10).to_html(classes='table table-striped', index=False))
                html_parts.append("<h3>Column dtypes</h3>")
                html_parts.append(df_s.dtypes.to_frame("dtype").to_html(classes='table table-striped'))
                html_parts.append("<h3>Null counts</h3>")
                nulls = df_s.isnull().sum()
                html_parts.append(nulls.to_frame("null_count").to_html(classes='table table-striped'))
                html_parts.append("<h3>Unique counts (top 10)</h3>")
                uniq = df_s.nunique().sort_values()
                html_parts.append(uniq.to_frame("unique_count").to_html(classes='table table-striped'))
                html_parts.append("<h3>Descriptive statistics (numeric)</h3>")
                html_parts.append(df_s.describe().T.to_html(classes='table table-striped'))
                try:
                    corr = df_s.corr()
                    html_parts.append("<h3>Correlation (pearson) — numeric cols</h3>")
                    html_parts.append(corr.to_html(classes='table table-striped'))
                except Exception:
                    pass
                return "<html><body style='font-family:Arial, Helvetica, sans-serif; padding:10px'>" + "".join(html_parts) + "</body></html>"
            except Exception:
                tb = traceback.format_exc()
                return f"<h3>Profiling fallback generation failed</h3><pre>{html.escape(tb)}</pre>"
    except Exception:
        tb = traceback.format_exc()
        return f"<h3>Profiling sanitization failed</h3><pre>{html.escape(tb)}</pre>"

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def safe_shap_explainer(fitted_model, background_data, feature_names=None):
    if not SHAP_AVAILABLE:
        return None
    try:
        if feature_names is not None:
            return shap.Explainer(fitted_model, background_data, feature_names=feature_names)
        return shap.Explainer(fitted_model, background_data)
    except Exception:
        try:
            return shap.Explainer(fitted_model, background_data)
        except Exception:
            return None

def ensure_shap_array_for_idx(vals_obj, idx):
    try:
        if hasattr(vals_obj, "values"):
            vals = vals_obj.values
        else:
            vals = vals_obj
        arr = np.array(vals)
        if arr.ndim == 1:
            return arr.ravel()
        if arr.ndim == 2:
            return arr[idx].ravel()
        if arr.ndim == 3:
            # attempt sample-first
            if arr.shape[0] == len(vals):
                return arr[idx].ravel()
            # fallback: choose class with largest mean abs
            s = arr[idx]
            if s.ndim == 2:
                out_idx = int(np.argmax(np.abs(s).mean(axis=1)))
                return s[out_idx].ravel()
            return s.ravel()
    except Exception:
        return None

def make_pdp_values(clf_pipeline, base_row_df, feature, grid):
    preds = []
    for v in grid:
        temp = base_row_df.copy()
        temp[feature] = v
        try:
            if hasattr(clf_pipeline, "predict_proba"):
                p = clf_pipeline.predict_proba(temp)
                preds.append(float(np.max(p)))
            else:
                p = clf_pipeline.predict(temp)
                preds.append(float(np.ravel(p)[0]))
        except Exception:
            preds.append(np.nan)
    return preds

def compact_shap_summary_plot(shap_values, X):
    plt.close("all")
    fig = plt.figure(figsize=(5, 3))
    try:
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        return fig
    except Exception:
        arr = np.abs(np.array(shap_values))
        if arr.ndim == 3:
            mean_imp = np.mean(np.mean(arr, axis=1), axis=0)
        else:
            mean_imp = np.mean(arr, axis=0)
        idxs = np.argsort(mean_imp)[-20:]
        feats = X.columns.tolist()
        sel = [feats[i] for i in idxs]
        vals = mean_imp[idxs]
        ax = fig.add_subplot(111)
        ax.barh(sel, vals)
        ax.set_xlabel("mean(|SHAP value|)")
        plt.tight_layout()
        return fig

def safe_filename_from_upload(name: str):
    base = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_\-]", "_", base)

def fallback_explainer_visuals(clf_pipeline, X_test_df, y_test_arr, le_target, class_names, idx):
    """
    If ExplainerDashboard cannot be constructed, show a set of fallback visuals:
    - Confusion matrix
    - ROC curve (if binary or probability available)
    - Precision-recall curve
    - Feature importances (if available)
    - SHAP summary + local contributions (if SHAP available)
    """
    st.subheader("Fallback Explainer Visuals")
    st.markdown("ExplainerDashboard failed to initialize in your environment; showing equivalent analysis here.")
    y_true = np.array(y_test_arr)
    y_pred = clf_pipeline.predict(X_test_df)
    try:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(cm)
        st.dataframe(cm_df)
    except Exception:
        st.info("Confusion matrix not available.")

    # ROC / AUC (if probability and binary)
    try:
        if hasattr(clf_pipeline, "predict_proba"):
            proba = clf_pipeline.predict_proba(X_test_df)
            if proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], "--", linewidth=0.7)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC curve")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("ROC curve requires binary problem or probability for positive class.")
        else:
            st.info("No predict_proba available for ROC curve.")
    except Exception:
        st.info("ROC curve generation failed.")

    # Precision-Recall
    try:
        if hasattr(clf_pipeline, "predict_proba"):
            proba = clf_pipeline.predict_proba(X_test_df)
            if proba.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_true, proba[:, 1])
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(recall, precision)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall")
                plt.tight_layout()
                st.pyplot(fig)
    except Exception:
        st.info("Precision-Recall generation failed.")

    # Feature importances (tree-based)
    try:
        # attempt to get classifier from pipeline
        if isinstance(clf_pipeline, Pipeline) or hasattr(clf_pipeline, "named_steps"):
            try:
                clf = clf_pipeline.named_steps.get("classifier", None) or clf_pipeline.steps[-1][1]
            except Exception:
                clf = clf_pipeline
        else:
            clf = clf_pipeline
        if hasattr(clf, "feature_importances_"):
            st.markdown("**Feature importances**")
            fi = np.array(clf.feature_importances_)
            # try to get feature names
            try:
                pre = clf_pipeline.named_steps.get("preprocessor", None)
                if pre is not None and hasattr(pre, "get_feature_names_out"):
                    feat_names = pre.get_feature_names_out()
                else:
                    feat_names = X_test_df.columns.tolist()
            except Exception:
                feat_names = X_test_df.columns.tolist()
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False).head(20)
            st.dataframe(fi_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
            plt.tight_layout()
            st.pyplot(fig)
    except Exception:
        pass

    # SHAP visuals (if SHAP available)
    if SHAP_AVAILABLE:
        try:
            pre = None
            try:
                pre = clf_pipeline.named_steps.get("preprocessor", None)
            except Exception:
                pre = None
            # background
            try:
                bg = pre.transform(X_test_df) if pre is not None else X_test_df.values
            except Exception:
                bg = X_test_df.values
            expl = safe_shap_explainer(clf_pipeline.named_steps.get("classifier", clf_pipeline) if hasattr(clf_pipeline, "named_steps") else clf_pipeline, bg, feature_names=(pre.get_feature_names_out() if pre is not None and hasattr(pre, "get_feature_names_out") else X_test_df.columns.tolist()))
            if expl is not None:
                st.markdown("**SHAP global summary (compact)**")
                X_for_shap = X_test_df.head(200)
                try:
                    shap_vals = expl(X_for_shap)
                    shap_vals_to_plot = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
                    fig = compact_shap_summary_plot(shap_vals_to_plot, X_for_shap)
                    st.pyplot(fig)
                except Exception:
                    st.info("SHAP global summary failed.")
                st.markdown("**SHAP local contributions (for selected index)**")
                try:
                    single = X_test_df.iloc[[idx]]
                    single_proc = pre.transform(single) if pre is not None else single.values
                    shap_out = expl(single_proc)
                    local_vals = shap_out.values if hasattr(shap_out, "values") else shap_out
                    local_arr = ensure_shap_array_for_idx(local_vals, 0)
                    if local_arr is not None:
                        feat_names = pre.get_feature_names_out() if pre is not None and hasattr(pre, "get_feature_names_out") else X_test_df.columns.tolist()
                        contrib_df = pd.DataFrame({"feature": list(feat_names), "contribution": local_arr})
                        contrib_df["abs_contrib"] = contrib_df["contribution"].abs()
                        contrib_df = contrib_df.sort_values("abs_contrib", ascending=False).head(20)
                        st.dataframe(contrib_df[["feature", "contribution"]], use_container_width=True)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.barh(contrib_df["feature"][::-1], contrib_df["contribution"][::-1])
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception:
                    st.info("SHAP local contributions failed.")
        except Exception:
            st.info("SHAP visuals generation failed or SHAP not supported for this model.")
    else:
        st.info("SHAP not installed; install 'shap' to see SHAP-based visuals.")

# -------------------------
# Sidebar: upload
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Profiling", "🚀 Model Development", "🔍 Model Evaluation & Explainer", "📈 Predictions"])

# -------------------------
# TAB 1: Data Profiling
# -------------------------
with tab1:
    st.header("📊 Data Profiling")
    st.markdown("Generate a profiling report. The app sanitizes the dataset to remove problematic columns.")
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset in the sidebar to begin profiling.")
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
        with st.expander("Quick preview"):
            st.dataframe(df.head())

        if st.button("🔍 Generate Comprehensive Profile Report"):
            if ProfileReport is None:
                st.error("Profiling library not installed. Add 'ydata-profiling' or 'pandas-profiling' to requirements.")
            else:
                with st.spinner("Generating profiling report..."):
                    try:
                        html_out = generate_profile_report_safe(df)
                        st.session_state.profile_html = html_out
                        st.session_state.profile_generated = True
                        st.success("✅ Profile generated (sanitized).")
                    except Exception as e:
                        st.error(f"Profile generation failed: {e}")
                        st.text(traceback.format_exc())

        if st.session_state.profile_generated and st.session_state.profile_html:
            st.subheader("📋 Profiling Report")
            if HAS_ST_PROFILE:
                try:
                    st_profile_report(st.session_state.profile_html)
                except Exception:
                    components.html(st.session_state.profile_html, height=700, scrolling=True)
            else:
                components.html(st.session_state.profile_html, height=700, scrolling=True)

# -------------------------
# TAB 2: Model Development
# -------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Choose target, features, algorithm, balance classes and train the model.")
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
        ohe = safe_onehot_encoder()
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
# TAB 3: Model Evaluation & Explainer
# -------------------------
with tab3:
    st.header("🔍 Model Evaluation & Explainer")
    st.markdown("Model performance, local explanations, PDP, and ExplainerDashboard (if available). The app will fall back to built-in visuals if the dashboard cannot initialize.")

    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Train a model in Model Development first.")
    else:
        clf_pipeline = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results
        X_test_df = st.session_state.trained_X_test.copy()
        y_test_arr = st.session_state.trained_y_test

        st.subheader("📊 Performance Summary")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
        with p2:
            st.metric("Test samples", len(y_test_arr))
        with p3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        st.subheader("📋 Detailed Performance Report")
        try:
            report = classification_report(y_test_arr, test_results["y_pred"], output_dict=True, target_names=class_names)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
        except Exception:
            try:
                report = classification_report(y_test_arr, test_results["y_pred"], output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
            except Exception:
                st.info("Classification report not available.")

        # Auto-select index
        if st.session_state.explainer_selected_index is None:
            if len(X_test_df) > 0:
                st.session_state.explainer_selected_index = int(random.randrange(0, len(X_test_df)))
            else:
                st.session_state.explainer_selected_index = None

        if st.session_state.explainer_selected_index is None:
            st.info("Test set empty — cannot show explainer details.")
        else:
            idx = st.slider("Select index for explanation", 0, max(0, len(X_test_df) - 1), value=int(st.session_state.explainer_selected_index), step=1)
            st.session_state.explainer_selected_index = int(idx)

            sel_col, pred_col = st.columns([1, 2])
            with sel_col:
                st.markdown("### Selected index")
                st.write(f"**{idx}**")
                st.markdown("**Guidance:** Pick an index to view local explanation & prediction breakdown.")
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

            # Try using ExplainerDashboard, but gracefully fallback if it raises (e.g., Plotly titlefont error)
            if EXPLAINERDASH_AVAILABLE:
                st.markdown("### Full ExplainerDashboard (attempting to embed)")
                st.markdown("If the dashboard cannot initialize due to environment/Plotly incompatibilities, the app will show equivalent fallback visuals below.")
                try:
                    sample_n = min(1000, len(X_test_df))
                    sample_idx = np.random.choice(len(X_test_df), size=sample_n, replace=False)
                    X_dash = X_test_df.iloc[sample_idx]
                    y_dash = y_test_arr[sample_idx]

                    fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)

                    expl = ClassifierExplainer(fitted_clf, X_dash, y_dash, labels=class_names if class_names else None, model_output="probability")

                    dashboard = ExplainerDashboard(
                        expl,
                        title="Model Performance Dashboard",
                        bootstrap="FLATLY",
                        whatif=True,
                        importances=True,
                        model_summary=True,
                        contributions=True,
                        shap_dependence=True,
                        shap_interaction=True,
                        decision_trees=isinstance(fitted_clf, DecisionTreeClassifier),
                        hide_poweredby=True,
                        fluid=True,
                    )

                    # If it constructs, get HTML
                    try:
                        html_str = dashboard.to_html()
                    except TypeError:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            html_path = Path(tmpdir) / "dashboard.html"
                            dashboard.to_html(filename=str(html_path))
                            html_str = html_path.read_text(encoding="utf-8")

                    # small patch for older/newer mismatches (best-effort)
                    patched = html_str.replace("titlefont", "title_font")
                    patched = patched.replace("title : {", "title: {")  # no-op harmless
                    components.html(patched, height=900, scrolling=True)
                except Exception as e:
                    st.error(f"ExplainerDashboard failed to build or render: {e}")
                    st.info("Showing fallback explainer visuals instead.")
                    # show fallback visuals that replicate dashboard analyses
                    fallback_explainer_visuals(clf_pipeline, X_test_df, y_test_arr, le_target, class_names, idx)
            else:
                # not available: show fallback visuals
                st.markdown("ExplainerDashboard not installed — showing fallback visuals.")
                fallback_explainer_visuals(clf_pipeline, X_test_df, y_test_arr, le_target, class_names, idx)

            st.markdown("---")
            # Partial Dependence Plot (approx)
            st.markdown("### Partial Dependence Plot (approx)")
            try:
                numeric_cols = X_test_df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.info("No numeric features available for PDP.")
                else:
                    pdp_feat = st.selectbox("Choose numeric feature for PDP", numeric_cols, index=0)
                    base_row = X_test_df.iloc[[idx]].copy().reset_index(drop=True)
                    grid = np.linspace(X_test_df[pdp_feat].min(), X_test_df[pdp_feat].max(), num=40)
                    pdp_preds = make_pdp_values(clf_pipeline, base_row, pdp_feat, grid)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(grid, pdp_preds, linewidth=1.6, marker="o", markersize=3)
                    ax.set_xlabel(pdp_feat, fontsize=10)
                    ax.set_ylabel("Predicted output / prob", fontsize=10)
                    ax.tick_params(axis="x", labelsize=9)
                    ax.tick_params(axis="y", labelsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"PDP failed: {e}")
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
                                    if trained_le_target is not None and hasattr(trained_le_target, "classes_"):
                                        prob_cols = [f"prob_{c}" for c in trained_le_target.classes_]
                                    else:
                                        prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
                                    for i, col in enumerate(prob_cols):
                                        results[col] = probs[:, i]

                                st.success(f"Predicted {len(results)} rows.")
                                st.dataframe(results, use_container_width=True)

                                base = safe_filename_from_upload(new_file.name)
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
