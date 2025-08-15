# app.py
"""
Predictive Analytics Streamlit App — robust ExplainerDashboard SHAP reshape fix
- Builds profiling and model pipeline UI.
- When building ExplainerDashboard: creates ClassifierExplainer (normal ctor),
  then inspects & normalizes explainer.shap_values in-place if it's 3D.
- Exports dashboard HTML using available API and embeds via components.html.
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

# -------------------------
# Guarded / optional imports
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
    ClassifierExplainer = None
    ExplainerDashboard = None
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
# Helper functions
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
        if df[col].dtype == object:
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
        out_html = "<h3>Profiling library not installed</h3><p>Install 'ydata-profiling' or 'pandas-profiling' to get full reports.</p>"
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        fd.write(out_html.encode("utf-8"))
        fd.close()
        return fd.name

    df_s = sanitize_dataframe_for_profiling(df)
    if df_s.shape[1] == 0:
        out_html = "<h3>No columns available to profile after sanitization.</h3>"
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        fd.write(out_html.encode("utf-8"))
        fd.close()
        return fd.name

    profile = ProfileReport(df_s, title="Dataset Profiling Report", explorative=True)
    try:
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        fd_path = fd.name
        fd.close()
        try:
            profile.to_file(fd_path)
            return fd_path
        except Exception:
            html_str = profile.to_html()
            with open(fd_path, "w", encoding="utf-8") as fh:
                fh.write(html_str)
            return fd_path
    except Exception:
        try:
            parts = []
            parts.append("<h2>Fallback Profiling Report</h2>")
            parts.append("<p>Profiler failed to export — presenting pandas summaries below.</p>")
            parts.append("<h3>Sample (first 10 rows)</h3>")
            parts.append(df_s.head(10).to_html(classes='table table-striped', index=False))
            parts.append("<h3>Column dtypes</h3>")
            parts.append(df_s.dtypes.to_frame("dtype").to_html(classes='table table-striped'))
            parts.append("<h3>Null counts</h3>")
            parts.append(df_s.isnull().sum().to_frame("null_count").to_html(classes='table table-striped'))
            parts.append("<h3>Unique counts</h3>")
            parts.append(df_s.nunique().to_frame("unique_count").to_html(classes='table table-striped'))
            parts.append("<h3>Descriptive stats (numeric)</h3>")
            try:
                parts.append(df_s.describe().T.to_html(classes='table table-striped'))
            except Exception:
                pass
            fallback_html = "<html><body style='font-family:Arial, Helvetica, sans-serif;padding:10px'>" + "".join(parts) + "</body></html>"
            fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
            with open(fd.name, "w", encoding="utf-8") as fh:
                fh.write(fallback_html)
            return fd.name
        except Exception:
            fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
            with open(fd.name, "w", encoding="utf-8") as fh:
                fh.write("<h3>Profiling generation failed and fallback generation also failed.</h3>")
            return fd.name

# -------------------------
# SHAP normalizer (for in-place fix)
# -------------------------
def _normalize_shap_values_to_2d(shap_values_raw, n_samples):
    """
    Attempt to convert many SHAP representations into a 2D array (n_samples, n_features).
    Returns tuple (shap_2d_or_none, meta)
    meta: {'original_shape': ..., 'selected_class_index': int or None}
    """
    meta = {"original_shape": None, "selected_class_index": None}
    if shap_values_raw is None:
        return None, meta

    # If list of arrays (common older shap outputs): stack into array (n_classes, n_samples, n_features)
    try:
        if isinstance(shap_values_raw, list):
            arrs = [np.asarray(a) for a in shap_values_raw]
            stacked = np.stack(arrs, axis=0)  # (n_classes, n_samples, n_features)
            meta["original_shape"] = stacked.shape
            # transpose to (n_samples, n_classes, n_features)
            stacked = np.transpose(stacked, (1, 0, 2))
            arr = stacked
        else:
            arr = np.asarray(shap_values_raw)
            meta["original_shape"] = arr.shape
            # common other shapes -> normalize to (n_samples, n_classes, n_features) if possible
            if arr.ndim == 3 and arr.shape[0] != n_samples and arr.shape[1] == n_samples:
                # (n_classes, n_samples, n_features)
                arr = np.transpose(arr, (1, 0, 2))
            # (n_samples, n_features, n_classes) -> (n_samples, n_classes, n_features)
            if arr.ndim == 3 and arr.shape[0] == n_samples and arr.shape[2] != arr.shape[1] and arr.shape[2] != n_samples:
                try:
                    arr = np.transpose(arr, (0, 2, 1))
                except Exception:
                    pass
    except Exception:
        return None, meta

    # If arr is 2D already
    if arr.ndim == 2:
        meta["original_shape"] = arr.shape
        return arr, meta

    # If 3D -> choose most important class by mean absolute
    if arr.ndim == 3:
        if arr.shape[0] == n_samples:
            n_classes = arr.shape[1]
            # mean abs per class across samples & features
            mean_abs_per_class = np.mean(np.abs(arr), axis=(0, 2))
            class_idx = int(np.argmax(mean_abs_per_class))
            meta["selected_class_index"] = class_idx
            meta["original_shape"] = arr.shape
            shap_2d = arr[:, class_idx, :]
            return shap_2d, meta
        else:
            # fallback: try to flatten last two axes into features
            try:
                flattened = arr.reshape(n_samples, -1)
                meta["original_shape"] = arr.shape
                return flattened, meta
            except Exception:
                return None, meta

    return None, meta

def _set_explainer_shap_values_inplace(explainer, shap_2d):
    """
    Try to set shap values on explainer in place. Different versions store them differently.
    """
    set_succeeded = False
    try:
        # preferred attribute
        if hasattr(explainer, "shap_values"):
            try:
                setattr(explainer, "shap_values", shap_2d)
                set_succeeded = True
            except Exception:
                pass
    except Exception:
        pass

    # try common internal attr names
    for attr in ("_shap_values", "shap_values_", "raw_shap_values"):
        try:
            if hasattr(explainer, attr):
                try:
                    setattr(explainer, attr, shap_2d)
                    set_succeeded = True
                except Exception:
                    pass
        except Exception:
            pass

    # Some internal caches may exist — try to clear them if present
    try:
        if hasattr(explainer, "_shap_values_df"):
            try:
                delattr = False
                try:
                    delattr = True
                    delattr  # just check
                except Exception:
                    pass
                # best-effort: attempt to remove if attribute present
                if hasattr(explainer, "_shap_values_df"):
                    try:
                        delattr(explainer, "_shap_values_df")
                    except Exception:
                        # alternative: set to None
                        try:
                            setattr(explainer, "_shap_values_df", None)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

    return set_succeeded

# -------------------------
# generate_explainer_dashboard_html_file (fixed approach)
# -------------------------
def generate_explainer_dashboard_html_file(fitted_clf, X_sample: pd.DataFrame, y_sample: np.ndarray, class_names: Optional[list] = None, tmp_dir: Optional[str] = None) -> str:
    """
    Build ExplainerDashboard and write to an HTML file. This function:
      - constructs ClassifierExplainer normally
      - inspects explainer.shap_values and normalizes it in-place to 2D if necessary
      - constructs ExplainerDashboard and exports to HTML using any available export API
    """
    if not EXPLAINERDASH_AVAILABLE or ClassifierExplainer is None:
        raise RuntimeError("explainerdashboard not installed in environment.")

    # prepare temp file
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
    fd_path = fd.name
    fd.close()

    # 1) create explainer normally
    expl = None
    try:
        expl = ClassifierExplainer(fitted_clf, X_sample, y_sample, labels=class_names if class_names else None, model_output="probability")
    except AssertionError as ae:
        # often raised if explainer tries to compute shap immediately and fails.
        # We'll try to proceed — sometimes the constructor still leaves an expl object; otherwise re-raise.
        if expl is None:
            raise
    except Exception:
        # bubble up
        raise

    # 2) if expl has shap_values attribute and it is 3D, normalize and set back in place
    try:
        sv = None
        if hasattr(expl, "shap_values"):
            sv = getattr(expl, "shap_values", None)
        # sometimes shap values are accessible through a method or property - attempt common alternatives
        if sv is None:
            for alt in ("raw_shap_values", "_shap_values", "shap_values_"):
                if hasattr(expl, alt):
                    sv = getattr(expl, alt)
                    if sv is not None:
                        break

        if sv is not None:
            # Attempt to normalize to 2D if necessary
            try:
                shap_2d, meta = _normalize_shap_values_to_2d(sv, n_samples=X_sample.shape[0])
                if shap_2d is not None:
                    ok = _set_explainer_shap_values_inplace(expl, shap_2d)
                    # If not ok, attempt to set via fallback: create attribute directly
                    if not ok:
                        try:
                            expl.shap_values = shap_2d
                        except Exception:
                            try:
                                setattr(expl, "_shap_values", shap_2d)
                            except Exception:
                                pass
            except Exception:
                # swallow - dashboard constructor will attempt to recompute or error later
                pass
        else:
            # shap not present yet — attempt to compute shap externally and set if possible
            if SHAP_AVAILABLE:
                try:
                    # try shap.Explainer then TreeExplainer fallback
                    try:
                        shap_expl = shap.Explainer(fitted_clf, X_sample)
                        shap_vals_obj = shap_expl(X_sample)
                        shap_values_raw = getattr(shap_vals_obj, "values", None)
                    except Exception:
                        shap_values_raw = None
                    if shap_values_raw is None:
                        try:
                            shap_values_raw = shap.TreeExplainer(fitted_clf).shap_values(X_sample)
                        except Exception:
                            shap_values_raw = None

                    if shap_values_raw is not None:
                        shap_2d, meta = _normalize_shap_values_to_2d(shap_values_raw, n_samples=X_sample.shape[0])
                        if shap_2d is not None:
                            _set_explainer_shap_values_inplace(expl, shap_2d)
                except Exception:
                    pass
    except Exception:
        # continue; we'll catch downstream errors when building dashboard / exporting
        pass

    # 3) Build dashboard object
    try:
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
    except Exception:
        # If this fails even after shap normalization, re-raise for caller to handle
        raise

    # 4) Export dashboard to HTML (try multiple methods)
    try:
        # Prefer to_html(path)
        try:
            dashboard.to_html(fd_path)
            return fd_path
        except TypeError:
            # method exists but signature mismatch
            pass
        except Exception:
            # maybe to_html doesn't take args; try to_file next
            pass

        try:
            # try to_file (older versions)
            dashboard.to_file(fd_path)
            return fd_path
        except Exception:
            # try to_html() returning string
            try:
                html_str = dashboard.to_html()
                if isinstance(html_str, str) and html_str.strip():
                    with open(fd_path, "w", encoding="utf-8") as fh:
                        fh.write(html_str)
                    return fd_path
                else:
                    raise RuntimeError("dashboard.to_html() returned empty or non-string.")
            except Exception as e:
                raise RuntimeError(f"Unable to export dashboard to HTML: {e}")
    except Exception:
        raise

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

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

# -------------------------
# Sidebar — upload
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
# TAB 1: DATA PROFILING (HTML-embedded)
# -------------------------
with tab1:
    st.header("📊 Data Profiling (HTML-embedded)")
    st.markdown("Generate a profiling report. The report is exported as a standalone HTML file and embedded into this app for reliable rendering.")
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
            st.dataframe(df.head(5))

        if st.button("🔍 Generate & Embed Profile Report (HTML)"):
            if ProfileReport is None:
                st.error("Profiling library not installed. Add 'ydata-profiling' or 'pandas-profiling' to your environment.")
            else:
                with st.spinner("Generating profile HTML (this may take time on large datasets)..."):
                    try:
                        tmp_dir = None
                        profile_html_path = generate_profile_html_file(df, tmp_dir=tmp_dir)
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
# TAB 2: MODEL DEVELOPMENT
# -------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Select target, features, algorithm, and train the model. Class balancing supported if imbalanced-learn is installed.")
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data in the sidebar first.")
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
# TAB 3: MODEL EVALUATION & EMBEDDED EXPLAINERDASHBOARD
# -------------------------
with tab3:
    st.header("🔍 Model Evaluation & Explainer")
    st.markdown("If available, ExplainerDashboard will be exported to a standalone HTML and embedded here. If not available or if dashboard construction fails, the app will show fallback diagnostics (no SHAP visuals).")

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

        st.subheader("📋 Classification Report")
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

        if st.session_state.explainer_selected_index is None:
            if len(X_test_df) > 0:
                st.session_state.explainer_selected_index = int(random.randrange(0, len(X_test_df)))
            else:
                st.session_state.explainer_selected_index = None

        if st.session_state.explainer_selected_index is None:
            st.info("Test set empty — cannot show explainer details.")
        else:
            idx = st.slider("Select index for explanation (used by the dashboard)", 0, max(0, len(X_test_df) - 1), value=int(st.session_state.explainer_selected_index), step=1)
            st.session_state.explainer_selected_index = int(idx)

            sel_col, pred_col = st.columns([1, 2])
            with sel_col:
                st.markdown("### Selected index")
                st.write(f"**{idx}**")
                st.markdown("**Guidance:** The embedded ExplainerDashboard's what-if & local explanation controls will reference this index when applicable.")
            with pred_col:
                st.markdown("### Prediction (selected index)")
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

            if EXPLAINERDASH_AVAILABLE:
                st.markdown("### ExplainerDashboard (HTML-embedded)")
                st.markdown("The app will export a standalone dashboard HTML and embed it here. This avoids many Streamlit/Plotly rendering mismatches.")
                try:
                    sample_n = min(1000, len(X_test_df))
                    sample_idx = np.random.choice(len(X_test_df), size=sample_n, replace=False)
                    X_dash = X_test_df.iloc[sample_idx]
                    y_dash = y_test_arr[sample_idx]

                    try:
                        fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                    except Exception:
                        fitted_clf = clf_pipeline

                    try:
                        dashboard_html_path = generate_explainer_dashboard_html_file(fitted_clf, X_dash, y_dash, class_names=class_names)
                        with open(dashboard_html_path, "r", encoding="utf-8") as fh:
                            dashboard_html = fh.read()
                        dashboard_html = dashboard_html.replace("titlefont", "title_font")
                        st.success("✅ ExplainerDashboard HTML generated and embedded below.")
                        components.html(dashboard_html, height=900, scrolling=True)
                        st.session_state.dashboard_html_path = dashboard_html_path
                    except Exception as dashboard_err:
                        st.error(f"ExplainerDashboard construction/export failed: {dashboard_err}")
                        st.info("Displaying fallback diagnostics (no SHAP visuals).")
                        st.text(traceback.format_exc())
                        # FALLBACK diagnostics...
                        try:
                            cm = confusion_matrix(y_test_arr, test_results["y_pred"])
                            st.markdown("**Confusion Matrix**")
                            st.dataframe(pd.DataFrame(cm))
                        except Exception:
                            st.info("Confusion matrix not available.")
                        try:
                            if hasattr(clf_pipeline, "predict_proba"):
                                proba = clf_pipeline.predict_proba(X_test_df)
                                if proba.shape[1] == 2:
                                    fpr, tpr, _ = roc_curve(y_test_arr, proba[:, 1])
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
                                    precision, recall, _ = precision_recall_curve(y_test_arr, proba[:, 1])
                                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                                    ax2.plot(recall, precision)
                                    ax2.set_xlabel("Recall")
                                    ax2.set_ylabel("Precision")
                                    ax2.set_title("Precision-Recall")
                                    plt.tight_layout()
                                    st.pyplot(fig2)
                                else:
                                    st.info("ROC/PR: multi-class or no binary probability available.")
                            else:
                                st.info("No predict_proba available for ROC/PR.")
                        except Exception:
                            st.info("ROC/PR generation failed.")
                        try:
                            clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                            if hasattr(clf, "feature_importances_"):
                                st.markdown("**Feature importances (top 20)**")
                                fi = np.array(clf.feature_importances_)
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
                except Exception as e:
                    st.error(f"Unexpected error while preparing/executing the dashboard: {e}")
                    st.text(traceback.format_exc())
            else:
                st.markdown("ExplainerDashboard not installed in this environment. Install `explainerdashboard` to enable a full interactive embedded dashboard.")
                st.info("Showing fallback diagnostics (no SHAP visuals).")
                # fallback diagnostics (same as above)
                try:
                    cm = confusion_matrix(y_test_arr, test_results["y_pred"])
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(pd.DataFrame(cm))
                except Exception:
                    st.info("Confusion matrix not available.")
                try:
                    if hasattr(clf_pipeline, "predict_proba"):
                        proba = clf_pipeline.predict_proba(X_test_df)
                        if proba.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y_test_arr, proba[:, 1])
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
                            precision, recall, _ = precision_recall_curve(y_test_arr, proba[:, 1])
                            fig2, ax2 = plt.subplots(figsize=(5, 3))
                            ax2.plot(recall, precision)
                            ax2.set_xlabel("Recall")
                            ax2.set_ylabel("Precision")
                            ax2.set_title("Precision-Recall")
                            plt.tight_layout()
                            st.pyplot(fig2)
                        else:
                            st.info("ROC/PR: multi-class or no binary probability available.")
                    else:
                        st.info("No predict_proba available for ROC/PR.")
                except Exception:
                    st.info("ROC/PR generation failed.")

            st.markdown("---")
            st.markdown("### Partial Dependence Plot (approx)")
            try:
                numeric_cols = X_test_df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.info("No numeric features available for a PDP.")
                else:
                    pdp_feat = st.selectbox("Choose numeric feature for PDP", numeric_cols, index=0)
                    base_row = X_test_df.iloc[[idx]].copy().reset_index(drop=True)
                    grid = np.linspace(X_test_df[pdp_feat].min(), X_test_df[pdp_feat].max(), num=40)
                    pdp_preds = make_pdp_values(clf_pipeline, base_row, pdp_feat, grid)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(grid, pdp_preds, linewidth=1.6, marker="o", markersize=3)
                    ax.set_xlabel(pdp_feat)
                    ax.set_ylabel("Predicted output / prob")
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception:
                st.info("PDP generation failed or not applicable.")

# -------------------------
# TAB 4: PREDICTIONS
# -------------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("Upload a file with the same features used in training and get predictions with a CSV download.")
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
