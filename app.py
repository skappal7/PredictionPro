# app.py
"""
Predictive Analytics Streamlit App
- Python 3.11 compatible
- Robust ydata/pandas-profiling fallback and sanitization
- Compact SHAP visualizations
- ExplainerDashboard embedding with user guidance
- Saves state across tabs
- Add to requirements.txt:
    streamlit
    pandas
    numpy
    scikit-learn
    imbalanced-learn
    ydata-profiling
    explainerdashboard
    shap
    matplotlib
    openpyxl
"""

import io
import tempfile
import traceback
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (5, 3),
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

# Optional profiling libs (ydata_profiling preferred)
try:
    from ydata_profiling import ProfileReport
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

# Imbalanced-learn
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    ImbPipeline = None
    SMOTE = RandomOverSampler = RandomUnderSampler = None

# SHAP
try:
    import shap
except Exception:
    shap = None

# ExplainerDashboard optional
try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    EXPLAINERDASH_AVAILABLE = True
except Exception:
    EXPLAINERDASH_AVAILABLE = False

# Streamlit page
st.set_page_config(page_title="Predictive Analytics App", layout="wide", page_icon="📊")
st.title("📊 Predictive Analytics — Explorer & Explainer")
st.markdown("Upload dataset → profile → train → explain (SHAP + ExplainerDashboard) → predict.")

# -----------------------
# Session state defaults
# -----------------------
defaults = {
    "data_uploaded": False,
    "profile_generated": False,
    "model_trained": False,
    "current_file": None,
    "data": None,
    "profile_html": None,
    "explainer_selected_index": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------
# Helpers
# -----------------------
def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    return None

def sanitize_dataframe_for_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is safe for ProfileReport:
    - Force DataFrame type
    - Drop all-NaN columns
    - Drop constant columns (nunique <= 1)
    - Reset index
    """
    df = pd.DataFrame(df)  # ensure DataFrame
    # drop all-NaN
    df = df.dropna(axis=1, how="all")
    # drop constant columns
    nunique = df.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    df = df.loc[:, keep_cols]
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def generate_profile_report(df: pd.DataFrame) -> str:
    """Return HTML string of profile report (ydata/pandas-profiling)."""
    if ProfileReport is None:
        return "<h3>Profiling library not available</h3><p>Install <code>ydata-profiling</code> or <code>pandas-profiling</code> in your environment.</p>"
    safe_df = sanitize_dataframe_for_profiling(df)
    profile = ProfileReport(safe_df, title="Dataset Profiling Report", explorative=True, minimal=False)
    return profile.to_html()

def safe_shap_explainer(fitted_model, background_data, feature_names=None):
    """Try shap.Explainer with fallbacks; returns explainer or None."""
    if shap is None:
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

def ensure_shap_array_for_idx(shap_values_obj, idx):
    """
    Make a best-effort to extract a 1D array of shap contributions for a single sample index.
    Works with: numpy arrays, list-of-arrays, shap.Explanation objects, tree explainer outputs.
    """
    try:
        # raw numpy
        if isinstance(shap_values_obj, np.ndarray):
            arr = shap_values_obj
            if arr.ndim == 2:
                return arr[idx].ravel()
            if arr.ndim == 3:
                # arr[class_index, sample_index, features] OR arr[sample_index, class_index, feature]
                # prefer sample-index-first if possible
                if arr.shape[0] > idx and arr.shape[1] != arr.shape[2]:
                    # treat as [sample, class, feature]
                    s = arr[idx]
                else:
                    # treat as [class, sample, feature] -> pick class with max mean abs
                    s = arr[:, idx, :]
                    s = s[np.argmax(np.abs(s).mean(axis=1))]
                return s.ravel()
            if arr.ndim == 1:
                return arr.ravel()
        # shap.Explanation or object with .values
        try:
            vals = getattr(shap_values_obj, "values", None)
        except Exception:
            vals = None
        if vals is None:
            vals = shap_values_obj
        vals = np.array(vals)
        if vals.ndim == 2:
            return vals[idx].ravel()
        if vals.ndim == 3:
            # [sample, classes, features] or [classes, samples, features]
            if vals.shape[0] > idx:
                sample_vals = vals[idx]
            else:
                sample_vals = vals[:, idx, :]
                sample_vals = sample_vals[np.argmax(np.abs(sample_vals).mean(axis=1))]
            # now sample_vals likely [classes, features] -> pick class with largest mean abs
            if sample_vals.ndim == 2:
                out_idx = int(np.argmax(np.abs(sample_vals).mean(axis=1)))
                return sample_vals[out_idx].ravel()
            return sample_vals.ravel()
        if vals.ndim == 1:
            return vals.ravel()
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

def st_small_plot(fig):
    """Helper to render matplotlib fig compactly."""
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------
# Sidebar - Upload
# -----------------------
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("Upload a CSV or Excel file to start.")
    uploaded_file = st.file_uploader("Choose dataset", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("Status")
    st.write(f"- Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"- Model trained: {st.session_state.model_trained}")

# Load data (preserve previous)
if uploaded_file:
    try:
        data = load_data(uploaded_file)
        st.session_state.data = data
        st.session_state.data_uploaded = True
        st.session_state.current_file = uploaded_file.name
        st.success("✅ Data uploaded.")
        st.write(f"**Shape:** {data.shape}")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(data.head(5))
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.session_state.data_uploaded = False
else:
    if st.session_state.data is None:
        st.session_state.data_uploaded = False

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Profiling",
    "🚀 Model Development",
    "🔍 Model Evaluation & Explainer",
    "📈 Predictions"
])

# -----------------------
# TAB 1: Data Profiling
# -----------------------
with tab1:
    st.header("📊 Data Profiling")
    st.markdown("Generate a comprehensive profiling report for your dataset. The profiler will automatically drop all-empty and constant columns to avoid errors.")
    if not st.session_state.data_uploaded:
        st.info("Upload a dataset from the sidebar to begin.")
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

        with st.expander("Quick data preview"):
            st.dataframe(df.head(5))

        if st.button("🔍 Generate Profile Report"):
            if ProfileReport is None:
                st.error("Profiling library not detected. Please add 'ydata-profiling' or 'pandas-profiling' to requirements.")
            else:
                with st.spinner("Generating profile..."):
                    try:
                        html = generate_profile_report(df)
                        st.session_state.profile_html = html
                        st.session_state.profile_generated = True
                        st.success("Profile generated.")
                    except Exception as e:
                        st.error(f"Profile generation failed: {e}")
                        st.text(traceback.format_exc())

        if st.session_state.profile_generated and st.session_state.profile_html:
            st.markdown("### Profiling Report")
            # large height because profile reports are long
            components.html(st.session_state.profile_html, height=800, scrolling=True)

# -----------------------
# TAB 2: Model Development
# -----------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Select target & features, pick a model, optionally balance classes, and train.")

    if not st.session_state.data_uploaded:
        st.info("Upload a dataset first.")
    else:
        data = st.session_state.data.copy()
        st.subheader("Target & Features")
        all_cols = list(data.columns)
        target_column = st.selectbox("Choose target column", all_cols, index=len(all_cols)-1 if len(all_cols)>0 else 0)
        available_features = [c for c in all_cols if c != target_column]

        feature_mode = st.radio("Feature selection:", ["Use all features", "Select features"], horizontal=True)
        if feature_mode == "Select features":
            feature_cols = st.multiselect("Pick features", available_features, default=available_features)
            if not feature_cols:
                st.warning("Select at least one feature.")
                st.stop()
        else:
            feature_cols = available_features

        st.info(f"Using {len(feature_cols)} features.")

        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()

        # Encode target if object-like or small unique set
        if y_raw.dtype == "object" or y_raw.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = [str(c) for c in le_target.classes_]
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = [str(c) for c in np.unique(y)]

        st.subheader("Target distribution")
        a1, a2 = st.columns(2)
        with a1:
            st.write(class_counts(y))
        with a2:
            try:
                dist_df = pd.DataFrame(list(class_counts(y).items()), columns=['class', 'count']).set_index('class')
                st.bar_chart(dist_df)
            except Exception:
                pass

        st.subheader("Model selection")
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        model_choice = st.selectbox("Algorithm", list(model_map.keys()), index=3)
        model = model_map[model_choice]

        with st.expander("Hyperparameters"):
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

        st.subheader("Class balancing")
        balance = st.selectbox("Balancing", ["None", "SMOTE", "Random Oversample", "Random Undersample"])
        sampler = None
        if balance == "SMOTE" and SMOTE is not None:
            sampler = SMOTE(random_state=42)
        elif balance == "Random Oversample" and RandomOverSampler is not None:
            sampler = RandomOverSampler(random_state=42)
        elif balance == "Random Undersample" and RandomUnderSampler is not None:
            sampler = RandomUnderSampler(random_state=42)
        elif balance != "None" and ImbPipeline is None:
            st.warning("imbalanced-learn not installed; balancing disabled.")

        st.subheader("Preprocessing & Train/Test split")
        test_pct = st.slider("Test size (%)", 10, 50, 20)

        numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
        categorical_feats = X.select_dtypes(exclude=np.number).columns.tolist()
        numeric_transformer = Pipeline([("scaler", StandardScaler())])

        # OneHotEncoder compatibility across sklearn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_transformer = Pipeline([("onehot", ohe)])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
        ], remainder="drop")

        # build pipeline (use imblearn pipeline if available and sampler requested)
        if sampler is None or ImbPipeline is None:
            clf_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        else:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("classifier", model)])

        st.markdown("---")
        if st.button("🚀 Train Model", use_container_width=True):
            unique = np.unique(y)
            if len(unique) < 2:
                st.error("At least 2 classes/unique values required to train.")
            else:
                with st.spinner("Training..."):
                    try:
                        stratify_arg = y if len(unique) >= 2 else None
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100.0, random_state=42, stratify=stratify_arg)
                        clf_pipeline.fit(X_train, y_train)
                        y_pred = clf_pipeline.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        st.session_state.model_trained = True
                        st.session_state.trained_clf = clf_pipeline
                        st.session_state.trained_le_target = le_target
                        st.session_state.trained_feature_cols = feature_cols
                        st.session_state.trained_class_names = class_names
                        st.session_state.test_results = {
                            "y_test": np.array(y_test),
                            "y_pred": np.array(y_pred),
                            "accuracy": float(acc)
                        }
                        st.session_state.trained_X_test = X_test.reset_index(drop=True)
                        st.session_state.trained_y_test = np.array(y_test)

                        st.success(f"✅ Model trained. Accuracy: {acc:.3f}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.text(traceback.format_exc())

# -----------------------
# TAB 3: Model Evaluation & Explainer
# -----------------------
with tab3:
    st.header("🔍 Model Evaluation & Explainer")
    st.markdown("Performance metrics, local/global SHAP contributions, PDP (approx), and a full ExplainerDashboard if available.")

    if not st.session_state.data_uploaded:
        st.info("Upload data first.")
    elif not st.session_state.model_trained:
        st.info("Train a model first.")
    else:
        clf_pipeline = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results
        X_test_df = st.session_state.trained_X_test.copy()
        y_test_arr = st.session_state.trained_y_test

        # Performance summary
        st.subheader("Performance summary")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
        with p2:
            st.metric("Test samples", len(y_test_arr))
        with p3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        st.subheader("Classification report")
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

        # Auto-select random index for explainer on first view
        if st.session_state.explainer_selected_index is None:
            if len(X_test_df) > 0:
                st.session_state.explainer_selected_index = int(random.randrange(0, len(X_test_df)))
            else:
                st.session_state.explainer_selected_index = None

        if st.session_state.explainer_selected_index is None:
            st.info("Test set is empty — cannot show explainer.")
        else:
            idx = st.slider("Select index for explanation", 0, max(0, len(X_test_df)-1),
                            value=int(st.session_state.explainer_selected_index), step=1)
            st.session_state.explainer_selected_index = int(idx)

            sel_col, pred_col = st.columns([1, 2])
            with sel_col:
                st.markdown("### Selected index")
                st.write(f"**{idx}**")
                st.markdown("**Guidance:** Choose an index to view local explanation, prediction & probability breakdown.")
            with pred_col:
                st.markdown("### Prediction & probabilities")
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
            st.markdown("### Contributions (SHAP-style)")
            st.markdown("Local feature contributions shown below. If SHAP is not supported for your model type, a difference-from-mean proxy will be used.")
            try:
                # extract preprocessor & fitted classifier when pipeline used
                preprocessor = None
                fitted_clf = clf_pipeline
                if isinstance(clf_pipeline, Pipeline) or (ImbPipeline is not None and isinstance(clf_pipeline, ImbPipeline)):
                    preprocessor = getattr(clf_pipeline, "named_steps", {}).get("preprocessor", None)
                    fitted_clf = getattr(clf_pipeline, "named_steps", {}).get("classifier", clf_pipeline)

                # create background (limited size)
                background_rows = X_test_df.sample(min(len(X_test_df), 200), replace=False)

                try:
                    if preprocessor is not None:
                        background_trans = preprocessor.transform(background_rows)
                    else:
                        background_trans = background_rows.values
                except Exception:
                    background_trans = background_rows.values

                # feature names extraction (preprocessor may provide get_feature_names_out)
                try:
                    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                        feat_names = preprocessor.get_feature_names_out()
                    else:
                        feat_names = X_test_df.columns.tolist()
                except Exception:
                    feat_names = X_test_df.columns.tolist()

                shap_vals_vector = None
                shap_explainer = None
                if shap is not None:
                    shap_explainer = safe_shap_explainer(fitted_clf, background_trans, feature_names=feat_names)
                    if shap_explainer is not None:
                        try:
                            if preprocessor is not None:
                                single_proc = preprocessor.transform(X_test_df.iloc[[idx]])
                            else:
                                single_proc = X_test_df.iloc[[idx]].values
                        except Exception:
                            single_proc = X_test_df.iloc[[idx]].values
                        try:
                            shap_res = shap_explainer(single_proc)
                            shap_vals_vector = ensure_shap_array_for_idx(shap_res, 0)
                        except Exception:
                            shap_vals_vector = None

                # Try TreeExplainer fallback if shap_vals_vector None
                if shap_vals_vector is None and shap is not None:
                    try:
                        tree_expl = shap.TreeExplainer(fitted_clf)
                        shap_vals_raw = tree_expl.shap_values(X_test_df if preprocessor is None else preprocessor.transform(X_test_df))
                        shap_vals_vector = ensure_shap_array_for_idx(shap_vals_raw, idx)
                    except Exception:
                        shap_vals_vector = None

                if shap_vals_vector is None:
                    st.info("SHAP not available; showing difference-from-mean as proxy contributions.")
                    base = X_test_df.mean()
                    diff = X_test_df.iloc[idx] - base
                    contrib_df = pd.DataFrame({
                        "feature": X_test_df.columns.tolist(),
                        "contribution": diff.values
                    })
                else:
                    contrib = np.asarray(shap_vals_vector).ravel()
                    # normalize to features length
                    if len(contrib) == len(feat_names):
                        feat_list = list(feat_names)
                    elif len(contrib) == len(X_test_df.columns):
                        feat_list = list(X_test_df.columns)
                    else:
                        feat_list = list(X_test_df.columns)
                        if len(contrib) > len(feat_list):
                            contrib = contrib[:len(feat_list)]
                        else:
                            pad_len = len(feat_list) - len(contrib)
                            contrib = np.concatenate([contrib, np.zeros(pad_len)])
                    contrib_df = pd.DataFrame({"feature": feat_list, "contribution": contrib})

                contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
                contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

                st.write("Top contributions (absolute impact)")
                st.dataframe(contrib_df[["feature", "contribution"]].head(20), use_container_width=True)

                # compact bar plot
                topn = min(10, len(contrib_df))
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(contrib_df.head(topn)["feature"][::-1], contrib_df.head(topn)["contribution"][::-1])
                ax.set_xlabel("Contribution (signed)", fontsize=10)
                ax.set_ylabel("Feature", fontsize=10)
                ax.tick_params(axis='x', labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
                plt.tight_layout()
                st_small_plot(fig)

            except Exception as e:
                st.error(f"Contributions failed: {e}")
                st.text(traceback.format_exc())

            st.markdown("---")
            st.markdown("### Partial Dependence Plot (approx)")
            st.markdown("PDP is approximated by varying a single numeric feature while keeping other features fixed for the selected instance.")
            try:
                numeric_cols = X_test_df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.info("No numeric features for PDP.")
                else:
                    pdp_feat = st.selectbox("Choose numeric feature for PDP", numeric_cols, index=0)
                    base_row = X_test_df.iloc[[idx]].copy().reset_index(drop=True)
                    grid = np.linspace(X_test_df[pdp_feat].min(), X_test_df[pdp_feat].max(), num=40)
                    pdp_preds = make_pdp_values(clf_pipeline, base_row, pdp_feat, grid)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(grid, pdp_preds, linewidth=1.6, marker='o', markersize=3)
                    ax.set_xlabel(pdp_feat, fontsize=10)
                    ax.set_ylabel("Predicted output / prob", fontsize=10)
                    ax.tick_params(axis='x', labelsize=9)
                    ax.tick_params(axis='y', labelsize=9)
                    plt.tight_layout()
                    st_small_plot(fig)
            except Exception as e:
                st.error(f"PDP failed: {e}")
                st.text(traceback.format_exc())

            # ExplainerDashboard embed
            st.markdown("---")
            st.markdown("### Full ExplainerDashboard")
            st.markdown(
                "Below: interactive ExplainerDashboard (if installed). It provides richer visuals: feature importances, "
                "SHAP dependence/interactions, what-if analysis, ROC/PR curves, confusion matrices, and more. "
                "If the dashboard fails to load here, try running it locally with the same environment versions."
            )
            if EXPLAINERDASH_AVAILABLE:
                try:
                    # sample to limit size
                    sample_n = min(1000, len(X_test_df))
                    sample_idx = np.random.choice(len(X_test_df), size=sample_n, replace=False)
                    X_dash = X_test_df.iloc[sample_idx]
                    y_dash = y_test_arr[sample_idx]

                    # If pipeline used, pass clf_pipeline (it can handle transforms), else pass classifier
                    explainer_input_model = clf_pipeline

                    # Build explainer (ClassifierExplainer expects raw X and y)
                    expl = ClassifierExplainer(
                        explainer_input_model,
                        X_dash,
                        y_dash,
                        labels=class_names if class_names is not None else None,
                        model_output="probability",
                    )

                    dashboard = ExplainerDashboard(
                        expl,
                        title="Model Explainer — Full Analysis",
                        whatif=True,
                        importances=True,
                        model_summary=True,
                        contributions=True,
                        shap_dependence=True,
                        shap_interaction=True,
                        decision_trees=isinstance(getattr(clf_pipeline, "named_steps", {}).get("classifier", clf_pipeline), DecisionTreeClassifier),
                        hide_poweredby=True,
                        fluid=True,
                    )

                    try:
                        html_str = dashboard.to_html()
                    except TypeError:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            html_path = Path(tmpdir) / "dashboard.html"
                            dashboard.to_html(filename=str(html_path))
                            html_str = html_path.read_text(encoding="utf-8")

                    components.html(html_str, height=900, scrolling=True)
                except Exception as e:
                    st.error(f"ExplainerDashboard failed to build or render: {e}")
                    st.info("The panels above still contain the main explanation content.")
                    st.text(traceback.format_exc())
            else:
                st.info("ExplainerDashboard not installed. Add 'explainerdashboard' to requirements to enable full interactive dashboard here.")

# -----------------------
# TAB 4: Predictions
# -----------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("Upload new data with the same training features to get predictions and download a CSV.")
    if not st.session_state.data_uploaded:
        st.info("Upload data first.")
    elif not st.session_state.model_trained:
        st.info("Train a model first.")
    else:
        st.success("Model ready for predictions.")
        clf_pipeline = st.session_state.trained_clf
        trained_feature_cols = st.session_state.trained_feature_cols
        trained_le_target = st.session_state.trained_le_target

        new_file = st.file_uploader("Upload CSV/XLSX with features", type=["csv", "xlsx"], key="pred_file")
        if new_file:
            try:
                if new_file.name.lower().endswith(".csv"):
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
                                csv_buf = io.StringIO()
                                results.to_csv(csv_buf, index=False)
                                st.download_button("📥 Download Predictions CSV", data=csv_buf.getvalue(), file_name=f"predictions_{new_file.name.split('.')[0]}.csv")
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                st.text(traceback.format_exc())
            except Exception as e:
                st.error(f"Failed to load prediction file: {e}")
                st.text(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | © 2025 CE Innovation Lab")
