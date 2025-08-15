# app_fixed.py
# Predictive Analytics App with Comprehensive ExplainerDashboard
# Cleaned & fixed version

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

# ExplainerDashboard (optional)
try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    EXPLAINERDASH_AVAILABLE = True
except Exception:
    EXPLAINERDASH_AVAILABLE = False

# -----------------------------
# Streamlit page config & session state defaults
# -----------------------------
st.set_page_config(page_title="Predictive Analytics App", layout="wide", page_icon="📊")

for key, val in {
    'data_uploaded': False,
    'profile_generated': False,
    'model_trained': False,
    'current_file': None,
    'data': None,
    'profile_html': None,
    'explainer_selected_index': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------
# Helper functions
# -----------------------------

def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    return None

@st.cache_data
def generate_profile_report(data):
    """Generate YData Profiling report"""
    if ProfileReport is None:
        return "<h1>Data Profiling library not available</h1><p>Please install ydata-profiling or pandas-profiling</p>"

    profile = ProfileReport(
        data,
        title="Dataset Profiling Report",
        explorative=True,
        minimal=False
    )
    return profile.to_html()


def safe_shap_explainer(fitted_model, background_data, feature_names=None):
    try:
        # shap.Explainer has different signatures across versions; guard it
        if feature_names is not None:
            return shap.Explainer(fitted_model, background_data, feature_names=feature_names)
        return shap.Explainer(fitted_model, background_data)
    except Exception:
        try:
            return shap.Explainer(fitted_model, background_data)
        except Exception:
            return None


def ensure_shap_array_for_idx(shap_values_obj, idx):
    try:
        # Handle numpy arrays directly
        if isinstance(shap_values_obj, np.ndarray):
            arr = shap_values_obj
            if arr.ndim == 2:
                return arr[idx].ravel()
            if arr.ndim == 3:
                if arr.shape[0] > idx:
                    s = arr[idx]
                    if s.ndim == 2:
                        out_idx = int(np.argmax(np.abs(s).mean(axis=1)))
                        return s[out_idx].ravel()
                    return s.ravel()
                else:
                    return arr[0].ravel()
            if arr.ndim == 1:
                return arr.ravel()
        # Try to access .values for shap.Explanation objects
        try:
            vals = shap_values_obj.values
        except Exception:
            vals = shap_values_obj
        vals = np.array(vals)
        if vals.ndim == 2:
            return vals[idx].ravel()
        if vals.ndim == 3:
            sample_vals = vals[idx]
            out_idx = int(np.argmax(np.abs(sample_vals).mean(axis=1)))
            return sample_vals[out_idx].ravel()
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
                # return top class probability as the PDP proxy
                preds.append(float(np.max(p)))
            else:
                p = clf_pipeline.predict(temp)
                preds.append(float(np.ravel(p)[0]))
        except Exception:
            preds.append(np.nan)
    return preds


def st_plt(fig):
    st.pyplot(fig)


# -----------------------------
# Main UI
# -----------------------------
st.title("📊 Predictive Analytics Application")
st.markdown("A comprehensive platform for data analysis, model development, evaluation, and predictions.")

# Sidebar - upload
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("Upload your dataset to begin the analysis workflow.")
    uploaded_file = st.file_uploader("Choose your dataset", type=["csv", "xlsx"], help="Upload a CSV or Excel file containing your dataset")
    st.markdown("---")
    st.write("Status:")
    st.write(f"Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"Model trained: {st.session_state.model_trained}")

if uploaded_file:
    try:
        data = load_data(uploaded_file)
        st.session_state.data = data
        st.session_state.data_uploaded = True
        st.session_state.current_file = uploaded_file.name
        st.success("✅ Data uploaded successfully!")
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {len(data.columns)}")
        with st.expander("Quick Preview"):
            st.dataframe(data.head(3))
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        st.session_state.data_uploaded = False
else:
    # keep previous data if present
    if st.session_state.data is None:
        st.session_state.data_uploaded = False

# Tabs
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
    st.markdown("""
    **What is Data Profiling?**
    Data profiling provides comprehensive insights into your dataset including:
    - Data types and missing values analysis
    - Statistical summaries and distributions
    - Correlation analysis between variables
    - Data quality assessment and anomaly detection
    """)
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
                st.error("Data profiling library not installed. Please add 'ydata-profiling' to requirements.txt")
            else:
                with st.spinner("Generating profiling report..."):
                    try:
                        html = generate_profile_report(data)
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
    st.markdown("""
    Select target, features, algorithm, balance classes, and train the model.
    """)
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset first.")
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

        numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
        categorical_feats = X.select_dtypes(exclude=np.number).columns.tolist()
        numeric_transformer = Pipeline([("scaler", StandardScaler())])

        # OneHotEncoder's sparse parameter name changed across sklearn versions; handle both
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_transformer = Pipeline([("onehot", ohe)])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
        ], remainder='drop')

        if sampler is None:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        else:
            clf_pipeline = ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("classifier", model)])

        st.markdown("---")
        if st.button("🚀 Train Model", use_container_width=True):
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
                        st.session_state.trained_X_test = X_test.reset_index(drop=True)
                        st.session_state.trained_y_test = np.array(y_test)
                        st.success(f"✅ Trained. Accuracy: {acc:.3f}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.text(traceback.format_exc())

# -----------------------------
# TAB 3: Model Evaluation (Explainer + SHAP + PDP + ExplainerDashboard)
# -----------------------------
with tab3:
    st.header("🔍 Model Evaluation & Explainer")
    st.markdown("""
    Model performance metrics, SHAP-style contributions, PDP, and a full ExplainerDashboard (if available).
    """)
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
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
        with p2:
            st.metric("Test samples", len(y_test_arr))
        with p3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        st.subheader("📋 Detailed Performance Report")
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

            sel_col, pred_col = st.columns([1, 2])
            with sel_col:
                st.markdown("### Selected index")
                st.write(f"**{idx}**")
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
            # Contributions (SHAP-style)
            st.markdown("### Contributions (SHAP-style)")
            try:
                try:
                    preprocessor = clf_pipeline.named_steps.get("preprocessor", None)
                    fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                except Exception:
                    preprocessor = None
                    fitted_clf = clf_pipeline

                background_rows = X_test_df.sample(min(len(X_test_df), 200), replace=False)

                try:
                    if preprocessor is not None:
                        background_trans = preprocessor.transform(background_rows)
                    else:
                        background_trans = background_rows.values
                except Exception:
                    background_trans = background_rows.values

                try:
                    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                        feat_names = preprocessor.get_feature_names_out()
                    else:
                        feat_names = X_test_df.columns.tolist()
                except Exception:
                    feat_names = X_test_df.columns.tolist()

                shap_explainer = safe_shap_explainer(fitted_clf, background_trans, feature_names=feat_names)
                shap_vals_vector = None
                if shap_explainer is not None:
                    try:
                        if preprocessor is not None:
                            single_proc = preprocessor.transform(X_test_df.iloc[[idx]])
                        else:
                            single_proc = X_test_df.iloc[[idx]].values
                    except Exception:
                        single_proc = X_test_df.iloc[[idx]].values
                    try:
                        shap_vals = shap_explainer(single_proc)
                        shap_vals_vector = ensure_shap_array_for_idx(shap_vals, 0)
                    except Exception:
                        shap_vals_vector = None

                if shap_vals_vector is None:
                    try:
                        tree_expl = shap.TreeExplainer(fitted_clf)
                        shap_vals_raw = tree_expl.shap_values(X_test_df)
                        shap_vals_vector = ensure_shap_array_for_idx(shap_vals_raw, idx)
                    except Exception:
                        shap_vals_vector = None

                if shap_vals_vector is None:
                    st.info("SHAP values not available for this pipeline/model. Showing difference-from-mean as proxy.")
                    base = X_test_df.mean()
                    diff = X_test_df.iloc[idx] - base
                    contrib_df = pd.DataFrame({
                        "feature": X_test_df.columns.tolist(),
                        "contribution": diff.values
                    })
                else:
                    contrib = np.asarray(shap_vals_vector).ravel()
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
                    contrib_df = pd.DataFrame({
                        "feature": feat_list,
                        "contribution": contrib
                    })

                contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
                contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

                st.write("Top contributions (absolute impact)")
                st.dataframe(contrib_df[["feature", "contribution"]].head(20), use_container_width=True)

                topn = min(10, len(contrib_df))
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(contrib_df.head(topn)["feature"][::-1], contrib_df.head(topn)["contribution"][::-1])
                ax.set_xlabel("Contribution (signed)", fontsize=10)
                ax.set_ylabel("Feature", fontsize=10)
                ax.tick_params(axis='x', labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
                plt.tight_layout()
                st_plt(fig)

            except Exception as e:
                st.error(f"Contributions failed: {e}")
                st.text(traceback.format_exc())

            st.markdown("---")
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
                    ax.tick_params(axis='x', labelsize=9)
                    ax.tick_params(axis='y', labelsize=9)
                    plt.tight_layout()
                    st_plt(fig)
            except Exception as e:
                st.error(f"PDP failed: {e}")
                st.text(traceback.format_exc())

            # Comprehensive ExplainerDashboard embed
            st.markdown("---")
            st.markdown("### Full ExplainerDashboard (from explainerdashboard)")
            if EXPLAINERDASH_AVAILABLE:
                try:
                    # Build sample for dashboard to limit size
                    sample_n = min(1000, len(X_test_df))
                    sample_idx = np.random.choice(len(X_test_df), size=sample_n, replace=False)
                    X_dash = X_test_df.iloc[sample_idx]
                    y_dash = y_test_arr[sample_idx]

                    # Try to pass classifier (not pipeline) and raw X_dash (ExplainerDashboard handles transforms differently across versions)
                    fitted_clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
                    # Create explainer
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
                    st.info("The custom panels above still provide prediction, contributions, and PDP information.")
                    st.text(traceback.format_exc())
            else:
                st.info("ExplainerDashboard not installed. Add 'explainerdashboard' to requirements to enable full dashboard embed.")

# -----------------------------
# TAB 4: Predictions
# -----------------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("""
    Upload new data (CSV/XLSX) with the same features used in training to get predictions and download results.
    """)
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
