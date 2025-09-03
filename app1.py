# app.py
"""
Predictive Analytics Streamlit App — NO ExplainerDashboard
- Full app: upload -> profile -> train -> evaluate -> predict
- ExplainerDashboard removed; replaced with polished fallback visuals + plain-language commentary
- Charts are compact and arranged with commentary next to each chart
- NEW: Added heavy CSV to Parquet conversion for faster processing
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
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# NEW: Import for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# sklearn / imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMB_AVAILABLE = True
except Exception:
    ImbPipeline = Pipeline
    SMOTE = RandomOverSampler = RandomUnderSampler = None
    IMB_AVAILABLE = False

# Optional profiling
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

# Page config
st.set_page_config(page_title="Predictive Analytics (No ExplainerDashboard)", layout="wide", page_icon="📊")
st.title("📊 Predictive Analytics Wizard")
st.markdown("Predictive modeling simplified: build the models, predict the outcomes and explain the results")

# Session defaults
_defaults = {
    "data_uploaded": False,
    "profile_generated": False,
    "model_trained": False,
    "current_file": None,
    "data": None,
    "profile_html_path": None,
    "trained_clf": None,
    "trained_feature_cols": None,
    "trained_le_target": None,
    "test_results": None,
    "trained_X_test": None,
    "trained_y_test": None,
    "explainer_selected_index": None,
    # NEW: Parquet-related session states
    "parquet_converted": False,
    "parquet_path": None,
    "original_csv_size": None,
    "parquet_size": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Helper utilities
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

# NEW: Heavy CSV to Parquet conversion functions
def get_file_size_mb(file_obj):
    """Get file size in MB"""
    try:
        current_pos = file_obj.tell()
        file_obj.seek(0, 2)  # Go to end of file
        size = file_obj.tell()
        file_obj.seek(current_pos)  # Reset to original position
        return size / (1024 * 1024)  # Convert to MB
    except:
        return 0

def convert_csv_to_parquet(uploaded_file, chunk_size=50000):
    """Convert heavy CSV to Parquet format with chunked processing"""
    if not PARQUET_AVAILABLE:
        st.error("PyArrow is required for Parquet conversion. Please install: pip install pyarrow")
        return None, None
    
    try:
        # Get original file size
        original_size = get_file_size_mb(uploaded_file)
        
        # Create temporary parquet file
        temp_dir = tempfile.mkdtemp()
        parquet_path = os.path.join(temp_dir, f"{safe_filename_base(uploaded_file.name)}.parquet")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read CSV in chunks and write to Parquet
        first_chunk = True
        total_rows = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Converting CSV to Parquet format..."):
            # First, count total rows for progress tracking
            uploaded_file.seek(0)
            total_lines = sum(1 for _ in uploaded_file) - 1  # Subtract header
            uploaded_file.seek(0)
            
            chunk_count = 0
            for chunk_df in pd.read_csv(uploaded_file, chunksize=chunk_size):
                chunk_count += 1
                total_rows += len(chunk_df)
                
                # Update progress
                progress = min(total_rows / total_lines, 1.0) if total_lines > 0 else 1.0
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {chunk_count}: {total_rows:,} rows processed")
                
                # Convert chunk to parquet
                if first_chunk:
                    # Create new parquet file
                    table = pa.Table.from_pandas(chunk_df)
                    pq.write_table(table, parquet_path)
                    first_chunk = False
                else:
                    # Append to existing parquet file
                    table = pa.Table.from_pandas(chunk_df)
                    # Read existing data
                    existing_table = pq.read_table(parquet_path)
                    # Concatenate tables
                    combined_table = pa.concat_tables([existing_table, table])
                    pq.write_table(combined_table, parquet_path)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Conversion complete! Processed {total_rows:,} rows")
        
        # Get parquet file size
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
        
        return parquet_path, {"original_size": original_size, "parquet_size": parquet_size, "rows": total_rows}
        
    except Exception as e:
        st.error(f"Failed to convert CSV to Parquet: {e}")
        return None, None

def load_parquet_file(parquet_path):
    """Load data from Parquet file"""
    try:
        if PARQUET_AVAILABLE:
            return pd.read_parquet(parquet_path)
        else:
            st.error("PyArrow is required to load Parquet files")
            return None
    except Exception as e:
        st.error(f"Failed to load Parquet file: {e}")
        return None

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
    # drop nested
    drop_cols = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(check_n).tolist()
        for v in sample_vals:
            if isinstance(v, (list, dict, set, tuple, np.ndarray, pd.Series)):
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
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        with open(fd.name, "w", encoding="utf-8") as fh:
            fh.write("<h3>Profiler failed to export.</h3>")
        return fd.name

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
# Sidebar upload
# -------------------------
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
    
    # NEW: Parquet conversion section
    if uploaded_file and uploaded_file.name.lower().endswith('.csv'):
        file_size = get_file_size_mb(uploaded_file)
        if file_size > 10:  # Show option for files larger than 10MB
            st.markdown("---")
            st.subheader("🚀 Heavy CSV Optimization")
            st.info(f"File size: {file_size:.1f} MB - Consider converting to Parquet for faster processing!")
            
            if PARQUET_AVAILABLE:
                if st.button("⚡ Convert to Parquet", help="Convert large CSV to Parquet format for 5-10x faster processing"):
                    parquet_path, conversion_info = convert_csv_to_parquet(uploaded_file)
                    if parquet_path and conversion_info:
                        st.session_state.parquet_converted = True
                        st.session_state.parquet_path = parquet_path
                        st.session_state.original_csv_size = conversion_info["original_size"]
                        st.session_state.parquet_size = conversion_info["parquet_size"]
                        
                        # Show conversion results
                        reduction = ((conversion_info["original_size"] - conversion_info["parquet_size"]) / conversion_info["original_size"]) * 100
                        st.success(f"✅ Converted successfully!")
                        st.metric("Size Reduction", f"{reduction:.1f}%")
                        st.metric("Original Size", f"{conversion_info['original_size']:.1f} MB")
                        st.metric("Parquet Size", f"{conversion_info['parquet_size']:.1f} MB")
                        
                # Option to use parquet if available
                if st.session_state.parquet_converted and st.session_state.parquet_path:
                    use_parquet = st.checkbox("🏎️ Use Parquet (Faster)", value=True, 
                                            help="Use the converted Parquet file for faster processing")
                    if use_parquet and st.button("📥 Load Parquet Data"):
                        df = load_parquet_file(st.session_state.parquet_path)
                        if df is not None:
                            st.session_state.data = df
                            st.session_state.data_uploaded = True
                            st.session_state.current_file = uploaded_file.name + " (Parquet)"
                            st.success("✅ Parquet data loaded successfully!")
            else:
                st.warning("Install PyArrow for Parquet conversion: `pip install pyarrow`")
    
    st.markdown("---")
    st.write("Status:")
    st.write(f"- Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"- Model trained: {st.session_state.model_trained}")
    # NEW: Show parquet status
    if st.session_state.parquet_converted:
        st.write("- Parquet converted: ✅")

if uploaded_file:
    # Only process if not using parquet or parquet not available
    if not (st.session_state.parquet_converted and st.session_state.get('use_parquet', False)):
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Profiling", "🚀 Model Development", "🔍 Model Evaluation & Explain", "📈 Predictions"])

# -------------------------
# TAB 1: DATA PROFILING
# -------------------------
with tab1:
    st.header("📊 Data Profiling") 
    st.markdown("Generate a profiling report and embed it here.")
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

        # NEW: Show file format info
        if st.session_state.parquet_converted and "Parquet" in st.session_state.current_file:
            st.info(f"📦 Using optimized Parquet format - Original CSV: {st.session_state.original_csv_size:.1f}MB → Parquet: {st.session_state.parquet_size:.1f}MB")

        with st.expander("Quick Preview"):
            st.dataframe(df.head(5))

        if st.button("🔍 Generate & Embed Profile Report (HTML)"):
            if ProfileReport is None:
                st.error("Profiling library not installed. Add 'ydata-profiling' or 'pandas-profiling' to your environment.")
            else:
                with st.spinner("Generating profile HTML..."):
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
# TAB 2: MODEL DEVELOPMENT
# -------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("Select target, features, algorithm, and train the model.")
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
# TAB 3: MODEL EVALUATION & FALLBACK VISUAL EXPLAINER
# -------------------------
with tab3:
    st.header("🔍 Model Evaluation & Visual Explain (fallback)")
    st.markdown("ExplainerDashboard removed. Below are polished visual diagnostics and plain-language commentary to help anyone understand model behavior.")

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

        # Summary metrics
        st.subheader("📊 Quick Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
        with c2:
            st.metric("Test samples", len(y_test_arr))
        with c3:
            st.metric("Features", len(st.session_state.trained_feature_cols))

        # Classification report table
        st.subheader("📋 Classification Report (numbers)")
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

        st.markdown("---")

        # Confusion matrix with explanation
        st.subheader("🔁 Confusion Matrix — what it means")
        try:
            cm = confusion_matrix(y_test_arr, test_results["y_pred"])
            cm_norm = cm.astype(float)
            # normalize rows to show proportions
            with np.errstate(all="ignore"):
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                cm_prop = np.divide(cm_norm, row_sums, where=row_sums != 0)
            # Plot heatmap (compact)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            im = ax_cm.imshow(cm_prop, interpolation="nearest", aspect="auto")
            ax_cm.set_title("Confusion matrix (proportions)")
            ticks = np.arange(len(cm))
            ax_cm.set_xticks(ticks)
            ax_cm.set_yticks(ticks)
            # labels if available
            try:
                xticklabels = [str(c) for c in class_names]
                yticklabels = [str(c) for c in class_names]
            except Exception:
                xticklabels = [str(i) for i in ticks]
                yticklabels = [str(i) for i in ticks]
            ax_cm.set_xticklabels(xticklabels, rotation=45, ha="right")
            ax_cm.set_yticklabels(yticklabels)
            # annotate values
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm_prop[i, j] if not np.isnan(cm_prop[i, j]) else 0.0
                    ax_cm.text(j, i, f"{val:.2f}\n({int(cm[i,j])})", ha="center", va="center", fontsize=8, color="black")
            plt.tight_layout()

            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.pyplot(fig_cm)
            with col2:
                # Plain-language explanation
                st.markdown("**What does this mean?**")
                st.write("Each row here is the actual class, and each column is the model's guess. Values show two things:")
                st.write("- Top: proportion of the actual class that the model predicted as that column (decimal).")
                st.write("- Bottom: raw count of samples.")
                st.write("If most of the shading is on the diagonal (top-left to bottom-right), the model is doing well.")
                st.write("Off-diagonal cells show where the model gets confused. For example, if many actual A → predicted B, that indicates those two look similar to the model.")
        except Exception:
            st.info("Confusion matrix not available.")
            st.text(traceback.format_exc())

        st.markdown("---")

        # ROC and Precision-Recall
        st.subheader("📈 ROC & Precision–Recall (how confident the model is)")
        try:
            clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
            proba = None
            if hasattr(clf_pipeline, "predict_proba"):
                proba = clf_pipeline.predict_proba(X_test_df)
            elif hasattr(clf, "decision_function"):
                try:
                    decision = clf.decision_function(X_test_df)
                    # attempt to scale to [0,1]
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    proba = scaler.fit_transform(np.atleast_2d(decision).T)
                except Exception:
                    proba = None

            if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                # binary case
                y_score = proba[:, 1]
                fpr, tpr, _ = roc_curve(y_test_arr, y_score)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
                ax_roc.plot(fpr, tpr, linewidth=1.8)
                ax_roc.plot([0, 1], [0, 1], "--", linewidth=0.9)
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"ROC curve (AUC = {roc_auc:.3f})")
                plt.tight_layout()

                precision, recall, _ = precision_recall_curve(y_test_arr, y_score)
                fig_pr, ax_pr = plt.subplots(figsize=(5, 3))
                ax_pr.plot(recall, precision, linewidth=1.6)
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision–Recall curve")
                plt.tight_layout()

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_roc)
                with col2:
                    st.pyplot(fig_pr)

                # plain language
                st.markdown("**What does this mean?**")
                st.write("- ROC AUC near 1.0 is great; 0.5 means guessing.")
                st.write("- Precision–Recall shows how precise positive predictions are across recall levels. Use it when classes are imbalanced.")
            elif proba is not None and proba.ndim == 2 and proba.shape[1] > 2:
                # multiclass: show per-class ROC micro-average if possible (one-vs-rest)
                n_classes = proba.shape[1]
                fig_mc, ax_mc = plt.subplots(figsize=(6, 3.5))
                has_any = False
                for i in range(n_classes):
                    try:
                        fpr, tpr, _ = roc_curve((y_test_arr == i).astype(int), proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax_mc.plot(fpr, tpr, linewidth=1.2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
                        has_any = True
                    except Exception:
                        pass
                if has_any:
                    ax_mc.plot([0, 1], [0, 1], "--", linewidth=0.8)
                    ax_mc.set_xlabel("False Positive Rate")
                    ax_mc.set_ylabel("True Positive Rate")
                    ax_mc.set_title("Per-class ROC (one-vs-rest)")
                    ax_mc.legend(fontsize=8, loc="lower right")
                    plt.tight_layout()
                    st.pyplot(fig_mc)
                    st.markdown("**What does this mean?** Each line shows how well the model separates one class from the rest. Higher is better.")
                else:
                    st.info("ROC per-class not available for this model/data.")
            else:
                st.info("Probability or score not available for ROC/PR plots.")
        except Exception:
            st.info("ROC/PR generation failed.")
            st.text(traceback.format_exc())

        st.markdown("---")

        # Feature importances
        st.subheader("⭐ Feature importance (what the model thinks matters)")
        try:
            clf = clf_pipeline.named_steps.get("classifier", clf_pipeline)
            feat_names = None
            try:
                pre = clf_pipeline.named_steps.get("preprocessor", None)
                if pre is not None and hasattr(pre, "get_feature_names_out"):
                    feat_names = pre.get_feature_names_out()
                else:
                    feat_names = X_test_df.columns.tolist()
            except Exception:
                feat_names = X_test_df.columns.tolist()

            imp_vals = None
            if hasattr(clf, "feature_importances_"):
                imp_vals = np.array(clf.feature_importances_)
            elif hasattr(clf, "coef_"):
                coef = np.array(clf.coef_)
                # for multiclass, take mean absolute coefficient across classes
                if coef.ndim == 2:
                    imp_vals = np.mean(np.abs(coef), axis=0)
                else:
                    imp_vals = np.abs(coef)
            if imp_vals is not None:
                # build df and show top 12
                imp_df = pd.DataFrame({"feature": feat_names, "importance": imp_vals})
                imp_df = imp_df.sort_values("importance", ascending=False).head(12)
                fig_fi, ax_fi = plt.subplots(figsize=(6, 3))
                ax_fi.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                ax_fi.set_title("Top features by importance")
                plt.tight_layout()
                col1, col2 = st.columns([1, 0.7])
                with col1:
                    st.pyplot(fig_fi)
                with col2:
                    st.markdown("**What does this mean?**")
                    st.write("Features at the top influence the model the most. Higher bars mean the model uses that column more when making decisions.")
                    st.write("If a surprising feature is important, consider checking data leakage or correlations.")
            else:
                st.info("No feature importances or coefficients available for this model type.")
        except Exception:
            st.info("Feature importance generation failed.")
            st.text(traceback.format_exc())

        st.markdown("---")

        # Partial Dependence Plot (compact)
        st.subheader("🔎 Partial Dependence — how a feature affects predictions")
        try:
            numeric_cols = X_test_df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("No numeric features available for PDP.")
            else:
                pdp_feat = st.selectbox("Choose numeric feature for PDP", numeric_cols, index=0, key="pdp_feat")
                base_row = X_test_df.iloc[[0]].copy().reset_index(drop=True)
                grid = np.linspace(X_test_df[pdp_feat].min(), X_test_df[pdp_feat].max(), num=40)
                pdp_preds = make_pdp_values(clf_pipeline, base_row, pdp_feat, grid)
                fig_pdp, ax_pdp = plt.subplots(figsize=(6, 3))
                ax_pdp.plot(grid, pdp_preds, linewidth=1.6, marker="o", markersize=3)
                ax_pdp.set_xlabel(pdp_feat)
                ax_pdp.set_ylabel("Predicted output / prob")
                ax_pdp.set_title(f"Partial dependence (approx) for {pdp_feat}")
                plt.tight_layout()
                col1, col2 = st.columns([1, 0.9])
                with col1:
                    st.pyplot(fig_pdp)
                with col2:
                    st.markdown("**What does this mean?**")
                    st.write("This plot shows how changing this feature (left to right) tends to change the model's predicted score/probability, holding other features near the selected sample.")
                    st.write("A rising line means higher values increase model confidence; a flat line means that feature doesn't change predictions much.")
        except Exception:
            st.info("PDP generation failed.")
            st.text(traceback.format_exc())

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

        # NEW: Enhanced prediction file upload with Parquet support
        st.subheader("📁 Upload Prediction Data")
        pred_file = st.file_uploader("Upload CSV/XLSX with same features", type=["csv", "xlsx"], key="pred_file")
        
        # NEW: Option to convert prediction CSV to Parquet too
        pred_parquet_path = None
        if pred_file and pred_file.name.lower().endswith('.csv'):
            pred_file_size = get_file_size_mb(pred_file)
            if pred_file_size > 5:  # Show option for prediction files larger than 5MB
                st.info(f"Prediction file size: {pred_file_size:.1f} MB")
                if PARQUET_AVAILABLE and st.button("⚡ Convert Prediction File to Parquet", key="pred_parquet"):
                    pred_parquet_path, pred_conversion_info = convert_csv_to_parquet(pred_file, chunk_size=25000)
                    if pred_parquet_path:
                        st.success(f"✅ Prediction file converted to Parquet!")
                        reduction = ((pred_conversion_info["original_size"] - pred_conversion_info["parquet_size"]) / pred_conversion_info["original_size"]) * 100
                        st.metric("Size Reduction", f"{reduction:.1f}%")
        
        if pred_file or pred_parquet_path:
            try:
                # Load data (Parquet or regular)
                if pred_parquet_path:
                    new_df = load_parquet_file(pred_parquet_path)
                    st.info("🏎️ Using Parquet format for faster processing")
                else:
                    name = pred_file.name.lower()
                    if name.endswith(".csv"):
                        new_df = pd.read_csv(pred_file)
                    else:
                        new_df = pd.read_excel(pred_file)
                
                if new_df is not None:
                    st.dataframe(new_df.head(), use_container_width=True)
                    missing = [c for c in trained_feature_cols if c not in new_df.columns]
                    extra = [c for c in new_df.columns if c not in trained_feature_cols]
                    
                    if missing:
                        st.error(f"Missing columns: {missing}")
                    else:
                        features_df = new_df[trained_feature_cols]
                        if extra:
                            st.warning(f"Ignoring extra columns: {extra[:5]}{'...' if len(extra) > 5 else ''}")
                        
                        # NEW: Enhanced prediction with batch processing for large files
                        predict_button_text = "🔮 Generate Predictions"
                        if len(new_df) > 50000:
                            predict_button_text += f" (Batch processing {len(new_df):,} rows)"
                        
                        if st.button(predict_button_text, use_container_width=True):
                            with st.spinner("Predicting..."):
                                try:
                                    # NEW: Batch processing for large datasets
                                    batch_size = 10000 if len(new_df) > 50000 else len(new_df)
                                    all_preds = []
                                    all_probs = []
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    for i in range(0, len(features_df), batch_size):
                                        batch = features_df.iloc[i:i+batch_size]
                                        
                                        # Update progress
                                        progress = min((i + len(batch)) / len(features_df), 1.0)
                                        progress_bar.progress(progress)
                                        status_text.text(f"Processing batch: {i+1:,} to {min(i+batch_size, len(features_df)):,} rows")
                                        
                                        # Predict batch
                                        batch_preds = clf_pipeline.predict(batch)
                                        all_preds.extend(batch_preds)
                                        
                                        # Try to get probabilities
                                        try:
                                            batch_probs = clf_pipeline.predict_proba(batch)
                                            if all_probs == []:  # First batch
                                                all_probs = batch_probs
                                            else:
                                                all_probs = np.vstack([all_probs, batch_probs])
                                        except Exception:
                                            all_probs = None
                                    
                                    progress_bar.progress(1.0)
                                    status_text.text("✅ Prediction complete!")
                                    
                                    # Convert to arrays
                                    preds = np.array(all_preds)
                                    probs = all_probs
                                    
                                    # Build results dataframe
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

                                    st.success(f"✅ Predicted {len(results):,} rows successfully!")
                                    
                                    # Show sample results
                                    st.subheader("📊 Prediction Results Sample")
                                    st.dataframe(results.head(10), use_container_width=True)
                                    
                                    # NEW: Enhanced download options
                                    st.subheader("📥 Download Options")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # CSV Download
                                        base = safe_filename_base(pred_file.name if pred_file else "predictions")
                                        fname_csv = f"predictions_{base}.csv"
                                        csv_buf = io.StringIO()
                                        results.to_csv(csv_buf, index=False)
                                        st.download_button("📄 Download as CSV", 
                                                         data=csv_buf.getvalue(), 
                                                         file_name=fname_csv,
                                                         mime="text/csv")
                                    
                                    with col2:
                                        # NEW: Parquet Download option for large files
                                        if PARQUET_AVAILABLE and len(results) > 10000:
                                            fname_parquet = f"predictions_{base}.parquet"
                                            parquet_buf = io.BytesIO()
                                            results.to_parquet(parquet_buf, index=False)
                                            st.download_button("📦 Download as Parquet (Faster)", 
                                                             data=parquet_buf.getvalue(), 
                                                             file_name=fname_parquet,
                                                             mime="application/octet-stream")
                                    
                                    # NEW: Prediction summary statistics
                                    if len(results) > 1000:
                                        st.subheader("📈 Prediction Summary")
                                        summary_cols = st.columns(3)
                                        
                                        with summary_cols[0]:
                                            st.metric("Total Predictions", f"{len(results):,}")
                                        
                                        with summary_cols[1]:
                                            if "prediction" in results.columns:
                                                unique_preds = results["prediction"].nunique()
                                                st.metric("Unique Predictions", unique_preds)
                                        
                                        with summary_cols[2]:
                                            # Show most common prediction
                                            if "prediction" in results.columns:
                                                most_common = results["prediction"].mode().iloc[0] if len(results["prediction"].mode()) > 0 else "N/A"
                                                st.metric("Most Common", str(most_common))
                                        
                                        # Show prediction distribution
                                        if "prediction" in results.columns and results["prediction"].nunique() < 20:
                                            pred_dist = results["prediction"].value_counts().head(10)
                                            st.bar_chart(pred_dist)
                                    
                                except Exception as e:
                                    st.error(f"Prediction failed: {e}")
                                    st.text(traceback.format_exc())
            except Exception as e:
                st.error(f"Failed to load prediction file: {e}")
                st.text(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | CE Innovation Team © 2025")

# NEW: Add installation instructions in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📦 Optional Dependencies")
    
    if not PARQUET_AVAILABLE:
        st.warning("⚡ **For faster processing of large files:**")
        st.code("pip install pyarrow", language="bash")
    
    if not IMB_AVAILABLE:
        st.warning("⚖️ **For class balancing:**")
        st.code("pip install imbalanced-learn", language="bash")
    
    if ProfileReport is None:
        st.warning("📊 **For data profiling:**")
        st.code("pip install ydata-profiling", language="bash")

# ====== OPTIMIZED HELPERS ADDED ======

import psutil  # NEW: system resource check

def get_system_memory_mb():
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None

def get_file_size_mb(file_obj):
    try:
        current_pos = file_obj.tell()
        file_obj.seek(0, 2)
        size = file_obj.tell()
        file_obj.seek(current_pos)
        return size / (1024 * 1024)
    except:
        return 0

# -------------------------
# Optimized CSV → Parquet
# -------------------------
def convert_csv_to_parquet(uploaded_file, chunk_size=50000):
    if not PARQUET_AVAILABLE:
        st.error("PyArrow is required for Parquet conversion. Please install: pip install pyarrow")
        return None, None
    try:
        original_size = get_file_size_mb(uploaded_file)
        temp_dir = tempfile.mkdtemp()
        parquet_path = os.path.join(temp_dir, f"{safe_filename_base(uploaded_file.name)}.parquet")

        uploaded_file.seek(0)
        total_rows = 0
        uploaded_file.seek(0)
        total_lines = sum(1 for _ in uploaded_file) - 1
        uploaded_file.seek(0)

        progress_bar = st.progress(0)
        status_text = st.empty()
        writer = None

        with st.spinner("Converting CSV to Parquet format..."):
            chunk_count = 0
            for chunk_df in pd.read_csv(uploaded_file, chunksize=chunk_size):
                chunk_count += 1
                total_rows += len(chunk_df)
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema)
                writer.write_table(table)
                progress = min(total_rows / total_lines, 1.0) if total_lines > 0 else 1.0
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {chunk_count}: {total_rows:,} rows")
            if writer is not None:
                writer.close()

        progress_bar.progress(1.0)
        status_text.text(f"✅ Conversion complete! Processed {total_rows:,} rows")
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        return parquet_path, {"original_size": original_size, "parquet_size": parquet_size, "rows": total_rows}
    except Exception as e:
        st.error(f"Failed to convert CSV to Parquet: {e}")
        return None, None

# -------------------------
# Optimized profiling
# -------------------------
def generate_profile_html_file(df: pd.DataFrame, tmp_dir: Optional[str] = None) -> str:
    if ProfileReport is None:
        out_html = "<h3>Profiling library not installed</h3>"
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=tmp_dir)
        fd.write(out_html.encode("utf-8"))
        fd.close()
        return fd.name
    use_light = df.shape[0] > 50000 or df.shape[1] > 50
    profile = ProfileReport(
        df,
        title="Dataset Profiling Report",
        explorative=True,
        minimal=use_light,
        correlations={"pearson": {"calculate": not use_light}},
    )
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

# -------------------------
# Optimized prediction batching
# -------------------------
def predict_in_batches(clf_pipeline, features_df, batch_size=10000):
    preds = []
    prob_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(0, len(features_df), batch_size):
        batch = features_df.iloc[i:i+batch_size]
        progress = min((i + len(batch)) / len(features_df), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing batch: {i+1:,} → {min(i+batch_size, len(features_df)):,}")
        batch_preds = clf_pipeline.predict(batch)
        preds.extend(batch_preds)
        try:
            batch_probs = clf_pipeline.predict_proba(batch)
            prob_list.append(batch_probs)
        except Exception:
            pass
    progress_bar.progress(1.0)
    status_text.text("✅ Prediction complete!")
    probs = np.concatenate(prob_list, axis=0) if prob_list else None
    return np.array(preds), probs
