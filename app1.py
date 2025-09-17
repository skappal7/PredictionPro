"""
Clean AutoML Streamlit App - Fixed Syntax Issues
Compatible with your requirements.txt libraries
"""

import io
import os
import tempfile
import traceback
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import hashlib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Your installed libraries
import plotly.express as px
import plotly.graph_objects as go
import shap
import pyarrow as pa
import pyarrow.parquet as pq

# Core ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, f1_score, mean_squared_error, r2_score)

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Data profiling
from ydata_profiling import ProfileReport
try:
    from streamlit_pandas_profiling import st_profile_report
    HAS_ST_PROFILE = True
except ImportError:
    HAS_ST_PROFILE = False

# Page configuration
st.set_page_config(
    page_title="AutoML Pro",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    defaults = {
        "data_uploaded": False,
        "profile_generated": False,
        "model_trained": False,
        "current_file": None,
        "data_hash": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Utility functions
@st.cache_data
def generate_data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_data
def load_and_process_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".csv", ".tsv")):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        return df, generate_data_hash(df)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None, None

@st.cache_data
def safe_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        df_clean = df.copy()
        df_clean = df_clean.dropna(axis=1, how='all')
        
        for col in df_clean.columns:
            try:
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                non_null_numeric = numeric_series.dropna()
                if len(non_null_numeric) >= 0.3 * len(df_clean[col].dropna()):
                    df_clean[col] = numeric_series
                    col_mean = df_clean[col].mean()
                    fill_value = col_mean if not pd.isna(col_mean) else 0
                    df_clean[col] = df_clean[col].fillna(fill_value)
                else:
                    if df_clean[col].dtype == 'object':
                        mode_val = df_clean[col].mode()
                        fill_value = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value).astype(str)
            except Exception:
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
        
    except Exception:
        try:
            simple_df = df.iloc[:, :min(5, df.shape[1])].copy()
            for col in simple_df.columns:
                simple_df[col] = simple_df[col].astype(str).fillna('Unknown')
            return simple_df
        except Exception:
            return pd.DataFrame({'default_col': ['No data available']})

@st.cache_data
def generate_profile_report(_df: pd.DataFrame, minimal: bool = False) -> Optional[ProfileReport]:
    try:
        df_clean = safe_data_cleaning(_df)
        
        if df_clean.empty:
            return None
        
        if len(df_clean) > 5000 and not minimal:
            df_clean = df_clean.sample(n=5000, random_state=42)
        
        config = {
            "title": "Dataset Profile Report",
            "minimal": minimal,
            "explorative": not minimal,
            "dark_mode": False,
            "lazy": False
        }
        
        profile = ProfileReport(df_clean, **config)
        return profile
        
    except Exception as e:
        st.warning(f"Profile generation issue: {e}")
        return None

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def create_model_package(model, feature_cols, le_target, model_name, metrics):
    try:
        model_package = {
            'model': model,
            'feature_columns': feature_cols,
            'label_encoder': le_target,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': '2.0'
        }
        
        buffer = io.BytesIO()
        joblib.dump(model_package, buffer, compress=3)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Model serialization failed: {e}")
        fallback_package = {
            'feature_columns': feature_cols,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'note': 'Model object could not be serialized'
        }
        buffer = io.BytesIO()
        joblib.dump(fallback_package, buffer)
        buffer.seek(0)
        return buffer.getvalue()

@st.cache_data
def create_shap_explanations(_model, _X_sample, model_type: str):
    try:
        if hasattr(_model, 'named_steps'):
            classifier = _model.named_steps.get('model', _model)
            X_transformed = _model.named_steps['preprocessor'].transform(_X_sample)
        else:
            classifier = _model
            X_transformed = _X_sample
        
        if model_type in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed[:100])
        else:
            explainer = shap.PermutationExplainer(classifier.predict, X_transformed[:50])
            shap_values = explainer(X_transformed[:20])
        
        return shap_values, explainer
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None, None

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Analytics Pro</h1>
    <p>Professional machine learning with SHAP, Plotly, and advanced analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    st.markdown("✅ SHAP Explanations")
    st.markdown("✅ Plotly Visualizations") 
    st.markdown("✅ PyArrow Support")
    st.markdown("✅ Imbalanced Learning")
    st.markdown("✅ Data Profiling")
    
    st.markdown("---")
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supports CSV, Excel, and Parquet files"
    )
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df, data_hash = load_and_process_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.session_state.data_uploaded = True
                st.session_state.data_hash = data_hash
                st.session_state.current_file = uploaded_file.name
                
                st.success("Data loaded successfully!")
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"""
                **File:** {uploaded_file.name}  
                **Size:** {file_size_mb:.1f} MB  
                **Shape:** {df.shape[0]:,} × {df.shape[1]}
                """)

# Main tabs
tab_names = ["📊 Data Explorer", "🚀 Model Lab", "🔍 SHAP Insights", "📈 Predictions"]
tabs = st.tabs(tab_names)

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("## 📊 Data Explorer")
    
    if not st.session_state.data_uploaded:
        st.info("Upload a dataset in the sidebar to start exploring")
    else:
        df = st.session_state.data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>📊 Rows</h3><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>📋 Columns</h3><h2>{df.shape[1]}</h2></div>', unsafe_allow_html=True)
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f'<div class="metric-card"><h3>💾 Memory</h3><h2>{memory_mb:.1f} MB</h2></div>', unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f'<div class="metric-card"><h3>❓ Missing</h3><h2>{missing_pct:.1f}%</h2></div>', unsafe_allow_html=True)
        
        # Data types visualization
        st.markdown("### 📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values, 
            names=[str(dtype) for dtype in dtype_counts.index],  # Convert dtype objects to strings
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
        # Data preview
        st.markdown("### 🔍 Data Preview")
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_pages = (len(df) - 1) // page_size + 1
        
        if total_pages > 1:
            page = st.number_input("Page", 1, total_pages, 1)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            display_df = df.iloc[start_idx:end_idx]
        else:
            display_df = df.head(page_size)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Profiling
        st.markdown("### 📈 Data Profile")
        profile_col1, profile_col2 = st.columns([1, 2])
        
        with profile_col1:
            profile_type = st.radio("Profile Type", ["Quick Overview", "Detailed Analysis"])
            
            if st.button("🚀 Generate Profile", type="primary"):
                with st.spinner("Generating profile report..."):
                    minimal = (profile_type == "Quick Overview")
                    profile = generate_profile_report(df, minimal=minimal)
                    
                    if profile is not None:
                        st.session_state.profile_report = profile
                        st.session_state.profile_generated = True
                        st.success("Profile report generated!")
                    else:
                        st.error("Failed to generate profile report")
        
        with profile_col2:
            if st.session_state.get('profile_generated', False):
                st.info("📊 Profile report will appear below")
        
        if st.session_state.get('profile_generated', False) and HAS_ST_PROFILE:
            if 'profile_report' in st.session_state:
                try:
                    st.markdown("---")
                    st_profile_report(st.session_state.profile_report)
                except Exception as e:
                    st.error(f"Failed to display profile: {e}")

# Tab 2: Model Lab
with tabs[1]:
    st.markdown("## 🚀 Model Development")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data first to start model development")
    else:
        data = st.session_state.data.copy()
        
        # Model configuration
        st.markdown("### ⚙️ Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            all_columns = list(data.columns)
            target_column = st.selectbox("🎯 Target Variable", all_columns)
        
        with config_col2:
            available_features = [col for col in all_columns if col != target_column]
            feature_mode = st.radio("🔧 Feature Selection", ["Auto-select all", "Manual selection"], horizontal=True)
        
        if feature_mode == "Manual selection":
            feature_cols = st.multiselect("Select features", available_features, default=available_features)
            if not feature_cols:
                st.warning("Please select at least one feature")
                st.stop()
        else:
            feature_cols = available_features
        
        # Data preparation
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()
        
        if y_raw.dtype == "object" or y_raw.nunique() <= 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_.tolist()
            problem_type = "classification"
            st.info(f"🎯 Detected: **Classification** with {len(class_names)} classes")
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
            st.info("🎯 Detected: **Regression** problem")
        
        # Target visualization
        st.markdown("### 📊 Target Analysis")
        
        if problem_type == "classification":
            target_counts = pd.Series(y).value_counts().sort_index()
            fig_target = px.bar(x=[class_names[i] for i in target_counts.index], y=target_counts.values, title="Target Distribution")
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            fig_target = px.histogram(y_raw, nbins=30, title="Target Distribution")
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            if problem_type == "classification":
                model_options = {
                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM": SVC(probability=True, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42)
                }
            else:
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import LinearRegression
                from sklearn.svm import SVR
                from sklearn.tree import DecisionTreeRegressor
                
                model_options = {
                    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression(),
                    "SVM": SVR(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42)
                }
            
            model_choice = st.selectbox("Algorithm", list(model_options.keys()))
            base_model = model_options[model_choice]
        
        with model_col2:
            if problem_type == "classification":
                balance_options = ["None", "SMOTE", "Random Oversample", "Random Undersample"]
                balance_method = st.selectbox("⚖️ Class Balancing", balance_options)
            else:
                balance_method = "None"
            
            test_size = st.slider("📊 Test Size (%)", 10, 50, 20, step=5)
        
        # Training
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Split data
                status_text.info("📊 Splitting data...")
                progress_bar.progress(20)
                
                if problem_type == "classification":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                
                # Create pipeline
                status_text.info("🔧 Building pipeline...")
                progress_bar.progress(40)
                
                numeric_features = X.select_dtypes(include=[np.number]).columns
                categorical_features = X.select_dtypes(exclude=[np.number]).columns
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', safe_onehot_encoder(), categorical_features)
                    ]
                )
                
                pipeline_steps = [('preprocessor', preprocessor)]
                
                if balance_method != "None" and problem_type == "classification":
                    if balance_method == "SMOTE":
                        sampler = SMOTE(random_state=42)
                    elif balance_method == "Random Oversample":
                        sampler = RandomOverSampler(random_state=42)
                    elif balance_method == "Random Undersample":
                        sampler = RandomUnderSampler(random_state=42)
                    pipeline_steps.append(('sampler', sampler))
                
                pipeline_steps.append(('model', base_model))
                
                if problem_type == "classification" and balance_method != "None":
                    model_pipeline = ImbPipeline(pipeline_steps)
                else:
                    model_pipeline = Pipeline(pipeline_steps)
                
                # Train
                status_text.info("🏋️ Training model...")
                progress_bar.progress(70)
                
                model_pipeline.fit(X_train, y_train)
                
                # Evaluate
                status_text.info("📊 Evaluating...")
                progress_bar.progress(90)
                
                y_pred = model_pipeline.predict(X_test)
                
                if problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    metrics = {'accuracy': accuracy, 'f1_score': f1}
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {'r2_score': r2, 'mse': mse}
                
                progress_bar.progress(100)
                status_text.empty()
                
                # Save results
                st.session_state.model_trained = True
                st.session_state.trained_model = model_pipeline
                st.session_state.trained_feature_cols = feature_cols
                st.session_state.trained_le_target = le_target
                st.session_state.trained_class_names = class_names
                st.session_state.test_results = {'y_test': y_test, 'y_pred': y_pred, 'metrics': metrics}
                st.session_state.trained_X_test = X_test
                st.session_state.model_name = model_choice
                st.session_state.problem_type = problem_type
                
                # Show results
                success_metric = list(metrics.items())[0]
                st.success(f"✅ Model trained! {success_metric[0].replace('_', ' ').title()}: {success_metric[1]:.3f}")
                
                metric_cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.3f}")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Training failed: {str(e)}")
                with st.expander("Show error details"):
                    st.text(traceback.format_exc())
        
        # Model download
        if st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### 📦 Download Model")
            
            model_package = create_model_package(
                st.session_state.trained_model,
                st.session_state.trained_feature_cols,
                st.session_state.trained_le_target,
                st.session_state.model_name,
                st.session_state.test_results['metrics']
            )
            
            filename = f"automl_model_{st.session_state.model_name.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            st.download_button(
                label="📦 Download Model Package",
                data=model_package,
                file_name=filename,
                mime="application/octet-stream"
            )

# Tab 3: SHAP Insights
with tabs[2]:
    st.markdown("## 🔍 SHAP Insights")
    
    if not st.session_state.model_trained:
        st.info("Train a model first to see SHAP explanations")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.test_results['y_test']
        y_pred = st.session_state.test_results['y_pred']
        model_name = st.session_state.model_name
        problem_type = st.session_state.problem_type
        
        # Performance overview
        st.markdown("### 📊 Performance")
        
        if problem_type == "classification":
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
            
            # ROC curve for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception:
                    pass
        else:
            # Regression plots
            fig_scatter = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted")
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect', line=dict(dash='dash')))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # SHAP explanations
        st.markdown("### 🎯 SHAP Explanations")
        
        with st.spinner("Generating SHAP explanations..."):
            shap_values, explainer = create_shap_explanations(model, X_test, model_name)
            
            if shap_values is not None:
                try:
                    # Summary plot
                    st.markdown("#### 📊 Feature Importance")
                    
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    
                    if hasattr(shap_values, 'values'):
                        shap.summary_plot(shap_values.values, X_test.iloc[:len(shap_values.values)], feature_names=X_test.columns, show=False, ax=ax)
                    else:
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            shap.summary_plot(shap_values[1], X_test.iloc[:len(shap_values[1])], feature_names=X_test.columns, show=False, ax=ax)
                        else:
                            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                            shap.summary_plot(shap_vals, X_test.iloc[:len(shap_vals)], feature_names=X_test.columns, show=False, ax=ax)
                    
                    plt.title("SHAP Feature Importance")
                    st.pyplot(fig_shap)
                    
                    # Individual explanation
                    st.markdown("#### 🔍 Individual Explanation")
                    
                    sample_idx = st.slider("Sample to explain", 0, min(20, len(X_test)-1), 0)
                    
                    try:
                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                        
                        if hasattr(shap_values, 'values'):
                            sample_shap = shap_values.values[sample_idx]
                        else:
                            if isinstance(shap_values, list):
                                sample_shap = shap_values[1][sample_idx] if len(shap_values) > 1 else shap_values[0][sample_idx]
                            else:
                                sample_shap = shap_values[sample_idx]
                        
                        # Waterfall plot
                        sorted_idx = np.argsort(np.abs(sample_shap))[-10:]
                        colors = ['red' if val > 0 else 'blue' for val in sample_shap[sorted_idx]]
                        
                        ax_waterfall.barh(range(len(sorted_idx)), sample_shap[sorted_idx], color=colors)
                        ax_waterfall.set_yticks(range(len(sorted_idx)))
                        ax_waterfall.set_yticklabels([X_test.columns[i] for i in sorted_idx])
                        ax_waterfall.set_xlabel("SHAP Value")
                        ax_waterfall.set_title(f"SHAP Values for Sample {sample_idx + 1}")
                        ax_waterfall.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        
                        st.pyplot(fig_waterfall)
                        
                        # Show feature values
                        sample_features = X_test.iloc[sample_idx]
                        feature_df = pd.DataFrame({
                            'Feature': sample_features.index,
                            'Value': sample_features.values
                        })
                        st.dataframe(feature_df, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Individual explanation failed: {e}")
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {e}")
            else:
                st.warning("SHAP explanations not available for this model")
        
        # Feature importance fallback
        st.markdown("### ⭐ Feature Importance")
        
        try:
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('model', model)
            else:
                classifier = model
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    try:
                        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                    except:
                        feature_names = X_test.columns
                else:
                    feature_names = X_test.columns
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(15)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Top 15 Features")
                st.plotly_chart(fig_importance, use_container_width=True)
                
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    try:
                        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                    except:
                        feature_names = X_test.columns
                else:
                    feature_names = X_test.columns
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': np.abs(coefficients)
                }).sort_values('Coefficient', ascending=True).tail(15)
                
                fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title="Top 15 Coefficients")
                st.plotly_chart(fig_coef, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Feature importance failed: {e}")

# Tab 4: Predictions
with tabs[3]:
    st.markdown("## 📈 Make Predictions")
    
    if not st.session_state.model_trained:
        st.info("Train a model first to make predictions")
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        problem_type = st.session_state.problem_type
        
        st.markdown("### 📁 Upload Prediction Data")
        
        pred_file = st.file_uploader(
            "Upload data for predictions",
            type=["csv", "xlsx", "parquet"],
            key="prediction_file"
        )
        
        if pred_file:
            try:
                # Load data
                if pred_file.name.lower().endswith('.csv'):
                    new_data = pd.read_csv(pred_file)
                elif pred_file.name.lower().endswith('.parquet'):
                    new_data = pd.read_parquet(pred_file)
                else:
                    new_data = pd.read_excel(pred_file)
                
                st.markdown("### 📊 Data Preview")
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Feature validation
                missing_features = set(feature_cols) - set(new_data.columns)
                extra_features = set(new_data.columns) - set(feature_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if missing_features:
                        st.error(f"Missing features: {list(missing_features)}")
                    else:
                        st.success("All required features present!")
                
                with col2:
                    if extra_features:
                        st.warning(f"Extra columns will be ignored: {len(extra_features)}")
                    st.info(f"Ready to predict on {len(new_data):,} rows")
                
                if not missing_features:
                    # Options
                    options_col1, options_col2 = st.columns(2)
                    
                    with options_col1:
                        include_probabilities = st.checkbox(
                            "Include probabilities",
                            value=True if problem_type == "classification" else False,
                            disabled=problem_type == "regression"
                        )
                    
                    with options_col2:
                        batch_size = st.selectbox("Batch size", [1000, 5000, 10000], index=1)
                    
                    # Make predictions
                    if st.button("🔮 Generate Predictions", type="primary"):
                        prediction_data = new_data[feature_cols]
                        
                        with st.spinner("Making predictions..."):
                            try:
                                all_predictions = []
                                all_probabilities = []
                                
                                progress_bar = st.progress(0)
                                
                                for i in range(0, len(prediction_data), batch_size):
                                    batch = prediction_data.iloc[i:i+batch_size]
                                    progress = min((i + len(batch)) / len(prediction_data), 1.0)
                                    progress_bar.progress(progress)
                                    
                                    # Predictions
                                    batch_preds = model.predict(batch)
                                    all_predictions.extend(batch_preds)
                                    
                                    # Probabilities
                                    if include_probabilities and hasattr(model, 'predict_proba'):
                                        try:
                                            batch_probs = model.predict_proba(batch)
                                            all_probabilities.append(batch_probs)
                                        except:
                                            pass
                                
                                progress_bar.progress(1.0)
                                
                                # Results
                                predictions = np.array(all_predictions)
                                
                                if all_probabilities:
                                    probabilities = np.vstack(all_probabilities)
                                else:
                                    probabilities = None
                                
                                results_df = new_data.copy()
                                
                                if le_target:
                                    results_df['prediction'] = le_target.inverse_transform(predictions)
                                    results_df['prediction_encoded'] = predictions
                                else:
                                    results_df['prediction'] = predictions
                                
                                if probabilities is not None:
                                    if le_target:
                                        class_names = le_target.classes_
                                    else:
                                        class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
                                    
                                    for i, class_name in enumerate(class_names):
                                        results_df[f'prob_{class_name}'] = probabilities[:, i]
                                
                                st.success(f"Predictions completed for {len(results_df):,} rows!")
                                
                                # Summary
                                summary_cols = st.columns(4)
                                with summary_cols[0]:
                                    st.metric("Total Predictions", f"{len(results_df):,}")
                                with summary_cols[1]:
                                    st.metric("Unique Values", results_df['prediction'].nunique())
                                with summary_cols[2]:
                                    if problem_type == "classification":
                                        most_common = results_df['prediction'].mode().iloc[0]
                                        st.metric("Most Common", str(most_common))
                                    else:
                                        st.metric("Mean", f"{results_df['prediction'].mean():.3f}")
                                with summary_cols[3]:
                                    if probabilities is not None:
                                        avg_conf = probabilities.max(axis=1).mean()
                                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                                    else:
                                        st.metric("Std Dev", f"{results_df['prediction'].std():.3f}")
                                
                                # Sample results
                                st.markdown("### Sample Results")
                                display_cols = feature_cols[:3] + ['prediction']
                                if probabilities is not None:
                                    prob_cols = [col for col in results_df.columns if col.startswith('prob_')][:3]
                                    display_cols.extend(prob_cols)
                                
                                st.dataframe(results_df[display_cols].head(10), use_container_width=True)
                                
                                # Distribution
                                if problem_type == "classification" and results_df['prediction'].nunique() <= 20:
                                    st.markdown("### Prediction Distribution")
                                    pred_dist = results_df['prediction'].value_counts()
                                    fig_dist = px.bar(x=pred_dist.index, y=pred_dist.values, title="Distribution")
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                elif problem_type == "regression":
                                    st.markdown("### Prediction Distribution")
                                    fig_hist = px.histogram(results_df['prediction'], nbins=50, title="Distribution")
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                # Downloads
                                st.markdown("### 📥 Download Results")
                                
                                download_cols = st.columns(3)
                                
                                with download_cols[0]:
                                    csv_buffer = io.StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    st.download_button(
                                        "📄 CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with download_cols[1]:
                                    excel_buffer = io.BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                    
                                    st.download_button(
                                        "📊 Excel",
                                        data=excel_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                
                                with download_cols[2]:
                                    if len(results_df) > 5000:
                                        parquet_buffer = io.BytesIO()
                                        results_df.to_parquet(parquet_buffer, index=False)
                                        
                                        st.download_button(
                                            "⚡ Parquet",
                                            data=parquet_buffer.getvalue(),
                                            file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                                            mime="application/octet-stream"
                                        )
                            
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                with st.expander("Error details"):
                                    st.text(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Failed to load file: {e}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**AutoML Pro v2.0**")
    st.caption("Built with your complete ML stack")

with col2:
    if st.session_state.model_trained:
        st.markdown("**Current Model**")
        st.caption(f"{st.session_state.model_name}")
    else:
        st.markdown("**Status**")
        st.caption("Ready for training")

with col3:
    st.markdown("**Libraries**")
    st.caption("SHAP • Plotly • PyArrow • Scikit-learn")

# Sidebar help
with st.sidebar:
    if not st.session_state.data_uploaded:
        st.markdown("---")
        st.markdown("### Quick Start")
        st.markdown("""
        1. Upload your dataset above
        2. Explore data in first tab
        3. Train model in second tab
        4. View SHAP insights in third tab
        5. Make predictions in fourth tab
        """)
    else:
        st.markdown("---")
        st.markdown("### Dataset Info")
        df = st.session_state.data
        st.markdown(f"""
        **File:** {st.session_state.current_file}  
        **Shape:** {df.shape[0]:,} × {df.shape[1]}  
        **Memory:** {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB
        """)
