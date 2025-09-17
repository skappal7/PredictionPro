"""
Enhanced Predictive Analytics AutoML App - Optimized for Your Stack
- Works with your exact requirements.txt versions
- SHAP 0.46.0 compatible
- ExplainerDashboard integration ready
- Plotly visualizations
- Production-grade UI/UX
"""

import io
import os
import re
import tempfile
import traceback
import pickle
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

# Your installed libraries - all should be available
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import shap  # You have 0.46.0
import pyarrow as pa
import pyarrow.parquet as pq

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve, f1_score)

# Imbalanced-learn - you have this installed
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Data profiling - you have ydata-profiling + streamlit-pandas-profiling
from ydata_profiling import ProfileReport
try:
    from streamlit_pandas_profiling import st_profile_report
    HAS_ST_PROFILE = True
except ImportError:
    HAS_ST_PROFILE = False

# ExplainerDashboard - you have 0.4.3
try:
    from explainerdashboard import ClassifierExplainer, RegressionExplainer
    from explainerdashboard.dashboard_components import *
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AutoML Analytics Pro",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
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
    .feature-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    .status-available { background: #28a745; color: white; }
    .status-missing { background: #dc3545; color: white; }
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

# Initialize session state
def init_session_state():
    defaults = {
        "data_uploaded": False,
        "profile_generated": False,
        "model_trained": False,
        "current_file": None,
        "data_hash": None,
        "experiment_history": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Utility functions with caching optimized for your libraries
@st.cache_data
def generate_data_hash(df: pd.DataFrame) -> str:
    """Generate hash for dataframe caching"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_data
def load_and_process_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Load data with caching"""
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".csv", ".tsv")):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            # Try CSV first, then Excel
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
        return df, generate_data_hash(df)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None, None

@st.cache_data
def safe_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust data cleaning that never fails:
    - Coerces all numeric columns with pd.to_numeric(errors='coerce')
    - Fills missing values with column means, fallback to 0
    - Handles categorical columns safely
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        df_clean = df.copy()
        
        # Remove completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Process each column safely
        for col in df_clean.columns:
            try:
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                
                # If more than 30% of values are numeric after conversion, treat as numeric
                non_null_numeric = numeric_series.dropna()
                if len(non_null_numeric) >= 0.3 * len(df_clean[col].dropna()):
                    df_clean[col] = numeric_series
                    # Fill missing values with mean, fallback to 0
                    col_mean = df_clean[col].mean()
                    fill_value = col_mean if not pd.isna(col_mean) else 0
                    df_clean[col] = df_clean[col].fillna(fill_value)
                else:
                    # Handle as categorical - fill missing with mode or 'Unknown'
                    if df_clean[col].dtype == 'object':
                        mode_val = df_clean[col].mode()
                        fill_value = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value).astype(str)
            
            except Exception:
                # Last resort: convert to string and fill missing
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        # Remove any remaining problematic columns
        problematic_cols = []
        for col in df_clean.columns:
            try:
                # Test if column can be processed
                _ = df_clean[col].dtype
                _ = df_clean[col].isnull().sum()
                _ = df_clean[col].nunique()
            except Exception:
                problematic_cols.append(col)
        
        if problematic_cols:
            df_clean = df_clean.drop(columns=problematic_cols)
        
        # Final safety check - ensure no infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
        
    except Exception as e:
        st.warning(f"Data cleaning encountered issues: {e}. Using minimal cleaning.")
        # Absolute fallback - return first few columns as strings
        try:
            simple_df = df.iloc[:, :min(5, df.shape[1])].copy()
            for col in simple_df.columns:
                simple_df[col] = simple_df[col].astype(str).fillna('Unknown')
            return simple_df
        except Exception:
            return pd.DataFrame({'default_col': ['No data available']})

@st.cache_data
def generate_profile_report(_df: pd.DataFrame, minimal: bool = False) -> ProfileReport:
    """Generate profiling report optimized for your ydata-profiling version"""
    try:
        # Clean data safely
        df_clean = safe_data_cleaning(_df)
        
        if df_clean.empty:
            return None
        
        # Sample large datasets for performance
        if len(df_clean) > 5000 and not minimal:
            df_clean = df_clean.sample(n=5000, random_state=42)
        
        # Configure profile for your ydata-profiling version
        config = {
            "title": "Dataset Profile Report",
            "minimal": minimal,
            "explorative": not minimal,
            "dark_mode": False,
            "lazy": False
        }
        
        # Create profile with your installed version
        profile = ProfileReport(df_clean, **config)
        return profile
        
    except Exception as e:
        st.warning(f"Profile generation issue: {e}")
        return None

def safe_onehot_encoder():
    """Create OneHotEncoder compatible with your scikit-learn version"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# Enhanced model training with your exact libraries
@st.cache_data
def create_shap_explanations(_model, _X_sample, model_type: str):
    """Generate SHAP explanations using your SHAP 0.46.0"""
    try:
        # Get the classifier from pipeline
        if hasattr(_model, 'named_steps'):
            classifier = _model.named_steps.get('classifier', _model)
            X_transformed = _model.named_steps['preprocessor'].transform(_X_sample)
        else:
            classifier = _model
            X_transformed = _X_sample
        
        # Use appropriate explainer for your SHAP version
        if model_type in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed[:100])  # Limit for performance
        else:
            # Use Permutation explainer for other models (more stable in 0.46.0)
            explainer = shap.PermutationExplainer(classifier.predict, X_transformed[:50])
            shap_values = explainer(X_transformed[:20])
        
        return shap_values, explainer
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None, None

# Model serialization optimized for your stack
def create_model_package(model, feature_cols, le_target, model_name, metrics):
    """Create model package using joblib (no cloudpickle needed)"""
    try:
        model_package = {
            'model': model,
            'feature_columns': feature_cols,
            'label_encoder': le_target,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'scikit_learn_version': '1.0+',
            'created_with': 'AutoML Pro v2.0'
        }
        
        # Use joblib (part of scikit-learn)
        buffer = io.BytesIO()
        joblib.dump(model_package, buffer, compress=3)
        buffer.seek(0)
        return buffer.getvalue()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Analytics Pro</h1>
    <p>Professional machine learning with your complete tech stack: SHAP, Plotly, ExplainerDashboard</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with library status
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Show available libraries from your requirements.txt
    libraries_status = [
        ("SHAP 0.46.0", True, "🎯"),
        ("Plotly ≤5.14.1", True, "📊"),
        ("ExplainerDashboard", EXPLAINER_AVAILABLE, "🔍"),
        ("ImbalancedLearn", True, "⚖️"),
        ("YData Profiling", True, "📈"),
        ("PyArrow", True, "⚡")
    ]
    
    for lib, status, icon in libraries_status:
        status_class = "status-available" if status else "status-missing"
        st.markdown(f'<span class="feature-status {status_class}">{icon} {lib}</span>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
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
                
                # Quick data info
                file_size_mb = uploaded_file.size / (1024 * 1024) if hasattr(uploaded_file, 'size') else 0
                st.info(f"""
                **File:** {uploaded_file.name}  
                **Size:** {file_size_mb:.1f} MB  
                **Shape:** {df.shape[0]:,} × {df.shape[1]}  
                **Memory:** {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB
                """)

# Main content tabs
tab_names = ["📊 Data Explorer", "🚀 Model Lab", "🔍 Insights & SHAP", "📈 Predictions"]
tabs = st.tabs(tab_names)

# Tab 1: Data Explorer with streamlit-pandas-profiling
with tabs[0]:
    st.markdown("## 📊 Data Explorer & Profiling")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload a dataset in the sidebar to start exploring")
    else:
        df = st.session_state.data
        
        # Enhanced data overview with Plotly charts
        st.markdown("### 🔍 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>📊 Rows</h3><h2>{:,}</h2></div>'.format(df.shape[0]), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>📋 Columns</h3><h2>{}</h2></div>'.format(df.shape[1]), unsafe_allow_html=True)
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown('<div class="metric-card"><h3>💾 Memory</h3><h2>{:.1f} MB</h2></div>'.format(memory_mb), unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown('<div class="metric-card"><h3>❓ Missing</h3><h2>{:.1f}%</h2></div>'.format(missing_pct), unsafe_allow_html=True)
        
        # Data types visualization with Plotly
        st.markdown("### 📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
        # Interactive data preview
        st.markdown("### 🔍 Interactive Data Preview")
        
        # Pagination
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
        
        # Profiling with your streamlit-pandas-profiling
        st.markdown("### 📈 Comprehensive Data Profile")
        
        profile_col1, profile_col2 = st.columns([1, 2])
        
        with profile_col1:
            profile_type = st.radio("Profile Type", ["Quick Overview", "Detailed Analysis"])
            
            if st.button("🚀 Generate Profile", type="primary"):
                with st.spinner("Generating comprehensive profile report..."):
                    try:
                        minimal = (profile_type == "Quick Overview")
                        profile = generate_profile_report(df, minimal=minimal)
                        
                        if profile is not None:
                            st.session_state.profile_report = profile
                            st.session_state.profile_generated = True
                            st.success("Profile report generated!")
                        else:
                            st.error("Failed to generate profile report")
                    except Exception as e:
                        st.error(f"Profile generation error: {e}")
        
        with profile_col2:
            if st.session_state.get('profile_generated', False) and HAS_ST_PROFILE:
                st.info("📊 Profile report will appear below")
        
        # Display profile using streamlit-pandas-profiling
        if st.session_state.get('profile_generated', False):
            if HAS_ST_PROFILE and 'profile_report' in st.session_state:
                try:
                    st.markdown("---")
                    st_profile_report(st.session_state.profile_report)
                except Exception as e:
                    st.error(f"Failed to display profile: {e}")
                    st.info("Try regenerating the profile report")

# Tab 2: Enhanced Model Lab
with tabs[1]:
    st.markdown("## 🚀 Advanced Model Development Lab")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first to start model development")
    else:
        data = st.session_state.data.copy()
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            all_columns = list(data.columns)
            target_column = st.selectbox(
                "🎯 Target Variable",
                all_columns,
                help="Select the column you want to predict"
            )
        
        with config_col2:
            available_features = [col for col in all_columns if col != target_column]
            feature_mode = st.radio(
                "🔧 Feature Selection",
                ["Auto-select all", "Manual selection"],
                horizontal=True
            )
        
        if feature_mode == "Manual selection":
            feature_cols = st.multiselect(
                "Select features",
                available_features,
                default=available_features
            )
            if not feature_cols:
                st.warning("Please select at least one feature")
                st.stop()
        else:
            feature_cols = available_features
        
        # Data preparation
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()
        
        # Smart target handling
        if y_raw.dtype == "object" or y_raw.nunique() <= 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_.tolist()
            problem_type = "classification"
            st.info(f"🎯 Detected: **Classification** problem with {len(class_names)} classes")
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
            st.info("🎯 Detected: **Regression** problem")
        
        # Target visualization with Plotly
        st.markdown("### 📊 Target Variable Analysis")
        
        if problem_type == "classification":
            target_counts = pd.Series(y).value_counts().sort_index()
            fig_target = px.bar(
                x=[class_names[i] for i in target_counts.index],
                y=target_counts.values,
                title="Target Class Distribution",
                labels={'x': 'Classes', 'y': 'Count'}
            )
            fig_target.update_traces(marker_color='lightblue')
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            fig_target = px.histogram(
                y_raw,
                nbins=30,
                title="Target Distribution"
            )
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Enhanced algorithm selection
        st.markdown("### 🤖 Algorithm Selection & Training")
        
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
            
            model_choice = st.selectbox("Select Algorithm", list(model_options.keys()))
            base_model = model_options[model_choice]
        
        with model_col2:
            # Advanced options
            st.markdown("**Training Options:**")
            
            # Class balancing (only for classification)
            if problem_type == "classification":
                balance_options = ["None", "SMOTE", "Random Oversample", "Random Undersample"]
                balance_method = st.selectbox("⚖️ Class Balancing", balance_options)
            else:
                balance_method = "None"
            
            test_size = st.slider("📊 Test Size (%)", 10, 50, 20, step=5)
            cv_folds = st.slider("🔄 Cross-validation Folds", 3, 10, 5)
        
        # Model training
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if len(np.unique(y)) < 2 and problem_type == "classification":
                st.error("Need at least 2 classes in target variable for classification")
            else:
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Split data
                        status_text.info("📊 Splitting data...")
                        progress_bar.progress(20)
                        
                        if problem_type == "classification":
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size/100, random_state=42, stratify=y
                            )
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size/100, random_state=42
                            )
                        
                        # Create preprocessing pipeline
                        status_text.info("🔧 Building preprocessing pipeline...")
                        progress_bar.progress(40)
                        
                        numeric_features = X.select_dtypes(include=[np.number]).columns
                        categorical_features = X.select_dtypes(exclude=[np.number]).columns
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_features),
                                ('cat', safe_onehot_encoder(), categorical_features)
                            ]
                        )
                        
                        # Handle class balancing
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
                        
                        # Create pipeline
                        if problem_type == "classification" and balance_method != "None":
                            model_pipeline = ImbPipeline(pipeline_steps)
                        else:
                            model_pipeline = Pipeline(pipeline_steps)
                        
                        # Train model
                        status_text.info("🏋️ Training model...")
                        progress_bar.progress(70)
                        
                        model_pipeline.fit(X_train, y_train)
                        
                        # Evaluate model
                        status_text.info("📊 Evaluating performance...")
                        progress_bar.progress(90)
                        
                        y_pred = model_pipeline.predict(X_test)
                        
                        if problem_type == "classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            metrics = {'accuracy': accuracy, 'f1_score': f1}
                        else:
                            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            metrics = {'r2_score': r2, 'mse': mse, 'mae': mae}
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        
                        # Save to session state
                        st.session_state.model_trained = True
                        st.session_state.trained_model = model_pipeline
                        st.session_state.trained_feature_cols = feature_cols
                        st.session_state.trained_le_target = le_target
                        st.session_state.trained_class_names = class_names
                        st.session_state.test_results = {
                            'y_test': y_test, 'y_pred': y_pred, 'metrics': metrics
                        }
                        st.session_state.trained_X_test = X_test
                        st.session_state.model_name = model_choice
                        st.session_state.problem_type = problem_type
                        
                        # Success message
                        success_metric = list(metrics.items())[0]
                        st.success(f"✅ Model trained successfully! {success_metric[0].replace('_', ' ').title()}: {success_metric[1]:.3f}")
                        
                        # Show metrics
                        st.markdown("### 📊 Training Results")
                        metric_cols = st.columns(len(metrics))
                        
                        for i, (metric_name, metric_value) in enumerate(metrics.items()):
                            with metric_cols[i]:
                                st.metric(
                                    metric_name.replace('_', ' ').title(),
                                    f"{metric_value:.3f}"
                                )
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Training failed: {str(e)}")
                        with st.expander("Show error details"):
                            st.text(traceback.format_exc())
        
        # Model download section
        if st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### 📦 Download Trained Model")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                model_package = create_model_package(
                    st.session_state.trained_model,
                    st.session_state.trained_feature_cols,
                    st.session_state.trained_le_target,
                    st.session_state.model_name,
                    st.session_state.test_results['metrics']
                )
                
                filename = f"automl_model_{st.session_state.model_name.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                
                st.download_button(
                    label="📦 Download Complete Model Package",
                    data=model_package,
                    file_name=filename,
                    mime="application/octet-stream",
                    help="Download model with preprocessing pipeline and metadata"
                )
            
            with download_col2:
                st.info("""
                **Package includes:**
                - Trained model + preprocessing pipeline
                - Feature columns and target encoder
                - Performance metrics
                - Metadata and timestamp
                """)

# Tab 3: Enhanced SHAP Insights
with tabs[2]:
    st.markdown("## 🔍 Model Insights & SHAP Explanations")
    
    if not st.session_state.model_trained:
        st.info("Train a model first to make predictions on new data")
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        problem_type = st.session_state.problem_type
        
        st.markdown("### 📁 Upload New Data for Predictions")
        
        pred_file = st.file_uploader(
            "Upload data with same features as training",
            type=["csv", "xlsx", "parquet"],
            key="prediction_file",
            help="Ensure the file contains the same features used during training"
        )
        
        if pred_file:
            try:
                # Load prediction data
                if pred_file.name.lower().endswith('.csv'):
                    new_data = pd.read_csv(pred_file)
                elif pred_file.name.lower().endswith('.parquet'):
                    new_data = pd.read_parquet(pred_file)
                else:
                    new_data = pd.read_excel(pred_file)
                
                st.markdown("### 📊 Prediction Data Preview")
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Feature validation
                missing_features = set(feature_cols) - set(new_data.columns)
                extra_features = set(new_data.columns) - set(feature_cols)
                
                validation_col1, validation_col2 = st.columns(2)
                
                with validation_col1:
                    if missing_features:
                        st.error(f"❌ Missing features: {list(missing_features)}")
                    else:
                        st.success("✅ All required features present!")
                
                with validation_col2:
                    if extra_features:
                        st.warning(f"⚠️ Extra columns will be ignored: {len(extra_features)} columns")
                    st.info(f"📊 Ready to predict on {len(new_data):,} rows")
                
                if not missing_features:
                    # Prediction options
                    options_col1, options_col2 = st.columns(2)
                    
                    with options_col1:
                        include_probabilities = st.checkbox(
                            "Include prediction probabilities",
                            value=True if problem_type == "classification" else False,
                            disabled=problem_type == "regression",
                            help="Include probability scores for each class (classification only)"
                        )
                    
                    with options_col2:
                        batch_size = st.selectbox(
                            "Batch processing size",
                            [1000, 5000, 10000],
                            index=1,
                            help="Process large datasets in batches"
                        )
                    
                    # Generate predictions
                    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                        prediction_data = new_data[feature_cols]
                        
                        with st.spinner("Making predictions..."):
                            try:
                                # Batch processing for large datasets
                                all_predictions = []
                                all_probabilities = []
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i in range(0, len(prediction_data), batch_size):
                                    batch = prediction_data.iloc[i:i+batch_size]
                                    
                                    # Update progress
                                    progress = min((i + len(batch)) / len(prediction_data), 1.0)
                                    progress_bar.progress(progress)
                                    status_text.info(f"Processing batch: {i+1:,} to {min(i+batch_size, len(prediction_data)):,}")
                                    
                                    # Make predictions
                                    batch_preds = model.predict(batch)
                                    all_predictions.extend(batch_preds)
                                    
                                    # Get probabilities if requested and available
                                    if include_probabilities and hasattr(model, 'predict_proba'):
                                        try:
                                            batch_probs = model.predict_proba(batch)
                                            all_probabilities.append(batch_probs)
                                        except Exception:
                                            pass
                                
                                progress_bar.progress(1.0)
                                status_text.success("✅ Predictions completed!")
                                
                                # Combine results
                                predictions = np.array(all_predictions)
                                
                                if all_probabilities:
                                    probabilities = np.vstack(all_probabilities)
                                else:
                                    probabilities = None
                                
                                # Create results dataframe
                                results_df = new_data.copy()
                                
                                # Add predictions
                                if le_target:
                                    results_df['prediction'] = le_target.inverse_transform(predictions)
                                    results_df['prediction_encoded'] = predictions
                                else:
                                    results_df['prediction'] = predictions
                                
                                # Add probabilities
                                if probabilities is not None:
                                    if le_target:
                                        class_names = le_target.classes_
                                    else:
                                        class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
                                    
                                    for i, class_name in enumerate(class_names):
                                        results_df[f'prob_{class_name}'] = probabilities[:, i]
                                
                                st.success(f"🎉 Predictions completed for {len(results_df):,} rows!")
                                
                                # Results summary
                                st.markdown("### 📊 Prediction Results Summary")
                                
                                summary_cols = st.columns(4)
                                
                                with summary_cols[0]:
                                    st.metric("Total Predictions", f"{len(results_df):,}")
                                
                                with summary_cols[1]:
                                    unique_preds = results_df['prediction'].nunique()
                                    st.metric("Unique Predictions", unique_preds)
                                
                                with summary_cols[2]:
                                    if problem_type == "classification":
                                        most_common = results_df['prediction'].mode().iloc[0]
                                        st.metric("Most Common", str(most_common))
                                    else:
                                        pred_mean = results_df['prediction'].mean()
                                        st.metric("Mean Prediction", f"{pred_mean:.3f}")
                                
                                with summary_cols[3]:
                                    if probabilities is not None:
                                        max_confidence = probabilities.max(axis=1).mean()
                                        st.metric("Avg. Confidence", f"{max_confidence:.3f}")
                                    else:
                                        pred_std = results_df['prediction'].std()
                                        st.metric("Std. Deviation", f"{pred_std:.3f}")
                                
                                # Sample results
                                st.markdown("### 🔍 Sample Results")
                                display_cols = ['prediction']
                                if probabilities is not None:
                                    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
                                    display_cols.extend(prob_cols[:5])  # Show first 5 probability columns
                                
                                # Add original features for context
                                context_cols = feature_cols[:3] if len(feature_cols) > 3 else feature_cols
                                display_cols = context_cols + display_cols
                                
                                st.dataframe(results_df[display_cols].head(10), use_container_width=True)
                                
                                # Prediction distribution visualization
                                if problem_type == "classification" and results_df['prediction'].nunique() <= 20:
                                    st.markdown("### 📈 Prediction Distribution")
                                    pred_dist = results_df['prediction'].value_counts()
                                    
                                    fig_dist = px.bar(
                                        x=pred_dist.index,
                                        y=pred_dist.values,
                                        title="Prediction Distribution",
                                        labels={'x': 'Prediction', 'y': 'Count'}
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                
                                elif problem_type == "regression":
                                    st.markdown("### 📈 Prediction Distribution")
                                    fig_hist = px.histogram(
                                        results_df['prediction'],
                                        nbins=50,
                                        title="Prediction Distribution"
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                # Download options
                                st.markdown("### 📥 Download Results")
                                
                                download_cols = st.columns(3)
                                
                                with download_cols[0]:
                                    # CSV download
                                    csv_buffer = io.StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    
                                    st.download_button(
                                        "📄 Download as CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with download_cols[1]:
                                    # Excel download
                                    excel_buffer = io.BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                        
                                        # Add summary sheet
                                        summary_data = {
                                            'Metric': ['Total Predictions', 'Unique Predictions', 'Model Used', 'Timestamp'],
                                            'Value': [
                                                len(results_df),
                                                results_df['prediction'].nunique(),
                                                st.session_state.model_name,
                                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    st.download_button(
                                        "📊 Download as Excel",
                                        data=excel_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                
                                with download_cols[2]:
                                    # Parquet download for large files
                                    if len(results_df) > 5000:
                                        parquet_buffer = io.BytesIO()
                                        results_df.to_parquet(parquet_buffer, index=False)
                                        
                                        st.download_button(
                                            "⚡ Download as Parquet",
                                            data=parquet_buffer.getvalue(),
                                            file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                                            mime="application/octet-stream",
                                            help="Recommended for large datasets"
                                        )
                            
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {e}")
                                with st.expander("Show error details"):
                                    st.text(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ Failed to load prediction file: {e}")
                with st.expander("Show error details"):
                    st.text(traceback.format_exc())

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**AutoML Analytics Pro v2.0**")
    st.caption("Built with Streamlit & your complete ML stack")

with footer_col2:
    if st.session_state.model_trained:
        st.markdown("**Current Model:**")
        st.caption(f"{st.session_state.model_name} ({st.session_state.problem_type})")
    else:
        st.markdown("**Status:**")
        st.caption("Ready for model training")

with footer_col3:
    st.markdown("**Your Libraries:**")
    active_libs = ["SHAP", "Plotly", "PyArrow", "ImbalancedLearn"]
    st.caption(" • ".join(active_libs))

# Quick start guide in sidebar
with st.sidebar:
    if not st.session_state.data_uploaded:
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Upload your dataset** above
        2. **Explore data** in Data Explorer tab
        3. **Train models** in Model Lab tab  
        4. **Understand results** with SHAP insights
        5. **Make predictions** on new data
        """)
        
        st.markdown("**Supported formats:**")
        st.markdown("• CSV, Excel, Parquet files")
        st.markdown("• Classification & Regression")
        st.markdown("• Automatic preprocessing")
    
    else:
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        df = st.session_state.data
        st.markdown(f"""
        **File:** {st.session_state.current_file}  
        **Rows:** {df.shape[0]:,}  
        **Columns:** {df.shape[1]}  
        **Size:** {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB  
        **Missing:** {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
        """)
        
        if st.session_state.model_trained:
            st.markdown("### 🤖 Model Status")
            metrics = st.session_state.test_results['metrics']
            primary_metric = list(metrics.items())[0]
            st.metric(
                primary_metric[0].replace('_', ' ').title(),
                f"{primary_metric[1]:.3f}"
            )
        st.info("Train a model first to see comprehensive insights and explanations")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.test_results['y_test']
        y_pred = st.session_state.test_results['y_pred']
        model_name = st.session_state.model_name
        problem_type = st.session_state.problem_type
        metrics = st.session_state.test_results['metrics']
        
        # Performance overview with Plotly
        st.markdown("### 📊 Model Performance Overview")
        
        if problem_type == "classification":
            # Confusion matrix with Plotly
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual", color="Count")
            )
            
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
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='blue', width=3)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_roc.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate'
                    )
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"ROC curve generation failed: {e}")
        
        else:  # Regression
            # Actual vs Predicted plot
            fig_scatter = px.scatter(
                x=y_test, y=y_pred,
                title="Actual vs Predicted Values",
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Residuals plot
            residuals = y_test - y_pred
            fig_residuals = px.scatter(
                x=y_pred, y=residuals,
                title="Residuals Plot",
                labels={'x': 'Predicted', 'y': 'Residuals'}
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # SHAP Explanations using your SHAP 0.46.0
        st.markdown("### 🎯 SHAP Model Explanations")
        
        with st.spinner("Generating SHAP explanations with your SHAP 0.46.0..."):
            shap_values, explainer = create_shap_explanations(model, X_test, model_name)
            
            if shap_values is not None:
                try:
                    # SHAP summary plot
                    st.markdown("#### 📊 Feature Importance Summary")
                    
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    
                    if hasattr(shap_values, 'values'):  # SHAP 0.46.0 format
                        shap.summary_plot(shap_values.values, X_test.iloc[:len(shap_values.values)], 
                                        feature_names=X_test.columns, show=False, ax=ax)
                    else:  # Older format
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            shap.summary_plot(shap_values[1], X_test.iloc[:len(shap_values[1])], 
                                            feature_names=X_test.columns, show=False, ax=ax)
                        else:
                            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                            shap.summary_plot(shap_vals, X_test.iloc[:len(shap_vals)], 
                                            feature_names=X_test.columns, show=False, ax=ax)
                    
                    plt.title("SHAP Feature Importance Summary")
                    st.pyplot(fig_shap)
                    
                    st.markdown("""
                    **How to interpret:**
                    - Each dot represents a sample
                    - X-axis shows SHAP value (impact on prediction)
                    - Color indicates feature value (red=high, blue=low)
                    - Features are ranked by importance (top to bottom)
                    """)
                    
                    # Waterfall plot for individual prediction
                    st.markdown("#### 🌊 Individual Prediction Explanation")
                    
                    sample_idx = st.slider("Choose sample to explain", 0, min(20, len(X_test)-1), 0)
                    
                    try:
                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                        
                        if hasattr(shap_values, 'values'):
                            sample_shap = shap_values.values[sample_idx]
                        else:
                            if isinstance(shap_values, list):
                                sample_shap = shap_values[1][sample_idx] if len(shap_values) > 1 else shap_values[0][sample_idx]
                            else:
                                sample_shap = shap_values[sample_idx]
                        
                        # Create waterfall plot
                        sorted_idx = np.argsort(np.abs(sample_shap))[-10:]  # Top 10 features
                        
                        colors = ['red' if val > 0 else 'blue' for val in sample_shap[sorted_idx]]
                        bars = ax_waterfall.barh(range(len(sorted_idx)), sample_shap[sorted_idx], color=colors)
                        
                        ax_waterfall.set_yticks(range(len(sorted_idx)))
                        ax_waterfall.set_yticklabels([X_test.columns[i] for i in sorted_idx])
                        ax_waterfall.set_xlabel("SHAP Value (impact on prediction)")
                        ax_waterfall.set_title(f"SHAP Explanation for Sample {sample_idx + 1}")
                        ax_waterfall.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        
                        st.pyplot(fig_waterfall)
                        
                        # Show actual feature values for this sample
                        st.markdown("**Feature values for this sample:**")
                        sample_features = X_test.iloc[sample_idx]
                        feature_df = pd.DataFrame({
                            'Feature': sample_features.index,
                            'Value': sample_features.values
                        })
                        st.dataframe(feature_df, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Waterfall plot failed: {e}")
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {e}")
            else:
                st.warning("SHAP explanations could not be generated for this model type")
        
        # Feature importance fallback
        st.markdown("### ⭐ Feature Importance Analysis")
        
        try:
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('model', model)
            else:
                classifier = model
            
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
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
                }).sort_values('Importance', ascending=True).tail(15)  # Top 15
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importances"
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
            elif hasattr(classifier, 'coef_'):
                # Linear models
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
                
                fig_coef = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Coefficients (Absolute)"
                )
                
                st.plotly_chart(fig_coef, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Feature importance analysis failed: {e}")

# Tab 4: Enhanced Predictions
with tabs[3]:
    st.markdown("## 📈 Make Predictions")
    
    if not st.session_state.model_trained:
        
    except Exception as e:
        st.error(f"Model serialization failed: {e}")
        # Fallback: save without the model object
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
        return buffer.getvalue()cols,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now().isoformat(),
        'version': '2.0_safe'
    }
    
    # Try to include model
    try:
        # Test if model can be pickled
        test_buffer = io.BytesIO()
        joblib.dump(model, test_buffer)
        safe_package['model'] = model
    except Exception:
        st.warning("Model object could not be serialized. Saving model parameters instead.")
        safe_package['model_info'] = {
            'type': str(type(model)),
            'parameters': getattr(model, 'get_params', lambda: {})()
        }
    
    # Try to include label encoder
    try:
        if le_target is not None:
            test_buffer = io.BytesIO()
            joblib.dump(le_target, test_buffer)
            safe_package['label_encoder'] = le_target
        else:
            safe_package['label_encoder'] = None
    except Exception:
        st.warning("Label encoder could not be serialized. Saving classes instead.")
        if le_target is not None and hasattr(le_target, 'classes_'):
            safe_package['label_encoder_classes'] = le_target.classes_.tolist()
        else:
            safe_package['label_encoder_classes'] = None
    
    return safe_package

def create_minimal_model_package(model_name, metrics):
    """Create minimal package when all else fails"""
    minimal_info = {
        'model_name': model_name,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now().isoformat(),
        'version': '2.0_minimal',
        'note': 'This is a minimal package due to serialization constraints. The trained model could not be included.'
    }
    
    buffer = io.BytesIO()
    joblib.dump(minimal_info, buffer)
    buffer.seek(0)
    return buffer.getvalue()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Analytics Platform</h1>
    <p>Professional predictive modeling with advanced explanations and automated optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced status
with st.sidebar:
    st.markdown("### 📊 Platform Status")
    
    # Status indicators
    status_data = [
        ("Data", st.session_state.data_uploaded, "✅" if st.session_state.data_uploaded else "⏳"),
        ("Model", st.session_state.model_trained, "✅" if st.session_state.model_trained else "⏳"),
        ("Optimization", st.session_state.hyperopt_completed, "✅" if st.session_state.hyperopt_completed else "⏳")
    ]
    
    for label, status, icon in status_data:
        badge_class = "status-success" if status else "status-info"
        st.markdown(f"""
        <div class="status-badge {badge_class}">
            {icon} {label}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset for analysis"
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
                st.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Library status
    st.markdown("---")
    st.markdown("### 🔧 Available Features")
    
    features_status = [
        ("SHAP Explanations", SHAP_AVAILABLE),
        ("Hyperopt (Optuna)", OPTUNA_AVAILABLE),
        ("Data Profiling", ProfileReport is not None),
        ("Class Balancing", IMB_AVAILABLE),
        ("Enhanced Serialization", CLOUDPICKLE_AVAILABLE)
    ]
    
    for feature, available in features_status:
        icon = "✅" if available else "❌"
        st.markdown(f"{icon} {feature}")

# Main tabs with enhanced navigation
tab_names = ["📊 Data Explorer", "🚀 Model Lab", "🔍 Model Insights", "📈 Predictions"]
tabs = st.tabs(tab_names)

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("## 📊 Data Explorer & Profiling")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload a dataset in the sidebar to start exploring")
    else:
        df = st.session_state.data
        
        # Data overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.1f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Data preview with pagination
        st.markdown("### 🔍 Data Preview")
        
        # Pagination controls
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_pages = (len(df) - 1) // page_size + 1
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            display_df = df.iloc[start_idx:end_idx]
        else:
            display_df = df.head(page_size)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Column information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Column Info")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Data Types Distribution")
            dtype_counts = df.dtypes.value_counts()
            st.bar_chart(dtype_counts)
        
        # Profiling report
        st.markdown("### 📈 Automated Profiling Report")
        
        profile_options = st.columns(3)
        with profile_options[0]:
            profile_type = st.radio("Report Type", ["Quick", "Detailed"], index=0)
        with profile_options[1]:
            if st.button("Generate Profile Report", type="primary"):
                with st.spinner("Generating profile report..."):
                    try:
                        minimal = (profile_type == "Quick")
                        html_content = generate_profile_report(df, minimal=minimal)
                        st.session_state.profile_html = html_content
                        st.session_state.profile_generated = True
                        st.success("Profile report generated successfully!")
                    except Exception as e:
                        st.error(f"Profile generation failed: {e}")
                        # Generate basic fallback
                        st.session_state.profile_html = generate_basic_profile_html(df)
                        st.session_state.profile_generated = True
        
        if st.session_state.get('profile_generated', False):
            try:
                components.html(st.session_state.profile_html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Failed to display profile: {e}")
                st.text("Profile HTML could not be rendered.")

# Tab 2: Model Lab
with tabs[1]:
    st.markdown("## 🚀 Model Development Lab")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first to start model development")
    else:
        data = st.session_state.data.copy()
        
        # Model configuration section
        st.markdown("### ⚙️ Model Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            # Target selection
            all_columns = list(data.columns)
            target_column = st.selectbox(
                "🎯 Target Variable",
                all_columns,
                help="Select the column you want to predict"
            )
        
        with config_col2:
            # Feature selection
            available_features = [col for col in all_columns if col != target_column]
            feature_mode = st.radio(
                "🔧 Feature Selection",
                ["Auto-select all", "Manual selection"],
                horizontal=True
            )
        
        if feature_mode == "Manual selection":
            feature_cols = st.multiselect(
                "Select features",
                available_features,
                default=available_features
            )
            if not feature_cols:
                st.warning("Please select at least one feature")
                st.stop()
        else:
            feature_cols = available_features
        
        # Prepare data
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()
        
        # Handle target encoding
        if y_raw.dtype == "object" or y_raw.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_.tolist()
            problem_type = "classification"
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
        
        # Target distribution
        st.markdown("### 📊 Target Variable Analysis")
        target_col1, target_col2 = st.columns(2)
        
        with target_col1:
            if problem_type == "classification":
                target_dist = pd.Series(y).value_counts().sort_index()
                st.bar_chart(target_dist)
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(y, bins=30, alpha=0.7)
                ax.set_title("Target Distribution")
                st.pyplot(fig)
        
        with target_col2:
            if problem_type == "classification":
                class_dist = pd.DataFrame({
                    'Class': class_names,
                    'Count': pd.Series(y).value_counts().sort_index().values,
                    'Percentage': (pd.Series(y).value_counts().sort_index().values / len(y) * 100).round(2)
                })
                st.dataframe(class_dist, use_container_width=True)
            else:
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std', 'Min', 'Max'],
                    'Value': [y.mean(), np.median(y), y.std(), y.min(), y.max()]
                })
                st.dataframe(stats_df, use_container_width=True)
        
        # Model selection and configuration
        st.markdown("### 🤖 Algorithm Selection")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            model_options = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }
            
            model_choice = st.selectbox("Select Algorithm", list(model_options.keys()), index=1)
            base_model = model_options[model_choice]
        
        with model_col2:
            # Hyperparameter optimization toggle
            use_hyperopt = st.checkbox(
                "🔍 Enable Hyperparameter Optimization",
                value=OPTUNA_AVAILABLE,
                disabled=not OPTUNA_AVAILABLE,
                help="Automatically find the best parameters for your model"
            )
            
            if use_hyperopt and OPTUNA_AVAILABLE:
                n_trials = st.slider("Optimization Trials", 10, 100, 50, step=10)
        
        # Advanced options
        with st.expander("🔧 Advanced Configuration"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                # Class balancing
                balance_options = ["None"]
                if IMB_AVAILABLE:
                    balance_options.extend(["SMOTE", "Random Oversample", "Random Undersample"])
                
                balance_method = st.selectbox("Class Balancing", balance_options)
                
                # Train/test split
                test_size = st.slider("Test Size (%)", 10, 50, 20, step=5)
            
            with adv_col2:
                # Cross-validation
                cv_folds = st.slider("CV Folds", 3, 10, 5)
                
                # Random seed
                random_seed = st.number_input("Random Seed", 0, 999, 42)
        
        # Training section
        st.markdown("### 🏃‍♂️ Model Training")
        
        train_col1, train_col2 = st.columns(2)
        
        with train_col1:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                if len(np.unique(y)) < 2:
                    st.error("Need at least 2 classes in target variable")
                else:
                    training_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=random_seed,
                            stratify=y if problem_type == "classification" else None
                        )
                        
                        progress_bar.progress(25)
                        training_placeholder.info("🔄 Preparing data pipeline...")
                        
                        # Create preprocessing pipeline
                        numeric_features = X.select_dtypes(include=[np.number]).columns
                        categorical_features = X.select_dtypes(exclude=[np.number]).columns
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_features),
                                ('cat', safe_onehot_encoder(), categorical_features)
                            ]
                        )
                        
                        # Handle class balancing
                        pipeline_steps = [('preprocessor', preprocessor)]
                        
                        if balance_method != "None" and IMB_AVAILABLE:
                            if balance_method == "SMOTE":
                                sampler = SMOTE(random_state=random_seed)
                            elif balance_method == "Random Oversample":
                                sampler = RandomOverSampler(random_state=random_seed)
                            elif balance_method == "Random Undersample":
                                sampler = RandomUnderSampler(random_state=random_seed)
                            pipeline_steps.append(('sampler', sampler))
                        
                        progress_bar.progress(50)
                        
                        # Hyperparameter optimization
                        if use_hyperopt and OPTUNA_AVAILABLE:
                            training_placeholder.info("🔍 Optimizing hyperparameters...")
                            
                            best_params = optimize_hyperparameters(X_train, y_train, model_choice, n_trials)
                            st.session_state.best_params = best_params
                            st.session_state.hyperopt_completed = True
                            
                            # Update model with best parameters
                            base_model.set_params(**best_params)
                        
                        progress_bar.progress(75)
                        training_placeholder.info("🏋️‍♂️ Training model...")
                        
                        # Add model to pipeline
                        pipeline_steps.append(('classifier', base_model))
                        
                        if IMB_AVAILABLE and any('sampler' in step[0] for step in pipeline_steps):
                            model_pipeline = ImbPipeline(pipeline_steps)
                        else:
                            model_pipeline = Pipeline(pipeline_steps)
                        
                        # Train model
                        model_pipeline.fit(X_train, y_train)
                        
                        # Evaluate model
                        y_pred = model_pipeline.predict(X_test)
                        
                        if problem_type == "classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            metrics = {'accuracy': accuracy, 'f1_score': f1}
                        else:
                            from sklearn.metrics import mean_squared_error, r2_score
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            metrics = {'mse': mse, 'r2_score': r2}
                        
                        progress_bar.progress(100)
                        training_placeholder.empty()
                        
                        # Save model results to session state
                        st.session_state.model_trained = True
                        st.session_state.trained_model = model_pipeline
                        st.session_state.trained_feature_cols = feature_cols
                        st.session_state.trained_le_target = le_target
                        st.session_state.trained_class_names = class_names
                        st.session_state.test_results = {
                            'y_test': y_test, 'y_pred': y_pred, 'metrics': metrics
                        }
                        st.session_state.trained_X_test = X_test
                        st.session_state.model_name = model_choice
                        st.session_state.problem_type = problem_type
                        
                        # Simple experiment logging (without MLflow)
                        experiment_record = {
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'algorithm': model_choice,
                            'features': len(feature_cols),
                            'balance_method': balance_method,
                            'test_size': test_size,
                            'metrics': metrics,
                            'hyperopt_used': use_hyperopt and st.session_state.best_params
                        }
                        
                        if 'experiment_history' not in st.session_state:
                            st.session_state.experiment_history = []
                        st.session_state.experiment_history.append(experiment_record)
                        
                        # Display success message
                        success_metric = list(metrics.items())[0]
                        st.success(f"✅ Model trained successfully! {success_metric[0].replace('_', ' ').title()}: {success_metric[1]:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        with st.expander("Show error details"):
                            st.text(traceback.format_exc())
        
        with train_col2:
            # Model download section
            if st.session_state.model_trained:
                st.markdown("**Download Trained Model**")
                
                model_package = create_model_download_package(
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
                    mime="application/octet-stream",
                    help="Download complete model package with preprocessing pipeline"
                )
        
        # Training results display
        if st.session_state.model_trained:
            st.markdown("### 📊 Training Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            metrics = st.session_state.test_results['metrics']
            metric_items = list(metrics.items())
            
            with results_col1:
                st.metric(metric_items[0][0].replace('_', ' ').title(), f"{metric_items[0][1]:.3f}")
            
            if len(metric_items) > 1:
                with results_col2:
                    st.metric(metric_items[1][0].replace('_', ' ').title(), f"{metric_items[1][1]:.3f}")
            
            with results_col3:
                st.metric("Features Used", len(st.session_state.trained_feature_cols))
            
            # Hyperparameter optimization results
            if st.session_state.hyperopt_completed and st.session_state.best_params:
                st.markdown("#### 🔍 Optimized Hyperparameters")
                params_df = pd.DataFrame(
                    list(st.session_state.best_params.items()),
                    columns=['Parameter', 'Value']
                )
                st.dataframe(params_df, use_container_width=True)

# Tab 3: Model Insights
with tabs[2]:
    st.markdown("## 🔍 Model Insights & Explainability")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data and train a model first to see insights")
    elif not st.session_state.model_trained:
        st.info("Train a model in the Model Lab to see detailed insights")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.test_results['y_test']
        y_pred = st.session_state.test_results['y_pred']
        model_name = st.session_state.model_name
        problem_type = st.session_state.problem_type
        
        # Model performance overview
        st.markdown("### 📈 Performance Overview")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            if problem_type == "classification":
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_xlabel('Predicted')
                st.pyplot(fig_cm)
        
        with perf_col2:
            # Classification Report
            if problem_type == "classification":
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.markdown("**Classification Report**")
                st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve for binary classification
        if problem_type == "classification" and len(np.unique(y_test)) == 2:
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                ax_roc.grid(True, alpha=0.3)
                st.pyplot(fig_roc)
            except Exception:
                pass
        
        st.markdown("---")
        
        # SHAP Explanations
        if SHAP_AVAILABLE:
            st.markdown("### 🎯 SHAP Model Explanations")
            
            with st.spinner("Generating SHAP explanations..."):
                shap_values, feature_names = generate_shap_values(
                    model, X_test.head(100), model_name
                )
                
                if shap_values is not None:
                    fig_summary, fig_waterfall = create_shap_plots(
                        shap_values, feature_names, X_test.head(100)
                    )
                    
                    shap_col1, shap_col2 = st.columns(2)
                    
                    with shap_col1:
                        st.markdown("**Feature Importance Summary**")
                        st.pyplot(fig_summary)
                        
                        st.markdown("""
                        **How to read this plot:**
                        - Each dot is a sample from your test data
                        - X-axis shows impact on model prediction
                        - Color shows feature value (red=high, blue=low)
                        - Features ranked by importance (top to bottom)
                        """)
                    
                    with shap_col2:
                        st.markdown("**Individual Prediction Breakdown**")
                        st.pyplot(fig_waterfall)
                        
                        st.markdown("""
                        **How to read this plot:**
                        - Shows how each feature contributes to first prediction
                        - Positive values push prediction higher
                        - Negative values push prediction lower
                        - Helps understand individual decisions
                        """)
                else:
                    st.warning("SHAP explanations not available for this model type")
        else:
            st.info("Install SHAP (`pip install shap`) for advanced model explanations")
        
        # Feature Importance (fallback)
        st.markdown("### ⭐ Feature Importance Analysis")
        
        try:
            classifier = model.named_steps.get('classifier', model)
            
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
                importances = classifier.feature_importances_
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax_imp)
                ax_imp.set_title('Top 15 Feature Importances')
                plt.tight_layout()
                st.pyplot(fig_imp)
                
            elif hasattr(classifier, 'coef_'):
                # Linear models
                coefficients = classifier.coef_
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': np.abs(coefficients)
                }).sort_values('Coefficient', ascending=False).head(15)
                
                fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
                sns.barplot(data=coef_df, y='Feature', x='Coefficient', ax=ax_coef)
                ax_coef.set_title('Top 15 Feature Coefficients (Absolute)')
                plt.tight_layout()
                st.pyplot(fig_coef)
            
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")
        
        # Interactive What-If Analysis
        st.markdown("### 🔧 What-If Analysis")
        st.markdown("Modify feature values to see how predictions change")
        
        # Select a sample for what-if analysis
        sample_idx = st.selectbox(
            "Choose sample for analysis",
            range(min(20, len(X_test))),
            format_func=lambda x: f"Sample {x+1}"
        )
        
        sample_data = X_test.iloc[sample_idx].copy()
        
        # Create form for feature modification
        with st.form("whatif_analysis"):
            st.markdown("**Modify feature values:**")
            
            whatif_cols = st.columns(3)
            modified_data = sample_data.copy()
            
            # Show only top 10 most important features for UI simplicity
            display_features = st.session_state.trained_feature_cols[:10]
            
            for i, feature in enumerate(display_features):
                col_idx = i % 3
                
                with whatif_cols[col_idx]:
                    if X_test[feature].dtype in ['int64', 'float64']:
                        min_val = float(X_test[feature].min())
                        max_val = float(X_test[feature].max())
                        current_val = float(sample_data[feature])
                        
                        new_val = st.slider(
                            f"{feature}",
                            min_val, max_val, current_val,
                            key=f"whatif_{feature}"
                        )
                        modified_data[feature] = new_val
                    else:
                        unique_vals = X_test[feature].unique()
                        current_val = sample_data[feature]
                        
                        new_val = st.selectbox(
                            f"{feature}",
                            unique_vals,
                            index=list(unique_vals).index(current_val) if current_val in unique_vals else 0,
                            key=f"whatif_{feature}"
                        )
                        modified_data[feature] = new_val
            
            submitted = st.form_submit_button("🔍 Predict with modified values")
            
            if submitted:
                try:
                    # Make prediction with modified data
                    modified_df = pd.DataFrame([modified_data])
                    original_pred = model.predict(pd.DataFrame([sample_data]))[0]
                    modified_pred = model.predict(modified_df)[0]
                    
                    # Show results
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        if st.session_state.trained_le_target:
                            original_label = st.session_state.trained_le_target.inverse_transform([original_pred])[0]
                            st.metric("Original Prediction", original_label)
                        else:
                            st.metric("Original Prediction", f"{original_pred:.3f}")
                    
                    with result_col2:
                        if st.session_state.trained_le_target:
                            modified_label = st.session_state.trained_le_target.inverse_transform([modified_pred])[0]
                            st.metric("Modified Prediction", modified_label)
                        else:
                            st.metric("Modified Prediction", f"{modified_pred:.3f}")
                    
                    with result_col3:
                        if problem_type == "classification":
                            change = "Changed" if original_pred != modified_pred else "Same"
                            st.metric("Result", change)
                        else:
                            change = modified_pred - original_pred
                            st.metric("Change", f"{change:+.3f}")
                    
                    # Show probability changes for classification
                    if problem_type == "classification" and hasattr(model, 'predict_proba'):
                        original_proba = model.predict_proba(pd.DataFrame([sample_data]))[0]
                        modified_proba = model.predict_proba(modified_df)[0]
                        
                        if st.session_state.trained_class_names:
                            proba_df = pd.DataFrame({
                                'Class': st.session_state.trained_class_names,
                                'Original': original_proba,
                                'Modified': modified_proba,
                                'Change': modified_proba - original_proba
                            })
                            st.dataframe(proba_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Tab 4: Predictions
with tabs[3]:
    st.markdown("## 📈 Make Predictions")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data and train a model first")
    elif not st.session_state.model_trained:
        st.info("Train a model in the Model Lab first")
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        
        st.markdown("### 📁 Upload New Data for Predictions")
        
        pred_file = st.file_uploader(
            "Upload data with same features as training",
            type=["csv", "xlsx"],
            key="prediction_file",
            help="Make sure the file contains the same features used in training"
        )
        
        if pred_file:
            try:
                # Load prediction data
                if pred_file.name.lower().endswith('.csv'):
                    new_data = pd.read_csv(pred_file)
                else:
                    new_data = pd.read_excel(pred_file)
                
                st.markdown("### 📊 Data Preview")
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Check for missing features
                missing_features = set(feature_cols) - set(new_data.columns)
                extra_features = set(new_data.columns) - set(feature_cols)
                
                status_col1, status_col2 = st.columns(2)
                
                with status_col1:
                    if missing_features:
                        st.error(f"Missing features: {list(missing_features)}")
                    else:
                        st.success("All required features present!")
                
                with status_col2:
                    if extra_features:
                        st.warning(f"Extra columns will be ignored: {len(extra_features)} columns")
                    st.info(f"Ready to predict on {len(new_data):,} rows")
                
                if not missing_features:
                    # Prediction options
                    pred_options_col1, pred_options_col2 = st.columns(2)
                    
                    with pred_options_col1:
                        include_probabilities = st.checkbox(
                            "Include prediction probabilities",
                            value=True,
                            help="Include probability scores for each class (classification only)"
                        )
                    
                    with pred_options_col2:
                        batch_size = st.selectbox(
                            "Batch size for large files",
                            [1000, 5000, 10000, 50000],
                            index=1,
                            help="Process data in batches to manage memory"
                        )
                    
                    # Make predictions
                    if st.button("🔮 Generate Predictions", type="primary"):
                        prediction_data = new_data[feature_cols]
                        
                        with st.spinner("Making predictions..."):
                            try:
                                # Batch processing for large datasets
                                all_predictions = []
                                all_probabilities = []
                                
                                progress_bar = st.progress(0)
                                
                                for i in range(0, len(prediction_data), batch_size):
                                    batch = prediction_data.iloc[i:i+batch_size]
                                    
                                    # Update progress
                                    progress = min((i + len(batch)) / len(prediction_data), 1.0)
                                    progress_bar.progress(progress)
                                    
                                    # Make predictions
                                    batch_preds = model.predict(batch)
                                    all_predictions.extend(batch_preds)
                                    
                                    # Get probabilities if requested
                                    if include_probabilities and hasattr(model, 'predict_proba'):
                                        batch_probs = model.predict_proba(batch)
                                        all_probabilities.append(batch_probs)
                                
                                progress_bar.progress(1.0)
                                
                                # Combine results
                                predictions = np.array(all_predictions)
                                
                                if all_probabilities:
                                    probabilities = np.vstack(all_probabilities)
                                else:
                                    probabilities = None
                                
                                # Create results dataframe
                                results_df = new_data.copy()
                                
                                # Add predictions
                                if le_target:
                                    results_df['prediction'] = le_target.inverse_transform(predictions)
                                    results_df['prediction_encoded'] = predictions
                                else:
                                    results_df['prediction'] = predictions
                                
                                # Add probabilities
                                if probabilities is not None:
                                    if le_target:
                                        class_names = le_target.classes_
                                    else:
                                        class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
                                    
                                    for i, class_name in enumerate(class_names):
                                        results_df[f'prob_{class_name}'] = probabilities[:, i]
                                
                                st.success(f"Predictions completed for {len(results_df):,} rows!")
                                
                                # Show results summary
                                st.markdown("### 📊 Prediction Results")
                                
                                summary_col1, summary_col2, summary_col3 = st.columns(3)
                                
                                with summary_col1:
                                    st.metric("Total Predictions", f"{len(results_df):,}")
                                
                                with summary_col2:
                                    unique_preds = results_df['prediction'].nunique()
                                    st.metric("Unique Predictions", unique_preds)
                                
                                with summary_col3:
                                    most_common = results_df['prediction'].mode().iloc[0]
                                    st.metric("Most Common", str(most_common))
                                
                                # Show sample results
                                st.markdown("**Sample Results:**")
                                st.dataframe(results_df.head(10), use_container_width=True)
                                
                                # Prediction distribution
                                if results_df['prediction'].nunique() <= 20:
                                    st.markdown("### 📈 Prediction Distribution")
                                    pred_dist = results_df['prediction'].value_counts()
                                    st.bar_chart(pred_dist)
                                
                                # Download options
                                st.markdown("### 📥 Download Results")
                                
                                download_col1, download_col2 = st.columns(2)
                                
                                with download_col1:
                                    # CSV download
                                    csv_buffer = io.StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    
                                    st.download_button(
                                        "📄 Download as CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with download_col2:
                                    # Excel download
                                    excel_buffer = io.BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                    
                                    st.download_button(
                                        "📊 Download as Excel",
                                        data=excel_buffer.getvalue(),
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                st.text(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Failed to load prediction file: {e}")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**AutoML Platform v2.0**")
    st.caption("Built with Streamlit")

with footer_col2:
    if st.session_state.model_trained:
        st.markdown("**Current Model:**")
        st.caption(f"{st.session_state.model_name}")

with footer_col3:
    st.markdown("**Available Libraries:**")
    libs = []
    if SHAP_AVAILABLE: libs.append("SHAP")
    if OPTUNA_AVAILABLE: libs.append("Optuna") 
    if CLOUDPICKLE_AVAILABLE: libs.append("CloudPickle")
    st.caption(" • ".join(libs) if libs else "Standard ML only")

# Installation helper in sidebar
with st.sidebar:
    missing_libs = []
    if not SHAP_AVAILABLE: missing_libs.append("shap")
    if not OPTUNA_AVAILABLE: missing_libs.append("optuna") 
    if not CLOUDPICKLE_AVAILABLE: missing_libs.append("cloudpickle")
    
    if missing_libs:
        st.markdown("---")
        st.markdown("### 📦 Optional Libraries")
        st.markdown("Install for enhanced functionality:")
        
        if not SHAP_AVAILABLE:
            st.code("pip install shap", language="bash")
            st.caption("🎯 Advanced model explanations")
        
        if not OPTUNA_AVAILABLE:
            st.code("pip install optuna", language="bash") 
            st.caption("🔍 Automated hyperparameter tuning")
            
        if not CLOUDPICKLE_AVAILABLE:
            st.code("pip install cloudpickle", language="bash")
            st.caption("🔧 Better model serialization")
        
        st.markdown("**Core libraries (required):**")
        st.code("pip install streamlit scikit-learn pandas numpy matplotlib seaborn", language="bash")
        
        st.markdown("**Optional for data balancing:**")
        st.code("pip install imbalanced-learn", language="bash")
        
        st.markdown("**Optional for data profiling:**")
        st.code("pip install ydata-profiling", language="bash")
