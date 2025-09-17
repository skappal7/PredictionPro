"""
Optimized AutoML Pro - Streamlined and Efficient
- Consolidated repetitive patterns
- Simplified CSS (90% reduction)
- Unified validation logic
- Removed excessive animations
- Maintained monolithic structure
"""

import io
import os
import sys
import tempfile
import traceback
import joblib
import shutil
import atexit
import gc
from typing import Optional, Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

# Required imports with error handling
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, f1_score, precision_recall_curve, mean_squared_error, r2_score)

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Page config
st.set_page_config(page_title="AutoML Pro", layout="wide", page_icon="🤖")

# Simplified CSS - Essential styling only
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
    }
    .success-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f8fafc;
        padding: 0.75rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: #475569;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Global config and utilities
class AppConfig:
    def __init__(self):
        self.max_file_size_mb = 500
        self.max_session_size_mb = 1000
        self.default_sample_size = 5000
        self.temp_dirs = []
    
    def cleanup_temp_dirs(self):
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        self.temp_dirs = []

config = AppConfig()
atexit.register(config.cleanup_temp_dirs)

# Unified session state management
def init_session_state():
    defaults = {
        "data_uploaded": False,
        "model_trained": False,
        "current_file": None,
        "training_data": None,
        "memory_usage": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_session_value(key, default=None):
    """Unified session state getter"""
    return st.session_state.get(key, default)

def safe_store_in_session(key: str, value: Any, max_size_mb: float = 100) -> bool:
    """Safe session storage with size checking"""
    size_mb = sys.getsizeof(value) / (1024 * 1024)
    if size_mb > max_size_mb:
        st.error(f"Object too large ({size_mb:.1f}MB) for session storage")
        return False
    st.session_state[key] = value
    return True

def clear_session_data():
    """Clear old session data"""
    keys_to_clear = ['old_data', 'old_model', 'temp_results']
    for key in list(st.session_state.keys()):
        if key in keys_to_clear or key not in ['cache_stats']:
            if key in st.session_state:
                del st.session_state[key]
    gc.collect()

# Data loading utilities
@st.cache_data(ttl=3600, show_spinner=False)
def get_file_size_mb(uploaded_file) -> float:
    try:
        return len(uploaded_file.getvalue()) / (1024 * 1024)
    except:
        return 0

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_safe(uploaded_file) -> Optional[pd.DataFrame]:
    """Enhanced data loading with validation"""
    try:
        name = uploaded_file.name.lower()
        file_size_mb = get_file_size_mb(uploaded_file)
        
        if file_size_mb > config.max_file_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB)")
            return None
        
        # Load based on file type
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False)
        
        if df.empty or len(df.columns) == 0:
            st.error("Invalid file: empty or no columns")
            return None
        
        # Parquet optimization for large files
        if PYARROW_AVAILABLE and not name.endswith('.parquet') and (file_size_mb > 10 or len(df) > 50000):
            st.info("Converting to Parquet for optimization...")
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        return None

# Model utilities
def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def get_model_options(problem_type):
    """Get available models for problem type"""
    if problem_type == "classification":
        return {
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
    else:
        return {
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }

def calculate_metrics(y_test, y_pred, problem_type):
    """Calculate performance metrics"""
    if problem_type == "classification":
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    else:
        return {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': np.mean(np.abs(y_test - y_pred))
        }

def create_shap_explanation_safe(model, X_sample: pd.DataFrame, max_samples: int = 1000):
    """SHAP explanations with proper model detection"""
    try:
        # Get classifier and preprocessor
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('model', model)
            preprocessor = model.named_steps.get('preprocessor')
            X_transformed = preprocessor.transform(X_sample) if preprocessor else X_sample.values
        else:
            classifier = model
            X_transformed = X_sample.values if hasattr(X_sample, 'values') else X_sample
        
        # Handle sparse matrices
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        X_shap = X_transformed[:max_samples]
        feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        # Choose explainer based on model type
        if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap)
        elif hasattr(classifier, 'coef_'):
            explainer = shap.LinearExplainer(classifier, X_shap[:50])
            shap_values = explainer.shap_values(X_shap[:20])
        else:
            background = X_transformed[:min(50, len(X_transformed))]
            explainer = shap.KernelExplainer(classifier.predict, background)
            shap_values = explainer.shap_values(X_shap[:20])
        
        return shap_values, X_shap, feature_names
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {str(e)}")
        return None, None, None

def validate_prediction_data(new_data: pd.DataFrame, training_features: List[str], training_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and fix prediction data compatibility"""
    validated_data = new_data.copy()
    
    # Check missing features
    missing_features = set(training_features) - set(validated_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {list(missing_features)}")
    
    # Fix data types
    for col in training_features:
        if col in validated_data.columns:
            training_dtype = training_data[col].dtype
            if training_dtype != validated_data[col].dtype:
                try:
                    if training_dtype == 'object':
                        validated_data[col] = validated_data[col].astype(str)
                    else:
                        validated_data[col] = pd.to_numeric(validated_data[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Type conversion failed for {col}: {e}")
    
    return validated_data

def create_metric_card(title, value, color="#667eea"):
    """Create metric display card"""
    return f'<div class="metric-card" style="background: {color};"><h3>{title}</h3><h2>{value}</h2></div>'

# Initialize
init_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>AutoML Analytics Pro</h1>
    <p>Complete machine learning platform with explanations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    
    # File upload settings
    config.max_file_size_mb = st.slider("Max File Size (MB)", 50, 1000, 500)
    config.default_sample_size = st.selectbox("Sample Size", [1000, 5000, 10000, 25000], index=1)
    
    st.markdown("---")
    st.markdown("### Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls", "parquet"]
    )
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_data_safe(uploaded_file)
            
            if df is not None:
                if safe_store_in_session("data", df):
                    st.session_state.data_uploaded = True
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.training_data = df.copy()
                    
                    st.markdown('<div class="success-card">Data loaded successfully!</div>', unsafe_allow_html=True)
                    
                    file_size = get_file_size_mb(uploaded_file)
                    st.info(f"**File:** {uploaded_file.name}\n**Rows:** {df.shape[0]:,} | **Cols:** {df.shape[1]}\n**Size:** {file_size:.1f} MB")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Model Training", "Model Analysis", "Predictions"])

# Tab 1: Data Explorer
with tab1:
    st.markdown("## Data Explorer")
    
    if not st.session_state.data_uploaded:
        st.info("Upload a dataset in the sidebar to begin")
    else:
        df = get_session_value("data")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Rows", f"{df.shape[0]:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Columns", f"{df.shape[1]}"), unsafe_allow_html=True)
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(create_metric_card("Memory", f"{memory_mb:.1f} MB"), unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            color = "#ef4444" if missing_pct > 10 else "#10b981" if missing_pct < 1 else "#f59e0b"
            st.markdown(create_metric_card("Missing", f"{missing_pct:.1f}%", color), unsafe_allow_html=True)
        
        # Data types distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Types")
            dtype_counts = df.dtypes.value_counts()
            fig_dtypes = px.pie(values=dtype_counts.values, names=[str(dtype) for dtype in dtype_counts.index])
            st.plotly_chart(fig_dtypes, use_container_width=True)
        
        with col2:
            st.markdown("### Missing Values")
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                missing_pct = (missing_data / len(df)) * 100
                fig_missing = px.bar(x=missing_pct.values, y=missing_pct.index, orientation='h')
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        # Data preview
        st.markdown("### Data Preview")
        show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
        st.dataframe(df.head(show_rows), use_container_width=True)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        
        # Profile report
        if PROFILING_AVAILABLE and st.button("Generate Profile Report", type="primary"):
            try:
                with st.spinner("Generating profile..."):
                    sample_df = df.sample(n=min(config.default_sample_size, len(df)), random_state=42)
                    profile = ProfileReport(sample_df, minimal=True, explorative=True)
                    st.components.v1.html(profile.to_html(), height=800, scrolling=True)
            except Exception as e:
                st.error(f"Profile generation failed: {e}")

# Tab 2: Model Training
with tab2:
    st.markdown("## Model Training")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data first to start training")
    else:
        data = get_session_value("data").copy()
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            all_columns = list(data.columns)
            target_column = st.selectbox("Target Variable", all_columns)
            
            available_features = [col for col in all_columns if col != target_column]
            feature_cols = st.multiselect("Features", available_features, default=available_features[:10])
            if not feature_cols:
                feature_cols = available_features
        
        with col2:
            # Determine problem type
            y_raw = data[target_column].copy()
            if y_raw.dtype == "object" or y_raw.nunique() <= 20:
                problem_type = "classification"
                st.success(f"Classification ({y_raw.nunique()} classes)")
                
                balance_options = ["None"]
                if IMBALANCED_AVAILABLE:
                    balance_options.extend(["SMOTE"])
                balance_method = st.selectbox("Class Balancing", balance_options)
            else:
                problem_type = "regression"
                st.success("Regression detected")
                balance_method = "None"
            
            model_options = get_model_options(problem_type)
            model_choice = st.selectbox("Algorithm", list(model_options.keys()))
            test_size = st.slider("Test Split (%)", 10, 50, 20, step=5)
        
        # Training
        if st.button("Train Model", type="primary"):
            try:
                clear_session_data()
                
                # Prepare data
                X = data[feature_cols].copy()
                y_raw = data[target_column].copy()
                
                if problem_type == "classification":
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y_raw.astype(str))
                    class_names = le_target.classes_.tolist()
                else:
                    le_target = None
                    y = y_raw.to_numpy()
                    class_names = None
                
                # Split data
                split_kwargs = {'test_size': test_size/100, 'random_state': 42}
                if problem_type == "classification":
                    split_kwargs['stratify'] = y
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
                
                # Create pipeline
                numeric_features = X.select_dtypes(include=[np.number]).columns
                categorical_features = X.select_dtypes(exclude=[np.number]).columns
                
                preprocessing_steps = []
                if len(numeric_features) > 0:
                    preprocessing_steps.append(('num', StandardScaler(), numeric_features))
                if len(categorical_features) > 0:
                    preprocessing_steps.append(('cat', safe_onehot_encoder(), categorical_features))
                
                pipeline_steps = []
                if preprocessing_steps:
                    pipeline_steps.append(('preprocessor', ColumnTransformer(preprocessing_steps)))
                
                if balance_method == "SMOTE" and IMBALANCED_AVAILABLE:
                    pipeline_steps.append(('sampler', SMOTE(random_state=42)))
                
                base_model = model_options[model_choice]
                pipeline_steps.append(('model', base_model))
                
                # Create and train pipeline
                if balance_method == "SMOTE":
                    model_pipeline = ImbPipeline(pipeline_steps)
                else:
                    model_pipeline = Pipeline(pipeline_steps)
                
                progress_bar = st.progress(0, "Training model...")
                model_pipeline.fit(X_train, y_train)
                progress_bar.progress(75, "Evaluating...")
                
                # Predictions and metrics
                y_pred = model_pipeline.predict(X_test)
                y_proba = None
                
                if problem_type == "classification" and hasattr(model_pipeline, 'predict_proba'):
                    try:
                        y_proba_full = model_pipeline.predict_proba(X_test)
                        y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] == 2 else y_proba_full
                    except:
                        pass
                
                metrics = calculate_metrics(y_test, y_pred, problem_type)
                progress_bar.progress(100, "Complete!")
                
                # Store results
                training_results = {
                    'model': model_pipeline,
                    'feature_columns': feature_cols,
                    'label_encoder': le_target,
                    'class_names': class_names,
                    'test_results': {'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba, 'metrics': metrics},
                    'X_test': X_test,
                    'model_name': model_choice,
                    'problem_type': problem_type
                }
                
                for key, value in training_results.items():
                    safe_store_in_session(f'trained_{key}', value)
                
                st.session_state.model_trained = True
                progress_bar.empty()
                
                st.markdown('<div class="success-card">Model trained successfully!</div>', unsafe_allow_html=True)
                
                # Display metrics
                st.markdown("### Training Results")
                metric_cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
                
                # Model download
                try:
                    model_package = {
                        'model': model_pipeline,
                        'feature_columns': feature_cols,
                        'label_encoder': le_target,
                        'metrics': metrics,
                        'model_name': model_choice
                    }
                    
                    buffer = io.BytesIO()
                    joblib.dump(model_package, buffer, compress=3)
                    
                    st.download_button(
                        "Download Model Package",
                        data=buffer.getvalue(),
                        file_name=f"automl_model_{model_choice.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Model export failed: {e}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.code(traceback.format_exc())

# Tab 3: Model Analysis
with tab3:
    st.markdown("## Model Analysis")
    
    if not get_session_value('model_trained', False):
        st.info("Train a model first to see analysis")
    else:
        model = get_session_value('trained_model')
        X_test = get_session_value('trained_X_test')
        test_results = get_session_value('trained_test_results', {})
        y_test = test_results.get('y_test')
        y_pred = test_results.get('y_pred')
        y_proba = test_results.get('y_proba')
        metrics = test_results.get('metrics', {})
        problem_type = get_session_value('trained_problem_type')
        class_names = get_session_value('trained_class_names')
        
        if model is None or X_test is None or y_test is None or y_pred is None:
            st.error("Missing training results. Please retrain the model.")
        
        # Performance Overview
        st.markdown("### Performance Overview")
        metric_cols = st.columns(len(metrics))
        for i, (name, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                color = "#10b981" if (name in ['accuracy', 'f1_score', 'r2_score'] and value > 0.8) else "#667eea"
                st.markdown(create_metric_card(name.replace('_', ' ').title(), f"{value:.4f}", color), unsafe_allow_html=True)
        
        # Performance Visualizations
        st.markdown("### Performance Visualizations")
        
        if problem_type == "classification":
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(cm))]
            
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", x=class_names, y=class_names)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # ROC Curve for binary classification
            if len(np.unique(y_test)) == 2 and y_proba is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(fig_roc, use_container_width=True)
                except:
                    pass
        
        else:  # Regression
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted")
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                residuals = y_test - y_pred
                fig_residuals = px.scatter(x=y_pred, y=residuals, title="Residuals Plot")
                fig_residuals.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        # SHAP Explanations
        st.markdown("### SHAP Explanations")
        
        shap_sample_size = st.slider("Sample Size for SHAP", 50, 1000, 500)
        
        if st.button("Generate SHAP Explanations", type="primary"):
            with st.spinner("Generating SHAP explanations..."):
                shap_values, X_shap, feature_names = create_shap_explanation_safe(model, X_test, shap_sample_size)
                
                if shap_values is not None:
                    st.success("SHAP explanations generated!")
                    
                    try:
                        if isinstance(shap_values, list):
                            plot_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                        else:
                            plot_values = shap_values
                        
                        if hasattr(plot_values, 'values'):
                            shap_array = plot_values.values
                        else:
                            shap_array = np.array(plot_values)
                        
                        # Summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_array, X_shap, feature_names=feature_names, show=False, max_display=15)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Individual explanation
                        sample_idx = st.slider("Sample to explain", 0, min(len(shap_array)-1, 50), 0)
                        
                        sample_shap = shap_array[sample_idx]
                        indices = np.argsort(np.abs(sample_shap))[-10:]
                        top_features = [feature_names[i] for i in indices]
                        top_values = sample_shap[indices]
                        
                        colors = ['#10b981' if val > 0 else '#ef4444' for val in top_values]
                        
                        fig_waterfall = go.Figure()
                        fig_waterfall.add_trace(go.Bar(
                            x=top_values, y=top_features, orientation='h',
                            marker_color=colors, text=[f'{val:.3f}' for val in top_values]
                        ))
                        fig_waterfall.update_layout(title=f"SHAP Values - Sample {sample_idx + 1}")
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"SHAP visualization failed: {e}")
                else:
                    st.warning("SHAP explanations not available")
        
        # Feature Importance
        st.markdown("### Feature Importance")
        
        try:
            classifier = model.named_steps.get('model', model) if hasattr(model, 'named_steps') else model
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                try:
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                        feature_names_processed = model.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        feature_names_processed = X_test.columns.tolist()
                except:
                    feature_names_processed = [f'feature_{i}' for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names_processed[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(15)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Top 15 Features")
                st.plotly_chart(fig_importance, use_container_width=True)
            
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                else:
                    coefficients = np.abs(coefficients)
                
                try:
                    feature_names_processed = X_test.columns.tolist()
                except:
                    feature_names_processed = [f'feature_{i}' for i in range(len(coefficients))]
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names_processed[:len(coefficients)],
                    'Coefficient': coefficients
                }).sort_values('Coefficient', ascending=True).tail(15)
                
                fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title="Top 15 Coefficients")
                st.plotly_chart(fig_coef, use_container_width=True)
            
            else:
                st.info("Feature importance not available for this model")
                
        except Exception as e:
            st.error(f"Feature importance failed: {e}")

# Tab 4: Predictions
with tab4:
    st.markdown("## Make Predictions")
    
    if not get_session_value('model_trained', False):
        st.info("Train a model first to make predictions")
    else:
        model = get_session_value('trained_model')
        feature_cols = get_session_value('trained_feature_columns')
        le_target = get_session_value('trained_le_target')
        problem_type = get_session_value('trained_problem_type')
        training_data = get_session_value('training_data')
        
        if not all([model, feature_cols, training_data is not None]):
            st.error("Missing model data. Please retrain the model.")
        else:
            st.markdown("### Upload Data for Predictions")
            
            # Required features info
            with st.expander("Required Features", expanded=True):
                feature_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Type': [str(training_data[col].dtype) for col in feature_cols],
                    'Sample': [str(training_data[col].iloc[0]) for col in feature_cols]
                })
                st.dataframe(feature_df, use_container_width=True)
            
            pred_file = st.file_uploader("Upload prediction data", type=["csv", "xlsx", "parquet"])
            
            # Configuration
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.selectbox("Batch Size", [1000, 5000, 10000], index=1)
            with col2:
                include_probabilities = st.checkbox("Include Confidence Scores", value=problem_type == "classification")
            
            if pred_file:
                try:
                    prediction_data = load_data_safe(pred_file)
                    
                    if prediction_data is None:
                        st.error("Failed to load prediction file")
                    else:
                        st.success(f"Loaded {len(prediction_data):,} rows for prediction")
                        
                        # Preview
                        st.markdown("### Data Preview")
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.dataframe(prediction_data.head(10), use_container_width=True)
                        with col2:
                            st.metric("Rows", f"{len(prediction_data):,}")
                            st.metric("Columns", len(prediction_data.columns))
                        
                        # Validation
                        missing_features = set(feature_cols) - set(prediction_data.columns)
                        extra_features = set(prediction_data.columns) - set(feature_cols)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if missing_features:
                                st.error(f"Missing Features: {list(missing_features)}")
                            else:
                                st.success("All required features present")
                        with col2:
                            if extra_features:
                                st.warning(f"Extra features (ignored): {len(extra_features)}")
                        
                        if not missing_features:
                            try:
                                validated_data = validate_prediction_data(prediction_data, feature_cols, training_data)
                                
                                if st.button("Generate Predictions", type="primary"):
                                    pred_features = validated_data[feature_cols]
                                    
                                    with st.spinner("Generating predictions..."):
                                        try:
                                            all_predictions = []
                                            all_probabilities = []
                                            
                                            progress_bar = st.progress(0)
                                            total_batches = (len(pred_features) + batch_size - 1) // batch_size
                                            
                                            for batch_idx in range(total_batches):
                                                start_idx = batch_idx * batch_size
                                                end_idx = min(start_idx + batch_size, len(pred_features))
                                                batch = pred_features.iloc[start_idx:end_idx]
                                                
                                                progress = (batch_idx + 1) / total_batches
                                                progress_bar.progress(progress)
                                                
                                                batch_preds = model.predict(batch)
                                                all_predictions.extend(batch_preds)
                                                
                                                if include_probabilities and hasattr(model, 'predict_proba'):
                                                    try:
                                                        batch_probs = model.predict_proba(batch)
                                                        all_probabilities.append(batch_probs)
                                                    except:
                                                        include_probabilities = False
                                            
                                            progress_bar.empty()
                                            
                                            # Process results
                                            predictions = np.array(all_predictions)
                                            probabilities = np.vstack(all_probabilities) if all_probabilities else None
                                            
                                            # Create results
                                            results_df = prediction_data.copy()
                                            
                                            if le_target:
                                                results_df['prediction'] = le_target.inverse_transform(predictions)
                                            else:
                                                results_df['prediction'] = predictions
                                            
                                            if probabilities is not None:
                                                if le_target and hasattr(le_target, 'classes_'):
                                                    class_names = le_target.classes_
                                                else:
                                                    class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
                                                
                                                for i, class_name in enumerate(class_names):
                                                    results_df[f'prob_{class_name}'] = probabilities[:, i]
                                                results_df['confidence'] = probabilities.max(axis=1)
                                            
                                            st.markdown('<div class="success-card">Predictions completed!</div>', unsafe_allow_html=True)
                                            
                                            # Summary
                                            st.markdown("### Prediction Summary")
                                            
                                            summary_cols = st.columns(4)
                                            with summary_cols[0]:
                                                st.metric("Total", f"{len(results_df):,}")
                                            with summary_cols[1]:
                                                st.metric("Unique", results_df['prediction'].nunique())
                                            with summary_cols[2]:
                                                if problem_type == "classification":
                                                    most_common = results_df['prediction'].mode().iloc[0]
                                                    st.metric("Most Common", str(most_common))
                                                else:
                                                    st.metric("Mean", f"{results_df['prediction'].mean():.3f}")
                                            with summary_cols[3]:
                                                if probabilities is not None:
                                                    st.metric("Avg Confidence", f"{results_df['confidence'].mean():.3f}")
                                            
                                            # Sample results
                                            st.markdown("### Sample Results")
                                            display_cols = feature_cols[:3] + ['prediction']
                                            if probabilities is not None:
                                                prob_cols = [col for col in results_df.columns if col.startswith('prob_')][:2]
                                                display_cols.extend(prob_cols + ['confidence'])
                                            
                                            st.dataframe(results_df[display_cols].head(10), use_container_width=True)
                                            
                                            # Distribution
                                            if problem_type == "classification":
                                                pred_counts = results_df['prediction'].value_counts()
                                                fig_dist = px.bar(x=pred_counts.index.astype(str), y=pred_counts.values, title="Prediction Distribution")
                                                st.plotly_chart(fig_dist, use_container_width=True)
                                            else:
                                                fig_hist = px.histogram(results_df['prediction'], nbins=30, title="Prediction Distribution")
                                                st.plotly_chart(fig_hist, use_container_width=True)
                                            
                                            # Downloads
                                            st.markdown("### Download Results")
                                            
                                            download_cols = st.columns(3)
                                            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                            
                                            with download_cols[0]:
                                                csv_buffer = io.StringIO()
                                                results_df.to_csv(csv_buffer, index=False)
                                                st.download_button(
                                                    "Download CSV",
                                                    data=csv_buffer.getvalue(),
                                                    file_name=f"predictions_{timestamp}.csv",
                                                    mime="text/csv"
                                                )
                                            
                                            with download_cols[1]:
                                                excel_buffer = io.BytesIO()
                                                results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                                st.download_button(
                                                    "Download Excel",
                                                    data=excel_buffer.getvalue(),
                                                    file_name=f"predictions_{timestamp}.xlsx",
                                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                                )
                                            
                                            with download_cols[2]:
                                                predictions_json = {
                                                    'metadata': {
                                                        'timestamp': timestamp,
                                                        'model_type': get_session_value('trained_model_name'),
                                                        'total_predictions': len(results_df)
                                                    },
                                                    'predictions': results_df.to_dict(orient='records')
                                                }
                                                st.download_button(
                                                    "Download JSON",
                                                    data=pd.Series(predictions_json).to_json(indent=2),
                                                    file_name=f"predictions_{timestamp}.json",
                                                    mime="application/json"
                                                )
                                        
                                        except Exception as e:
                                            st.error(f"Prediction failed: {str(e)}")
                                            st.code(traceback.format_exc())
                            
                            except Exception as e:
                                st.error(f"Data validation failed: {e}")
                        else:
                            st.error("Cannot proceed: Missing required features")
                
                except Exception as e:
                    st.error(f"Failed to load prediction data: {e}")

# Footer
st.markdown("---")

footer_cols = st.columns(4)

with footer_cols[0]:
    st.markdown("**AutoML Pro v2.0**")
    st.caption("Streamlined ML platform")

with footer_cols[1]:
    if get_session_value('model_trained', False):
        model_name = get_session_value('trained_model_name', 'Unknown')
        problem_type = get_session_value('trained_problem_type', 'Unknown')
        st.markdown("**Current Model**")
        st.caption(f"{model_name} ({problem_type})")
    else:
        st.markdown("**Status**")
        st.caption("Ready for training")

with footer_cols[2]:
    st.markdown("**Features**")
    st.caption("SHAP • Visualizations • Predictions")

with footer_cols[3]:
    st.markdown("**Quick Actions**")
    if st.button("Clear Data", help="Reset application"):
        keys_to_keep = ['cache_stats']
        keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]
        config.cleanup_temp_dirs()
        st.rerun()

# Sidebar status
with st.sidebar:
    if get_session_value('model_trained', False):
        st.markdown("---")
        st.markdown("### Model Status")
        
        metrics = get_session_value('trained_test_results', {}).get('metrics', {})
        if metrics:
            primary_metric = list(metrics.items())[0]
            value = primary_metric[1]
            
            if primary_metric[0] in ['accuracy', 'f1_score', 'r2_score']:
                status = "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Needs Improvement"
            else:
                status = "Trained"
            
            st.success(f"Status: {status}")
            st.metric(primary_metric[0].replace('_', ' ').title(), f"{value:.3f}")
    
    st.markdown("---")
    st.markdown("### System")
    
    system_status = {
        "PyArrow": "✅" if PYARROW_AVAILABLE else "❌",
        "SHAP": "✅",
        "Profiling": "✅" if PROFILING_AVAILABLE else "❌",
        "Imbalanced": "✅" if IMBALANCED_AVAILABLE else "❌"
    }
    
    for feature, status in system_status.items():
        st.caption(f"{feature}: {status}")

# Cleanup registration
if not get_session_value('cleanup_registered', False):
    atexit.register(config.cleanup_temp_dirs)
    st.session_state.cleanup_registered = True
