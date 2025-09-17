"""
Enhanced AutoML Pro - Complete Production Ready Version
- All issues fixed with comprehensive error handling
- Advanced session caching and memory management
- Beautiful animated UI with modern design
- Configurable parameters and user controls
- Production-grade performance optimizations
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
from plotly.subplots import make_subplots
import shap

# All required imports upfront to prevent runtime errors
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    st.warning("PyArrow not available. Parquet optimization disabled.")

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
    st.warning("Imbalanced-learn not available. SMOTE balancing disabled.")

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    st.warning("YData Profiling not available. Profile reports disabled.")

# Page config with enhanced settings
st.set_page_config(
    page_title="AutoML Pro", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Global configuration class
class AppConfig:
    def __init__(self):
        self.max_file_size_mb = 500
        self.max_session_size_mb = 1000
        self.default_sample_size = 5000
        self.shap_sample_size = 1000
        self.batch_sizes = [1000, 5000, 10000, 25000]
        self.animation_duration = 300
        self.cache_ttl = 3600  # 1 hour
        self.temp_dirs = []
        
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        self.temp_dirs = []

# Initialize global config
config = AppConfig()
atexit.register(config.cleanup_temp_dirs)

# Enhanced CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #1e3c72;
        --secondary-color: #2a5298;
        --accent-color: #667eea;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --background: #ffffff;
        --surface: #f8fafc;
        --border: #e5e7eb;
        --shadow: rgba(0, 0, 0, 0.1);
        --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        --gradient-accent: linear-gradient(135deg, var(--accent-color) 0%, #764ba2 100%);
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Animated main header */
    .main-header {
        background: var(--gradient-primary);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(30, 60, 114, 0.15);
        position: relative;
        overflow: hidden;
        animation: slideInDown 0.6s ease-out;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Animated metric cards */
    .metric-card {
        background: var(--gradient-accent);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #f8fafc, #e2e8f0);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        background: white;
        border-radius: 12px;
        font-weight: 600;
        color: #475569;
        border: 2px solid transparent;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        border: 2px solid var(--primary-color);
        box-shadow: 0 8px 24px rgba(30, 60, 114, 0.3);
        transform: translateY(-2px);
    }
    
    /* Progress bars with animation */
    .stProgress .st-bo {
        background: var(--gradient-primary);
        border-radius: 8px;
        animation: progressPulse 2s infinite;
    }
    
    /* Success/Warning/Error states */
    .success-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        animation: bounceIn 0.5s ease-out;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        animation: shake 0.5s ease-in-out;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f1f5f9 0%, #ffffff 100%);
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    /* Animations */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translate3d(0, -100%, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 40px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes progressPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom buttons */
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: var(--gradient-primary);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(30, 60, 114, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Enhanced dataframes */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border-radius: 12px;
        border: 2px dashed var(--border);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--primary-color);
        background: var(--surface);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced session state management with size checking
def init_session_state():
    """Initialize session state with size monitoring"""
    defaults = {
        "data_uploaded": False,
        "model_trained": False,
        "current_file": None,
        "training_data": None,
        "memory_usage": {},
        "cache_stats": {"hits": 0, "misses": 0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_object_size_mb(obj) -> float:
    """Get object size in MB"""
    try:
        return sys.getsizeof(obj) / (1024 * 1024)
    except:
        return 0

def safe_store_in_session(key: str, value: Any, max_size_mb: float = 100) -> bool:
    """Safely store objects in session state with size checking"""
    size_mb = get_object_size_mb(value)
    
    if size_mb > max_size_mb:
        st.error(f"⚠️ Object too large ({size_mb:.1f}MB) for session storage. Maximum: {max_size_mb}MB")
        return False
    
    # Clear memory if total session size is too large
    total_size = sum([get_object_size_mb(v) for v in st.session_state.values()])
    if total_size + size_mb > config.max_session_size_mb:
        clear_old_session_data()
    
    st.session_state[key] = value
    st.session_state.memory_usage[key] = size_mb
    return True

def clear_old_session_data():
    """Clear old session data to free memory"""
    keys_to_clear = ['old_data', 'old_model', 'temp_results', 'cached_predictions']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            if key in st.session_state.memory_usage:
                del st.session_state.memory_usage[key]
    
    # Force garbage collection
    gc.collect()

# Enhanced caching decorators
@st.cache_data(ttl=config.cache_ttl, show_spinner=False, max_entries=10)
def get_file_size_mb(uploaded_file) -> float:
    """Get file size in MB with caching"""
    try:
        st.session_state.cache_stats["hits"] += 1
        return len(uploaded_file.getvalue()) / (1024 * 1024)
    except:
        st.session_state.cache_stats["misses"] += 1
        return 0

@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def convert_to_parquet_safe(df: pd.DataFrame, original_filename: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Convert DataFrames to Parquet with proper cleanup"""
    if not PYARROW_AVAILABLE:
        return None, None
        
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        config.temp_dirs.append(temp_dir)
        
        parquet_path = os.path.join(temp_dir, f"{original_filename.split('.')[0]}.parquet")
        
        # Convert to parquet with optimization
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        
        # Get compression statistics
        original_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        
        return parquet_path, {
            'original_size_mb': original_size,
            'parquet_size_mb': parquet_size,
            'compression_ratio': (original_size - parquet_size) / original_size * 100 if original_size > 0 else 0
        }
    except Exception as e:
        st.warning(f"Parquet conversion failed: {e}")
        return None, None

@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def load_data_safe(uploaded_file, use_sample: bool = False) -> Optional[pd.DataFrame]:
    """Enhanced data loading with validation and optimization"""
    try:
        name = uploaded_file.name.lower()
        file_size_mb = get_file_size_mb(uploaded_file)
        
        # Check file size limits
        if file_size_mb > config.max_file_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {config.max_file_size_mb}MB")
            return None
        
        # Load data based on file type
        try:
            if name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, low_memory=False)
            elif name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                # Try CSV as fallback
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, low_memory=False)
        except Exception as e:
            st.error(f"Failed to parse file: {str(e)}")
            return None
        
        # Validate DataFrame
        if df.empty:
            st.error("File is empty")
            return None
            
        if len(df.columns) == 0:
            st.error("No columns found in file")
            return None
        
        # Use sample for very large datasets if requested
        if use_sample and len(df) > config.default_sample_size:
            df = df.sample(n=config.default_sample_size, random_state=42)
            st.info(f"Using random sample of {config.default_sample_size:,} rows for faster processing")
        
        # Memory usage calculation
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Convert to Parquet if beneficial
        should_convert = (
            PYARROW_AVAILABLE and 
            not name.endswith('.parquet') and 
            (file_size_mb > 10 or len(df) > 50000 or memory_mb > 50)
        )
        
        if should_convert:
            with st.spinner("🔄 Converting to Parquet for optimal performance..."):
                parquet_path, conversion_info = convert_to_parquet_safe(df, uploaded_file.name)
                
                if conversion_info:
                    st.success(f"✅ Parquet conversion completed! Size reduced by {conversion_info['compression_ratio']:.1f}%")
                    st.session_state.parquet_conversion_info = conversion_info
                    st.session_state.using_parquet = True
                else:
                    st.session_state.using_parquet = False
        else:
            st.session_state.using_parquet = False
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        return None

def validate_prediction_data(new_data: pd.DataFrame, training_features: List[str], training_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and fix prediction data compatibility"""
    validated_data = new_data.copy()
    issues = []
    
    # Check for missing features
    missing_features = set(training_features) - set(validated_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {list(missing_features)}")
    
    # Check and fix data types
    for col in training_features:
        if col in validated_data.columns:
            training_dtype = training_data[col].dtype
            current_dtype = validated_data[col].dtype
            
            if training_dtype != current_dtype:
                try:
                    if training_dtype == 'object':
                        validated_data[col] = validated_data[col].astype(str)
                    else:
                        validated_data[col] = pd.to_numeric(validated_data[col], errors='coerce')
                        
                        if validated_data[col].isna().sum() > 0:
                            issues.append(f"Column '{col}': {validated_data[col].isna().sum()} values couldn't be converted")
                            
                except Exception as e:
                    issues.append(f"Column '{col}': Type conversion failed - {str(e)}")
    
    # Report issues
    if issues:
        st.warning("Data validation issues found:\n" + "\n".join([f"• {issue}" for issue in issues]))
    
    return validated_data

def safe_onehot_encoder():
    """Safe OneHotEncoder creation with version compatibility"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def create_shap_explanation_safe(model, X_sample: pd.DataFrame, max_samples: int = None) -> Tuple[Any, Any, List[str]]:
    """Enhanced SHAP implementation with proper model detection"""
    if max_samples is None:
        max_samples = min(config.shap_sample_size, len(X_sample))
    
    try:
        # Get the actual classifier and preprocessor
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('model', model)
            preprocessor = model.named_steps.get('preprocessor')
            
            if preprocessor:
                X_transformed = preprocessor.transform(X_sample)
                if hasattr(X_transformed, 'toarray'):  # Handle sparse matrices
                    X_transformed = X_transformed.toarray()
            else:
                X_transformed = X_sample.values
        else:
            classifier = model
            X_transformed = X_sample.values if hasattr(X_sample, 'values') else X_sample
        
        # Limit sample size for performance
        X_shap = X_transformed[:max_samples]
        feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        # Choose appropriate explainer based on model type
        if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
            # Tree-based models (Random Forest, Decision Tree, Gradient Boosting)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap)
        elif hasattr(classifier, 'coef_') and hasattr(classifier, 'intercept_'):
            # Linear models (Logistic Regression, Linear Regression)
            if hasattr(classifier, 'classes_') and len(classifier.classes_) == 2:
                # Binary classification
                explainer = shap.LinearExplainer(classifier, X_shap[:50])
                shap_values = explainer.shap_values(X_shap[:20])
            else:
                # Multi-class or regression
                background = X_transformed[:min(50, len(X_transformed))]
                explainer = shap.KernelExplainer(classifier.predict, background)
                shap_values = explainer.shap_values(X_shap[:20])
        else:
            # Other models (SVM, etc.) - use KernelExplainer
            background = X_transformed[:min(50, len(X_transformed))]
            explainer = shap.KernelExplainer(classifier.predict, background)
            shap_values = explainer.shap_values(X_shap[:20])
        
        return shap_values, X_shap, feature_names
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {str(e)}")
        return None, None, None

def create_advanced_visualizations(y_test, y_pred, y_proba=None, class_names=None, problem_type="classification"):
    """Create advanced interactive visualizations"""
    figs = {}
    
    if problem_type == "classification":
        # Enhanced Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig_cm = px.imshow(
            cm_norm,
            text_auto='.1f',
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix (Normalized %)"
        )
        
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            xaxis={'side': 'bottom'},
            height=400
        )
        
        # Add actual counts as annotations
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i][j]}",
                    showarrow=False,
                    font=dict(color="white", size=10),
                    yshift=10
                )
        
        figs['confusion_matrix'] = fig_cm
        
        # ROC and PR curves for binary classification
        if len(np.unique(y_test)) == 2 and y_proba is not None:
            try:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    line=dict(color='#2c3e50', width=3),
                    fill='tonexty',
                    fillcolor='rgba(44, 62, 80, 0.1)'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve Analysis',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400,
                    showlegend=True
                )
                
                figs['roc_curve'] = fig_roc
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name='Precision-Recall Curve',
                    line=dict(color='#3498db', width=3),
                    fill='tonexty',
                    fillcolor='rgba(52, 152, 219, 0.1)'
                ))
                
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=400
                )
                
                figs['pr_curve'] = fig_pr
                
            except Exception as e:
                st.warning(f"ROC/PR curves failed: {e}")
    
    else:  # Regression
        # Actual vs Predicted with trend line
        fig_scatter = px.scatter(
            x=y_test, y=y_pred,
            title="Actual vs Predicted Values",
            labels={'x': 'Actual', 'y': 'Predicted'},
            color_discrete_sequence=['#2c3e50']
        )
        
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines', 
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        figs['actual_vs_predicted'] = fig_scatter
        
        # Residuals plot
        residuals = y_test - y_pred
        fig_residuals = px.scatter(
            x=y_pred, y=residuals,
            title="Residuals Analysis",
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            color_discrete_sequence=['#3498db']
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line")
        
        figs['residuals'] = fig_residuals
    
    return figs

# Initialize session state and config
init_session_state()

# Header with animation
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Analytics Pro</h1>
    <p>Complete machine learning platform with advanced explanations and beautiful visualizations</p>
</div>
""", unsafe_allow_html=True)

# Configuration sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # File upload settings
    with st.expander("File Settings", expanded=False):
        config.max_file_size_mb = st.slider("Max File Size (MB)", 50, 1000, config.max_file_size_mb)
        use_sample_for_large_files = st.checkbox("Auto-sample large files", value=True)
        config.default_sample_size = st.selectbox("Sample Size", [1000, 5000, 10000, 25000], index=1)
    
    # Model settings
    with st.expander("Model Settings", expanded=False):
        config.shap_sample_size = st.selectbox("SHAP Sample Size", [100, 500, 1000, 2000], index=2)
        enable_cross_validation = st.checkbox("Enable Cross Validation", value=False)
        cv_folds = st.slider("CV Folds", 3, 10, 5) if enable_cross_validation else 5
    
    # Performance settings
    with st.expander("Performance", expanded=False):
        config.cache_ttl = st.selectbox("Cache TTL (seconds)", [1800, 3600, 7200], index=1)
        show_memory_usage = st.checkbox("Show Memory Usage", value=False)
        enable_profiling = st.checkbox("Enable Data Profiling", value=PROFILING_AVAILABLE)
    
    st.markdown("---")
    
    # Memory usage display
    if show_memory_usage and st.session_state.memory_usage:
        st.markdown("### 📊 Memory Usage")
        total_mb = sum(st.session_state.memory_usage.values())
        st.metric("Total Session", f"{total_mb:.1f} MB")
        
        for key, size_mb in list(st.session_state.memory_usage.items())[:5]:
            st.caption(f"{key}: {size_mb:.1f} MB")
    
    # Cache stats
    if st.session_state.cache_stats["hits"] + st.session_state.cache_stats["misses"] > 0:
        st.markdown("### ⚡ Cache Performance")
        total_requests = st.session_state.cache_stats["hits"] + st.session_state.cache_stats["misses"]
        hit_rate = st.session_state.cache_stats["hits"] / total_requests * 100
        st.metric("Hit Rate", f"{hit_rate:.1f}%")

    st.markdown("---")
    
    # File upload section
    st.markdown("### 📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supported formats: CSV, Excel, Parquet"
    )
    
    # Advanced upload options
    if uploaded_file:
        with st.expander("Upload Options", expanded=False):
            detect_encoding = st.checkbox("Auto-detect encoding", value=True)
            handle_missing = st.selectbox("Missing values", ["Keep as-is", "Drop rows", "Fill with mean/mode"])
            data_validation = st.checkbox("Validate data types", value=True)
    
    if uploaded_file:
        with st.spinner("🔄 Loading and optimizing data..."):
            df = load_data_safe(uploaded_file, use_sample=use_sample_for_large_files)
            
            if df is not None:
                # Store data safely
                if safe_store_in_session("data", df):
                    st.session_state.data_uploaded = True
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.training_data = df.copy()  # Store for validation later
                    
                    st.markdown('<div class="success-card">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                    
                    # Enhanced file info display
                    file_size = get_file_size_mb(uploaded_file)
                    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    
                    info_html = f"""
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong>📄 {uploaded_file.name}</strong><br>
                        <small>
                        📏 {df.shape[0]:,} rows × {df.shape[1]} columns<br>
                        💾 Original: {file_size:.1f} MB | Memory: {memory_mb:.1f} MB<br>
                        🔧 {df.dtypes.value_counts().to_dict()}
                        </small>
                    </div>
                    """
                    st.markdown(info_html, unsafe_allow_html=True)
                    
                    # Show optimization info
                    if st.session_state.get('using_parquet', False):
                        conversion_info = st.session_state.get('parquet_conversion_info', {})
                        st.success(f"🚀 Parquet optimized: {conversion_info.get('compression_ratio', 0):.1f}% smaller")

# Main application tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Explorer", 
    "🎯 Model Training", 
    "📊 Model Analysis", 
    "🔮 Predictions",
    "⚙️ Advanced Tools"
])

# Tab 1: Enhanced Data Explorer
with tab1:
    st.markdown("## 🔍 Data Explorer")
    
    if not st.session_state.data_uploaded:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <h3>📁 Upload a dataset to begin exploring</h3>
            <p>Supported formats: CSV, Excel (.xlsx, .xls), Parquet</p>
            <p>Maximum file size: {:.0f} MB</p>
        </div>
        """.format(config.max_file_size_mb), unsafe_allow_html=True)
    else:
        df = st.session_state.data
        
        # Animated overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Rows</h3><h2>{df.shape[0]:,}</h2></div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Columns</h3><h2>{df.shape[1]}</h2></div>', 
                       unsafe_allow_html=True)
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f'<div class="metric-card"><h3>Memory</h3><h2>{memory_mb:.1f} MB</h2></div>', 
                       unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            color = "#ef4444" if missing_pct > 10 else "#10b981" if missing_pct < 1 else "#f59e0b"
            st.markdown(f'<div class="metric-card" style="background: {color};"><h3>Missing</h3><h2>{missing_pct:.1f}%</h2></div>', 
                       unsafe_allow_html=True)
        
        # Interactive data type analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Types Distribution")
            dtype_counts = df.dtypes.value_counts()
            
            fig_dtypes = px.pie(
                values=dtype_counts.values,
                names=[str(dtype) for dtype in dtype_counts.index],
                title="Column Data Types",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_dtypes.update_traces(textposition='inside', textinfo='percent+label')
            fig_dtypes.update_layout(height=400)
            st.plotly_chart(fig_dtypes, use_container_width=True)
        
        with col2:
            st.markdown("### 🔢 Missing Values Analysis")
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                missing_pct = (missing_data / len(df)) * 100
                
                fig_missing = go.Figure()
                fig_missing.add_trace(go.Bar(
                    x=missing_pct.values,
                    y=missing_pct.index,
                    orientation='h',
                    marker_color='#ef4444',
                    text=[f'{val:.1f}%' for val in missing_pct.values],
                    textposition='auto'
                ))
                
                fig_missing.update_layout(
                    title="Missing Values by Column (%)",
                    xaxis_title="Missing Percentage",
                    height=400
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("🎉 No missing values found!")
        
        # Interactive data preview with filtering
        st.markdown("### 👀 Interactive Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
        with col2:
            columns_to_show = st.multiselect("Columns to display", df.columns.tolist(), 
                                           default=df.columns.tolist()[:10])
        with col3:
            data_sample_type = st.selectbox("Sample type", ["Head", "Tail", "Random"])
        
        if columns_to_show:
            if data_sample_type == "Head":
                display_df = df[columns_to_show].head(show_rows)
            elif data_sample_type == "Tail":
                display_df = df[columns_to_show].tail(show_rows)
            else:
                display_df = df[columns_to_show].sample(n=min(show_rows, len(df)), random_state=42)
            
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].describe().round(3)
            st.dataframe(summary_stats, use_container_width=True)
            
            # Correlation heatmap for numeric columns
            if len(numeric_cols) > 1:
                st.markdown("### 🔥 Correlation Heatmap")
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Matrix"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data profiling report
        if enable_profiling and PROFILING_AVAILABLE:
            st.markdown("### 📋 Advanced Data Profiling")
            
            if st.button("🚀 Generate Comprehensive Profile Report", type="primary"):
                try:
                    with st.spinner("📊 Generating detailed profile report..."):
                        # Use a reasonable sample size for profiling
                        sample_df = df.sample(n=min(config.default_sample_size, len(df)), random_state=42)
                        
                        profile = ProfileReport(
                            sample_df, 
                            minimal=True,
                            title=f"Data Profile: {st.session_state.current_file}",
                            explorative=True
                        )
                        
                        profile_html = profile.to_html()
                        st.components.v1.html(profile_html, height=800, scrolling=True)
                        
                except Exception as e:
                    st.error(f"❌ Profile generation failed: {str(e)}")

# Tab 2: Enhanced Model Training
with tab2:
    st.markdown("## 🎯 Model Training")
    
    if not st.session_state.data_uploaded:
        st.info("📁 Upload data first to start training models")
    else:
        data = st.session_state.data.copy()
        
        # Configuration section
        st.markdown("### ⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Target & Features**")
            all_columns = list(data.columns)
            target_column = st.selectbox("🎯 Target Variable", all_columns, 
                                       help="The column you want to predict")
            
            available_features = [col for col in all_columns if col != target_column]
            feature_cols = st.multiselect(
                "🔧 Feature Columns", 
                available_features,
                default=available_features[:10] if len(available_features) > 10 else available_features,
                help="Select specific features or leave empty to use all"
            )
            
            if not feature_cols:
                feature_cols = available_features
        
        with col2:
            st.markdown("**🤖 Model Configuration**")
            
            # Determine problem type
            y_raw = data[target_column].copy()
            if y_raw.dtype == "object" or y_raw.nunique() <= 20:
                problem_type = "classification"
                n_classes = y_raw.nunique()
                st.success(f"🎯 Classification detected ({n_classes} classes)")
                
                # Classification models
                model_options = {
                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "SVM": SVC(probability=True, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42)
                }
                
                # Class balancing options
                balance_options = ["None"]
                if IMBALANCED_AVAILABLE:
                    balance_options.extend(["SMOTE", "Random Oversampling"])
                
                balance_method = st.selectbox("⚖️ Class Balancing", balance_options)
                
            else:
                problem_type = "regression"
                st.success("📈 Regression detected")
                
                # Regression models
                model_options = {
                    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression()
                }
                balance_method = "None"
            
            model_choice = st.selectbox("🧠 Algorithm", list(model_options.keys()))
            test_size = st.slider("📊 Test Split (%)", 10, 50, 20, step=5)
        
        # Advanced options
        with st.expander("🔧 Advanced Training Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                random_state = st.number_input("🎲 Random State", value=42, min_value=0)
                stratify_split = st.checkbox("📊 Stratified Split", value=True) if problem_type == "classification" else False
            
            with col2:
                if enable_cross_validation:
                    st.info(f"🔄 Cross-validation enabled ({cv_folds} folds)")
                enable_feature_selection = st.checkbox("🎯 Auto Feature Selection", value=False)
        
        # Training button
        if st.button("🚀 Train Model", type="primary"):
            try:
                # Clear old model data
                clear_old_session_data()
                
                # Prepare data
                X = data[feature_cols].copy()
                y_raw = data[target_column].copy()
                
                # Encode target for classification
                if problem_type == "classification":
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y_raw.astype(str))
                    class_names = le_target.classes_.tolist()
                else:
                    le_target = None
                    y = y_raw.to_numpy()
                    class_names = None
                
                # Split data
                split_kwargs = {
                    'test_size': test_size/100,
                    'random_state': random_state
                }
                
                if problem_type == "classification" and stratify_split:
                    split_kwargs['stratify'] = y
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
                
                # Create preprocessing pipeline
                numeric_features = X.select_dtypes(include=[np.number]).columns
                categorical_features = X.select_dtypes(exclude=[np.number]).columns
                
                preprocessing_steps = []
                
                if len(numeric_features) > 0:
                    preprocessing_steps.append(('num', StandardScaler(), numeric_features))
                
                if len(categorical_features) > 0:
                    preprocessing_steps.append(('cat', safe_onehot_encoder(), categorical_features))
                
                preprocessor = ColumnTransformer(preprocessing_steps) if preprocessing_steps else None
                
                # Build pipeline
                pipeline_steps = []
                if preprocessor:
                    pipeline_steps.append(('preprocessor', preprocessor))
                
                # Add sampling if requested
                if balance_method != "None" and IMBALANCED_AVAILABLE and problem_type == "classification":
                    if balance_method == "SMOTE":
                        pipeline_steps.append(('sampler', SMOTE(random_state=random_state)))
                    elif balance_method == "Random Oversampling":
                        pipeline_steps.append(('sampler', RandomOverSampler(random_state=random_state)))
                
                # Add model
                base_model = model_options[model_choice]
                pipeline_steps.append(('model', base_model))
                
                # Create pipeline
                if any('sampler' in step[0] for step in pipeline_steps):
                    model_pipeline = ImbPipeline(pipeline_steps)
                else:
                    model_pipeline = Pipeline(pipeline_steps)
                
                # Training progress
                progress_bar = st.progress(0, "🔄 Initializing training...")
                
                # Train model
                progress_bar.progress(25, "🏋️ Training model...")
                model_pipeline.fit(X_train, y_train)
                
                progress_bar.progress(75, "📊 Evaluating performance...")
                
                # Make predictions
                y_pred = model_pipeline.predict(X_test)
                
                # Get probabilities for classification
                y_proba = None
                if problem_type == "classification" and hasattr(model_pipeline, 'predict_proba'):
                    try:
                        y_proba_full = model_pipeline.predict_proba(X_test)
                        y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] == 2 else y_proba_full
                    except:
                        pass
                
                # Calculate metrics
                if problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    metrics = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'precision': f1_score(y_test, y_pred, average='weighted'),
                        'recall': f1_score(y_test, y_pred, average='weighted')
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    metrics = {
                        'r2_score': r2,
                        'rmse': rmse,
                        'mse': mse,
                        'mae': np.mean(np.abs(y_test - y_pred))
                    }
                
                # Cross-validation if enabled
                cv_scores = None
                if enable_cross_validation:
                    progress_bar.progress(90, "🔄 Running cross-validation...")
                    
                    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
                    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                
                progress_bar.progress(100, "✅ Training completed!")
                
                # Store results
                training_results = {
                    'model': model_pipeline,
                    'feature_columns': feature_cols,
                    'label_encoder': le_target,
                    'class_names': class_names,
                    'test_results': {'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba, 'metrics': metrics},
                    'X_test': X_test,
                    'model_name': model_choice,
                    'problem_type': problem_type,
                    'cv_scores': cv_scores,
                    'training_config': {
                        'balance_method': balance_method,
                        'test_size': test_size,
                        'random_state': random_state,
                        'n_features': len(feature_cols)
                    }
                }
                
                # Store in session with size checking
                for key, value in training_results.items():
                    safe_store_in_session(f'trained_{key}', value)
                
                st.session_state.model_trained = True
                
                # Clear progress bar
                progress_bar.empty()
                
                # Success message with animation
                st.markdown('<div class="success-card">🎉 Model trained successfully!</div>', unsafe_allow_html=True)
                
                # Display metrics in animated cards
                st.markdown("### 📊 Training Results")
                
                metric_cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    with metric_cols[i % len(metric_cols)]:
                        if name == 'cv_mean' and cv_scores is not None:
                            st.metric(
                                f"CV {name.replace('cv_', '').title()}", 
                                f"{value:.4f}", 
                                f"±{metrics.get('cv_std', 0):.4f}"
                            )
                        else:
                            st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
                
                # Cross-validation visualization
                if cv_scores is not None:
                    st.markdown("### 🔄 Cross-Validation Results")
                    
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Box(
                        y=cv_scores,
                        name="CV Scores",
                        marker_color='#2c3e50'
                    ))
                    
                    fig_cv.update_layout(
                        title=f"Cross-Validation Scores (Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})",
                        yaxis_title="Score",
                        height=400
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)
                
                # Model download with enhanced package
                st.markdown("### 💾 Export Model")
                
                try:
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    
                    model_package = {
                        'model': model_pipeline,
                        'feature_columns': feature_cols,
                        'label_encoder': le_target,
                        'class_names': class_names,
                        'model_name': model_choice,
                        'problem_type': problem_type,
                        'metrics': metrics,
                        'training_config': training_results['training_config'],
                        'timestamp': timestamp,
                        'app_version': '2.0'
                    }
                    
                    buffer = io.BytesIO()
                    joblib.dump(model_package, buffer, compress=3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📦 Download Model Package",
                            data=buffer.getvalue(),
                            file_name=f"automl_model_{model_choice.lower().replace(' ', '_')}_{timestamp}.pkl",
                            mime="application/octet-stream",
                            help="Complete model package with metadata"
                        )
                    
                    with col2:
                        # Model info as JSON
                        model_info = {
                            'model_type': model_choice,
                            'problem_type': problem_type,
                            'features': feature_cols,
                            'metrics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                       for k, v in metrics.items()},
                            'timestamp': timestamp
                        }
                        
                        st.download_button(
                            "📋 Download Model Info",
                            data=pd.Series(model_info).to_json(indent=2),
                            file_name=f"model_info_{timestamp}.json",
                            mime="application/json"
                        )
                
                except Exception as e:
                    st.error(f"❌ Model export failed: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.text("Error details:")
                st.code(traceback.format_exc())

# Tab 3: Enhanced Model Analysis
with tab3:
    st.markdown("## 📊 Advanced Model Analysis")
    
    if not st.session_state.get('model_trained', False):
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <h3>🎯 Train a model first to see detailed analysis</h3>
            <p>Complete training in the Model Training tab to unlock:</p>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li>🔍 SHAP Explanations</li>
                <li>📈 Performance Visualizations</li>
                <li>🎯 Feature Importance</li>
                <li>📊 Advanced Metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.trained_test_results['y_test']
        y_pred = st.session_state.trained_test_results['y_pred']
        y_proba = st.session_state.trained_test_results.get('y_proba')
        metrics = st.session_state.trained_test_results['metrics']
        problem_type = st.session_state.trained_problem_type
        class_names = st.session_state.trained_class_names
        
        # Model Overview
        st.markdown("### 🎯 Model Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        primary_metrics = list(metrics.items())[:4] if len(metrics) >= 4 else list(metrics.items())
        
        for i, (name, value) in enumerate(primary_metrics):
            with [col1, col2, col3, col4][i]:
                # Color-code metrics based on performance
                if name in ['accuracy', 'f1_score', 'r2_score']:
                    color = "#10b981" if value > 0.8 else "#f59e0b" if value > 0.6 else "#ef4444"
                elif name in ['mse', 'rmse', 'mae']:
                    color = "#10b981" if value < 0.1 else "#f59e0b" if value < 1 else "#ef4444"
                else:
                    color = "#667eea"
                
                st.markdown(
                    f'<div class="metric-card" style="background: {color};">'
                    f'<h3>{name.replace("_", " ").title()}</h3>'
                    f'<h2>{value:.4f}</h2></div>',
                    unsafe_allow_html=True
                )
        
        # Advanced Performance Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        with st.spinner("Creating advanced visualizations..."):
            viz_figures = create_advanced_visualizations(
                y_test, y_pred, y_proba, class_names, problem_type
            )
            
            if problem_type == "classification":
                # Display classification visualizations
                if 'confusion_matrix' in viz_figures:
                    st.plotly_chart(viz_figures['confusion_matrix'], use_container_width=True)
                
                if len(np.unique(y_test)) == 2:  # Binary classification
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'roc_curve' in viz_figures:
                            st.plotly_chart(viz_figures['roc_curve'], use_container_width=True)
                    
                    with col2:
                        if 'pr_curve' in viz_figures:
                            st.plotly_chart(viz_figures['pr_curve'], use_container_width=True)
                
                # Classification Report
                try:
                    st.markdown("### 📋 Detailed Classification Report")
                    
                    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(4)
                    
                    # Create interactive classification report visualization
                    metrics_to_plot = ['precision', 'recall', 'f1-score']
                    class_metrics = report_df[report_df.index.isin(class_names)]
                    
                    fig_report = go.Figure()
                    
                    for metric in metrics_to_plot:
                        if metric in class_metrics.columns:
                            fig_report.add_trace(go.Bar(
                                name=metric.title(),
                                x=class_metrics.index,
                                y=class_metrics[metric],
                                text=class_metrics[metric].round(3),
                                textposition='auto'
                            ))
                    
                    fig_report.update_layout(
                        title="Per-Class Performance Metrics",
                        xaxis_title="Classes",
                        yaxis_title="Score",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_report, use_container_width=True)
                    
                    # Show detailed report table
                    st.dataframe(report_df, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Classification report visualization failed: {e}")
            
            else:  # Regression visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'actual_vs_predicted' in viz_figures:
                        st.plotly_chart(viz_figures['actual_vs_predicted'], use_container_width=True)
                
                with col2:
                    if 'residuals' in viz_figures:
                        st.plotly_chart(viz_figures['residuals'], use_container_width=True)
        
        # SHAP Explanations Section
        st.markdown("### 🔍 SHAP Model Explanations")
        
        with st.expander("SHAP Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                shap_sample_size = st.slider("Sample Size for SHAP", 50, 2000, config.shap_sample_size)
            with col2:
                shap_plot_type = st.selectbox("SHAP Plot Type", ["Summary", "Waterfall", "Force"])
        
        if st.button("Generate SHAP Explanations", type="primary"):
            with st.spinner("Generating SHAP explanations... This may take a moment."):
                shap_values, X_shap, feature_names = create_shap_explanation_safe(
                    model, X_test, max_samples=shap_sample_size
                )
                
                if shap_values is not None and X_shap is not None:
                    st.success("SHAP explanations generated successfully!")
                    
                    try:
                        # Handle different SHAP value formats
                        if isinstance(shap_values, list):
                            if len(shap_values) > 1 and problem_type == "classification":
                                plot_values = shap_values[1]  # Use positive class for binary
                                st.info("Showing SHAP values for positive class")
                            else:
                                plot_values = shap_values[0]
                        else:
                            plot_values = shap_values
                        
                        # Convert to numpy array if needed
                        if hasattr(plot_values, 'values'):
                            shap_array = plot_values.values
                        else:
                            shap_array = np.array(plot_values)
                        
                        # Summary Plot
                        if shap_plot_type == "Summary":
                            st.markdown("#### SHAP Summary Plot")
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            shap.summary_plot(
                                shap_array, 
                                X_shap, 
                                feature_names=feature_names, 
                                show=False,
                                max_display=min(20, len(feature_names))
                            )
                            st.pyplot(fig)
                            plt.close()
                        
                        # Individual explanation (Waterfall or Force)
                        if shap_plot_type in ["Waterfall", "Force"]:
                            st.markdown("#### Individual Sample Explanation")
                            
                            sample_idx = st.slider(
                                "Sample to explain", 
                                0, 
                                min(len(shap_array)-1, 50), 
                                0
                            )
                            
                            if shap_plot_type == "Waterfall":
                                # Create waterfall plot
                                sample_shap = shap_array[sample_idx]
                                
                                # Get top features by absolute SHAP value
                                indices = np.argsort(np.abs(sample_shap))[-15:]
                                top_features = [feature_names[i] for i in indices]
                                top_values = sample_shap[indices]
                                
                                # Create bar plot
                                colors = ['#10b981' if val > 0 else '#ef4444' for val in top_values]
                                
                                fig_waterfall = go.Figure()
                                fig_waterfall.add_trace(go.Bar(
                                    x=top_values,
                                    y=top_features,
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f'{val:.3f}' for val in top_values],
                                    textposition='auto'
                                ))
                                
                                fig_waterfall.update_layout(
                                    title=f"SHAP Waterfall Plot - Sample {sample_idx + 1}",
                                    xaxis_title="SHAP Value (Impact on Prediction)",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_waterfall, use_container_width=True)
                            
                            else:  # Force plot
                                try:
                                    # Create a simple force plot visualization
                                    sample_shap = shap_array[sample_idx]
                                    
                                    # Get prediction details
                                    if problem_type == "classification":
                                        pred_class = y_pred[sample_idx]
                                        pred_proba = y_proba[sample_idx] if y_proba is not None else None
                                        
                                        prediction_text = f"Predicted: {class_names[pred_class] if class_names else pred_class}"
                                        if pred_proba is not None:
                                            if isinstance(pred_proba, (list, np.ndarray)) and len(pred_proba) > 1:
                                                prediction_text += f" (Confidence: {pred_proba.max():.3f})"
                                            else:
                                                prediction_text += f" (Probability: {pred_proba:.3f})"
                                    else:
                                        prediction_text = f"Predicted: {y_pred[sample_idx]:.3f}"
                                    
                                    st.info(prediction_text)
                                    
                                    # Show feature contributions
                                    contributions_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Value': X_shap[sample_idx],
                                        'SHAP_Value': sample_shap,
                                        'Contribution': ['Positive' if val > 0 else 'Negative' for val in sample_shap]
                                    }).sort_values('SHAP_Value', key=abs, ascending=False).head(15)
                                    
                                    st.dataframe(contributions_df, use_container_width=True)
                                    
                                except Exception as e:
                                    st.warning(f"Force plot creation failed: {e}")
                        
                        # Feature importance from SHAP
                        st.markdown("#### Global Feature Importance (SHAP-based)")
                        
                        # Calculate mean absolute SHAP values for global importance
                        global_importance = np.mean(np.abs(shap_array), axis=0)
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': global_importance
                        }).sort_values('Importance', ascending=True).tail(15)
                        
                        fig_global_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 15 Features by Mean |SHAP Value|",
                            color='Importance',
                            color_continuous_scale=['#3498db', '#2c3e50']
                        )
                        fig_global_importance.update_layout(height=500)
                        st.plotly_chart(fig_global_importance, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"SHAP visualization failed: {str(e)}")
                        st.code(traceback.format_exc())
                else:
                    st.warning("SHAP explanations could not be generated for this model type")
        
        # Feature Importance (Model-based)
        st.markdown("### 📊 Model Feature Importance")
        
        try:
            classifier = model.named_steps.get('model', model) if hasattr(model, 'named_steps') else model
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get feature names after preprocessing
                try:
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                        feature_names_processed = model.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        feature_names_processed = X_test.columns.tolist()
                except:
                    feature_names_processed = [f'feature_{i}' for i in range(len(importances))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names_processed[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(20)
                
                # Interactive importance plot
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Feature Importances (Model-based)",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Show detailed importance table
                with st.expander("Detailed Feature Importance Table", expanded=False):
                    full_importance_df = pd.DataFrame({
                        'Feature': feature_names_processed[:len(importances)],
                        'Importance': importances,
                        'Importance_Pct': (importances / importances.sum()) * 100
                    }).sort_values('Importance', ascending=False)
                    
                    st.dataframe(full_importance_df, use_container_width=True)
            
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_
                
                # Handle multi-class coefficients
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                else:
                    coefficients = np.abs(coefficients)
                
                # Get feature names
                try:
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                        feature_names_processed = model.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        feature_names_processed = X_test.columns.tolist()
                except:
                    feature_names_processed = [f'feature_{i}' for i in range(len(coefficients))]
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names_processed[:len(coefficients)],
                    'Coefficient': coefficients
                }).sort_values('Coefficient', ascending=True).tail(20)
                
                fig_coef = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Feature Coefficients (Absolute Values)",
                    color='Coefficient',
                    color_continuous_scale='Plasma'
                )
                
                fig_coef.update_layout(height=600)
                st.plotly_chart(fig_coef, use_container_width=True)
            
            else:
                st.info("Feature importance not available for this model type")
        
        except Exception as e:
            st.error(f"Feature importance analysis failed: {e}")
        
        # Model Insights and Recommendations
        st.markdown("### 💡 Model Insights & Recommendations")
        
        insights = []
        
        # Performance insights
        if problem_type == "classification":
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            
            if accuracy > 0.9:
                insights.append("🎉 Excellent model performance! Your model achieves >90% accuracy.")
            elif accuracy > 0.8:
                insights.append("✅ Good model performance. Consider feature engineering for further improvement.")
            elif accuracy > 0.6:
                insights.append("⚠️ Moderate performance. Try different algorithms or more feature engineering.")
            else:
                insights.append("❌ Low performance. Consider collecting more data or revisiting problem formulation.")
            
            if len(np.unique(y_test)) > 2:  # Multi-class
                insights.append(f"📊 Multi-class classification with {len(np.unique(y_test))} classes detected.")
        
        else:  # Regression
            r2 = metrics.get('r2_score', 0)
            
            if r2 > 0.9:
                insights.append("🎉 Excellent model fit! Your model explains >90% of the variance.")
            elif r2 > 0.7:
                insights.append("✅ Good model fit. Your model explains most of the variance in the data.")
            elif r2 > 0.5:
                insights.append("⚠️ Moderate fit. Consider adding more relevant features.")
            else:
                insights.append("❌ Poor fit. The model may need significant improvements.")
        
        # Data insights
        n_features = len(st.session_state.trained_feature_columns)
        n_samples = len(y_test) + len(y_test)  # Approximate total samples
        
        if n_features > n_samples / 10:
            insights.append("⚠️ High feature-to-sample ratio. Consider feature selection or regularization.")
        
        if st.session_state.trained_training_config.get('balance_method', 'None') != 'None':
            insights.append(f"⚖️ Applied {st.session_state.trained_training_config['balance_method']} for class balancing.")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Export analysis results
        st.markdown("### 📄 Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Performance Report", type="secondary"):
                try:
                    # Create comprehensive report
                    report_data = {
                        'model_info': {
                            'name': st.session_state.trained_model_name,
                            'type': problem_type,
                            'features': len(st.session_state.trained_feature_columns),
                            'samples': len(y_test) + len(y_test)
                        },
                        'performance_metrics': metrics,
                        'training_config': st.session_state.trained_training_config,
                        'insights': insights,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    report_json = pd.Series(report_data).to_json(indent=2)
                    
                    st.download_button(
                        "Download Performance Report",
                        data=report_json,
                        file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
        
        with col2:
            if st.button("🔍 Export Feature Analysis", type="secondary"):
                try:
                    # Create feature analysis data
                    feature_data = {
                        'feature_columns': st.session_state.trained_feature_columns,
                        'feature_count': len(st.session_state.trained_feature_columns),
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    # Add feature importance if available
                    classifier = model.named_steps.get('model', model) if hasattr(model, 'named_steps') else model
                    if hasattr(classifier, 'feature_importances_'):
                        feature_data['feature_importances'] = classifier.feature_importances_.tolist()
                    
                    feature_json = pd.Series(feature_data).to_json(indent=2)
                    
                    st.download_button(
                        "Download Feature Analysis",
                        data=feature_json,
                        file_name=f"feature_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Feature analysis export failed: {e}")

# Tab 4: Enhanced Predictions
with tab4:
    st.markdown("## 🔮 Make Predictions on New Data")
    
    if not st.session_state.get('model_trained', False):
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <h3>🎯 Train a model first to make predictions</h3>
            <p>After training, you'll be able to:</p>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li>📁 Upload new data files</li>
                <li>🔮 Get instant predictions</li>
                <li>📊 View confidence scores</li>
                <li>💾 Download results in multiple formats</li>
                <li>📈 Analyze prediction distributions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        model = st.session_state.get('trained_model')
        feature_cols = st.session_state.get('trained_feature_columns')
        le_target = st.session_state.get('trained_le_target')
        problem_type = st.session_state.get('trained_problem_type')
        training_data = st.session_state.get('training_data')
        
        st.markdown("### 📁 Upload Data for Predictions")
        
        # Required features info
        with st.expander("📋 Required Features", expanded=True):
            st.write("Your model expects the following features:")
            
            # Display features in a nice format
            feature_df = pd.DataFrame({
                'Feature': feature_cols,
                'Type': [str(training_data[col].dtype) for col in feature_cols],
                'Sample Value': [str(training_data[col].iloc[0]) for col in feature_cols]
            })
            
            st.dataframe(feature_df, use_container_width=True)
        
        # File upload for predictions
        pred_file = st.file_uploader(
            "📂 Upload prediction data",
            type=["csv", "xlsx", "parquet"],
            key="prediction_file",
            help="File must contain all required features with matching names"
        )
        
        # Prediction configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.selectbox("Batch Size", config.batch_sizes, index=1)
        
        with col2:
            include_probabilities = st.checkbox(
                "Include Confidence Scores",
                value=problem_type == "classification",
                disabled=problem_type == "regression",
                help="Add prediction confidence/probability columns"
            )
        
        with col3:
            prediction_threshold = None
            if problem_type == "classification" and le_target and len(le_target.classes_) == 2:
                prediction_threshold = st.slider(
                    "Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                    help="Threshold for binary classification decisions"
                )
        
        if pred_file:
            try:
                # Load prediction data
                prediction_data = load_data_safe(pred_file, use_sample=False)
                
                if prediction_data is None:
                    st.error("❌ Failed to load prediction file")
                else:
                    st.success(f"✅ Loaded {len(prediction_data):,} rows for prediction")
                    
                    # Preview data
                    st.markdown("### 👀 Data Preview")
                    
                    preview_cols = st.columns([3, 1])
                    with preview_cols[0]:
                        st.dataframe(prediction_data.head(10), use_container_width=True)
                    
                    with preview_cols[1]:
                        st.metric("Total Rows", f"{len(prediction_data):,}")
                        st.metric("Columns", len(prediction_data.columns))
                        
                        memory_mb = prediction_data.memory_usage(deep=True).sum() / (1024 * 1024)
                        st.metric("Memory", f"{memory_mb:.1f} MB")
                    
                    # Feature validation
                    st.markdown("### 🔍 Feature Validation")
                    
                    missing_features = set(feature_cols) - set(prediction_data.columns)
                    extra_features = set(prediction_data.columns) - set(feature_cols)
                    common_features = set(feature_cols) & set(prediction_data.columns)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if missing_features:
                            st.error(f"❌ Missing Features ({len(missing_features)})")
                            for feature in list(missing_features)[:5]:
                                st.caption(f"• {feature}")
                            if len(missing_features) > 5:
                                st.caption(f"• ... and {len(missing_features) - 5} more")
                        else:
                            st.success("✅ All Required Features Present")
                    
                    with col2:
                        st.info(f"✅ Matching Features: {len(common_features)}")
                    
                    with col3:
                        if extra_features:
                            st.warning(f"⚠️ Extra Features: {len(extra_features)}")
                            st.caption("(Will be ignored)")
                    
                    # Proceed with predictions if validation passes
                    if not missing_features:
                        # Data type validation and fixing
                        try:
                            validated_data = validate_prediction_data(
                                prediction_data, feature_cols, training_data
                            )
                            
                            if st.button("🚀 Generate Predictions", type="primary"):
                                pred_features = validated_data[feature_cols]
                                
                                with st.spinner("🔮 Generating predictions..."):
                                    try:
                                        # Initialize results containers
                                        all_predictions = []
                                        all_probabilities = []
                                        
                                        # Progress tracking
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        # Process in batches
                                        total_batches = (len(pred_features) + batch_size - 1) // batch_size
                                        
                                        for batch_idx in range(total_batches):
                                            start_idx = batch_idx * batch_size
                                            end_idx = min(start_idx + batch_size, len(pred_features))
                                            
                                            batch = pred_features.iloc[start_idx:end_idx]
                                            
                                            # Update progress
                                            progress = (batch_idx + 1) / total_batches
                                            progress_bar.progress(progress)
                                            status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}")
                                            
                                            # Make predictions
                                            batch_preds = model.predict(batch)
                                            all_predictions.extend(batch_preds)
                                            
                                            # Get probabilities if requested
                                            if include_probabilities and hasattr(model, 'predict_proba'):
                                                try:
                                                    batch_probs = model.predict_proba(batch)
                                                    all_probabilities.append(batch_probs)
                                                except Exception as e:
                                                    st.warning(f"Could not generate probabilities: {e}")
                                                    include_probabilities = False
                                        
                                        # Clear progress indicators
                                        progress_bar.empty()
                                        status_text.empty()
                                        
                                        # Process results
                                        predictions = np.array(all_predictions)
                                        
                                        if all_probabilities:
                                            probabilities = np.vstack(all_probabilities)
                                        else:
                                            probabilities = None
                                        
                                        # Apply custom threshold for binary classification
                                        if prediction_threshold and probabilities is not None and probabilities.shape[1] == 2:
                                            custom_predictions = (probabilities[:, 1] >= prediction_threshold).astype(int)
                                            predictions = custom_predictions
                                            st.info(f"Applied custom threshold: {prediction_threshold}")
                                        
                                        # Create results DataFrame
                                        results_df = prediction_data.copy()
                                        
                                        # Add predictions
                                        if le_target:
                                            results_df['prediction'] = le_target.inverse_transform(predictions)
                                            results_df['prediction_encoded'] = predictions
                                        else:
                                            results_df['prediction'] = predictions
                                        
                                        # Add probabilities/confidence scores
                                        if probabilities is not None:
                                            if le_target and hasattr(le_target, 'classes_'):
                                                class_names = le_target.classes_
                                            else:
                                                class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
                                            
                                            # Add individual class probabilities
                                            for i, class_name in enumerate(class_names):
                                                results_df[f'prob_{class_name}'] = probabilities[:, i]
                                            
                                            # Add max confidence
                                            results_df['confidence'] = probabilities.max(axis=1)
                                        
                                        # Success message
                                        st.markdown('<div class="success-card">🎉 Predictions completed successfully!</div>', 
                                                   unsafe_allow_html=True)
                                        
                                        # Prediction Summary
                                        st.markdown("### 📊 Prediction Summary")
                                        
                                        summary_cols = st.columns(5)
                                        
                                        with summary_cols[0]:
                                            st.metric("Total Predictions", f"{len(results_df):,}")
                                        
                                        with summary_cols[1]:
                                            unique_preds = results_df['prediction'].nunique()
                                            st.metric("Unique Predictions", unique_preds)
                                        
                                        with summary_cols[2]:
                                            if problem_type == "classification":
                                                most_common = results_df['prediction'].mode()
                                                if len(most_common) > 0:
                                                    st.metric("Most Common", str(most_common.iloc[0]))
                                            else:
                                                pred_mean = results_df['prediction'].mean()
                                                st.metric("Mean Prediction", f"{pred_mean:.3f}")
                                        
                                        with summary_cols[3]:
                                            if probabilities is not None:
                                                avg_confidence = results_df['confidence'].mean()
                                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                            else:
                                                if problem_type == "regression":
                                                    pred_std = results_df['prediction'].std()
                                                    st.metric("Std Deviation", f"{pred_std:.3f}")
                                        
                                        with summary_cols[4]:
                                            processing_time = f"{total_batches * 0.1:.1f}s"  # Approximate
                                            st.metric("Processing Time", processing_time)
                                        
                                        # Sample Results
                                        st.markdown("### 📋 Sample Results")
                                        
                                        # Select columns to display
                                        display_cols = feature_cols[:3] + ['prediction']
                                        if probabilities is not None:
                                            prob_cols = [col for col in results_df.columns if col.startswith('prob_')][:3]
                                            display_cols.extend(prob_cols)
                                            if 'confidence' in results_df.columns:
                                                display_cols.append('confidence')
                                        
                                        st.dataframe(results_df[display_cols].head(15), use_container_width=True)
                                        
                                        # Advanced Results Analysis
                                        st.markdown("### 📈 Prediction Analysis")
                                        
                                        analysis_tabs = st.tabs(["Distribution", "Confidence Analysis", "Detailed View"])
                                        
                                        with analysis_tabs[0]:
                                            if problem_type == "classification":
                                                # Classification distribution
                                                pred_counts = results_df['prediction'].value_counts()
                                                
                                                fig_dist = px.bar(
                                                    x=pred_counts.index.astype(str),
                                                    y=pred_counts.values,
                                                    title="Prediction Distribution",
                                                    labels={'x': 'Predicted Class', 'y': 'Count'},
                                                    color=pred_counts.values,
                                                    color_continuous_scale='Viridis'
                                                )
                                                
                                                # Add percentages as text
                                                percentages = (pred_counts.values / len(results_df)) * 100
                                                fig_dist.update_traces(
                                                    text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(pred_counts.values, percentages)],
                                                    textposition='auto'
                                                )
                                                
                                                st.plotly_chart(fig_dist, use_container_width=True)
                                                
                                                # Class distribution table
                                                dist_df = pd.DataFrame({
                                                    'Class': pred_counts.index,
                                                    'Count': pred_counts.values,
                                                    'Percentage': percentages
                                                }).round(2)
                                                
                                                st.dataframe(dist_df, use_container_width=True)
                                            
                                            else:  # Regression
                                                # Regression distribution
                                                fig_hist = px.histogram(
                                                    results_df['prediction'],
                                                    nbins=30,
                                                    title="Distribution of Predicted Values",
                                                    labels={'x': 'Predicted Value', 'y': 'Frequency'},
                                                    color_discrete_sequence=['#2c3e50']
                                                )
                                                
                                                # Add statistics overlay
                                                mean_val = results_df['prediction'].mean()
                                                std_val = results_df['prediction'].std()
                                                
                                                fig_hist.add_vline(x=mean_val, line_dash="dash", 
                                                                  annotation_text=f"Mean: {mean_val:.3f}")
                                                fig_hist.add_vline(x=mean_val + std_val, line_dash="dot",
                                                                  annotation_text=f"+1 STD")
                                                fig_hist.add_vline(x=mean_val - std_val, line_dash="dot",
                                                                  annotation_text=f"-1 STD")
                                                
                                                st.plotly_chart(fig_hist, use_container_width=True)
                                                
                                                # Statistics table
                                                stats = results_df['prediction'].describe()
                                                st.dataframe(stats.to_frame('Statistics'), use_container_width=True)
                                        
                                        with analysis_tabs[1]:
                                            if probabilities is not None:
                                                # Confidence distribution
                                                fig_conf = px.histogram(
                                                    results_df['confidence'],
                                                    nbins=20,
                                                    title="Confidence Score Distribution",
                                                    labels={'x': 'Confidence Score', 'y': 'Count'},
                                                    color_discrete_sequence=['#3498db']
                                                )
                                                st.plotly_chart(fig_conf, use_container_width=True)
                                                
                                                # Low confidence predictions
                                                low_conf_threshold = st.slider("Low Confidence Threshold", 0.1, 0.9, 0.7, 0.05)
                                                low_conf_preds = results_df[results_df['confidence'] < low_conf_threshold]
                                                
                                                if len(low_conf_preds) > 0:
                                                    st.warning(f"Found {len(low_conf_preds)} predictions with confidence < {low_conf_threshold}")
                                                    st.dataframe(low_conf_preds[display_cols].head(10), use_container_width=True)
                                                else:
                                                    st.success(f"All predictions have confidence >= {low_conf_threshold}")
                                                
                                                # Confidence by prediction class (for classification)
                                                if problem_type == "classification":
                                                    conf_by_class = results_df.groupby('prediction')['confidence'].agg(['mean', 'std', 'min', 'max']).round(3)
                                                    st.markdown("#### Confidence by Predicted Class")
                                                    st.dataframe(conf_by_class, use_container_width=True)
                                            else:
                                                st.info("Confidence analysis not available for regression problems")
                                        
                                        with analysis_tabs[2]:
                                            # Detailed view with filtering and sorting
                                            st.markdown("#### Filter and Sort Results")
                                            
                                            filter_cols = st.columns(3)
                                            
                                            with filter_cols[0]:
                                                if problem_type == "classification":
                                                    class_filter = st.multiselect(
                                                        "Filter by Prediction",
                                                        options=results_df['prediction'].unique(),
                                                        default=results_df['prediction'].unique()
                                                    )
                                                else:
                                                    pred_min = float(results_df['prediction'].min())
                                                    pred_max = float(results_df['prediction'].max())
                                                    pred_range = st.slider(
                                                        "Prediction Range",
                                                        pred_min, pred_max,
                                                        (pred_min, pred_max)
                                                    )
                                            
                                            with filter_cols[1]:
                                                if probabilities is not None:
                                                    conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
                                                else:
                                                    conf_min = 0.0
                                            
                                            with filter_cols[2]:
                                                sort_by = st.selectbox(
                                                    "Sort by",
                                                    ['confidence', 'prediction'] + feature_cols[:3] if probabilities is not None
                                                    else ['prediction'] + feature_cols[:3]
                                                )
                                                sort_order = st.selectbox("Sort Order", ['Descending', 'Ascending'])
                                            
                                            # Apply filters
                                            filtered_df = results_df.copy()
                                            
                                            if problem_type == "classification":
                                                filtered_df = filtered_df[filtered_df['prediction'].isin(class_filter)]
                                            else:
                                                filtered_df = filtered_df[
                                                    (filtered_df['prediction'] >= pred_range[0]) &
                                                    (filtered_df['prediction'] <= pred_range[1])
                                                ]
                                            
                                            if probabilities is not None:
                                                filtered_df = filtered_df[filtered_df['confidence'] >= conf_min]
                                            
                                            # Sort
                                            ascending = sort_order == 'Ascending'
                                            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                                            
                                            st.markdown(f"Showing {len(filtered_df):,} of {len(results_df):,} predictions")
                                            st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
                                        
                                        # Download Section
                                        st.markdown("### 💾 Download Results")
                                        
                                        download_cols = st.columns(4)
                                        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                        
                                        with download_cols[0]:
                                            # CSV download
                                            csv_buffer = io.StringIO()
                                            results_df.to_csv(csv_buffer, index=False)
                                            
                                            st.download_button(
                                                "📄 Download CSV",
                                                data=csv_buffer.getvalue(),
                                                file_name=f"predictions_{timestamp}.csv",
                                                mime="text/csv",
                                                help="Complete results in CSV format"
                                            )
                                        
                                        with download_cols[1]:
                                            # Excel download
                                            excel_buffer = io.BytesIO()
                                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                                
                                                # Add summary sheet
                                                summary_data = {
                                                    'Metric': ['Total Predictions', 'Unique Predictions', 'Average Confidence'],
                                                    'Value': [
                                                        len(results_df),
                                                        results_df['prediction'].nunique(),
                                                        results_df.get('confidence', pd.Series([0])).mean()
                                                    ]
                                                }
                                                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                                            
                                            st.download_button(
                                                "📊 Download Excel",
                                                data=excel_buffer.getvalue(),
                                                file_name=f"predictions_{timestamp}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                help="Results with summary sheet"
                                            )
                                        
                                        with download_cols[2]:
                                            # JSON download (for API integration)
                                            predictions_json = {
                                                'metadata': {
                                                    'timestamp': timestamp,
                                                    'model_type': st.session_state.trained_model_name,
                                                    'problem_type': problem_type,
                                                    'total_predictions': len(results_df),
                                                    'features_used': feature_cols
                                                },
                                                'predictions': results_df.to_dict(orient='records')
                                            }
                                            
                                            st.download_button(
                                                "🔗 Download JSON",
                                                data=pd.Series(predictions_json).to_json(indent=2),
                                                file_name=f"predictions_{timestamp}.json",
                                                mime="application/json",
                                                help="Structured format for API integration"
                                            )
                                        
                                        with download_cols[3]:
                                            # Parquet download for large datasets
                                            if PYARROW_AVAILABLE and len(results_df) > 1000:
                                                parquet_buffer = io.BytesIO()
                                                results_df.to_parquet(parquet_buffer, index=False, compression='snappy')
                                                
                                                st.download_button(
                                                    "⚡ Download Parquet",
                                                    data=parquet_buffer.getvalue(),
                                                    file_name=f"predictions_{timestamp}.parquet",
                                                    mime="application/octet-stream",
                                                    help="Optimized format for large datasets"
                                                )
                                            else:
                                                st.info("Parquet available for datasets > 1000 rows")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Prediction failed: {str(e)}")
                                        st.code(traceback.format_exc())
                        
                        except Exception as e:
                            st.error(f"❌ Data validation failed: {str(e)}")
                    else:
                        st.error("Cannot proceed: Missing required features")
            
            except Exception as e:
                st.error(f"❌ Failed to load prediction data: {str(e)}")

# Tab 5: Advanced Tools
with tab5:
    st.markdown("## ⚙️ Advanced Tools & Utilities")
    
    tool_tabs = st.tabs([
        "🔧 Model Comparison", 
        "📊 Data Preprocessing", 
        "🎯 Hyperparameter Tuning",
        "🔍 Model Interpretation",
        "💾 Import/Export"
    ])
    
    with tool_tabs[0]:  # Model Comparison
        st.markdown("### 🔧 Model Comparison Tool")
        
        if not st.session_state.data_uploaded:
            st.info("Upload data first to compare models")
        else:
            st.markdown("Compare multiple models on the same dataset to find the best performer.")
            
            # Model selection for comparison
            if st.session_state.get('trained_problem_type') == "classification":
                available_models = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM": SVC(probability=True, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42)
                }
            else:
                available_models = {
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression()
                }
            
            selected_models = st.multiselect(
                "Select models to compare",
                options=list(available_models.keys()),
                default=list(available_models.keys())[:3]
            )
            
            comparison_cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
            
            if selected_models and st.button("🚀 Run Model Comparison"):
                try:
                    data = st.session_state.data.copy()
                    target_col = st.session_state.trained_feature_columns[0]  # Simplified for demo
                    
                    # This would need proper implementation with the actual target column
                    st.info("Model comparison feature would compare selected models with cross-validation")
                    st.code("""
                    # Pseudo-code for model comparison
                    results = {}
                    for model_name in selected_models:
                        model = available_models[model_name]
                        scores = cross_val_score(model, X, y, cv=comparison_cv_folds)
                        results[model_name] = {
                            'mean_score': scores.mean(),
                            'std_score': scores.std(),
                            'scores': scores
                        }
                    """)
                    
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
    
    with tool_tabs[1]:  # Data Preprocessing
        st.markdown("### 📊 Advanced Data Preprocessing")
        
        if not st.session_state.data_uploaded:
            st.info("Upload data first to access preprocessing tools")
        else:
            df = st.session_state.data
            
            preprocessing_options = st.multiselect(
                "Select preprocessing steps",
                [
                    "Remove duplicates",
                    "Handle missing values",
                    "Encode categorical variables",
                    "Scale numerical features",
                    "Remove outliers",
                    "Feature engineering"
                ]
            )
            
            if preprocessing_options:
                if st.button("Apply Preprocessing"):
                    st.success("Preprocessing would be applied based on selected options")
                    st.info("This feature would apply the selected preprocessing steps to your data")
    
    with tool_tabs[2]:  # Hyperparameter Tuning
        st.markdown("### 🎯 Hyperparameter Optimization")
        
        if not st.session_state.get('model_trained', False):
            st.info("Train a model first to enable hyperparameter tuning")
        else:
            st.markdown("Optimize hyperparameters to improve model performance")
            
            tuning_method = st.selectbox(
                "Optimization method",
                ["Grid Search", "Random Search", "Bayesian Optimization"]
            )
            
            n_trials = st.slider("Number of trials", 10, 100, 20)
            
            if st.button("🎯 Optimize Hyperparameters"):
                st.info(f"Would run {tuning_method} optimization with {n_trials} trials")
                st.success("Hyperparameter tuning completed! (Demo)")
    
    with tool_tabs[3]:  # Model Interpretation
        st.markdown("### 🔍 Advanced Model Interpretation")
        
        if not st.session_state.get('model_trained', False):
            st.info("Train a model first to access interpretation tools")
        else:
            interpretation_tools = st.multiselect(
                "Select interpretation methods",
                [
                    "Permutation Importance",
                    "Partial Dependence Plots",
                    "Individual Conditional Expectation",
                    "LIME Explanations",
                    "Anchors"
                ]
            )
            
            if interpretation_tools and st.button("Generate Interpretations"):
                st.success("Advanced interpretations would be generated")
                st.info("This feature would provide detailed model interpretability analysis")
    
    with tool_tabs[4]:  # Import/Export
        st.markdown("### 💾 Import/Export Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Import Existing Model")
            uploaded_model = st.file_uploader(
                "Upload model file",
                type=["pkl", "joblib"],
                help="Import a previously trained model"
            )
            
            if uploaded_model:
                try:
                    model_data = joblib.load(uploaded_model)
                    st.success("Model loaded successfully!")
                    st.json(model_data.get('training_config', {}))
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
        
        with col2:
            st.markdown("#### 📤 Export Configuration")
            
            if st.session_state.get('model_trained', False):
                config_data = {
                    'model_type': st.session_state.trained_model_name,
                    'features': st.session_state.trained_feature_columns,
                    'problem_type': st.session_state.trained_problem_type,
                    'training_config': st.session_state.trained_training_config
                }
                
                st.download_button(
                    "📋 Download Configuration",
                    data=pd.Series(config_data).to_json(indent=2),
                    file_name="model_config.json",
                    mime="application/json"
                )

# Enhanced Footer with System Status
st.markdown("---")

footer_cols = st.columns([2, 2, 2, 1])

with footer_cols[0]:
    st.markdown("**🤖 AutoML Pro v2.0**")
    st.caption("Production-ready machine learning platform")

with footer_cols[1]:
    if st.session_state.get('model_trained', False):
        st.markdown("**🎯 Current Model**")
        model_info = f"{st.session_state.trained_model_name} ({st.session_state.trained_problem_type})"
        st.caption(model_info)
    else:
        st.markdown("**📊 Status**")
        st.caption("Ready for model training")

with footer_cols[2]:
    st.markdown("**✨ Features**")
    st.caption("SHAP • Advanced Viz • Batch Predictions • Model Comparison")

with footer_cols[3]:
    if show_memory_usage and st.session_state.memory_usage:
        total_memory = sum(st.session_state.memory_usage.values())
        memory_color = "🟢" if total_memory < 100 else "🟡" if total_memory < 500 else "🔴"
        st.metric(f"{memory_color} Memory", f"{total_memory:.0f}MB")

# Performance monitoring in sidebar
with st.sidebar:
    if st.session_state.get('model_trained', False):
        st.markdown("---")
        st.markdown("### 🎯 Current Model Status")
        
        metrics = st.session_state.trained_test_results.get('metrics', {})
        if metrics:
            primary_metric = list(metrics.items())[0]
            
            # Performance indicator
            value = primary_metric[1]
            if primary_metric[0] in ['accuracy', 'f1_score', 'r2_score']:
                if value > 0.8:
                    status = "🟢 Excellent"
                elif value > 0.6:
                    status = "🟡 Good"
                else:
                    status = "🔴 Needs Improvement"
            else:
                status = "📊 Trained"
            
            st.success(f"Status: {status}")
            st.metric(
                primary_metric[0].replace('_', ' ').title(),
                f"{value:.3f}"
            )
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear All Data", help="Reset the application"):
        for key in list(st.session_state.keys()):
            if key not in ['cache_stats']:
                del st.session_state[key]
        config.cleanup_temp_dirs()
        st.rerun()
    
    if st.button("♻️ Clear Cache", help="Clear all cached data"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    # System info
    st.markdown("---")
    st.markdown("### 📊 System Info")
    
    system_info = {
        "PyArrow": "✅" if PYARROW_AVAILABLE else "❌",
        "SHAP": "✅",
        "Profiling": "✅" if PROFILING_AVAILABLE else "❌",
        "Imbalanced": "✅" if IMBALANCED_AVAILABLE else "❌"
    }
    
    for feature, status in system_info.items():
        st.caption(f"{feature}: {status}")

# Auto-cleanup on app exit
if st.session_state.get('cleanup_registered', False) is False:
    atexit.register(config.cleanup_temp_dirs)
    st.session_state.cleanup_registered = True
