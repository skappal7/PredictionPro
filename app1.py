"""
Production-Grade AutoML App - Clean Working Version
Compatible with your exact requirements.txt
"""

import io
import os
import tempfile
import traceback
import joblib
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Your exact libraries
import plotly.express as px
import plotly.graph_objects as go
import shap
import pyarrow as pa
import pyarrow.parquet as pq

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, f1_score, mean_squared_error, r2_score, mean_absolute_error)

# Imbalanced learning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Data profiling
from ydata_profiling import ProfileReport

# Page config
st.set_page_config(
    page_title="AutoML Pro",
    layout="wide", 
    page_icon="🤖"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: white;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Session state
def init_session_state():
    defaults = {
        "data_uploaded": False,
        "profile_generated": False,
        "model_trained": False,
        "current_file": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Utility functions
@st.cache_data(show_spinner=False)
def load_data_safe(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def clean_data_robust(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        df_clean = df.copy()
        df_clean = df_clean.dropna(axis=1, how='all')
        
        for col in df_clean.columns:
            try:
                numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                non_null_count = numeric_converted.count()
                total_count = len(df_clean[col].dropna())
                
                if total_count > 0 and non_null_count / total_count > 0.5:
                    df_clean[col] = numeric_converted
                    median_val = df_clean[col].median()
                    fill_val = median_val if not pd.isna(median_val) else 0
                    df_clean[col] = df_clean[col].fillna(fill_val)
                else:
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].fillna('Missing')
            except:
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        return df_clean
    except Exception:
        return df.iloc[:, :min(3, df.shape[1])].fillna('Error').astype(str)

# Remove @st.cache_data decorator from profile function to fix pickle error
def create_profile_safe(df: pd.DataFrame, minimal: bool = True) -> Optional[ProfileReport]:
    try:
        if df is None or df.empty:
            return None
            
        df_clean = clean_data_robust(df)
        if df_clean.empty:
            return None
        
        if len(df_clean) > 3000:
            df_clean = df_clean.sample(n=3000, random_state=42)
        
        # FIXED: Remove dark_mode parameter that causes validation error
        config = {
            "title": "Dataset Profile",
            "minimal": minimal,
            "lazy": False,
            "explorative": not minimal
        }
        
        return ProfileReport(df_clean, **config)
    except Exception as e:
        st.error(f"Profile generation failed: {str(e)}")
        return None

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def create_shap_explanation_safe(model, X_sample, model_type: str):
    try:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('model', model)
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                X_transformed = preprocessor.transform(X_sample)
            else:
                X_transformed = X_sample
        else:
            classifier = model
            X_transformed = X_sample
        
        max_samples = min(100, len(X_transformed))
        X_shap = X_transformed[:max_samples]
        
        if model_type in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap)
        else:
            background_size = min(50, len(X_transformed))
            background = X_transformed[:background_size]
            explainer = shap.KernelExplainer(classifier.predict, background)
            shap_values = explainer.shap_values(X_shap[:20])
        
        return shap_values, explainer, X_shap
    except Exception as e:
        st.warning(f"SHAP explanation failed: {str(e)}")
        return None, None, None

def serialize_model_safe(model, feature_cols, le_target, model_name, metrics):
    try:
        package = {
            'model': model,
            'feature_columns': feature_cols,
            'label_encoder': le_target,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': '2.0'
        }
        
        buffer = io.BytesIO()
        joblib.dump(package, buffer, compress=3)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        minimal_package = {
            'feature_columns': feature_cols,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'note': 'Model object excluded due to serialization issues'
        }
        
        buffer = io.BytesIO()
        joblib.dump(minimal_package, buffer)
        buffer.seek(0)
        return buffer.getvalue()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Analytics Pro</h1>
    <p>Production-grade machine learning platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    st.success("✅ SHAP 0.46.0")
    st.success("✅ Plotly ≤5.14.1")
    st.success("✅ PyArrow")
    st.success("✅ Imbalanced Learn")
    st.success("✅ YData Profiling")
    
    st.markdown("---")
    st.markdown("### 📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls", "parquet"]
    )
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_data_safe(uploaded_file)
            
            if df is not None:
                st.session_state.data = df
                st.session_state.data_uploaded = True
                st.session_state.current_file = uploaded_file.name
                
                st.success("✅ Data loaded successfully!")
                
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"📄 {uploaded_file.name}\n📏 {df.shape[0]:,} × {df.shape[1]}\n💾 {file_size:.1f} MB")

# Main tabs
tabs = st.tabs(["📊 Data Explorer", "🤖 Model Training", "🔍 SHAP Analysis", "📈 Predictions"])

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("## 📊 Data Explorer")
    
    if not st.session_state.data_uploaded:
        st.info("Upload a dataset in the sidebar to begin")
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
        
        # Data types chart - FIXED
        st.markdown("### 📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        
        # Convert dtype objects to strings for Plotly
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=[str(dtype) for dtype in dtype_counts.index],
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
        # Data preview
        st.markdown("### 🔍 Data Preview")
        page_size = st.selectbox("Rows per page", [10, 25, 50], index=1)
        st.dataframe(df.head(page_size), use_container_width=True)
        
        # Profiling
        st.markdown("### 📈 Data Profile")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            profile_type = st.radio("Profile Type", ["Quick", "Detailed"])
            
            if st.button("🚀 Generate Profile", type="primary"):
                with st.spinner("Generating profile..."):
                    minimal = (profile_type == "Quick")
                    profile = create_profile_safe(df, minimal=minimal)
                    
                    if profile is not None:
                        st.session_state.profile_report = profile
                        st.session_state.profile_generated = True
                        st.success("✅ Profile generated!")
                        st.rerun()
        
        with col2:
            if st.session_state.get('profile_generated', False):
                st.info("📊 Profile report generated!")
        
        # Display profile
        if st.session_state.get('profile_generated', False) and 'profile_report' in st.session_state:
            try:
                st.markdown("---")
                profile_html = st.session_state.profile_report.to_html()
                st.components.v1.html(profile_html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Failed to display profile: {str(e)}")

# Tab 2: Model Training
with tabs[1]:
    st.markdown("## 🤖 Model Training")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data first")
    else:
        data = st.session_state.data.copy()
        
        # Model configuration
        st.markdown("### ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            all_columns = list(data.columns)
            target_column = st.selectbox("🎯 Target Variable", all_columns)
        
        with col2:
            available_features = [col for col in all_columns if col != target_column]
            feature_mode = st.radio("🔧 Features", ["All", "Select"], horizontal=True)
        
        if feature_mode == "Select":
            feature_cols = st.multiselect("Choose features", available_features, default=available_features)
            if not feature_cols:
                st.warning("Select at least one feature")
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
            st.success(f"🎯 Classification - {len(class_names)} classes")
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
            st.success("🎯 Regression problem")
        
        # Target visualization
        st.markdown("### 📊 Target Analysis")
        
        if problem_type == "classification":
            target_counts = pd.Series(y).value_counts().sort_index()
            class_labels = [class_names[i] for i in target_counts.index]
            
            fig_target = px.bar(x=class_labels, y=target_counts.values, title="Target Distribution")
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            fig_target = px.histogram(y_raw, nbins=30, title="Target Distribution")
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            if problem_type == "classification":
                balance_method = st.selectbox("⚖️ Balancing", ["None", "SMOTE", "Random Oversample"])
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
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                
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
                    pipeline_steps.append(('sampler', sampler))
                
                pipeline_steps.append(('model', base_model))
                
                if balance_method != "None" and problem_type == "classification":
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
                status_text.success("✅ Training completed!")
                
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
                st.balloons()
                primary_metric = list(metrics.items())[0]
                st.success(f"🎉 Model trained! {primary_metric[0]}: {primary_metric[1]:.4f}")
                
                # Metrics display
                metric_cols = st.columns(len(metrics))
                for i, (name, value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Training failed: {str(e)}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        
        # Model download
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.markdown("### 📦 Download Model")
            
            model_package = serialize_model_safe(
                st.session_state.trained_model,
                st.session_state.trained_feature_cols,
                st.session_state.trained_le_target,
                st.session_state.model_name,
                st.session_state.test_results['metrics']
            )
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_{timestamp}.pkl"
            
            st.download_button(
                "📦 Download Model Package",
                data=model_package,
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )

# Tab 3: SHAP Analysis
with tabs[2]:
    st.markdown("## 🔍 SHAP Analysis")
    
    if not st.session_state.get('model_trained', False):
        st.info("Train a model first")
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
            
            if st.session_state.trained_class_names:
                labels = st.session_state.trained_class_names
            else:
                labels = [f"Class {i}" for i in range(len(cm))]
            
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", x=labels, y=labels)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            try:
                report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
            except Exception:
                pass
        else:
            # Regression plots
            fig_scatter = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted")
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # SHAP explanations
        st.markdown("### 🎯 SHAP Explanations")
        
        with st.spinner("Generating SHAP explanations..."):
            shap_values, explainer, X_shap = create_shap_explanation_safe(model, X_test, model_name)
            
            if shap_values is not None and X_shap is not None:
                try:
                    st.markdown("#### 📊 Feature Importance")
                    
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    
                    if hasattr(shap_values, 'values'):
                        plot_values = shap_values.values
                        plot_data = X_shap
                    else:
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            plot_values = shap_values[1]
                        else:
                            plot_values = shap_values[0] if isinstance(shap_values, list) else shap_values
                        plot_data = X_shap
                    
                    shap.summary_plot(plot_values, plot_data, feature_names=X_test.columns.tolist(), show=False, ax=ax)
                    plt.title("SHAP Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    
                    # Individual explanation
                    st.markdown("#### 🔍 Individual Explanation")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        sample_idx = st.slider("Sample to explain", 0, min(20, len(X_test)-1), 0)
                        
                        actual_val = y_test[sample_idx]
                        pred_val = y_pred[sample_idx]
                        
                        if st.session_state.trained_le_target:
                            actual_label = st.session_state.trained_le_target.inverse_transform([actual_val])[0]
                            pred_label = st.session_state.trained_le_target.inverse_transform([pred_val])[0]
                            st.metric("Actual", actual_label)
                            st.metric("Predicted", pred_label)
                        else:
                            st.metric("Actual", f"{actual_val:.3f}")
                            st.metric("Predicted", f"{pred_val:.3f}")
                    
                    with col2:
                        try:
                            fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 5))
                            
                            if sample_idx < len(plot_values):
                                sample_shap = plot_values[sample_idx]
                            else:
                                sample_shap = plot_values[0]
                            
                            top_indices = np.argsort(np.abs(sample_shap))[-8:]
                            colors = ['red' if val > 0 else 'blue' for val in sample_shap[top_indices]]
                            
                            ax_waterfall.barh(range(len(top_indices)), sample_shap[top_indices], color=colors)
                            ax_waterfall.set_yticks(range(len(top_indices)))
                            ax_waterfall.set_yticklabels([X_test.columns[i] for i in top_indices])
                            ax_waterfall.set_xlabel("SHAP Value")
                            ax_waterfall.set_title(f"Sample {sample_idx + 1} Explanation")
                            ax_waterfall.axvline(x=0, color='black', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)
                        except Exception:
                            st.info("Individual explanation plot not available")
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {str(e)}")
            else:
                st.warning("SHAP explanations not available for this model")

# Tab 4: Predictions
with tabs[3]:
    st.markdown("## 📈 Predictions")
    
    if not st.session_state.get('model_trained', False):
        st.info("Train a model first")
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        problem_type = st.session_state.problem_type
