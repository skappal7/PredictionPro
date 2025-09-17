"""
Working AutoML App - All Issues Fixed
- Complete prediction functionality
- Working SHAP explanations
- ROC curves and model evaluation
- File upload for predictions
- Professional UI
"""

import io
import os
import tempfile
import traceback
import joblib
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

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
                           roc_curve, auc, f1_score, precision_recall_curve, mean_squared_error, r2_score)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from ydata_profiling import ProfileReport

# Page config
st.set_page_config(page_title="AutoML Pro", layout="wide", page_icon="🤖")

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #34495e 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 6px;
        font-weight: 600;
        color: #2c3e50;
        border: 1px solid #bdc3c7;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        border: 1px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Session state
def init_session_state():
    defaults = {
        "data_uploaded": False,
        "model_trained": False,
        "current_file": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Utility functions
@st.cache_data(show_spinner=False)
def load_data_safe(uploaded_file):
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
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        return None

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def create_shap_explanation_safe(model, X_sample):
    """Working SHAP implementation"""
    try:
        # Get the actual classifier
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('model', model)
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                X_transformed = preprocessor.transform(X_sample)
            else:
                X_transformed = X_sample.values
        else:
            classifier = model
            X_transformed = X_sample.values
        
        # Limit sample size
        max_samples = min(100, len(X_transformed))
        X_shap = X_transformed[:max_samples]
        
        # Tree explainer for tree models
        if hasattr(classifier, 'feature_importances_'):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap)
        else:
            # Kernel explainer for other models
            background = X_transformed[:min(50, len(X_transformed))]
            explainer = shap.KernelExplainer(classifier.predict, background)
            shap_values = explainer.shap_values(X_shap[:20])
        
        return shap_values, X_shap, X_sample.columns.tolist()
        
    except Exception as e:
        st.warning(f"SHAP failed: {str(e)}")
        return None, None, None

# Header
st.markdown("""
<div class="main-header">
    <h1>AutoML Analytics Pro</h1>
    <p>Complete machine learning platform with explanations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Upload Dataset")
    
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
                
                st.success("Data loaded successfully!")
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"File: {uploaded_file.name}\nRows: {df.shape[0]:,}\nColumns: {df.shape[1]}\nSize: {file_size:.1f} MB")

# Main tabs
tabs = st.tabs(["Data Explorer", "Model Training", "Model Analysis", "Predictions"])

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("## Data Explorer")
    
    if not st.session_state.data_uploaded:
        st.info("Upload a dataset in the sidebar to begin")
    else:
        df = st.session_state.data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Rows</h3><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Columns</h3><h2>{df.shape[1]}</h2></div>', unsafe_allow_html=True)
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f'<div class="metric-card"><h3>Memory</h3><h2>{memory_mb:.1f} MB</h2></div>', unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f'<div class="metric-card"><h3>Missing</h3><h2>{missing_pct:.1f}%</h2></div>', unsafe_allow_html=True)
        
        # Data types chart
        st.markdown("### Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=[str(dtype) for dtype in dtype_counts.index],
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(25), use_container_width=True)
        
        # Basic profiling
        if st.button("Generate Profile Report"):
            try:
                profile = ProfileReport(df.sample(min(1000, len(df))), minimal=True, title="Dataset Profile")
                profile_html = profile.to_html()
                st.components.v1.html(profile_html, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Profiling failed: {e}")

# Tab 2: Model Training
with tabs[1]:
    st.markdown("## Model Training")
    
    if not st.session_state.data_uploaded:
        st.info("Upload data first")
    else:
        data = st.session_state.data.copy()
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            all_columns = list(data.columns)
            target_column = st.selectbox("Target Variable", all_columns)
        
        with col2:
            available_features = [col for col in all_columns if col != target_column]
            feature_cols = st.multiselect("Features (leave empty for all)", available_features)
            if not feature_cols:
                feature_cols = available_features
        
        # Data preparation
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()
        
        if y_raw.dtype == "object" or y_raw.nunique() <= 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_.tolist()
            problem_type = "classification"
            st.success(f"Classification problem detected - {len(class_names)} classes")
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
            st.success("Regression problem detected")
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            if problem_type == "classification":
                model_options = {
                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM": SVC(probability=True, random_state=42)
                }
            else:
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import LinearRegression
                model_options = {
                    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression()
                }
            
            model_choice = st.selectbox("Algorithm", list(model_options.keys()))
            base_model = model_options[model_choice]
        
        with col2:
            if problem_type == "classification":
                balance_method = st.selectbox("Class Balancing", ["None", "SMOTE"])
            else:
                balance_method = "None"
            
            test_size = st.slider("Test Size (%)", 10, 50, 20)
        
        # Training
        if st.button("Train Model", type="primary"):
            try:
                # Split data
                if problem_type == "classification":
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                
                # Create pipeline
                numeric_features = X.select_dtypes(include=[np.number]).columns
                categorical_features = X.select_dtypes(exclude=[np.number]).columns
                
                preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), numeric_features),
                    ('cat', safe_onehot_encoder(), categorical_features)
                ])
                
                pipeline_steps = [('preprocessor', preprocessor)]
                
                if balance_method == "SMOTE" and problem_type == "classification":
                    pipeline_steps.append(('sampler', SMOTE(random_state=42)))
                
                pipeline_steps.append(('model', base_model))
                
                if balance_method == "SMOTE":
                    model_pipeline = ImbPipeline(pipeline_steps)
                else:
                    model_pipeline = Pipeline(pipeline_steps)
                
                # Train
                with st.spinner("Training model..."):
                    model_pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model_pipeline.predict(X_test)
                
                if problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    metrics = {'accuracy': accuracy, 'f1_score': f1}
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {'r2_score': r2, 'mse': mse}
                
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
                st.success(f"Model trained successfully!")
                
                col1, col2 = st.columns(2)
                for i, (name, value) in enumerate(metrics.items()):
                    with col1 if i % 2 == 0 else col2:
                        st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
                
                # Model download
                st.markdown("### Download Model")
                try:
                    package = {
                        'model': model_pipeline,
                        'feature_columns': feature_cols,
                        'label_encoder': le_target,
                        'model_name': model_choice,
                        'metrics': metrics
                    }
                    
                    buffer = io.BytesIO()
                    joblib.dump(package, buffer)
                    
                    st.download_button(
                        "Download Model Package",
                        data=buffer.getvalue(),
                        file_name=f"model_{model_choice.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Model serialization failed: {e}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

# Tab 3: Model Analysis
with tabs[2]:
    st.markdown("## Model Analysis & Explanations")
    
    if not st.session_state.get('model_trained', False):
        st.info("Train a model first to see analysis")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.test_results['y_test']
        y_pred = st.session_state.test_results['y_pred']
        problem_type = st.session_state.problem_type
        
        # Performance metrics
        st.markdown("### Model Performance")
        
        if problem_type == "classification":
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            if st.session_state.trained_class_names:
                labels = st.session_state.trained_class_names
            else:
                labels = [f"Class {i}" for i in range(len(cm))]
            
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", x=labels, y=labels)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # ROC Curve for binary classification
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
                        line=dict(color='#2c3e50', width=3)
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
                    
                    # Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=recall, y=precision,
                        mode='lines',
                        name='Precision-Recall Curve',
                        line=dict(color='#3498db', width=3)
                    ))
                    
                    fig_pr.update_layout(
                        title='Precision-Recall Curve',
                        xaxis_title='Recall',
                        yaxis_title='Precision'
                    )
                    
                    st.plotly_chart(fig_pr, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"ROC/PR curves failed: {e}")
            
            # Classification Report
            try:
                report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
            except:
                pass
        
        else:
            # Regression plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted")
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                               mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                residuals = y_test - y_pred
                fig_residuals = px.scatter(x=y_pred, y=residuals, title="Residuals Plot")
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        # SHAP Explanations
        st.markdown("### SHAP Explanations")
        
        with st.spinner("Generating SHAP explanations..."):
            shap_values, X_shap, feature_names = create_shap_explanation_safe(model, X_test)
            
            if shap_values is not None:
                try:
                    st.success("SHAP explanations generated successfully")
                    
                    # Handle different SHAP formats
                    if isinstance(shap_values, list):
                        if len(shap_values) > 1:
                            plot_values = shap_values[1]  # Use positive class for binary
                        else:
                            plot_values = shap_values[0]
                    else:
                        plot_values = shap_values
                    
                    # Summary plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Convert to numpy if needed
                    if hasattr(plot_values, 'values'):
                        plot_array = plot_values.values
                    else:
                        plot_array = np.array(plot_values)
                    
                    shap.summary_plot(plot_array, X_shap, feature_names=feature_names, show=False)
                    st.pyplot(fig)
                    
                    # Individual explanation
                    st.markdown("#### Individual Sample Explanation")
                    sample_idx = st.slider("Sample to explain", 0, min(19, len(plot_array)-1), 0)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    
                    # Get SHAP values for selected sample
                    sample_shap = plot_array[sample_idx]
                    
                    # Create bar plot
                    indices = np.argsort(np.abs(sample_shap))[-10:]  # Top 10 features
                    colors = ['#2c3e50' if val > 0 else '#e74c3c' for val in sample_shap[indices]]
                    
                    plt.barh(range(len(indices)), sample_shap[indices], color=colors)
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel("SHAP Value")
                    plt.title(f"SHAP Values for Sample {sample_idx + 1}")
                    plt.axvline(x=0, color='black', alpha=0.3)
                    
                    st.pyplot(fig2)
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {str(e)}")
            else:
                st.warning("SHAP explanations not available")
        
        # Feature Importance Fallback
        st.markdown("### Feature Importance")
        
        try:
            classifier = model.named_steps.get('model', model)
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                try:
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                except:
                    feature_names = X_test.columns.tolist()
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(15)
                
                fig_importance = px.bar(
                    importance_df, x='Importance', y='Feature', orientation='h',
                    title="Top 15 Feature Importances",
                    color='Importance', color_continuous_scale=['#3498db', '#2c3e50']
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                
                coef_df = pd.DataFrame({
                    'Feature': X_test.columns[:len(coefficients)],
                    'Coefficient': np.abs(coefficients)
                }).sort_values('Coefficient', ascending=True).tail(15)
                
                fig_coef = px.bar(
                    coef_df, x='Coefficient', y='Feature', orientation='h',
                    title="Top 15 Feature Coefficients",
                    color='Coefficient', color_continuous_scale=['#3498db', '#2c3e50']
                )
                st.plotly_chart(fig_coef, use_container_width=True)
            
            else:
                st.info("No feature importance available for this model type")
                
        except Exception as e:
            st.error(f"Feature importance analysis failed: {e}")

# Tab 4: Predictions
with tabs[3]:
    st.markdown("## Make Predictions")
    
    if not st.session_state.get('model_trained', False):
        st.info("Train a model first to make predictions on new data")
        
        st.markdown("""
        ### What you can do after training:
        - Upload new data with the same features
        - Get predictions with confidence scores
        - Download results in multiple formats
        """)
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        problem_type = st.session_state.problem_type
        
        st.markdown("### Upload New Data for Predictions")
        st.info(f"Required features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
        
        pred_file = st.file_uploader(
            "Upload prediction data",
            type=["csv", "xlsx", "parquet"],
            key="prediction_file"
        )
        
        if pred_file:
            try:
                prediction_data = load_data_safe(pred_file)
                
                if prediction_data is None:
                    st.error("Failed to load file")
                else:
                    st.success(f"Loaded {len(prediction_data):,} rows for prediction")
                    
                    # Preview data
                    st.markdown("### Data Preview")
                    st.dataframe(prediction_data.head(10), use_container_width=True)
                    
                    # Check features
                    missing_features = set(feature_cols) - set(prediction_data.columns)
                    extra_features = set(prediction_data.columns) - set(feature_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if missing_features:
                            st.error(f"Missing required features: {list(missing_features)}")
                        else:
                            st.success("All required features present!")
                    
                    with col2:
                        if extra_features:
                            st.warning(f"Extra columns will be ignored: {len(extra_features)}")
                        st.info(f"Ready to predict on {len(prediction_data):,} rows")
                    
                    if not missing_features:
                        # Prediction options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            include_probabilities = st.checkbox(
                                "Include prediction probabilities",
                                value=problem_type == "classification",
                                disabled=problem_type == "regression"
                            )
                        
                        with col2:
                            batch_size = st.selectbox("Batch processing size", [1000, 5000, 10000], index=0)
                        
                        # Generate predictions
                        if st.button("Generate Predictions", type="primary"):
                            pred_features = prediction_data[feature_cols]
                            
                            with st.spinner("Making predictions..."):
                                try:
                                    all_predictions = []
                                    all_probabilities = []
                                    
                                    progress_bar = st.progress(0)
                                    
                                    # Process in batches
                                    for i in range(0, len(pred_features), batch_size):
                                        batch = pred_features.iloc[i:i+batch_size]
                                        progress = min((i + len(batch)) / len(pred_features), 1.0)
                                        progress_bar.progress(progress)
                                        
                                        # Make predictions
                                        batch_preds = model.predict(batch)
                                        all_predictions.extend(batch_preds)
                                        
                                        # Get probabilities
                                        if include_probabilities and hasattr(model, 'predict_proba'):
                                            try:
                                                batch_probs = model.predict_proba(batch)
                                                all_probabilities.append(batch_probs)
                                            except:
                                                pass
                                    
                                    progress_bar.progress(1.0)
                                    
                                    # Combine results
                                    predictions = np.array(all_predictions)
                                    
                                    if all_probabilities:
                                        probabilities = np.vstack(all_probabilities)
                                    else:
                                        probabilities = None
                                    
                                    # Create results dataframe
                                    results_df = prediction_data.copy()
                                    
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
                                    
                                    # Summary statistics
                                    st.markdown("### Prediction Summary")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Total Predictions", f"{len(results_df):,}")
                                    
                                    with col2:
                                        unique_preds = results_df['prediction'].nunique()
                                        st.metric("Unique Predictions", unique_preds)
                                    
                                    with col3:
                                        if problem_type == "classification":
                                            most_common = results_df['prediction'].mode().iloc[0]
                                            st.metric("Most Common", str(most_common))
                                        else:
                                            pred_mean = results_df['prediction'].mean()
                                            st.metric("Mean Prediction", f"{pred_mean:.3f}")
                                    
                                    with col4:
                                        if probabilities is not None:
                                            avg_confidence = probabilities.max(axis=1).mean()
                                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                        else:
                                            if problem_type == "regression":
                                                pred_std = results_df['prediction'].std()
                                                st.metric("Std Deviation", f"{pred_std:.3f}")
                                    
                                    # Sample results
                                    st.markdown("### Sample Results")
                                    display_cols = feature_cols[:3] + ['prediction']
                                    if probabilities is not None:
                                        prob_cols = [col for col in results_df.columns if col.startswith('prob_')][:3]
                                        display_cols.extend(prob_cols)
                                    
                                    st.dataframe(results_df[display_cols].head(10), use_container_width=True)
                                    
                                    # Distribution visualization
                                    if problem_type == "classification" and results_df['prediction'].nunique() <= 20:
                                        st.markdown("### Prediction Distribution")
                                        pred_counts = results_df['prediction'].value_counts()
                                        
                                        fig_dist = px.bar(
                                            x=pred_counts.index,
                                            y=pred_counts.values,
                                            title="Distribution of Predictions",
                                            color_discrete_sequence=['#2c3e50']
                                        )
                                        st.plotly_chart(fig_dist, use_container_width=True)
                                    
                                    elif problem_type == "regression":
                                        st.markdown("### Prediction Distribution")
                                        fig_hist = px.histogram(
                                            results_df['prediction'],
                                            nbins=30,
                                            title="Distribution of Predicted Values",
                                            color_discrete_sequence=['#2c3e50']
                                        )
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                    
                                    # Download options
                                    st.markdown("### Download Results")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                    
                                    with col1:
                                        # CSV download
                                        csv_buffer = io.StringIO()
                                        results_df.to_csv(csv_buffer, index=False)
                                        
                                        st.download_button(
                                            "Download as CSV",
                                            data=csv_buffer.getvalue(),
                                            file_name=f"predictions_{timestamp}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    with col2:
                                        # Excel download
                                        excel_buffer = io.BytesIO()
                                        results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                        
                                        st.download_button(
                                            "Download as Excel",
                                            data=excel_buffer.getvalue(),
                                            file_name=f"predictions_{timestamp}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    
                                    with col3:
                                        # Parquet download for large files
                                        if len(results_df) > 1000:
                                            import pyarrow as pa
                                            import pyarrow.parquet as pq
                                            
                                            parquet_buffer = io.BytesIO()
                                            results_df.to_parquet(parquet_buffer, index=False)
                                            
                                            st.download_button(
                                                "Download as Parquet",
                                                data=parquet_buffer.getvalue(),
                                                file_name=f"predictions_{timestamp}.parquet",
                                                mime="application/octet-stream"
                                            )
                                        else:
                                            st.info("Parquet download available for datasets > 1000 rows")
                                
                                except Exception as e:
                                    st.error(f"Prediction failed: {str(e)}")
                                    st.text(traceback.format_exc())
                
            except Exception as e:
                st.error(f"Failed to load prediction file: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**AutoML Pro v2.0**")
    st.caption("Complete machine learning platform")

with col2:
    if st.session_state.get('model_trained', False):
        st.markdown("**Current Model**")
        st.caption(f"{st.session_state.model_name} ({st.session_state.problem_type})")
    else:
        st.markdown("**Status**")
        st.caption("Ready for training")

with col3:
    st.markdown("**Features**")
    st.caption("SHAP • ROC Curves • Model Download • Predictions")

# Sidebar info
with st.sidebar:
    if st.session_state.data_uploaded:
        st.markdown("---")
        st.markdown("### Dataset Info")
        df = st.session_state.data
        st.markdown(f"""
        **File:** {st.session_state.current_file}  
        **Shape:** {df.shape[0]:,} × {df.shape[1]}  
        **Memory:** {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB
        """)
        
        if st.session_state.get('model_trained', False):
            st.markdown("### Model Performance")
            metrics = st.session_state.test_results['metrics']
            primary_metric = list(metrics.items())[0]
            st.metric(primary_metric[0].replace('_', ' ').title(), f"{primary_metric[1]:.4f}")
    else:
        st.markdown("---")
        st.markdown("### Quick Start")
        st.markdown("""
        1. Upload your dataset above
        2. Explore data in first tab
        3. Train model in second tab
        4. View analysis and SHAP in third tab
        5. Make predictions in fourth tab
        """)
        
        st.markdown("### Key Features")
        st.markdown("✅ Multiple algorithms")
        st.markdown("✅ SHAP explanations")
        st.markdown("✅ ROC/PR curves")
        st.markdown("✅ Model downloads")
        st.markdown("✅ Batch predictions")
