# -----------------------------
# app.py (Tab-based Predictive Analytics with YData Profiling)
# -----------------------------
# 1) Matplotlib: headless + safe font *before* any plotting
import matplotlib
matplotlib.use("Agg")  # Headless backend for Replit/Streamlit Cloud

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.family": "DejaVu Sans",  # bundled with matplotlib
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 2) Core libs
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import io

# 3) Data Profiling
from ydata_profiling import ProfileReport

# 4) ML stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 5) ExplainerDashboard
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# 6) Imbalanced learning (class balancing)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# -----------------------------
# Streamlit Page Config & Session State Initialization
# -----------------------------
st.set_page_config(page_title="Predictive Analytics App", layout="wide", page_icon="📊")

# Initialize session state variables
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'profile_generated' not in st.session_state:
    st.session_state.profile_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'show_warning' not in st.session_state:
    st.session_state.show_warning = False
if 'warning_message' not in st.session_state:
    st.session_state.warning_message = ""

# Helper functions
def reset_downstream_states(from_step):
    """Reset all states downstream from the given step"""
    if from_step <= 1:  # Data upload changed
        st.session_state.profile_generated = False
        st.session_state.model_trained = False
        st.session_state.show_warning = True
        st.session_state.warning_message = "⚠️ New data uploaded. All previous work has been reset."
    elif from_step <= 2:  # Model development changed
        st.session_state.model_trained = False
        st.session_state.show_warning = True
        st.session_state.warning_message = "⚠️ Model settings changed. Please retrain the model to continue."

def class_counts(y_arr):
    vc = pd.Series(y_arr).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    return None

@st.cache_data
def generate_profile_report(data):
    """Generate YData Profiling report"""
    profile = ProfileReport(
        data, 
        title="Dataset Profiling Report",
        explorative=True,
        minimal=False
    )
    return profile.to_html()

# -----------------------------
# Main App Layout
# -----------------------------
st.title("📊 Predictive Analytics Application")
st.markdown("A comprehensive platform for data analysis, model development, and predictions.")

# -----------------------------
# Sidebar - Data Upload
# -----------------------------
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("Upload your dataset to begin the analysis workflow.")
    
    uploaded_file = st.file_uploader(
        "Choose your dataset", 
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing your dataset"
    )
    
    if uploaded_file:
        # Check if this is a new file
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            reset_downstream_states(1)
            st.session_state.current_file = uploaded_file.name
        
        data = load_data(uploaded_file)
        if data is not None:
            st.session_state.data = data
            st.session_state.data_uploaded = True
            
            st.success("✅ Data uploaded successfully!")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Columns:** {len(data.columns)}")
            
            # Show basic info
            with st.expander("Quick Preview"):
                st.dataframe(data.head(3))
    else:
        st.session_state.data_uploaded = False

# Display warning message if exists
if st.session_state.show_warning:
    st.warning(st.session_state.warning_message)
    if st.button("✓ Acknowledge", key="ack_warning"):
        st.session_state.show_warning = False
        st.rerun()

# -----------------------------
# Tab Structure
# -----------------------------
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
    
    This step helps you understand your data before building predictive models.
    """)
    
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset in the sidebar to begin profiling.")
    else:
        data = st.session_state.data
        
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        if st.button("🔍 Generate Comprehensive Profile Report", type="primary"):
            with st.spinner("Generating detailed profiling report... This may take a moment."):
                try:
                    profile_html = generate_profile_report(data)
                    st.session_state.profile_html = profile_html
                    st.session_state.profile_generated = True
                    st.success("✅ Profile report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating profile: {str(e)}")
        
        # Display profile report if generated
        if st.session_state.profile_generated and 'profile_html' in st.session_state:
            st.subheader("📋 Comprehensive Data Profile")
            components.html(st.session_state.profile_html, height=800, scrolling=True)

# -----------------------------
# TAB 2: Model Development
# -----------------------------
with tab2:
    st.header("🚀 Model Development")
    st.markdown("""
    **Model Development Process:**
    1. **Target Selection**: Choose the variable you want to predict
    2. **Feature Selection**: Select input variables for your model
    3. **Model Selection**: Choose the algorithm that best fits your problem
    4. **Hyperparameter Tuning**: Optimize model parameters for better performance
    5. **Class Balancing**: Handle imbalanced datasets if needed
    
    This step creates your predictive model using machine learning algorithms.
    """)
    
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset first.")
    else:
        data = st.session_state.data
        
        # Target Selection
        st.subheader("🎯 Target Variable Selection")
        st.markdown("Select the column you want to predict (dependent variable).")
        
        target_column = st.selectbox(
            "Choose Target Column", 
            data.columns,
            help="This is the variable your model will learn to predict"
        )
        
        # Check if target changed
        if 'previous_target' not in st.session_state or st.session_state.previous_target != target_column:
            if 'previous_target' in st.session_state:  # Not first time
                reset_downstream_states(2)
            st.session_state.previous_target = target_column
        
        # Feature Selection
        st.subheader("🔧 Feature Selection")
        st.markdown("Choose which variables to use as inputs for your model (independent variables).")
        
        available_features = [c for c in data.columns if c != target_column]
        feature_selection_mode = st.radio(
            "Feature selection method:",
            ["Use all available features", "Select specific features"],
            horizontal=True,
            help="Choose whether to use all features or manually select specific ones"
        )
        
        if feature_selection_mode == "Select specific features":
            feature_cols = st.multiselect(
                "Select features for training:",
                available_features,
                default=available_features,
                help="Choose which columns to use as inputs for your model"
            )
            
            if not feature_cols:
                st.warning("⚠️ Please select at least one feature to continue.")
                st.stop()
        else:
            feature_cols = available_features
        
        # Check if features changed
        if 'previous_features' not in st.session_state or st.session_state.previous_features != feature_cols:
            if 'previous_features' in st.session_state:  # Not first time
                reset_downstream_states(2)
            st.session_state.previous_features = feature_cols
        
        st.info(f"✓ Selected {len(feature_cols)} features: {feature_cols[:3]}{'...' if len(feature_cols) > 3 else ''}")
        
        # Prepare data
        X = data[feature_cols]
        y_raw = data[target_column]
        
        # Encode target if needed
        if y_raw.dtype == "object" or y_raw.nunique() < 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw)
            class_names = [str(c) for c in le_target.classes_]
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = [str(c) for c in np.unique(y)]
        
        # Show class distribution
        st.subheader("📊 Target Variable Distribution")
        st.markdown("Understanding the distribution of your target variable helps in model selection.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Counts:**")
            class_dist = class_counts(y)
            st.json(class_dist)
        
        with col2:
            # Simple bar chart for class distribution
            dist_df = pd.DataFrame(list(class_dist.items()), columns=['Class', 'Count'])
            st.bar_chart(dist_df.set_index('Class'))
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        st.markdown("""
        **Algorithm Options:**
        - **Logistic Regression**: Simple, interpretable, good baseline
        - **SVM**: Effective for high-dimensional data
        - **Decision Tree**: Highly interpretable, handles non-linear relationships
        - **Random Forest**: Robust ensemble method, reduces overfitting
        - **Gradient Boosting**: Powerful ensemble method, often high performance
        """)
        
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(),
        }
        
        model_choice = st.selectbox("Select Algorithm", list(model_options.keys()))
        model = model_options[model_choice]
        
        # Hyperparameters
        with st.expander("⚙️ Hyperparameter Tuning (Optional)", expanded=False):
            st.markdown("Fine-tune your model parameters for better performance.")
            
            if model_choice in ["Logistic Regression", "SVM"]:
                C_val = st.slider("C (Regularization Strength)", 0.01, 10.0, 1.0, 
                                help="Higher values = less regularization")
                model.set_params(C=C_val)
            
            if model_choice == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"],
                                    help="The kernel function used by SVM")
                model.set_params(kernel=kernel)
            
            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 50, 500, 200, step=50)
                max_depth = st.slider("Max Depth (0 = No limit)", 0, 30, 0, step=1)
                model.set_params(
                    n_estimators=n_estimators, 
                    max_depth=(None if max_depth == 0 else max_depth)
                )
        
        # Class Balancing
        st.subheader("⚖️ Class Balancing")
        st.markdown("""
        **Why Balance Classes?**
        When your target classes are imbalanced, models may be biased toward the majority class.
        Balancing techniques help create more fair and accurate predictions.
        """)
        
        balance_method = st.selectbox(
            "Choose balancing method:",
            ["None", "SMOTE", "Random Oversample", "Random Undersample"],
            help="SMOTE creates synthetic samples, Oversample duplicates minority class, Undersample reduces majority class"
        )
        
        # Train-Test Split
        st.subheader("📊 Data Splitting")
        st.markdown("Split your data into training and testing sets to evaluate model performance.")
        
        test_size = st.slider("Test Set Size (%)", 10, 50, 20,
                             help="Percentage of data to use for testing")
        
        # Validation checks
        unique, counts = np.unique(y, return_counts=True)
        too_small = np.any(counts < 2)
        stratify_ok = not too_small and len(unique) >= 2
        
        if too_small:
            st.warning("⚠️ At least one class has fewer than 2 samples. Stratified split not possible.")
        
        # Build pipeline
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])
        
        # Add sampler if selected
        sampler = None
        if balance_method == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif balance_method == "Random Oversample":
            sampler = RandomOverSampler(random_state=42)
        elif balance_method == "Random Undersample":
            sampler = RandomUnderSampler(random_state=42)
        
        if sampler is None:
            clf = ImbPipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        else:
            clf = ImbPipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("classifier", model)])
        
        # Train Model Button
        st.markdown("---")
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if len(np.unique(y)) < 2:
                st.error("❌ Training failed: Only one class present in target variable.")
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=(y if stratify_ok else None)
                )
                
                # Train model
                with st.spinner("Training model... Please wait."):
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                
                # Store results in session state
                st.session_state.model_trained = True
                st.session_state.trained_clf = clf
                st.session_state.trained_le_target = le_target
                st.session_state.trained_feature_cols = feature_cols
                st.session_state.trained_class_names = class_names
                st.session_state.test_results = {
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'accuracy': acc
                }
                
                st.success(f"✅ Model trained successfully! Accuracy: **{acc:.3f}**")
                
                if sampler is not None:
                    st.info(f"✓ Applied balancing: **{balance_method}**")
                
                st.balloons()

# -----------------------------
# TAB 3: Model Evaluation
# -----------------------------
with tab3:
    st.header("🔍 Model Evaluation")
    st.markdown("""
    **Model Evaluation Components:**
    - **Performance Metrics**: Accuracy, precision, recall, F1-score for each class
    - **Interactive Dashboard**: Explore model predictions, feature importance, and decision boundaries
    - **Model Interpretability**: Understand how your model makes predictions
    
    This step helps you understand how well your model performs and why it makes certain predictions.
    """)
    
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Please train a model in the Model Development tab first.")
    else:
        # Get stored results
        clf = st.session_state.trained_clf
        le_target = st.session_state.trained_le_target
        class_names = st.session_state.trained_class_names
        test_results = st.session_state.test_results
        
        y_test = test_results['y_test']
        y_pred = test_results['y_pred']
        accuracy = test_results['accuracy']
        
        # Performance Summary
        st.subheader("📊 Model Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Test Samples", len(y_test))
        with col3:
            st.metric("Features Used", len(st.session_state.trained_feature_cols))
        
        # Classification Report
        st.subheader("📋 Detailed Performance Report")
        st.markdown("Comprehensive performance metrics for each class:")
        
        try:
            report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
        except Exception:
            report = classification_report(y_test, y_pred, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
        
        # Interactive Dashboard
        st.subheader("🖥️ Interactive Model Explorer")
        st.markdown("""
        The dashboard below provides interactive exploration of your model including:
        - Feature importance analysis
        - Individual prediction explanations
        - Model performance visualizations
        - Decision boundary exploration
        """)
        
        try:
            # Prepare data for dashboard
            data = st.session_state.data
            X = data[st.session_state.trained_feature_cols]
            y_raw = data[st.session_state.get('previous_target')]
            
            # Get preprocessed data
            fitted_pre = clf.named_steps["preprocessor"]
            fitted_clf = clf.named_steps["classifier"]
            
            X_test_proc = fitted_pre.transform(X.iloc[test_results['y_test'].shape[0]:])
            
            try:
                feat_names = fitted_pre.get_feature_names_out()
            except Exception:
                feat_names = [f"feature_{i}" for i in range(X_test_proc.shape[1])]
            
            # Sample for performance
            sample_n = min(300, len(y_test))
            idx = np.random.choice(len(y_test), size=sample_n, replace=False)
            
            X_dash = pd.DataFrame(fitted_pre.transform(X.iloc[idx]), columns=feat_names)
            y_dash = y_test[idx]
            
            # Create explainer
            explainer = ClassifierExplainer(
                fitted_clf,
                X_dash,
                y_dash,
                labels=class_names,
                model_output="probability",
            )
            
            # Create dashboard
            dashboard = ExplainerDashboard(
                explainer,
                title="Model Performance Dashboard",
                shap_interaction=False,
                whatif=True,
                importances=True,
                shap_dependence=True,
                decision_trees=isinstance(fitted_clf, DecisionTreeClassifier),
            )
            
            # Render dashboard
            try:
                html_str = dashboard.to_html()
            except TypeError:
                with tempfile.TemporaryDirectory() as tmpdir:
                    html_path = Path(tmpdir) / "dashboard.html"
                    dashboard.to_html(filename=str(html_path))
                    html_str = html_path.read_text(encoding="utf-8")
            
            components.html(html_str, height=900, scrolling=True)
            
        except Exception as e:
            st.error(f"Dashboard generation failed: {str(e)}")
            st.info("Dashboard requires additional model information. Please retrain your model.")

# -----------------------------
# TAB 4: Predictions
# -----------------------------
with tab4:
    st.header("📈 Predictions")
    st.markdown("""
    **Making Predictions on New Data:**
    1. **Upload New Data**: Provide a dataset with the same features used in training
    2. **Automatic Processing**: The model automatically preprocesses your data
    3. **Generate Predictions**: Get class predictions and probability scores
    4. **Download Results**: Export predictions as CSV for further use
    
    Your new data must have the same column structure as the training data.
    """)
    
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload a dataset first.")
    elif not st.session_state.model_trained:
        st.info("🚀 Please train a model first.")
    else:
        # Model ready for predictions
        st.success("✅ Model is ready for predictions!")
        
        trained_clf = st.session_state.trained_clf
        trained_le_target = st.session_state.trained_le_target
        trained_feature_cols = st.session_state.trained_feature_cols
        
        st.subheader("📋 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Required Features:** {len(trained_feature_cols)}")
            with st.expander("View Feature List"):
                st.write(trained_feature_cols)
        
        with col2:
            st.info(f"**Accuracy:** {st.session_state.test_results['accuracy']:.3f}")
        
        # File upload for predictions
        st.subheader("📁 Upload New Data for Prediction")
        new_file = st.file_uploader(
            "Choose your prediction dataset",
            type=["csv", "xlsx"],
            key="prediction_data",
            help="Upload data with the same columns as your training dataset"
        )
        
        if new_file:
            # Load new data
            if new_file.name.endswith(".csv"):
                new_df = pd.read_csv(new_file)
            else:
                new_df = pd.read_excel(new_file)
            
            st.subheader("📊 New Data Preview")
            st.dataframe(new_df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", new_df.shape[0])
            with col2:
                st.metric("Columns", new_df.shape[1])
            with col3:
                st.metric("Missing Values", new_df.isnull().sum().sum())
            
            # Validation and Prediction
            try:
                # Check for required features
                missing_cols = [col for col in trained_feature_cols if col not in new_df.columns]
                extra_cols = [col for col in new_df.columns if col not in trained_feature_cols]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("Please ensure your new data has the same features as training data.")
                else:
                    # Prepare data for prediction
                    new_data_features = new_df[trained_feature_cols]
                    
                    if extra_cols:
                        st.warning(f"⚠️ Extra columns will be ignored: {extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}")
                    
                    # Make Predictions Button
                    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                        with st.spinner("Generating predictions... Please wait."):
                            # Make predictions
                            preds = trained_clf.predict(new_data_features)
                            probs = trained_clf.predict_proba(new_data_features) if hasattr(trained_clf.named_steps["classifier"], "predict_proba") else None
                            
                            # Prepare results
                            results_df = new_df.copy()
                            
                            # Add predictions
                            if trained_le_target is not None:
                                preds_decoded = trained_le_target.inverse_transform(preds)
                                results_df['prediction'] = preds_decoded
                                results_df['prediction_encoded'] = preds
                            else:
                                results_df['prediction'] = preds
                            
                            # Add probabilities
                            if probs is not None:
                                if trained_le_target is not None:
                                    prob_cols = [f"prob_{cls}" for cls in trained_le_target.classes_]
                                else:
                                    prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
                                
                                for i, col in enumerate(prob_cols):
                                    results_df[col] = probs[:, i]
                        
                        st.success(f"✅ Generated predictions for {len(results_df)} samples!")
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📊 Prediction Summary")
                        if trained_le_target is not None:
                            pred_counts = pd.Series(preds_decoded).value_counts()
                        else:
                            pred_counts = pd.Series(preds).value_counts()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Prediction Counts:**")
                            st.dataframe(pred_counts.reset_index())
                        
                        with col2:
                            st.write("**Prediction Distribution:**")
                            st.bar_chart(pred_counts)
                        
                        # Download button
                        st.subheader("💾 Download Results")
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv_data,
                            file_name=f"predictions_{new_file.name.split('.')[0]}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.balloons()
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.write("**Troubleshooting:**")
                st.write("- Ensure column names match exactly")
                st.write("- Check for data type consistency")
                st.write("- Verify categorical values exist in training data")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | © 2025 CE Team Prediction Pro Analytics Platform")
