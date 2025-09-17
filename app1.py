# Step 4: Model training
                    status_text.info("🏋️ Training the model... (this may take a moment)")
                    progress_bar.progress(65)
                    
                    model_pipeline.fit(X_train, y_train)
                    
                    # Step 5: Model evaluation
                    status_text.info("📊 Evaluating model performance...")
                    progress_bar.progress(85)
                    
                    y_pred = model_pipeline.predict(X_test)
                    
                    # Calculate metrics based on problem type
                    if problem_type == "classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        metrics = {
                            'accuracy': accuracy,
                            'f1_score': f1
                        }
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        metrics = {
                            'r2_score': r2,
                            'mse': mse,
                            'mae': mae
                        }
                    
                    # Complete training
                    progress_bar.progress(100)
                    status_text.success("✅ Training completed successfully!")
                    
                    # Save results to session state
                    st.session_state.model_trained = True
                    st.session_state.trained_model = model_pipeline
                    st.session_state.trained_feature_cols = feature_cols
                    st.session_state.trained_le_target = le_target
                    st.session_state.trained_class_names = class_names
                    st.session_state.test_results = {
                        'y_test': y_test, 
                        'y_pred': y_pred, 
                        'metrics': metrics
                    }
                    st.session_state.trained_X_test = X_test
                    st.session_state.model_name = model_choice
                    st.session_state.problem_type = problem_type
                    
                    # Display results
                    st.balloons()
                    
                    primary_metric = list(metrics.items())[0]
                    st.success(
                        f"🎉 Model trained successfully! "
                        f"{primary_metric[0].replace('_', ' ').title()}: {primary_metric[1]:.4f}"
                    )
                    
                    # Metrics display
                    st.markdown("### 📊 Performance Metrics")
                    metric_cols = st.columns(len(metrics))
                    
                    for i, (metric_name, metric_value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            # Format metric display
                            display_name = metric_name.replace('_', ' ').title()
                            if metric_name in ['accuracy', 'f1_score', 'r2_score']:
                                display_value = f"{metric_value:.4f}"
                                if metric_value > 0.8:
                                    delta_color = "normal"
                                elif metric_value > 0.6:
                                    delta_color = "off"
                                else:
                                    delta_color = "inverse"
                            else:
                                display_value = f"{metric_value:.4f}"
                                delta_color = "normal"
                            
                            st.metric(display_name, display_value)
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Training failed: {str(e)}")
                    
                    with st.expander("🔍 View detailed error information"):
                        st.code(traceback.format_exc())
                        
                        # Troubleshooting suggestions
                        st.markdown("**Possible solutions:**")
                        st.markdown("- Check if target variable has at least 2 unique values")
                        st.markdown("- Ensure features contain valid data (no all-NaN columns)")
                        st.markdown("- Try a different algorithm or reduce data complexity")
        
        # Model download section
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.markdown("### 📦 Download Trained Model")
            
            download_col1, download_col2 = st.columns([2, 1])
            
            with download_col1:
                model_package = serialize_model_safe(
                    st.session_state.trained_model,
                    st.session_state.trained_feature_cols,
                    st.session_state.trained_le_target,
                    st.session_state.model_name,
                    st.session_state.test_results['metrics']
                )
                
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                model_name_clean = st.session_state.model_name.lower().replace(' ', '_')
                filename = f"automl_model_{model_name_clean}_{timestamp}.pkl"
                
                st.download_button(
                    label="📦 Download Complete Model Package",
                    data=model_package,
                    file_name=filename,
                    mime="application/octet-stream",
                    help="Download trained model with preprocessing pipeline and metadata",
                    use_container_width=True
                )
            
            with download_col2:
                st.info("""
                **Package Contents:**
                - Trained model
                - Preprocessing pipeline
                - Feature columns list
                - Label encoder (if applicable)
                - Performance metrics
                - Training metadata
                """)

# Tab 3: SHAP Analysis
with tabs[2]:
    st.markdown("## 🔍 SHAP Model Analysis")
    
    if not st.session_state.get('model_trained', False):
        st.info("👈 Train a model first to see SHAP explanations")
        
        st.markdown("""
        ### What you'll get with SHAP analysis:
        
        **🎯 Feature Importance**
        - See which features matter most for predictions
        - Understand positive vs negative impact
        
        **🔍 Individual Explanations** 
        - Explain any single prediction in detail
        - See exactly how each feature contributed
        
        **📊 Model Behavior**
        - Visualize model decision patterns
        - Compare feature effects across samples
        """)
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.trained_X_test
        y_test = st.session_state.test_results['y_test']
        y_pred = st.session_state.test_results['y_pred']
        model_name = st.session_state.model_name
        problem_type = st.session_state.problem_type
        metrics = st.session_state.test_results['metrics']
        
        # Performance overview
        st.markdown("### 📊 Model Performance Overview")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            if problem_type == "classification":
                # Confusion matrix with Plotly
                cm = confusion_matrix(y_test, y_pred)
                
                # Create labels for confusion matrix
                if st.session_state.trained_class_names:
                    labels = st.session_state.trained_class_names
                else:
                    labels = [f"Class {i}" for i in range(len(cm))]
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    color_continuous_scale='Blues'
                )
                
                fig_cm.update_traces(texttemplate="%{z}", textfont_size=12)
                st.plotly_chart(fig_cm, use_container_width=True)
                
            else:
                # Actual vs Predicted scatter plot for regression
                fig_scatter = px.scatter(
                    x=y_test, 
                    y=y_pred,
                    title="Actual vs Predicted Values",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    opacity=0.6
                )
                
                # Add perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash', width=2)
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with perf_col2:
            if problem_type == "classification":
                # Classification report
                try:
                    class_names = st.session_state.trained_class_names
                    if class_names:
                        target_names = class_names
                    else:
                        target_names = None
                    
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names,
                        output_dict=True
                    )
                    
                    # Convert to DataFrame and display
                    report_df = pd.DataFrame(report).transpose()
                    # Round to 3 decimal places
                    numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                    for col in numeric_cols:
                        if col in report_df.columns:
                            if col == 'support':
                                report_df[col] = report_df[col].astype(int)
                            else:
                                report_df[col] = report_df[col].round(3)
                    
                    st.markdown("**Classification Report**")
                    st.dataframe(report_df, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate classification report: {e}")
            
            else:
                # Regression residuals plot
                residuals = y_test - y_pred
                
                fig_residuals = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title="Residuals Plot",
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    opacity=0.6
                )
                
                fig_residuals.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Perfect Prediction"
                )
                
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        # ROC curve for binary classification
        if problem_type == "classification" and len(np.unique(y_test)) == 2:
            try:
                st.markdown("### 📈 ROC Curve Analysis")
                
                # Get prediction probabilities
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC plot
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
                    name='Random Classifier',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
                
            except Exception as e:
                st.warning(f"ROC curve generation failed: {e}")
        
        # SHAP Explanations
        st.markdown("### 🎯 SHAP Model Explanations")
        
        with st.spinner("Generating SHAP explanations... (this may take a moment)"):
            shap_values, explainer, X_shap = create_shap_explanation_safe(
                model, X_test, model_name
            )
            
            if shap_values is not None and X_shap is not None:
                try:
                    # Feature importance summary
                    st.markdown("#### 📊 Global Feature Importance")
                    
                    fig_shap, ax = plt.subplots(figsize=(12, 8))
                    
                    # Handle different SHAP value formats
                    if hasattr(shap_values, 'values'):
                        # SHAP 0.46.0 Explanation object
                        plot_values = shap_values.values
                        plot_data = X_shap
                    else:
                        # Older format or array
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            # Multi-class: use class 1 for visualization
                            plot_values = shap_values[1]
                        else:
                            # Binary or single output
                            plot_values = shap_values[0] if isinstance(shap_values, list) else shap_values
                        plot_data = X_shap
                    
                    # Create SHAP summary plot
                    shap.summary_plot(
                        plot_values, 
                        plot_data,
                        feature_names=X_test.columns.tolist(),
                        show=False,
                        ax=ax
                    )
                    
                    plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    
                    # Explanation of the plot
                    st.markdown("""
                    **How to interpret this plot:**
                    - Features are ranked by importance (top = most important)
                    - Each dot represents one sample from your test data
                    - **Color indicates feature value:** Red = High, Blue = Low
                    - **X-axis shows impact:** Positive = increases prediction, Negative = decreases prediction
                    - **Spread shows variability:** Wide spread = feature has different impacts for different samples
                    """)
                    
                    # Individual explanation
                    st.markdown("#### 🔍 Individual Prediction Explanation")
                    
                    sample_col1, sample_col2 = st.columns([1, 2])
                    
                    with sample_col1:
                        max_samples = min(50, len(X_test))
                        sample_idx = st.slider(
                            "Choose sample to explain",
                            0, max_samples - 1, 0,
                            help="Select which prediction to explain in detail"
                        )
                        
                        # Show actual vs predicted for this sample
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
                    
                    with sample_col2:
                        try:
                            # Create waterfall plot for individual explanation
                            fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                            
                            # Get SHAP values for the selected sample
                            if sample_idx < len(plot_values):
                                sample_shap = plot_values[sample_idx]
                            else:
                                sample_shap = plot_values[0]  # Fallback to first sample
                            
                            # Get top 10 most important features for this sample
                            feature_importance = np.abs(sample_shap)
                            top_indices = np.argsort(feature_importance)[-10:]
                            
                            # Create horizontal bar plot
                            colors = ['red' if val > 0 else 'blue' for val in sample_shap[top_indices]]
                            bars = ax_waterfall.barh(
                                range(len(top_indices)), 
                                sample_shap[top_indices], 
                                color=colors,
                                alpha=0.7
                            )
                            
                            # Customize plot
                            ax_waterfall.set_yticks(range(len(top_indices)))
                            ax_waterfall.set_yticklabels([X_test.columns[i] for i in top_indices])
                            ax_waterfall.set_xlabel("SHAP Value (Impact on Prediction)")
                            ax_waterfall.set_title(f"Feature Contributions for Sample {sample_idx + 1}")
                            ax_waterfall.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add value labels on bars
                            for bar, value in zip(bars, sample_shap[top_indices]):
                                width = bar.get_width()
                                ax_waterfall.text(
                                    width + (0.01 * np.sign(width) if width != 0 else 0.01),
                                    bar.get_y() + bar.get_height()/2,
                                    f'{value:.3f}',
                                    ha='left' if width >= 0 else 'right',
                                    va='center',
                                    fontsize=9
                                )
                            
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)
                            
                        except Exception as e:
                            st.warning(f"Individual explanation plot failed: {e}")
                    
                    # Feature values for selected sample
                    st.markdown("**Feature values for this sample:**")
                    sample_features = X_test.iloc[sample_idx]
                    
                    # Create a nice display of feature values
                    feature_df = pd.DataFrame({
                        'Feature': sample_features.index,
                        'Value': sample_features.values,
                        'SHAP Impact': sample_shap[:len(sample_features)] if len(sample_shap) >= len(sample_features) else [0] * len(sample_features)
                    })
                    
                    # Sort by absolute SHAP impact
                    feature_df['abs_impact'] = np.abs(feature_df['SHAP Impact'])
                    feature_df = feature_df.sort_values('abs_impact', ascending=False).drop('abs_impact', axis=1)
                    
                    st.dataframe(feature_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {str(e)}")
                    st.info("This might be due to model complexity or data format. The model still works for predictions.")
                    
            else:
                st.warning("⚠️ SHAP explanations could not be generated for this model type or data format.")
                st.info("SHAP works best with tree-based models (Random Forest, Gradient Boosting) and standard data formats.")
        
        # Feature importance fallback
        st.markdown("### ⭐ Alternative Feature Importance Analysis")
        
        try:
            # Extract the actual model from pipeline
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('model', model)
                preprocessor = model.named_steps.get('preprocessor')
            else:
                classifier = model
                preprocessor = None
            
            # Get feature names after preprocessing
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = X_test.columns.tolist()
            else:
                feature_names = X_test.columns.tolist()
            
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
                importances = classifier.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],  # Ensure lengths match
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(15)  # Top 15
                
                # Create horizontal bar chart
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importances (Built-in)",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)
                
            elif hasattr(classifier, 'coef_'):
                # Linear models
                coefficients = classifier.coef_
                
                # Handle multi-class case
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                else:
                    coefficients = np.abs(coefficients)
                
                # Create coefficient DataFrame
                coef_df = pd.DataFrame({
                    'Feature': feature_names[:len(coefficients)],
                    'Coefficient': coefficients
                }).sort_values('Coefficient', ascending=True).tail(15)
                
                # Create horizontal bar chart
                fig_coef = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Coefficients (Absolute)",
                    color='Coefficient',
                    color_continuous_scale='Plasma'
                )
                
                fig_coef.update_layout(height=500)
                st.plotly_chart(fig_coef, use_container_width=True)
                
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.warning(f"Feature importance analysis failed: {str(e)}")

# Tab 4: Predictions
with tabs[3]:
    st.markdown("## 📈 Make Predictions on New Data")
    
    if not st.session_state.get('model_trained', False):
        st.info("👈 Train a model first to make predictions on new data")
        
        st.markdown("""
        ### Once you have a trained model, you can:
        
        **📁 Upload new data** with the same features as your training data
        
        **🔮 Get predictions** for classification or regression problems
        
        **📊 View results** with confidence scores and distributions
        
        **📥 Download results** in multiple formats (CSV, Excel, Parquet)
        """)
    else:
        model = st.session_state.trained_model
        feature_cols = st.session_state.trained_feature_cols
        le_target = st.session_state.trained_le_target
        problem_type = st.session_state.problem_type
        
        st.markdown("### 📁 Upload Prediction Data")
        
        st.markdown(f"""
        **Required features:** {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}
        
        **Total features needed:** {len(feature_cols)}
        """)
        
        pred_file = st.file_uploader(
            "Upload file for predictions",
            type=["csv", "xlsx", "parquet"],
            key="prediction_file",
            help="File must contain the same features used during model training"
        )
        
        if pred_file:
            try:
                # Load prediction data
                prediction_data = load_data_safe(pred_file)
                
                if prediction_data is None:
                    st.error("❌ Failed to load prediction file")
                    st.stop()
                
                st.success(f"✅ Loaded {len(prediction_data):,} rows for prediction")
                
                # Preview data
                st.markdown("### 📊 Data Preview")
                st.dataframe(prediction_data.head(), use_container_width=True)
                
                # Feature validation
                missing_features = set(feature_cols) - set(prediction_data.columns)
                extra_features = set(prediction_data.columns) - set(feature_cols)
                
                validation_col1, validation_col2 = st.columns(2)
                
                with validation_col1:
                    if missing_features:
                        st.error(f"❌ **Missing Required Features:**")
                        for feature in sorted(missing_features):
                            st.write(f"• {feature}")
                    else:
                        st.success("✅ **All required features present!**")
                
                with validation_col2:
                    if extra_features:
                        st.warning(f"⚠️ **Extra columns found (will be ignored):**")
                        extra_list = sorted(list(extra_features))
                        st.write(f"{len(extra_list)} columns: {', '.join(extra_list[:3])}{'...' if len(extra_list) > 3 else ''}")
                    
                    st.info(f"📊 **Ready to predict on {len(prediction_data):,} rows**")
                
                # Only proceed if we have all required features
                if not missing_features:
                    # Prediction options
                    st.markdown("### ⚙️ Prediction Options")
                    
                    options_col1, options_col2 = st.columns(2)
                    
                    with options_col1:
                        include_probabilities = st.checkbox(
                            "Include prediction probabilities",
                            value=True if problem_type == "classification" else False,
                            disabled=problem_type == "regression",
                            help="Show confidence scores for each class (classification only)"
                        )
                    
                    with options_col2:
                        batch_size = st.selectbox(
                            "Processing batch size",
                            [1000, 5000, 10000],
                            index=1,
                            help="Larger batches are faster but use more memory"
                        )
                    
                    # Generate predictions
                    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                        pred_features = prediction_data[feature_cols]
                        
                        with st.spinner("Processing predictions..."):
                            try:
                                # Initialize containers for results
                                all_predictions = []
                                all_probabilities = []
                                
                                # Progress tracking
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Process in batches
                                for i in range(0, len(pred_features), batch_size):
                                    batch = pred_features.iloc[i:i+batch_size]
                                    
                                    # Update progress
                                    progress = min((i + len(batch)) / len(pred_features), 1.0)
                                    progress_bar.progress(progress)
                                    status_text.info(f"Processing batch: {i+1:,} to {min(i+batch_size, len(pred_features)):,}")
                                    
                                    # Make predictions
                                    batch_preds = model.predict(batch)
                                    all_predictions.extend(batch_preds)
                                    
                                    # Get probabilities if requested and available
                                    if include_probabilities and hasattr(model, 'predict_proba'):
                                        try:
                                            batch_probs = model.predict_proba(batch)
                                            all_probabilities.append(batch_probs)
                                        except Exception:
                                            pass  # Skip if prediction probabilities fail
                                
                                # Complete processing
                                progress_bar.progress(1.0)
                                status_text.success("✅ Predictions completed!")
                                
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
                                    # Convert encoded predictions back to original labels
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
                                
                                # Success message
                                st.balloons()
                                st.success(f"🎉 Successfully generated predictions for {len(results_df):,} rows!")
                                
                                # Results summary
                                st.markdown("### 📊 Prediction Summary")
                                
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
                                        avg_confidence = probabilities.max(axis=1).mean()
                                        st.metric("Avg. Confidence", f"{avg_confidence:.3f}")
                                    else:
                                        if problem_type == "regression":
                                            pred_std = results_df['prediction'].std()
                                            st.metric("Std. Deviation", f"{pred_std:.3f}")
                                        else:
                                            st.metric("Model Type", "Classification")
                                
                                # Sample results display
                                st.markdown("### 🔍 Sample Results")
                                
                                # Select columns to display
                                display_cols = []
                                
                                # Add some original features for context
                                context_features = feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
                                display_cols.extend(context_features)
                                
                                # Add prediction columns
                                display_cols.append('prediction')
                                
                                # Add probability columns if available
                                if probabilities is not None:
                                    prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
                                    display_cols.extend(prob_cols[:3])  # Show first 3 probability columns
                                
                                # Display sample
                                st.dataframe(results_df[display_cols].head(10), use_container_width=True)
                                
                                # Prediction distribution visualization
                                if problem_type == "classification" and results_df['prediction'].nunique() <= 20:
                                    st.markdown("### 📈 Prediction Distribution")
                                    
                                    pred_counts = results_df['prediction'].value_counts()
                                    
                                    fig_dist = px.bar(
                                        x=pred_counts.index,
                                        y=pred_counts.values,
                                        title="Distribution of Predictions",
                                        labels={'x': 'Prediction', 'y': 'Count'},
                                        color=pred_counts.values,
                                        color_continuous_scale='Viridis'
                                    )
                                    
                                    fig_dist.update_layout(showlegend=False)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                    
                                elif problem_type == "regression":
                                    st.markdown("### 📈 Prediction Distribution")
                                    
                                    fig_hist = px.histogram(
                                        results_df['prediction'],
                                        nbins=50,
                                        title="Distribution of Predicted Values",
                                        labels={'x': 'Predicted Value', 'y': 'Frequency'}
                                    )
                                    
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                # Download section
                                st.markdown("### 📥 Download Results")
                                
                                download_cols = st.columns(3)
                                
                                with download_cols[0]:
                                    # CSV download
                                    csv_buffer = io.StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    
                                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                    
                                    st.download_button(
                                        "📄 Download as CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"predictions_{timestamp}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with download_cols[1]:
                                    # Excel download
                                    excel_buffer = io.BytesIO()
                                    
                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                        
                                        # Add summary sheet
                                        summary_data = {
                                            'Metric': [
                                                'Total Predictions', 
                                                'Unique Values', 
                                                'Model Used', 
                                                'Problem Type',
                                                'Generated On'
                                            ],
                                            'Value': [
                                                len(results_df),
                                                results_df['prediction'].nunique(),
                                                st.session_state.model_name,
                                                problem_type.title(),
                                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                            ]
                                        }
                                        
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    st.download_button(
                                        "📊 Download as Excel",
                                        data=excel_buffer.getvalue(),
                                        file_name=f"predictions_{timestamp}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                
                                with download_cols[2]:
                                    # Parquet download for large files
                                    if len(results_df) > 5000:
                                        parquet_buffer = io.BytesIO()
                                        results_df.to_parquet(parquet_buffer, index=False)
                                        
                                        st.download_button(
                                            "⚡ Download as Parquet",
                                            data=parquet_buffer.getvalue(),
                                            file_name=f"predictions_{timestamp}.parquet",
                                            mime="application/octet-stream",
                                            help="Recommended for large datasets (faster loading)",
                                            use_container_width=True
                                        )
                                    else:
                                        st.info("Parquet download available for datasets > 5,000 rows")
                                
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {str(e)}")
                                
                                with st.expander("🔍 View detailed error"):
                                    st.code(traceback.format_exc())
                                    
                                    st.markdown("**Possible solutions:**")
                                    st.markdown("- Ensure all required features are present")
                                    st.markdown("- Check that data types match training data")
                                    st.markdown("- Verify no missing or invalid values in key features")
                
            except Exception as e:
                st.error(f"❌ Failed to process prediction file: {str(e)}")

# Footer
st.markdown("---")

footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown("**AutoML Analytics Pro v2.0**")
    st.caption("Production-grade machine learning platform")

with footer_cols[1]:
    if st.session_state.get('model_trained', False):
        st.markdown("**Current Model**")
        model_info = f"{st.session_state.model_name} ({st.session_state.problem_type})"
        st.caption(model_info)
    else:
        st.markdown("**Status**")
        st.caption("Ready for model training")

with footer_cols[2]:
    st.markdown("**Tech Stack**")
    st.caption("SHAP • Plotly • PyArrow • Scikit-learn")

# Sidebar help and info
with st.sidebar:
    if not st.session_state.data_uploaded:
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        
        steps = [
            "Upload your dataset above",
            "Explore data in the first tab",
            "Train your model in the second tab", 
            "Analyze with SHAP in the third tab",
            "Make predictions in the fourth tab"
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
            
        st.markdown("---")
        st.markdown("### 📋 Supported Features")
        st.markdown("✅ Classification & Regression")
        st.markdown("✅ Automatic preprocessing")
        st.markdown("✅ Class balancing (SMOTE)")
        st.markdown("✅ SHAP explanations")
        st.markdown("✅ Model download")
        st.markdown("✅ Batch predictions")
        
    else:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset")
        df = st.session_state.data
        
        st.markdown(f"""
        **File:** {st.session_state.current_file}  
        **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns  
        **Memory:** {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB  
        **Missing:** {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
        """)
        
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.markdown("### 🤖 Model Performance")
            
            metrics = st.session_state.test_results['metrics']
            primary_metric = list(metrics.items())[0]
            
            st.metric(
                primary_metric[0].replace('_', ' ').title(),
                f"{primary_metric[1]:.4f}"
            )
            
            # Show additional metrics if available
            if len(metrics) > 1:
                secondary_metric = list(metrics.items())[1]
                st.metric(
                    secondary_metric[0].replace('_', ' ').title(),
                    f"{secondary_metric[1]:.4f}"
                )"""
Production-Grade AutoML App - Bulletproof Version
- All known issues fixed
- Optimized for your exact requirements.txt
- Professional UI/UX
- Robust error handling
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

# Core libraries from your requirements.txt
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

# Data profiling - FIXED configuration
from ydata_profiling import ProfileReport

# Page config
st.set_page_config(
    page_title="AutoML Pro",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: white;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state - simplified and bulletproof
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

# Utility functions - bulletproof implementations
@st.cache_data(show_spinner=False)
def load_data_safe(uploaded_file) -> Optional[pd.DataFrame]:
    """Load data with maximum compatibility"""
    try:
        name = uploaded_file.name.lower()
        
        if name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            # Try CSV first, then Excel
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
    """Bulletproof data cleaning"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        df_clean = df.copy()
        
        # Remove completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Clean each column safely
        for col in df_clean.columns:
            try:
                # Try numeric conversion
                numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                non_null_count = numeric_converted.count()
                total_count = len(df_clean[col].dropna())
                
                # If >50% can be converted to numeric, treat as numeric
                if total_count > 0 and non_null_count / total_count > 0.5:
                    df_clean[col] = numeric_converted
                    # Fill missing with median (more robust than mean)
                    median_val = df_clean[col].median()
                    fill_val = median_val if not pd.isna(median_val) else 0
                    df_clean[col] = df_clean[col].fillna(fill_val)
                else:
                    # Treat as categorical
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].fillna('Missing')
                    
            except:
                # Last resort: convert to string
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        # Handle infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        return df_clean
        
    except Exception:
        # Ultimate fallback
        return df.iloc[:, :min(3, df.shape[1])].fillna('Error').astype(str)

@st.cache_data(show_spinner=False)
def create_profile_safe(_df: pd.DataFrame, minimal: bool = True) -> Optional[ProfileReport]:
    """Generate profile with proper ydata-profiling configuration"""
    try:
        if _df is None or _df.empty:
            return None
            
        # Clean data first
        df_clean = clean_data_robust(_df)
        if df_clean.empty:
            return None
        
        # Sample large datasets
        if len(df_clean) > 3000:
            df_clean = df_clean.sample(n=3000, random_state=42)
        
        # FIXED: Correct configuration for ydata-profiling
        config = {
            "title": "Dataset Profile",
            "minimal": minimal,
            "lazy": False,
            "explorative": not minimal,
            # Remove dark_mode - not supported in your version
            # "dark_mode": False,  # REMOVED - causes validation error
        }
        
        return ProfileReport(df_clean, **config)
        
    except Exception as e:
        st.error(f"Profile generation failed: {str(e)}")
        return None

def safe_onehot_encoder():
    """Create compatible OneHotEncoder"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def create_shap_explanation_safe(model, X_sample, model_type: str):
    """SHAP explanations with proper error handling"""
    try:
        # Get classifier from pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('model', model)
            # Transform data through preprocessing
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                X_transformed = preprocessor.transform(X_sample)
            else:
                X_transformed = X_sample
        else:
            classifier = model
            X_transformed = X_sample
        
        # Limit sample size for performance
        max_samples = min(100, len(X_transformed))
        X_shap = X_transformed[:max_samples]
        
        # Use appropriate explainer
        if model_type in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap)
        else:
            # Use sample for background
            background_size = min(50, len(X_transformed))
            background = X_transformed[:background_size]
            explainer = shap.KernelExplainer(classifier.predict, background)
            shap_values = explainer.shap_values(X_shap[:20])  # Even smaller for Kernel
        
        return shap_values, explainer, X_shap
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {str(e)}")
        return None, None, None

def serialize_model_safe(model, feature_cols, le_target, model_name, metrics):
    """Bulletproof model serialization"""
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
        # Fallback: save without model object
        minimal_package = {
            'feature_columns': feature_cols,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'note': 'Minimal package - model object excluded due to serialization issues'
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
    
    # Status indicators
    status_items = [
        ("SHAP 0.46.0", "🎯"),
        ("Plotly ≤5.14.1", "📊"), 
        ("PyArrow", "⚡"),
        ("Imbalanced Learn", "⚖️"),
        ("YData Profiling", "📈")
    ]
    
    for item, icon in status_items:
        st.markdown(f"""
        <div class="status-card">
            {icon} <strong>{item}</strong> - Ready
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Upload CSV, Excel, or Parquet files"
    )
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_data_safe(uploaded_file)
            
            if df is not None:
                st.session_state.data = df
                st.session_state.data_uploaded = True
                st.session_state.current_file = uploaded_file.name
                
                st.success("✅ Data loaded successfully!")
                
                # File info
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.markdown(f"""
                <div class="status-card">
                    <strong>📄 {uploaded_file.name}</strong><br>
                    📏 {df.shape[0]:,} rows × {df.shape[1]} columns<br>
                    💾 {file_size:.1f} MB
                </div>
                """, unsafe_allow_html=True)

# Main tabs
tabs = st.tabs(["📊 Data Explorer", "🤖 Model Training", "🔍 SHAP Analysis", "📈 Predictions"])

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("## 📊 Data Explorer & Profiling")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload a dataset in the sidebar to begin")
        st.markdown("""
        ### Supported Formats:
        - **CSV files** (.csv)
        - **Excel files** (.xlsx, .xls) 
        - **Parquet files** (.parquet)
        
        ### Features:
        - Automatic data type detection
        - Missing value handling
        - Interactive data profiling
        - Statistical summaries
        """)
    else:
        df = st.session_state.data
        
        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Rows</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Columns</h3>
                <h2>{df.shape[1]}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 Memory</h3>
                <h2>{memory_mb:.1f} MB</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>❓ Missing</h3>
                <h2>{missing_pct:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Data types chart - FIXED
        st.markdown("### 📊 Data Types Overview")
        dtype_counts = df.dtypes.value_counts()
        
        # FIXED: Convert dtype objects to strings for Plotly
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=[str(dtype) for dtype in dtype_counts.index],  # Convert to strings
            title="Data Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_dtypes.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_dtypes, use_container_width=True)
        
        # Data preview with pagination
        st.markdown("### 🔍 Data Preview")
        
        # Pagination controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            
        with col2:
            total_pages = (len(df) - 1) // page_size + 1
            if total_pages > 1:
                page = st.slider("Page", 1, total_pages, 1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                display_df = df.iloc[start_idx:end_idx]
            else:
                display_df = df.head(page_size)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Column information
        st.markdown("### 📋 Column Analysis")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dtype) for dtype in df.dtypes],  # Convert to strings
            'Non-Null': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(1),
            'Unique Values': df.nunique()
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Profiling section
        st.markdown("### 📈 Comprehensive Data Profiling")
        
        profile_col1, profile_col2 = st.columns([1, 2])
        
        with profile_col1:
            st.markdown("**Profile Options:**")
            profile_type = st.radio(
                "Analysis depth",
                ["Quick (Minimal)", "Detailed (Full)"],
                help="Quick for fast overview, Detailed for complete analysis"
            )
            
            if st.button("🚀 Generate Profile Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive profile..."):
                    minimal = (profile_type == "Quick (Minimal)")
                    profile = create_profile_safe(df, minimal=minimal)
                    
                    if profile is not None:
                        st.session_state.profile_report = profile
                        st.session_state.profile_generated = True
                        st.success("✅ Profile report generated!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate profile report")
        
        with profile_col2:
            if st.session_state.get('profile_generated', False):
                st.info("📊 Profile report generated! Displaying below...")
            else:
                st.markdown("""
                **What you'll get:**
                - Statistical summaries for all columns
                - Distribution plots and histograms  
                - Correlation analysis
                - Missing value patterns
                - Data quality warnings
                - Variable relationships
                """)
        
        # Display profile report
        if st.session_state.get('profile_generated', False) and 'profile_report' in st.session_state:
            try:
                st.markdown("---")
                st.markdown("### 📊 Automated Profile Report")
                
                # Display as HTML component
                profile_html = st.session_state.profile_report.to_html()
                st.components.v1.html(profile_html, height=800, scrolling=True)
                
            except Exception as e:
                st.error(f"Failed to display profile: {str(e)}")

# Tab 2: Model Training
with tabs[1]:
    st.markdown("## 🤖 Model Training Laboratory")
    
    if not st.session_state.data_uploaded:
        st.info("👈 Upload data first to start model development")
    else:
        data = st.session_state.data.copy()
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**Target Variable Selection**")
            all_columns = list(data.columns)
            target_column = st.selectbox(
                "Choose target column",
                all_columns,
                help="Select the column you want to predict"
            )
        
        with config_col2:
            st.markdown("**Feature Selection**")
            available_features = [col for col in all_columns if col != target_column]
            feature_mode = st.radio(
                "Feature selection mode",
                ["Use all features", "Select manually"],
                horizontal=True
            )
        
        if feature_mode == "Select manually":
            feature_cols = st.multiselect(
                "Choose features for training",
                available_features,
                default=available_features,
                help="Select which columns to use as input features"
            )
            if not feature_cols:
                st.warning("⚠️ Please select at least one feature to continue")
                st.stop()
        else:
            feature_cols = available_features
        
        # Data preparation and analysis
        X = data[feature_cols].copy()
        y_raw = data[target_column].copy()
        
        # Smart problem type detection
        if y_raw.dtype == "object" or y_raw.nunique() <= 20:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_.tolist()
            problem_type = "classification"
            
            st.success(f"🎯 **Classification Problem Detected** - {len(class_names)} classes")
        else:
            le_target = None
            y = y_raw.to_numpy()
            class_names = None
            problem_type = "regression"
            
            st.success("🎯 **Regression Problem Detected**")
        
        # Target distribution visualization
        st.markdown("### 📊 Target Variable Analysis")
        
        if problem_type == "classification":
            target_counts = pd.Series(y).value_counts().sort_index()
            class_labels = [class_names[i] for i in target_counts.index]
            
            fig_target = px.bar(
                x=class_labels,
                y=target_counts.values,
                title="Target Class Distribution",
                labels={'x': 'Classes', 'y': 'Count'},
                color=class_labels,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_target.update_layout(showlegend=False)
            st.plotly_chart(fig_target, use_container_width=True)
            
            # Show class balance info
            total_samples = len(y)
            balance_info = pd.DataFrame({
                'Class': class_labels,
                'Count': target_counts.values,
                'Percentage': (target_counts.values / total_samples * 100).round(1)
            })
            st.dataframe(balance_info, use_container_width=True)
            
        else:
            fig_target = px.histogram(
                y_raw,
                nbins=30,
                title="Target Distribution",
                labels={'x': 'Value', 'y': 'Frequency'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
            # Show regression stats
            stats_data = {
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{y.mean():.3f}",
                    f"{np.median(y):.3f}",
                    f"{y.std():.3f}",
                    f"{y.min():.3f}",
                    f"{y.max():.3f}",
                    f"{y.max() - y.min():.3f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        # Algorithm selection and configuration
        st.markdown("### 🤖 Algorithm Selection")
        
        algo_col1, algo_col2 = st.columns(2)
        
        with algo_col1:
            st.markdown("**Choose Algorithm**")
            
            if problem_type == "classification":
                model_options = {
                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Support Vector Machine": SVC(probability=True, random_state=42),
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
                    "Support Vector Machine": SVR(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42)
                }
            
            model_choice = st.selectbox("Select algorithm", list(model_options.keys()))
            base_model = model_options[model_choice]
        
        with algo_col2:
            st.markdown("**Training Options**")
            
            # Class balancing (only for classification)
            if problem_type == "classification":
                balance_options = ["None", "SMOTE", "Random Oversample", "Random Undersample"]
                balance_method = st.selectbox(
                    "⚖️ Class Balancing",
                    balance_options,
                    help="Handle imbalanced datasets"
                )
            else:
                balance_method = "None"
            
            # Train-test split
            test_size = st.slider(
                "📊 Test Size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            )
        
        # Training section
        st.markdown("### 🏋️ Model Training")
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            # Validation
            if len(np.unique(y)) < 2 and problem_type == "classification":
                st.error("❌ Need at least 2 classes in target variable for classification")
                st.stop()
            
            # Training container
            training_container = st.container()
            
            with training_container:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Data splitting
                    status_text.info("📊 Splitting data into train/test sets...")
                    progress_bar.progress(15)
                    
                    if problem_type == "classification" and len(np.unique(y)) >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=test_size/100, 
                            random_state=42, 
                            stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=test_size/100, 
                            random_state=42
                        )
                    
                    # Step 2: Preprocessing pipeline
                    status_text.info("🔧 Building preprocessing pipeline...")
                    progress_bar.progress(30)
                    
                    # Identify feature types
                    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    # Create preprocessing pipeline
                    preprocessor_steps = []
                    
                    if numeric_features:
                        preprocessor_steps.append(('num', StandardScaler(), numeric_features))
                    
                    if categorical_features:
                        preprocessor_steps.append(('cat', safe_onehot_encoder(), categorical_features))
                    
                    if preprocessor_steps:
                        preprocessor = ColumnTransformer(
                            transformers=preprocessor_steps,
                            remainder='drop'
                        )
                    else:
                        preprocessor = 'passthrough'
                    
                    # Step 3: Handle class balancing
                    status_text.info("⚖️ Configuring class balancing...")
                    progress_bar.progress(45)
                    
                    pipeline_steps = []
                    
                    if preprocessor != 'passthrough':
                        pipeline_steps.append(('preprocessor', preprocessor))
                    
                    # Add sampling if needed
                    if balance_method != "None" and problem_type == "classification":
                        if balance_method == "SMOTE":
                            sampler = SMOTE(random_state=42)
                        elif balance_method == "Random Oversample":
                            sampler = RandomOverSampler(random_state=42)
                        elif balance_method == "Random Undersample":
                            sampler = RandomUnderSampler(random_state=42)
                        
                        pipeline_steps.append(('sampler', sampler))
                    
                    # Add model
                    pipeline_steps.append(('model', base_model))
                    
                    # Create appropriate pipeline
                    if balance_method != "None" and problem_type == "classification":
                        model_pipeline = ImbPipeline(pipeline_steps)
                    else:
                        model_pipeline = Pipeline(pipeline_steps)
                    
                    # Step 4: Model training
                    status_text.info("🏋️ Training the model... (this may take a moment)")
                    progress_bar.progress(65)
