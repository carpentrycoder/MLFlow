import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, r2_score, mean_squared_error,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def run_model_selector(df):
    """
    DataSutra Model Selector - The thread that binds your data intelligence
    Handles both Classification and Regression with intelligent preprocessing
    """
    
    # Apply DataSutra styling
    st.markdown("""
    <div class="datasutra-card fade-in-up">
        <h1>🧠 Model Selector</h1>
        <p class="datasutra-tagline">Intelligent model selection and training pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Validate input dataframe
        if df is None or df.empty:
            st.error("❌ No data available. Please upload a dataset first.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # === STEP 1: Problem Type Selection ===
        status_text.text("Step 1/7: Configuring problem type...")
        progress_bar.progress(1/7)
        
        st.markdown("""
        <div class="datasutra-panel">
            <h3>🔍 Problem Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            problem_type = st.radio(
                "Select Problem Type", 
                ["Classification", "Regression"],
                help="Choose based on your target variable type"
            )
        
        with col2:
            # Auto-suggest problem type based on target
            if st.button("🤖 Auto-Detect Problem Type"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(numeric_cols) > len(categorical_cols):
                    st.info("💡 Suggestion: Regression (more numeric columns detected)")
                else:
                    st.info("💡 Suggestion: Classification (more categorical columns detected)")
        
        # === STEP 2: Target Column Selection ===
        status_text.text("Step 2/7: Selecting target variable...")
        progress_bar.progress(2/7)
        
        target_column = st.selectbox(
            "🎯 Select Target Column", 
            df.columns,
            help="Choose the column you want to predict"
        )
        
        # Display target column info
        if target_column:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Values", df[target_column].nunique())
            with col2:
                st.metric("Missing Values", df[target_column].isnull().sum())
            with col3:
                st.metric("Data Type", str(df[target_column].dtype))
        
        # === STEP 3: PCA Feature Selection ===
        status_text.text("Step 3/7: Feature selection...")
        progress_bar.progress(3/7)
        
        st.markdown("""
        <div class="datasutra-panel">
            <h3>✨ Feature Selection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        use_pca_features = "No"
        if "top_pca_features" in st.session_state and st.session_state["top_pca_features"]:
            use_pca_features = st.radio(
                "Use PCA-Selected Features?", 
                ["Yes", "No"], 
                horizontal=True,
                help="Use features selected from PCA analysis"
            )
            
            if use_pca_features == "Yes":
                st.success(f"🎯 Using {len(st.session_state['top_pca_features'])} PCA-selected features")
            
            if st.button("🔄 Clear PCA Selection"):
                del st.session_state["top_pca_features"]
                st.success("✅ PCA selection cleared")
                st.experimental_rerun()
        
        # === STEP 4: Data Preprocessing ===
        status_text.text("Step 4/7: Preprocessing data...")
        progress_bar.progress(4/7)
        
        # Feature selection logic
        if use_pca_features == "Yes" and "top_pca_features" in st.session_state:
            available_features = [col for col in st.session_state["top_pca_features"] if col in df.columns and col != target_column]
            if available_features:
                X = df[available_features].copy()
                st.success(f"✅ Using {len(available_features)} PCA-selected features")
            else:
                st.warning("⚠️ PCA features not found. Using all available features.")
                X = df.drop(columns=[target_column]).copy()
        else:
            X = df.drop(columns=[target_column]).copy()
        
        y = df[target_column].copy()
        
        # Handle missing values in features
        if X.isnull().sum().sum() > 0:
            st.warning(f"⚠️ Found {X.isnull().sum().sum()} missing values in features. Filling with median/mode.")
            # Fill numeric columns with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            # Fill categorical columns with mode
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Handle missing values in target
        if y.isnull().sum() > 0:
            st.error(f"❌ Found {y.isnull().sum()} missing values in target column. Please clean your data first.")
            return
        
        # Encode categorical target for classification
        label_encoder = None
        original_labels = None
        if problem_type == "Classification":
            if y.dtype == "object" or y.dtype.name == 'category':
                label_encoder = LabelEncoder()
                original_labels = y.unique()
                y = label_encoder.fit_transform(y)
                st.info(f"🏷️ Encoded categorical target: {len(original_labels)} classes")
        
        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            st.info(f"🔄 Encoding {len(categorical_features)} categorical features")
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Ensure all data is numeric
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            st.error("❌ No numeric features available after preprocessing.")
            return
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Validate data shapes
        if X_scaled.shape[0] != len(y):
            st.error("❌ Mismatch between features and target dimensions.")
            return
        
        st.success(f"✅ Preprocessed data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        # === STEP 5: Train-Test Split ===
        status_text.text("Step 5/7: Splitting data...")
        progress_bar.progress(5/7)
        
        test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, help="For reproducible results")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=test_size, 
                random_state=int(random_state),
                stratify=y if problem_type == "Classification" and len(np.unique(y)) > 1 else None
            )
        except Exception as e:
            st.error(f"❌ Error in train-test split: {str(e)}")
            return
        
        # === STEP 6: Model Selection and Training ===
        status_text.text("Step 6/7: Training models...")
        progress_bar.progress(6/7)
        
        st.markdown("""
        <div class="datasutra-panel">
            <h3>🤖 Model Selection & Hyperparameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection based on problem type
        if problem_type == "Classification":
            model_options = {
                "Random Forest": RandomForestClassifier,
                "Logistic Regression": LogisticRegression,
                "Support Vector Machine": SVC
            }
        else:
            model_options = {
                "Random Forest": RandomForestRegressor,
                "Linear Regression": LinearRegression,
                "Support Vector Regression": SVR
            }
        
        model_choice = st.selectbox("🔧 Choose Model Algorithm", list(model_options.keys()))
        
        # Model-specific hyperparameters
        model_params = {}
        if "Random Forest" in model_choice:
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Trees", 10, 200, 100)
                model_params['n_estimators'] = n_estimators
            with col2:
                max_depth = st.slider("Max Depth", 3, 20, 10)
                model_params['max_depth'] = max_depth if max_depth < 20 else None
        elif "Logistic Regression" in model_choice:
            C_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            model_params['C'] = C_value
            model_params['max_iter'] = 1000
        elif "SVM" in model_choice or "SVR" in model_choice:
            C_value = st.slider("C Parameter", 0.1, 10.0, 1.0)
            model_params['C'] = C_value
            if "SVM" in model_choice:
                model_params['probability'] = True
        
        model_params['random_state'] = int(random_state)
        
        # Train model
        try:
            model_class = model_options[model_choice]
            model = model_class(**model_params)
            
            with st.spinner("🔄 Training model..."):
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
        except Exception as e:
            st.error(f"❌ Error during model training: {str(e)}")
            st.error("💡 Try adjusting hyperparameters or checking your data quality.")
            return
        
        # === STEP 7: Model Evaluation ===
        status_text.text("Step 7/7: Evaluating model...")
        progress_bar.progress(7/7)
        
        st.markdown("""
        <div class="feature-highlight">
            <h2>📊 Model Performance</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if problem_type == "Classification":
            # Classification metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1-Score", f"{f1:.4f}")
            
            # Classification report
            with st.expander("📋 Detailed Classification Report"):
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm, 
                text_auto=True, 
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{mae:.4f}")
            with col2:
                st.metric("MSE", f"{mse:.4f}")
            with col3:
                st.metric("RMSE", f"{rmse:.4f}")
            with col4:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Actual vs Predicted Plot
            st.subheader("📈 Actual vs Predicted")
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title="Actual vs Predicted Values"
            )
            # Add perfect prediction line
            min_val, max_val = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals Plot
            residuals = y_test - y_pred
            st.subheader("📉 Residuals Analysis")
            fig_residuals = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title="Residuals vs Predicted Values"
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            importances = pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False)
            
            top_k = st.slider("Show Top K Important Features", 5, min(20, len(importances)), 10)
            
            fig_importance = px.bar(
                x=importances.head(top_k).values,
                y=importances.head(top_k).index,
                orientation='h',
                title=f"Top {top_k} Feature Importances",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model Export
        st.markdown("""
        <div class="feature-highlight">
            <h3>💾 Export Trained Model</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_columns': list(X.columns),
            'target_column': target_column,
            'problem_type': problem_type,
            'label_encoder': label_encoder,
            'original_labels': original_labels,
            'model_type': model_choice,
            'preprocessing_info': {
                'categorical_features_encoded': len(categorical_features),
                'total_features': X_scaled.shape[1],
                'training_samples': X_train.shape[0]
            }
        }
        
        # Save model to bytes
        model_bytes = io.BytesIO()
        joblib.dump(model_package, model_bytes)
        model_bytes.seek(0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Trained Model",
                data=model_bytes.getvalue(),
                file_name=f"datasutra_{model_choice.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                help="Download the complete model package including preprocessing steps"
            )
        
        with col2:
            if st.button("💾 Save to Session"):
                st.session_state['trained_model'] = model_package
                st.success("✅ Model saved to session for predictions!")
        
        # Model Summary
        with st.expander("📋 Model Summary"):
            st.json({
                "Model Type": model_choice,
                "Problem Type": problem_type,
                "Training Samples": X_train.shape[0],
                "Test Samples": X_test.shape[0],
                "Features Used": X_scaled.shape[1],
                "Target Column": target_column,
                "Preprocessing": f"Standardized, {len(categorical_features)} categorical features encoded"
            })
        
        status_text.text("✅ Model training completed successfully!")
        progress_bar.progress(1.0)
        
        st.success("🎉 Model training pipeline completed successfully!")
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.error("💡 Please check your data quality and try again.")
        # Log the full error for debugging
        st.expander("🐛 Debug Information").exception(e)