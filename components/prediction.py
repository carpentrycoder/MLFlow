import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go

def run_prediction():
    """
    DataSutra Prediction Engine - The thread that binds your data intelligence
    Supports both file upload and manual input for predictions
    """
    
    # Apply DataSutra styling
    st.markdown("""
    <div class="datasutra-card fade-in-up">
        <h1>🔮 Prediction Engine</h1>
        <p class="datasutra-tagline">Make intelligent predictions on new data</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if trained model exists
    if "trained_model" not in st.session_state:
        st.markdown("""
        <div class="status-warning">
            <h3>⚠️ No Trained Model Found</h3>
            <p>Please train a model in the Model Selector first before making predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Go to Model Selector"):
            st.switch_page("Model Selector")
        return

    # Get model package from session
    model_package = st.session_state["trained_model"]
    model = model_package["model"]
    scaler = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    target_column = model_package["target_column"]
    problem_type = model_package["problem_type"]
    label_encoder = model_package.get("label_encoder")
    original_labels = model_package.get("original_labels")
    model_type = model_package["model_type"]

    # Display model information
    st.markdown("""
    <div class="datasutra-panel">
        <h3>🤖 Current Model Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", model_type)
    with col2:
        st.metric("Problem Type", problem_type)
    with col3:
        st.metric("Features", len(feature_columns))
    with col4:
        st.metric("Target", target_column)

    # Prediction method selection
    st.markdown("""
    <div class="feature-highlight">
        <h3>📊 Choose Prediction Method</h3>
    </div>
    """, unsafe_allow_html=True)
    
    prediction_method = st.radio(
        "Select how you want to input data for prediction:",
        ["📄 Upload CSV File", "✍️ Manual Input Form", "📋 Batch Manual Input"],
        horizontal=True,
        help="Choose between file upload or manual data entry"
    )

    if prediction_method == "📄 Upload CSV File":
        handle_file_upload_prediction(model_package)
    elif prediction_method == "✍️ Manual Input Form":
        handle_manual_input_prediction(model_package)
    else:  # Batch Manual Input
        handle_batch_manual_input(model_package)

def handle_file_upload_prediction(model_package):
    """Handle predictions from uploaded CSV file"""
    
    model = model_package["model"]
    scaler = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    problem_type = model_package["problem_type"]
    label_encoder = model_package.get("label_encoder")
    original_labels = model_package.get("original_labels")
    
    st.markdown("""
    <div class="datasutra-panel">
        <h3>📤 File Upload Prediction</h3>
        <p>Upload a CSV file with the same structure as your training features (excluding target column)</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=["csv"],
        help="Make sure your CSV has the same column names as the training data"
    )
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            # Display preview
            st.subheader("📄 Data Preview")
            st.dataframe(new_data.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(new_data))
            with col2:
                st.metric("Columns", len(new_data.columns))
            with col3:
                st.metric("Missing Values", new_data.isnull().sum().sum())
            
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("🔄 Processing predictions..."):
                    predictions_df = make_predictions(new_data, model_package)
                    
                    if predictions_df is not None:
                        display_predictions(predictions_df, problem_type, original_labels)
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.error("💡 Please ensure your CSV file is properly formatted")

def handle_manual_input_prediction(model_package):
    """Handle single prediction from manual input form"""
    
    feature_columns = model_package["feature_columns"]
    problem_type = model_package["problem_type"]
    original_labels = model_package.get("original_labels")
    
    st.markdown("""
    <div class="datasutra-panel">
        <h3>✍️ Manual Input Form</h3>
        <p>Enter values for each feature to get an instant prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    input_data = {}
    
    # Organize inputs in columns for better layout
    n_cols = min(3, len(feature_columns))
    cols = st.columns(n_cols)
    
    for i, feature in enumerate(feature_columns):
        col_idx = i % n_cols
        with cols[col_idx]:
            # Determine input type based on feature name
            if any(keyword in feature.lower() for keyword in ['age', 'year', 'count', 'number', 'id']):
                input_data[feature] = st.number_input(
                    f"📊 {feature}", 
                    value=0.0, 
                    key=f"manual_{feature}"
                )
            elif any(keyword in feature.lower() for keyword in ['rate', 'ratio', 'percent', 'score']):
                input_data[feature] = st.slider(
                    f"📈 {feature}", 
                    0.0, 1.0, 0.5, 
                    key=f"manual_{feature}"
                )
            elif any(keyword in feature.lower() for keyword in ['price', 'cost', 'amount', 'salary', 'income']):
                input_data[feature] = st.number_input(
                    f"💰 {feature}", 
                    value=0.0, 
                    min_value=0.0,
                    key=f"manual_{feature}"
                )
            else:
                # Default to number input
                input_data[feature] = st.number_input(
                    f"🔢 {feature}", 
                    value=0.0, 
                    key=f"manual_{feature}"
                )
    
    # Create prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            with st.spinner("🔄 Generating prediction..."):
                prediction_df = make_predictions(input_df, model_package)
                
                if prediction_df is not None:
                    # Display single prediction prominently
                    prediction_value = prediction_df['Prediction'].iloc[0]
                    
                    if problem_type == "Classification" and original_labels is not None:
                        predicted_class = original_labels[int(prediction_value)]
                        st.markdown(f"""
                        <div class="feature-highlight text-center">
                            <h2>🎯 Prediction Result</h2>
                            <h1 style="color: var(--accent-blue); font-size: 3rem;">{predicted_class}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="feature-highlight text-center">
                            <h2>🎯 Prediction Result</h2>
                            <h1 style="color: var(--accent-blue); font-size: 3rem;">{prediction_value:.4f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        st.json(input_data)
                    
                    # Show prediction confidence (if available)
                    if hasattr(model_package["model"], "predict_proba") and problem_type == "Classification":
                        try:
                            input_scaled = prepare_data_for_prediction(input_df, model_package)
                            probabilities = model_package["model"].predict_proba(input_scaled)[0]
                            
                            st.subheader("📊 Prediction Confidence")
                            conf_df = pd.DataFrame({
                                'Class': original_labels if original_labels is not None else range(len(probabilities)),
                                'Probability': probabilities
                            })
                            
                            fig = px.bar(
                                conf_df, 
                                x='Class', 
                                y='Probability',
                                title="Class Probabilities"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.warning("Could not calculate prediction probabilities")

def handle_batch_manual_input(model_package):
    """Handle multiple predictions from manual batch input"""
    
    feature_columns = model_package["feature_columns"]
    problem_type = model_package["problem_type"]
    original_labels = model_package.get("original_labels")
    
    st.markdown("""
    <div class="datasutra-panel">
        <h3>📋 Batch Manual Input</h3>
        <p>Enter multiple rows of data for batch predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Number of rows to input
    num_rows = st.slider("Number of predictions to make", 1, 10, 3)
    
    # Create data editor
    empty_data = pd.DataFrame(
        {col: [0.0] * num_rows for col in feature_columns}
    )
    
    st.subheader("📝 Enter Data")
    edited_data = st.data_editor(
        empty_data,
        use_container_width=True,
        num_rows="dynamic",
        key="batch_input"
    )
    
    if st.button("🚀 Generate Batch Predictions", type="primary"):
        if not edited_data.empty:
            with st.spinner("🔄 Processing batch predictions..."):
                prediction_df = make_predictions(edited_data, model_package)
                
                if prediction_df is not None:
                    display_predictions(prediction_df, problem_type, original_labels)
        else:
            st.warning("⚠️ Please enter some data first")

def prepare_data_for_prediction(new_data, model_package):
    """Prepare new data for prediction with proper preprocessing"""
    
    scaler = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    
    # Handle missing values
    new_data_copy = new_data.copy()
    
    # Fill missing values
    numeric_cols = new_data_copy.select_dtypes(include=[np.number]).columns
    new_data_copy[numeric_cols] = new_data_copy[numeric_cols].fillna(new_data_copy[numeric_cols].median())
    
    categorical_cols = new_data_copy.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        new_data_copy[col] = new_data_copy[col].fillna(new_data_copy[col].mode()[0] if not new_data_copy[col].mode().empty else 'Unknown')
    
    # Encode categorical variables
    if len(categorical_cols) > 0:
        new_data_copy = pd.get_dummies(new_data_copy, columns=categorical_cols, drop_first=True)
    
    # Handle missing columns (add them with 0 values)
    missing_cols = set(feature_columns) - set(new_data_copy.columns)
    for col in missing_cols:
        new_data_copy[col] = 0
    
    # Remove extra columns and reorder
    new_data_copy = new_data_copy[feature_columns]
    
    # Handle infinite values
    new_data_copy = new_data_copy.replace([np.inf, -np.inf], np.nan)
    new_data_copy = new_data_copy.fillna(0)
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data_copy)
    
    return new_data_scaled

def make_predictions(new_data, model_package):
    """Make predictions on new data"""
    
    try:
        model = model_package["model"]
        label_encoder = model_package.get("label_encoder")
        original_labels = model_package.get("original_labels")
        problem_type = model_package["problem_type"]
        
        # Prepare data
        new_data_scaled = prepare_data_for_prediction(new_data, model_package)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        
        # Decode predictions if classification with label encoder
        if problem_type == "Classification" and label_encoder is not None:
            try:
                predictions = label_encoder.inverse_transform(predictions.astype(int))
            except:
                # If decoding fails, use original labels mapping
                if original_labels is not None:
                    predictions = [original_labels[int(p)] if int(p) < len(original_labels) else f"Class_{int(p)}" for p in predictions]
        
        # Create result DataFrame
        result_df = new_data.copy()
        result_df['Prediction'] = predictions
        
        # Add confidence scores for classification
        if problem_type == "Classification" and hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(new_data_scaled)
                max_probs = np.max(probabilities, axis=1)
                result_df['Confidence'] = max_probs
            except:
                pass
        
        return result_df
        
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        st.error("💡 Please ensure your data matches the training format")
        return None

def display_predictions(predictions_df, problem_type, original_labels=None):
    """Display prediction results with visualizations"""
    
    st.markdown("""
    <div class="feature-highlight">
        <h2>🎯 Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display results table
    st.subheader("📊 Results Table")
    st.dataframe(predictions_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(predictions_df))
    
    if problem_type == "Classification":
        with col2:
            unique_predictions = predictions_df['Prediction'].nunique()
            st.metric("Unique Classes", unique_predictions)
        with col3:
            if 'Confidence' in predictions_df.columns:
                avg_confidence = predictions_df['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Prediction distribution
        if len(predictions_df) > 1:
            st.subheader("📈 Prediction Distribution")
            pred_counts = predictions_df['Prediction'].value_counts()
            
            fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Distribution of Predicted Classes"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        with col2:
            mean_pred = predictions_df['Prediction'].mean()
            st.metric("Mean Prediction", f"{mean_pred:.4f}")
        with col3:
            std_pred = predictions_df['Prediction'].std()
            st.metric("Std Deviation", f"{std_pred:.4f}")
        
        # Prediction distribution for regression
        if len(predictions_df) > 1:
            st.subheader("📈 Prediction Distribution")
            fig = px.histogram(
                predictions_df,
                x='Prediction',
                title="Distribution of Predicted Values",
                nbins=min(20, len(predictions_df))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.subheader("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"datasutra_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON download
        json_str = predictions_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_str,
            file_name=f"datasutra_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Detailed analysis
    with st.expander("🔍 Detailed Analysis"):
        st.subheader("Statistical Summary")
        
        if problem_type == "Regression":
            st.write("**Prediction Statistics:**")
            st.write(predictions_df['Prediction'].describe())
        else:
            st.write("**Class Distribution:**")
            st.write(predictions_df['Prediction'].value_counts())
            
            if 'Confidence' in predictions_df.columns:
                st.write("**Confidence Statistics:**")
                st.write(predictions_df['Confidence'].describe())
        
        st.subheader("Sample Predictions")
        st.dataframe(predictions_df.head(10))
    
    st.success("✅ Predictions completed successfully!")