"""
ML Assignment 2: Heart Disease Risk Prediction - Streamlit App
Interactive Web Application for Model Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">❤️ Heart Disease Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Classification Models Comparison</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Model Configuration")

# Load pre-trained models and artifacts
@st.cache_resource
def load_models_and_artifacts():
    """Load all trained models and preprocessing artifacts"""
    models = {}
    model_names = [
        'logistic_regression_model.pkl',
        'decision_tree_model.pkl',
        'k_nearest_neighbor_model.pkl',
        'naive_bayes_model.pkl',
        'random_forest_model.pkl',
        'xgboost_model.pkl'
    ]
    
    for model_file in model_names:
        try:
            with open(f'model/{model_file}', 'rb') as f:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                models[model_name] = pickle.load(f)
        except:
            st.error(f"Could not load {model_file}")
    
    # Load scaler and encoders
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('model/detailed_results.pkl', 'rb') as f:
            detailed_results = pickle.load(f)
        with open('model/model_results.csv', 'rb') as f:
            results_df = pd.read_csv(f)
    except:
        scaler = None
        encoders = None
        detailed_results = None
        results_df = None
    
    return models, scaler, encoders, detailed_results, results_df

models, scaler, encoders, detailed_results, results_df = load_models_and_artifacts()

# Model selection
model_display_names = {
    'Logistic Regression': 'Logistic Regression',
    'Decision Tree': 'Decision Tree',
    'K Nearest Neighbor': 'K-Nearest Neighbor',
    'Naive Bayes': 'Naive Bayes',
    'Random Forest': 'Random Forest (Ensemble)',
    'Xgboost': 'XGBoost (Ensemble)'
}

selected_model_key = st.sidebar.selectbox(
    "🎯 Select Model",
    options=list(models.keys()),
    format_func=lambda x: model_display_names.get(x, x)
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Dataset Upload")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with test data for predictions"
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Performance", "🔮 Predictions", "📊 Model Comparison", "ℹ️ About"])

with tab1:
    st.header(f"Model Performance: {model_display_names.get(selected_model_key, selected_model_key)}")
    
    if detailed_results and selected_model_key in detailed_results:
        result = detailed_results[selected_model_key]
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{result['accuracy']:.4f}")
            st.metric("📐 Precision", f"{result['precision']:.4f}")
        
        with col2:
            st.metric("📊 AUC Score", f"{result['auc']:.4f}")
            st.metric("🔍 Recall", f"{result['recall']:.4f}")
        
        with col3:
            st.metric("⚖️ F1 Score", f"{result['f1']:.4f}")
            st.metric("🔢 MCC Score", f"{result['mcc']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔲 Confusion Matrix")
            cm = result['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'Confusion Matrix - {selected_model_key}')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📋 Classification Report")
            st.text(result['classification_report'])
    else:
        st.warning("Model performance data not available")

with tab2:
    st.header("🔮 Make Predictions")
    
    if uploaded_file is not None:
        try:
            # Load data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(test_data.head(10))
            
            # Preprocess button
            if st.button("🚀 Run Predictions"):
                with st.spinner("Processing predictions..."):
                    # Preprocess data
                    df_pred = test_data.copy()
                    
                    # Check if target exists
                    has_target = 'Heart_Disease' in df_pred.columns
                    if has_target:
                        y_true = df_pred['Heart_Disease']
                        df_pred = df_pred.drop(['Heart_Disease'], axis=1)
                    
                    # Encode categorical variables
                    if encoders:
                        if 'Gender' in df_pred.columns:
                            df_pred['Gender'] = encoders['gender'].transform(df_pred['Gender'])
                        if 'ECG_Result' in df_pred.columns:
                            df_pred['ECG_Result'] = encoders['ecg'].transform(df_pred['ECG_Result'])
                        if 'Smoking_Status' in df_pred.columns:
                            df_pred['Smoking_Status'] = encoders['smoking'].transform(df_pred['Smoking_Status'])
                        if 'Physical_Activity_Level' in df_pred.columns:
                            df_pred['Physical_Activity_Level'] = encoders['activity'].transform(df_pred['Physical_Activity_Level'])
                        if 'Risk_Level' in df_pred.columns:
                            df_pred['Risk_Level'] = encoders['risk'].transform(df_pred['Risk_Level'])
                    
                    # Scale features
                    if scaler:
                        X_scaled = scaler.transform(df_pred)
                    else:
                        X_scaled = df_pred.values
                    
                    # Make predictions
                    model = models[selected_model_key]
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Display results
                    st.success("✅ Predictions completed!")
                    
                    # Results dataframe
                    results_display = test_data.copy()
                    results_display['Predicted_Risk'] = predictions
                    results_display['Predicted_Label'] = ['Disease' if p == 1 else 'No Disease' for p in predictions]
                    if probabilities is not None:
                        results_display['Probability'] = probabilities
                    
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_display.head(20))
                    
                    # Summary statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                        st.metric("Predicted Disease Cases", int(predictions.sum()))
                    with col2:
                        st.metric("Predicted Healthy Cases", int((predictions == 0).sum()))
                        if probabilities is not None:
                            st.metric("Avg Risk Probability", f"{probabilities.mean():.4f}")
                    
                    # If ground truth available, show metrics
                    if has_target:
                        st.subheader("📈 Performance on Test Data")
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        acc = accuracy_score(y_true, predictions)
                        prec = precision_score(y_true, predictions, zero_division=0)
                        rec = recall_score(y_true, predictions, zero_division=0)
                        f1 = f1_score(y_true, predictions, zero_division=0)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{acc:.4f}")
                        col2.metric("Precision", f"{prec:.4f}")
                        col3.metric("Recall", f"{rec:.4f}")
                        col4.metric("F1 Score", f"{f1:.4f}")
                    
                    # Download predictions
                    csv = results_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct format and column names")
    else:
        st.info("👆 Please upload a CSV file in the sidebar to make predictions")
        
        # Show expected format
        st.subheader("📋 Expected Data Format")
        st.markdown("""
        Your CSV should contain the following columns:
        - Age, Gender, Resting_BP, Cholesterol, Fasting_Blood_Sugar
        - Max_Heart_Rate, ECG_Result, Smoking_Status, Alcohol_Consumption
        - Physical_Activity_Level, Diet_Quality_Score, Sleep_Hours, BMI
        - Diabetes, Hypertension, Family_History, Risk_Score, Risk_Level
        - (Optional) Heart_Disease (if you want to evaluate predictions)
        """)

with tab3:
    st.header("📊 All Models Comparison")
    
    if results_df is not None:
        st.subheader("📋 Performance Metrics Table")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']), use_container_width=True)
        
        # Visualizations
        st.subheader("📈 Visual Comparison")
        
        # Metrics selection
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        # Bar chart
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['ML Model Name'],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart
        st.subheader("🕸️ Radar Chart Comparison")
        
        selected_models = st.multiselect(
            "Select models to compare",
            options=results_df['ML Model Name'].tolist(),
            default=results_df['ML Model Name'].tolist()[:3]
        )
        
        if selected_models:
            fig = go.Figure()
            
            for model in selected_models:
                model_data = results_df[results_df['ML Model Name'] == model].iloc[0]
                fig.add_trace(go.Scatterpolar(
                    r=[model_data['Accuracy'], model_data['Precision'], model_data['Recall'],
                       model_data['F1'], model_data['AUC'], model_data['MCC']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC'],
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Project Overview
    This application demonstrates the implementation and deployment of multiple machine learning 
    classification models for heart disease risk prediction.
    
    ### 📊 Dataset
    **Heart Disease Risk Prediction Dataset**
    - **Source**: Kaggle/UCI (Synthetic)
    - **Total Instances**: 12,000
    - **Features**: 18 predictive features
    - **Target**: Heart_Disease (Binary: 0 = No Disease, 1 = Disease)
    
    ### 🤖 Models Implemented
    1. **Logistic Regression** - Linear probabilistic classifier
    2. **Decision Tree** - Tree-based rule classifier
    3. **K-Nearest Neighbor** - Instance-based learning
    4. **Naive Bayes** - Probabilistic classifier based on Bayes theorem
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting ensemble method
    
    ### 📈 Evaluation Metrics
    - **Accuracy**: Overall correctness
    - **AUC Score**: Area under ROC curve
    - **Precision**: Positive prediction accuracy
    - **Recall**: True positive detection rate
    - **F1 Score**: Harmonic mean of precision and recall
    - **MCC Score**: Matthews correlation coefficient
    
    ### 🛠️ Technologies Used
    - **Python** 3.x
    - **Scikit-learn** for ML models
    - **Streamlit** for web interface
    - **Plotly** for interactive visualizations
    - **Pandas** for data manipulation
    
    ### 👨‍🎓 Assignment Details
    - **Course**: M.Tech (AIML/DSE) - Machine Learning
    - **Institution**: BITS Pilani Work Integrated Learning
    - **Assignment**: Classification & Deployment
    
    ### 📝 Repository Structure
    ```
    project/
    ├── app.py                    # Streamlit application
    ├── model/
    │   ├── train_models.py       # Model training script
    │   ├── *.pkl                 # Saved models
    │   └── model_results.csv     # Performance results
    ├── requirements.txt          # Python dependencies
    └── README.md                 # Documentation
    ```
    
    ### 🚀 Deployment
    This application is deployed on **Streamlit Community Cloud** for easy access and demonstration.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ for ML Assignment 2 | BITS Pilani")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Need Help?")
st.sidebar.info("Upload a CSV file with the correct format to get predictions!")
