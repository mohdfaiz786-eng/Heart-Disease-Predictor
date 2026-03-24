# app.py - Main Application

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import numpy as np
import base64
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from models import ModelTrainer

# Import custom modules
from database import Database
from feature_selector import AutoFeatureSelector
from models import ModelTrainer
from report_generator import ReportGenerator

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    with open('styles.css', 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Session state init
def init_session():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "df" not in st.session_state:
        st.session_state.df = None
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = None
    if "feature_importance" not in st.session_state:
        st.session_state.feature_importance = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "history" not in st.session_state:
        st.session_state.history = []

# Login page
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem;">🫀</div>
            <h1 class="gradient-text">CardioAI Pro</h1>
            <p>Advanced Heart Disease Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        db = Database()
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        if db.verify_user(username, password):
                            st.session_state.auth = True
                            st.session_state.user = username
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("⚠️ Please fill all fields")
        
        with tab2:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Enter full name")
                email = st.text_input("Email", placeholder="Enter email")
                username = st.text_input("Username", placeholder="Choose username")
                password = st.text_input("Password", type="password", placeholder="Choose password (min 6 chars)")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                submit = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit:
                    if all([full_name, email, username, password, confirm]):
                        if len(password) >= 6:
                            if password == confirm:
                                if db.add_user(username, password, email, full_name):
                                    st.success("✅ Account created! Please login.")
                                    st.balloons()
                                else:
                                    st.error("❌ Username already exists")
                            else:
                                st.error("❌ Passwords don't match")
                        else:
                            st.warning("⚠️ Password must be at least 6 characters")
                    else:
                        st.warning("⚠️ Please fill all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Dashboard
def dashboard_page():
    st.markdown('<h1 class="gradient-text">Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>5</h2>
            <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅" if st.session_state.trainer else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h2>{status}</h2>
            <p>Models Trained</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(st.session_state.history)}</h2>
            <p>Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        features = len(st.session_state.selected_features) if st.session_state.selected_features else 0
        st.markdown(f"""
        <div class="metric-card">
            <h2>{features}</h2>
            <p>Features Selected</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Auto Feature Selection", use_container_width=True):
            st.session_state.page = "Feature"
            st.rerun()
    with col2:
        if st.button("🚀 Train Models", use_container_width=True):
            st.session_state.page = "Train"
            st.rerun()
    
    st.markdown("---")
    
    st.info("""
    **CardioAI Pro - Complete Workflow:**
    
    1. 📁 **Upload Dataset** - Upload CSV file
    2. 🔍 **Auto Feature Selection** - Automatically select best features
    3. 🤖 **Train 5 Models** - Train all models simultaneously
    4. 🫀 **Make Predictions** - Get instant risk assessment
    5. 📄 **Generate Reports** - Download professional reports
    """)

# Feature Selection Page
def feature_page():
    st.markdown('<h1 class="gradient-text">Auto Feature Selection</h1>', unsafe_allow_html=True)
    
    with st.expander("📁 Upload Dataset", expanded=st.session_state.df is None):
        uploaded = st.file_uploader("Choose CSV Dataset", type=["csv"])
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
            
            # Auto detect target
            possible_targets = []
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['heart', 'disease', 'target', 'class', 'label']):
                    possible_targets.append(col)
            
            mode = st.radio("Target Selection", ["Auto Detect", "Manual Select"])
            
            if mode == "Auto Detect" and possible_targets:
                target = possible_targets[0]
                st.success(f"✅ Auto detected target: {target}")
            else:
                target = st.selectbox("Select Target Column", df.columns.tolist())
            
            st.session_state.target_col = target
            
            # Show distribution
            st.subheader("Target Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.write(df[target].value_counts())
            with col2:
                fig = px.pie(df, names=target, title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature selection
            st.subheader("Feature Selection Method")
            
            method = st.selectbox(
                "Select Method",
                ["Auto (Combine All)", "Correlation", "Mutual Information", "Random Forest", "RFE"]
            )
            
            k = st.slider("Number of Features", 3, min(20, len(df.columns)-1), 10)
            
            if st.button("🚀 Run Feature Selection", use_container_width=True):
                with st.spinner("Selecting features..."):
                    method_map = {
                        "Auto (Combine All)": "auto",
                        "Correlation": "correlation",
                        "Mutual Information": "mutual_info",
                        "Random Forest": "random_forest",
                        "RFE": "rfe"
                    }
                    
                    selector = AutoFeatureSelector(df, target)
                    selector.prepare_data()
                    selected = selector.auto_select_features(method_map[method], k)
                    
                    st.session_state.selected_features = selected
                    st.session_state.feature_importance = selector.feature_importances
                    
                    st.success(f"✅ Selected {len(selected)} features")
                    
                    # Display features
                    st.subheader("Selected Features")
                    cols = st.columns(4)
                    for i, feat in enumerate(selected):
                        with cols[i % 4]:
                            st.markdown(f"✅ {feat}")
                    
                    # Feature importance
                    if selector.feature_importances is not None:
                        st.subheader("Feature Importance")
                        fig = px.bar(selector.feature_importances.head(15), 
                                   x="Importance" if "Importance" in selector.feature_importances.columns else "MI_Score",
                                   y="Feature", orientation='h')
                        st.plotly_chart(fig, use_container_width=True)

# Train Page
def train_page():
    st.markdown('<h1 class="gradient-text">Train 5 Models</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload dataset first!")
        if st.button("📁 Go to Upload"):
            st.session_state.page = "Feature"
            st.rerun()
    elif st.session_state.selected_features is None:
        st.warning("⚠️ Please run feature selection first!")
        if st.button("🔍 Go to Feature Selection"):
            st.session_state.page = "Feature"
            st.rerun()
    else:
        st.success(f"✅ Dataset: {st.session_state.df.shape[0]} rows")
        st.success(f"✅ Selected Features: {len(st.session_state.selected_features)}")
        st.success(f"✅ Target: {st.session_state.target_col}")
        
        # Show selected features
        st.subheader("Features for Training")
        cols = st.columns(5)
        for i, feat in enumerate(st.session_state.selected_features[:10]):
            with cols[i % 5]:
                st.markdown(f"✅ {feat}")
        
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("🚀 Train All 5 Models", use_container_width=True):
            with st.spinner("Training models..."):
                df = st.session_state.df
                target = st.session_state.target_col
                features = st.session_state.selected_features
                
                X = df[features].copy()
                y = df[target].copy()
                
                # Handle categorical
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                X = X.fillna(X.median())
                
                trainer = ModelTrainer()
                results = trainer.train_all(X, y, test_size)
                
                st.session_state.trainer = trainer
                st.session_state.results = results
                
                st.success("✅ All models trained!")
                st.balloons()
                
                # Display results
                st.subheader("Training Results")
                display_df = results.drop(columns=['Model Object', 'Accuracy_Score'])
                st.dataframe(display_df, use_container_width=True)
                
                # Best model
                best = results.iloc[0]["Model"]
                best_acc = results.iloc[0]["Accuracy"]
                st.markdown(f"""
                <div class="best-model-card">
                    <h3>🏆 Best Model: {best}</h3>
                    <p>Accuracy: {best_acc}</p>
                </div>
                """, unsafe_allow_html=True)

# Predict Page
def predict_page():
    st.markdown('<h1 class="gradient-text">Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    if st.session_state.trainer is None:
        st.warning("⚠️ No models trained! Please train models first.")
        if st.button("🚀 Go to Training"):
            st.session_state.page = "Train"
            st.rerun()
    else:
        best_name, best_model = st.session_state.trainer.get_best_model()
        st.success(f"🎯 Best Model: **{best_name}**")
        
        # Input form
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Patient Information")
        
        cols = st.columns(2)
        inputs = {}
        
        features = st.session_state.selected_features[:12]
        
        for i, feat in enumerate(features):
            with cols[i % 2]:
                default_val = float(st.session_state.df[feat].median()) if st.session_state.df[feat].dtype in ['int64', 'float64'] else 0.0
                inputs[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=default_val,
                    step=1.0,
                    format="%.2f"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model selection
        model_options = list(st.session_state.trainer.models.keys())
        selected_model = st.selectbox("Select Model", model_options, index=model_options.index(best_name))
        
        if st.button("🔍 Analyze Risk", use_container_width=True):
            with st.spinner(f"Analyzing with {selected_model}..."):
                # Prepare input
                df_input = pd.DataFrame([inputs])
                
                for feat in features:
                    if feat not in df_input.columns:
                        df_input[feat] = 0
                
                df_input = df_input[features]
                
                # Handle categorical
                for col in df_input.columns:
                    if df_input[col].dtype == 'object':
                        df_input[col] = LabelEncoder().fit_transform(df_input[col].astype(str))
                
                df_input = df_input.fillna(0)
                
                # Predict
                pred, proba = st.session_state.trainer.predict(selected_model, df_input)
                
                if pred is not None:
                    risk_percent = proba[0][1] * 100 if proba is not None else 0
                    risk_level = "High Risk" if pred[0] == 1 else "Low Risk"
                    
                    # Save to database
                    db = Database()
                    db.save_prediction(st.session_state.user, risk_level, risk_percent, selected_model, inputs)
                    
                    # Save to session
                    st.session_state.history.insert(0, {
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Model": selected_model,
                        "Risk": risk_level,
                        "Probability": f"{risk_percent:.1f}%",
                        **inputs
                    })
                    
                    # Display result
                    if pred[0] == 1:
                        st.markdown(f"""
                        <div class="error-alert">
                            <h2>⚠️ {risk_level}</h2>
                            <p style="font-size: 24px;">Risk Probability: {risk_percent:.1f}%</p>
                            <p>Please consult a healthcare professional immediately!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📋 Detailed Recommendations"):
                            st.markdown("""
                            ### 🏥 Medical Recommendations
                            - **Immediate:** Schedule cardiologist appointment
                            - **Lifestyle:** 30 min exercise daily, reduce salt intake
                            - **Monitoring:** Check BP weekly
                            - **Medication:** Take prescribed medicines
                            - **Diet:** Mediterranean diet
                            """)
                    else:
                        st.markdown(f"""
                        <div class="success-alert">
                            <h2>✅ {risk_level}</h2>
                            <p style="font-size: 24px;">Health Probability: {100-risk_percent:.1f}%</p>
                            <p>Maintain healthy lifestyle habits!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("💪 Preventive Recommendations"):
                            st.markdown("""
                            ### 💪 Healthy Habits
                            - **Exercise:** 150 min moderate activity/week
                            - **Diet:** Fruits, vegetables, whole grains
                            - **Sleep:** 7-8 hours quality sleep
                            - **Stress:** Meditation or yoga
                            - **Checkups:** Annual health screening
                            """)

# Comparison Page
def comparison_page():
    st.markdown('<h1 class="gradient-text">Model Comparison</h1>', unsafe_allow_html=True)
    
    if st.session_state.results is not None:
        display_df = st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'])
        st.dataframe(display_df, use_container_width=True)
        
        # Best model
        best = st.session_state.results.iloc[0]["Model"]
        st.markdown(f"""
        <div class="best-model-card">
            <h3>🏆 Best Model: {best}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(st.session_state.results, x="Model", y="Accuracy_Score",
                        title="Accuracy Comparison", color="Model")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            metrics = st.session_state.results.melt(id_vars=["Model"],
                                                   value_vars=["Accuracy_Score", "Precision", "Recall", "F1-Score"],
                                                   var_name="Metric", value_name="Score")
            metrics['Score'] = pd.to_numeric(metrics['Score'], errors='coerce')
            metrics = metrics.dropna()
            
            fig = px.line_polar(metrics, r="Score", theta="Metric",
                               color="Model", line_close=True,
                               title="Radar Chart")
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        if st.session_state.trainer and st.session_state.trainer.X_test is not None:
            st.subheader("Confusion Matrices")
            cols = st.columns(3)
            models = list(st.session_state.trainer.models.keys())[:3]
            
            for idx, model_name in enumerate(models):
                with cols[idx]:
                    cm = st.session_state.trainer.get_confusion_matrix(model_name)
                    if cm is not None:
                        fig = px.imshow(cm, text_auto=True, title=model_name)
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No models trained yet!")

# Analytics Page
def analytics_page():
    st.markdown('<h1 class="gradient-text">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is not None and st.session_state.target_col:
        df = st.session_state.df
        target = st.session_state.target_col
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names=target, title="Target Distribution", hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistics")
            st.write(f"**Total:** {len(df)}")
            st.write(f"**Positive:** {df[target].sum()}")
            st.write(f"**Negative:** {len(df) - df[target].sum()}")
        
        # Correlation
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 1:
            st.subheader("Correlation Matrix")
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("📊 No dataset loaded. Please upload a dataset first.")

# History Page
def history_page():
    st.markdown('<h1 class="gradient-text">Prediction History</h1>', unsafe_allow_html=True)
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("📝 No predictions yet. Make your first prediction!")

# Reports Page
# Reports Page (Fixed)
def reports_page():
    st.markdown('<h1 class="gradient-text">Generate Reports</h1>', unsafe_allow_html=True)
    
    if st.session_state.results is not None and len(st.session_state.results) > 0:
        st.success("✅ Models trained! Ready to generate report.")
        
        # Dataset info
        dataset_info = {
            'samples': len(st.session_state.df) if st.session_state.df is not None else 0,
            'features': len(st.session_state.selected_features) if st.session_state.selected_features else 0,
            'target': st.session_state.target_col if st.session_state.target_col else 'N/A',
            'positive': int(st.session_state.df[st.session_state.target_col].sum()) if st.session_state.df is not None and st.session_state.target_col else 0,
            'negative': int(len(st.session_state.df) - st.session_state.df[st.session_state.target_col].sum()) if st.session_state.df is not None and st.session_state.target_col else 0
        }
        
        # Feature importance
        if st.session_state.feature_importance is not None and len(st.session_state.feature_importance) > 0:
            feature_imp = st.session_state.feature_importance
        else:
            # Create default feature importance if not available
            if st.session_state.selected_features:
                feature_imp = pd.DataFrame({
                    'Feature': st.session_state.selected_features,
                    'Importance': [1.0/len(st.session_state.selected_features)] * len(st.session_state.selected_features)
                }).sort_values('Importance', ascending=False)
            else:
                feature_imp = pd.DataFrame({'Feature': [], 'Importance': []})
        
        # Best model
        best_model = st.session_state.results.iloc[0]["Model"] if len(st.session_state.results) > 0 else "N/A"
        
        # Report options
        st.subheader("Report Options")
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Select Report Format",
                ["HTML Report", "CSV Report"]
            )
        
        with col2:
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    try:
                        report_gen = ReportGenerator()
                        
                        if report_type == "HTML Report":
                            # Generate HTML
                            html = report_gen.generate_html_report(
                                st.session_state.results,
                                dataset_info,
                                feature_imp,
                                best_model
                            )
                            
                            # Create download link
                            b64 = base64.b64encode(html.encode()).decode()
                            filename = f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 20px;">
                                <a href="data:text/html;base64,{b64}" 
                                   download="{filename}"
                                   style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                          color: white;
                                          padding: 12px 24px;
                                          text-decoration: none;
                                          border-radius: 10px;
                                          display: inline-block;
                                          font-weight: 600;">
                                    📥 Download HTML Report
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("✅ Report generated successfully!")
                            st.balloons()
                            
                            # Preview
                            with st.expander("📄 Preview Report"):
                                st.components.v1.html(html, height=600, scrolling=True)
                        
                        else:  # CSV Report
                            csv_data = report_gen.generate_csv_report(
                                st.session_state.results,
                                dataset_info,
                                feature_imp
                            )
                            
                            b64 = base64.b64encode(csv_data.encode()).decode()
                            filename = f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 20px;">
                                <a href="data:file/csv;base64,{b64}" 
                                   download="{filename}"
                                   style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                          color: white;
                                          padding: 12px 24px;
                                          text-decoration: none;
                                          border-radius: 10px;
                                          display: inline-block;
                                          font-weight: 600;">
                                    📥 Download CSV Report
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("✅ CSV Report generated successfully!")
                    
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Please make sure all data is available and try again.")
        
        # Quick Export Options
        st.markdown("---")
        st.subheader("Quick Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Model Results", use_container_width=True):
                csv = st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'], errors='ignore').to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Ready to download!")
        
        with col2:
            if st.button("📈 Export Feature Importance", use_container_width=True):
                if len(feature_imp) > 0:
                    csv = feature_imp.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="feature_importance.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Ready to download!")
                else:
                    st.warning("No feature importance data available")
        
        with col3:
            if st.button("📋 Export All Data", use_container_width=True):
                all_data = {
                    "Model Results": st.session_state.results.drop(columns=['Model Object', 'Accuracy_Score'], errors='ignore').to_dict(),
                    "Dataset Info": dataset_info,
                    "Feature Importance": feature_imp.to_dict() if len(feature_imp) > 0 else {}
                }
                import json
                json_data = json.dumps(all_data, indent=2, default=str)
                b64 = base64.b64encode(json_data.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="all_data.json">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Ready to download!")
    
    else:
        st.warning("⚠️ No models trained yet! Please train models first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Go to Feature Selection"):
                st.session_state.page = "Feature"
                st.rerun()
        with col2:
            if st.button("🤖 Go to Training"):
                st.session_state.page = "Train"
                st.rerun()
        
        st.info("""
        **How to generate report:**
        1. First upload dataset
        2. Run feature selection
        3. Train models
        4. Then come back here to generate report
        """)
# Settings Page
def settings_page():
    st.markdown('<h1 class="gradient-text">Settings</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    db = Database()
    user_data = db.get_user_stats(st.session_state.user)
    
    if user_data:
        st.write(f"**👤 Name:** {user_data[0]}")
        st.write(f"**📧 Email:** {user_data[1]}")
        st.write(f"**🔑 Total Logins:** {user_data[2]}")
    
    st.markdown("---")
    
    st.subheader("System Info")
    st.write("**Version:** 4.0 Pro")
    st.write("**Models:** 5 ML Models")
    st.write("**Feature Selection:** 4 Methods + Auto")
    
    st.markdown("---")
    
    if st.button("🔄 Reset All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main
def main():
    load_css()
    init_session()
    
    if not st.session_state.auth:
        login_page()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">👤</div>
            <h3>{st.session_state.user}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        pages = {
            "🏠 Dashboard": "Dashboard",
            "🔍 Feature Selection": "Feature",
            "🤖 Train Models": "Train",
            "🫀 Predict": "Predict",
            "📈 Comparison": "Comparison",
            "📊 Analytics": "Analytics",
            "📜 History": "History",
            "📄 Reports": "Reports",
            "⚙️ Settings": "Settings"
        }
        
        for display, page in pages.items():
            if st.button(display, use_container_width=True):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth = False
            st.rerun()
    
    # Page routing
    page = st.session_state.page
    
    if page == "Dashboard":
        dashboard_page()
    elif page == "Feature":
        feature_page()
    elif page == "Train":
        train_page()
    elif page == "Predict":
        predict_page()
    elif page == "Comparison":
        comparison_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "History":
        history_page()
    elif page == "Reports":
        reports_page()
    elif page == "Settings":
        settings_page()

if __name__ == "__main__":
    main()
