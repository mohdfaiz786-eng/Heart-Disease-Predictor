# app.py - CardioAI Pro Final (Theme Working)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import numpy as np
import base64
import json
import sqlite3
import hashlib
import requests
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="CardioAI Pro",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS - MOBILE FRIENDLY + PROFESSIONAL (WITH THEME SUPPORT)
# ---------------------------
def load_css():
    """Load CSS with dynamic theme support"""
    
    # Base CSS
    base_css = """
    <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* Base Styles */
        .stApp {
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        /* Cards */
        .card {
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.45);
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 15px;
            padding: 1rem;
            color: white;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: scale(1.05);
        }
        
        /* Alert Boxes */
        .success-alert {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 1.5rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }
        
        .error-alert {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            padding: 1.5rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-radius: 12px;
            transition: all 0.3s ease;
            width: 100%;
            cursor: pointer;
            min-height: 48px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #334155;
            padding: 12px;
            transition: all 0.3s ease;
            background: #1e293b;
            color: white;
            font-size: 16px;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            outline: none;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #0a0f1a 100%);
            min-width: 280px;
        }
        
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        
        /* Bottom Navigation - Mobile Only */
        .bottom-nav {
            display: none;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #0f172a;
            padding: 12px;
            z-index: 1000;
            border-top: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .bottom-nav-item {
            flex: 1;
            text-align: center;
            padding: 8px;
            border-radius: 12px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .bottom-nav-item:hover {
            background: rgba(59, 130, 246, 0.2);
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1f2937;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 10px;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .gradient-text {
                font-size: 1.5rem !important;
            }
            
            .metric-card {
                padding: 0.75rem !important;
                margin: 5px 0 !important;
            }
            
            .metric-card h2 {
                font-size: 1.2rem !important;
            }
            
            .card {
                padding: 1rem !important;
                margin: 10px 0 !important;
            }
            
            .stButton > button {
                padding: 0.75rem !important;
                font-size: 1rem !important;
            }
            
            [data-testid="stSidebar"] {
                display: none !important;
            }
            
            .bottom-nav {
                display: flex !important;
            }
            
            .main .block-container {
                padding-bottom: 80px !important;
            }
        }
        
        /* Glass Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem;
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* Best Model Card */
        .best-model-card {
            border: 2px solid #10b981;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
            animation: pulse 2s infinite;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 4px;
            flex-wrap: wrap;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
        }
        
        /* Recommendation Cards */
        .rec-card {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .rec-card:hover {
            background: rgba(59, 130, 246, 0.2);
            transform: translateX(5px);
        }
        
        .rec-icon {
            font-size: 1.5rem;
            margin-right: 12px;
        }
        
        /* Risk Meter */
        .risk-meter {
            width: 100%;
            height: 12px;
            background: #334155;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .risk-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease;
        }
        
        .risk-high { background: linear-gradient(90deg, #ef4444, #dc2626); }
        .risk-low { background: linear-gradient(90deg, #10b981, #059669); }
        
        /* Loading Animation */
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(59, 130, 246, 0.3);
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)
    
    # ============ THEME APPLICATION - FIXED ============
    theme_css = ""
    
    # Background
    if "theme_bg" in st.session_state and st.session_state.theme_bg:
        bg_value = st.session_state.theme_bg
        theme_css += f"""
        <style>
            .stApp {{
                background: {bg_value} !important;
            }}
        </style>
        """
    
    # Font
    if "theme_font" in st.session_state and st.session_state.theme_font:
        font_family = st.session_state.theme_font
        theme_css += f"""
        <style>
            * {{
                font-family: {font_family} !important;
            }}
        </style>
        """
    
    # Text Color
    if "theme_text_color" in st.session_state and st.session_state.theme_text_color:
        text_color = st.session_state.theme_text_color
        theme_css += f"""
        <style>
            .stMarkdown, .stText, p, .stTitle, label, div {{
                color: {text_color} !important;
            }}
        </style>
        """
    
    # Heading Color
    if "theme_heading_color" in st.session_state and st.session_state.theme_heading_color:
        heading_color = st.session_state.theme_heading_color
        theme_css += f"""
        <style>
            h1, h2, h3, h4, h5, h6, .gradient-text {{
                color: {heading_color} !important;
            }}
            .gradient-text {{
                -webkit-text-fill-color: {heading_color} !important;
                background: none !important;
            }}
        </style>
        """
    
    # Card Background
    if "theme_card_bg" in st.session_state and st.session_state.theme_card_bg:
        card_bg = st.session_state.theme_card_bg
        theme_css += f"""
        <style>
            .card {{
                background: {card_bg} !important;
            }}
        </style>
        """
    
    # Apply theme CSS
    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)
    
    # Bottom Navigation HTML
    bottom_nav_html = """
    <div class="bottom-nav">
        <div class="bottom-nav-item" onclick="window.location.href='?page=Dashboard'">🏠</div>
        <div class="bottom-nav-item" onclick="window.location.href='?page=Predict'">❤️</div>
        <div class="bottom-nav-item" onclick="window.location.href='?page=Analytics'">📊</div>
        <div class="bottom-nav-item" onclick="window.location.href='?page=History'">📜</div>
        <div class="bottom-nav-item" onclick="window.location.href='?page=Settings'">⚙️</div>
    </div>
    """
    st.markdown(bottom_nav_html, unsafe_allow_html=True)

# ---------------------------
# ENHANCED RECOMMENDATION SYSTEM
# ---------------------------
class RecommendationEngine:
    @staticmethod
    def get_recommendations(risk_percent, age, sex, chol, bp, thalach):
        recommendations = []
        
        if risk_percent > 70:
            recommendations.append({"icon": "🚨", "title": "Immediate Action Required", "desc": "Consult a cardiologist within 24 hours.", "priority": "critical"})
        elif risk_percent > 50:
            recommendations.append({"icon": "⚠️", "title": "High Priority", "desc": "Schedule doctor's appointment this week.", "priority": "high"})
        elif risk_percent > 30:
            recommendations.append({"icon": "📋", "title": "Moderate Risk", "desc": "Schedule check-up within 3 months.", "priority": "medium"})
        else:
            recommendations.append({"icon": "✅", "title": "Low Risk", "desc": "Maintain healthy habits.", "priority": "low"})
        
        if age > 60:
            recommendations.append({"icon": "👴", "title": "Age Consideration", "desc": "Check blood pressure weekly.", "priority": "high"})
        elif age > 45:
            recommendations.append({"icon": "👨", "title": "Age Alert", "desc": "Annual ECG recommended.", "priority": "medium"})
        
        if chol > 240:
            recommendations.append({"icon": "🍔", "title": "High Cholesterol", "desc": "Reduce saturated fats. Target: <200 mg/dL", "priority": "high"})
        elif chol > 200:
            recommendations.append({"icon": "🥗", "title": "Borderline Cholesterol", "desc": "Increase fiber-rich foods.", "priority": "medium"})
        
        if bp > 140:
            recommendations.append({"icon": "💓", "title": "High Blood Pressure", "desc": "Reduce salt, monitor BP daily.", "priority": "high"})
        elif bp > 120:
            recommendations.append({"icon": "🩺", "title": "Elevated BP", "desc": "Monitor BP weekly.", "priority": "medium"})
        
        recommendations.append({"icon": "🥦", "title": "Heart-Healthy Diet", "desc": "Mediterranean diet recommended.", "priority": "general"})
        recommendations.append({"icon": "😴", "title": "Sleep Quality", "desc": "Aim for 7-8 hours of sleep.", "priority": "general"})
        recommendations.append({"icon": "🧘", "title": "Stress Management", "desc": "Practice meditation or yoga.", "priority": "general"})
        
        return recommendations

# ---------------------------
# SESSION INIT
# ---------------------------
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
    
    # Theme settings
    if "theme_bg" not in st.session_state:
        st.session_state.theme_bg = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    if "theme_font" not in st.session_state:
        st.session_state.theme_font = "'Inter', sans-serif"
    if "theme_text_color" not in st.session_state:
        st.session_state.theme_text_color = "#ffffff"
    if "theme_heading_color" not in st.session_state:
        st.session_state.theme_heading_color = "#60a5fa"
    if "theme_card_bg" not in st.session_state:
        st.session_state.theme_card_bg = "rgba(30, 41, 59, 0.95)"
    
    # Profile settings
    if "profile_pic" not in st.session_state:
        st.session_state.profile_pic = None
    if "profile_name" not in st.session_state:
        st.session_state.profile_name = None
    if "profile_bio" not in st.session_state:
        st.session_state.profile_bio = "Heart Health Enthusiast"

# ---------------------------
# DATABASE CLASS
# ---------------------------
class Database:
    def __init__(self):
        self.db_path = "users.db"
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            login_count INTEGER DEFAULT 0
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_level TEXT,
            risk_percentage REAL,
            model_used TEXT,
            patient_data TEXT
        )
        """)
        conn.commit()
        conn.close()
    
    def hash_pwd(self, pwd):
        return hashlib.sha256((pwd + "salt2024").encode()).hexdigest()
    
    def add_user(self, username, password, email, full_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email, full_name) VALUES (?,?,?,?)",
                     (username, self.hash_pwd(password), email, full_name))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()
    
    def verify_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        if data and data[0] == self.hash_pwd(password):
            self.update_login(username)
            return True
        return False
    
    def update_login(self, username):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE users SET login_count = login_count + 1 WHERE username=?", (username,))
        conn.commit()
        conn.close()
    
    def get_user_stats(self, username):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT full_name, email, login_count, created_at FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        return data
    
    def save_prediction(self, username, risk_level, risk_percentage, model_used, patient_data):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO predictions (username, risk_level, risk_percentage, model_used, patient_data) VALUES (?,?,?,?,?)",
                (username, risk_level, risk_percentage, model_used, json.dumps(patient_data))
            )
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()

# ---------------------------
# AUTO FEATURE SELECTOR
# ---------------------------
class AutoFeatureSelector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.feature_importances = None
        self.selected_features = None
    
    def prepare_data(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median())
        return X, y
    
    def auto_select_features(self, method='auto', k=10):
        X, y = self.prepare_data()
        
        if method == 'correlation':
            correlations = []
            for col in X.columns:
                corr = abs(X[col].corr(y))
                correlations.append((col, corr))
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected = [col for col, corr in correlations[:k]]
        
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=False)
            selected = imp_df.head(k)['Feature'].tolist()
            self.feature_importances = imp_df
        
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=False)
            selected = imp_df.head(k)['Feature'].tolist()
            self.feature_importances = imp_df
        
        self.selected_features = selected
        return selected

# ---------------------------
# MODEL TRAINER
# ---------------------------
class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.results = None
        self.trained_models = {}
        self.X_test = None
        self.y_test = None
    
    def train_all(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = []
        for name, model in self.models.items():
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                pipeline.fit(self.X_train, self.y_train)
                y_pred = pipeline.predict(self.X_test)
                
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    "Model": name,
                    "Accuracy": f"{acc:.4f}",
                    "Accuracy_Score": acc,
                    "Precision": f"{prec:.4f}",
                    "Recall": f"{rec:.4f}",
                    "F1-Score": f"{f1:.4f}"
                })
                self.trained_models[name] = pipeline
            except Exception as e:
                results.append({"Model": name, "Accuracy": "Error", "Accuracy_Score": 0, "Precision": "Error", "Recall": "Error", "F1-Score": "Error"})
        
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values('Accuracy_Score', ascending=False)
        return self.results
    
    def get_best_model(self):
        if len(self.results) > 0:
            return self.results.iloc[0]["Model"], self.trained_models.get(self.results.iloc[0]["Model"])
        return None, None
    
    def predict(self, model_name, X):
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            return pred, proba
        return None, None

# ---------------------------
# REPORT GENERATOR
# ---------------------------
class ReportGenerator:
    def generate_html_report(self, results_df, dataset_info, feature_importance, best_model):
        results_html = results_df.to_html(index=False) if results_df is not None else "<p>No results</p>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>CardioAI Report</title>
        <style>
            body {{ font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; }}
            .container {{ max-width: 1200px; margin: auto; background: white; border-radius: 20px; padding: 30px; }}
            h1 {{ color: #667eea; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #667eea; color: white; padding: 12px; }}
            td {{ border: 1px solid #ddd; padding: 10px; }}
            .best {{ background: #10b981; color: white; padding: 15px; border-radius: 10px; text-align: center; }}
        </style>
        </head>
        <body>
        <div class="container">
            <h1>❤️ CardioAI Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>Dataset Info</h2>
            <p>Samples: {dataset_info.get('samples', 0)} | Features: {dataset_info.get('features', 0)} | Target: {dataset_info.get('target', 'N/A')}</p>
            <h2>Model Results</h2>
            {results_html}
            <div class="best"><h3>🏆 Best Model: {best_model}</h3></div>
        </div>
        </body>
        </html>
        """
        return html
    
    def generate_csv_report(self, dataset_info, feature_importance, results_df):
        import io
        output = io.StringIO()
        output.write(f"CardioAI Report,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        output.write(f"Total Samples,{dataset_info.get('samples', 0)}\n")
        output.write(f"Features,{dataset_info.get('features', 0)}\n")
        output.write(f"Target,{dataset_info.get('target', 'N/A')}\n\n")
        if results_df is not None:
            results_df.to_csv(output, index=False)
        return output.getvalue()

# ---------------------------
# LOGIN PAGE
# ---------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem;">❤️</div>
            <h1 class="gradient-text">CardioAI Pro</h1>
            <p>Advanced Heart Disease Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        db = Database()
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    if db.verify_user(username, password):
                        st.session_state.auth = True
                        st.session_state.user = username
                        st.session_state.profile_name = username
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Sign Up", use_container_width=True):
                    if all([full_name, email, username, password, confirm]) and password == confirm and len(password) >= 6:
                        if db.add_user(username, password, email, full_name):
                            st.success("Account created! Please login.")
                        else:
                            st.error("Username exists")
                    else:
                        st.warning("Fill all fields correctly")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# DASHBOARD PAGE
# ---------------------------
def dashboard_page():
    st.markdown('<h1 class="gradient-text">Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>5</h2><p>ML Models</p></div>', unsafe_allow_html=True)
    with col2:
        status = "✅" if st.session_state.trainer else "❌"
        st.markdown(f'<div class="metric-card"><h2>{status}</h2><p>Models Trained</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{len(st.session_state.history)}</h2><p>Predictions</p></div>', unsafe_allow_html=True)
    with col4:
        features = len(st.session_state.selected_features) if st.session_state.selected_features else 0
        st.markdown(f'<div class="metric-card"><h2>{features}</h2><p>Features</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Feature Selection", use_container_width=True):
            st.session_state.page = "Feature"
            st.rerun()
    with col2:
        if st.button("🚀 Train Models", use_container_width=True):
            st.session_state.page = "Train"
            st.rerun()
    
    st.info("Workflow: Upload Dataset → Feature Selection → Train Models → Predict → Reports")

# ---------------------------
# FEATURE SELECTION PAGE
# ---------------------------
def feature_page():
    st.markdown('<h1 class="gradient-text">Auto Feature Selection</h1>', unsafe_allow_html=True)
    
    with st.expander("📁 Upload Dataset", expanded=st.session_state.df is None):
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
            
            possible_targets = [col for col in df.columns if any(w in col.lower() for w in ['heart', 'disease', 'target', 'class'])]
            mode = st.radio("Target Selection", ["Auto Detect", "Manual Select"])
            target = possible_targets[0] if mode == "Auto Detect" and possible_targets else st.selectbox("Select Target", df.columns)
            st.session_state.target_col = target
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(df[target].value_counts())
            with col2:
                st.plotly_chart(px.pie(df, names=target, title="Class Distribution"), use_container_width=True)
            
            method = st.selectbox("Method", ["Auto", "Random Forest", "Correlation"])
            k = st.slider("Number of Features", 3, min(20, len(df.columns)-1), 10)
            
            if st.button("🚀 Run Selection", use_container_width=True):
                with st.spinner("Selecting..."):
                    selector = AutoFeatureSelector(df, target)
                    selected = selector.auto_select_features(method.lower().replace(" ", "_"), k)
                    st.session_state.selected_features = selected
                    st.session_state.feature_importance = selector.feature_importances
                    st.success(f"✅ Selected {len(selected)} features")
                    
                    cols = st.columns(4)
                    for i, feat in enumerate(selected):
                        with cols[i % 4]:
                            st.markdown(f"✅ {feat}")
                    
                    if selector.feature_importances is not None:
                        fig = px.bar(selector.feature_importances.head(15), x="Importance", y="Feature", orientation='h')
                        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TRAIN MODELS PAGE
# ---------------------------
def train_page():
    st.markdown('<h1 class="gradient-text">Train 5 Models</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Upload dataset first!")
        if st.button("Go to Upload"):
            st.session_state.page = "Feature"
            st.rerun()
    elif st.session_state.selected_features is None:
        st.warning("Run feature selection first!")
        if st.button("Go to Feature Selection"):
            st.session_state.page = "Feature"
            st.rerun()
    else:
        st.success(f"Dataset: {st.session_state.df.shape[0]} rows | Features: {len(st.session_state.selected_features)}")
        
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Train All Models", use_container_width=True):
            with st.spinner("Training..."):
                X = st.session_state.df[st.session_state.selected_features].copy()
                y = st.session_state.df[st.session_state.target_col].copy()
                
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
                
                display_df = results.drop(columns=['Accuracy_Score'])
                st.dataframe(display_df, use_container_width=True)
                
                best = results.iloc[0]["Model"]
                st.markdown(f'<div class="best-model-card"><h3>🏆 Best: {best}</h3></div>', unsafe_allow_html=True)

# ---------------------------
# PREDICT PAGE WITH ENHANCED RECOMMENDATIONS
# ---------------------------
def predict_page():
    st.markdown('<h1 class="gradient-text">Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    if st.session_state.trainer is None:
        st.warning("Train models first!")
        if st.button("Go to Training"):
            st.session_state.page = "Train"
            st.rerun()
    else:
        best_name, _ = st.session_state.trainer.get_best_model()
        st.success(f"🎯 Best Model: {best_name}")
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Patient Information")
        
        cols = st.columns(2)
        inputs = {}
        features = st.session_state.selected_features[:12]
        
        default_age, default_sex, default_chol, default_bp, default_thalach = 55, 1, 220, 130, 138
        
        for i, feat in enumerate(features):
            with cols[i % 2]:
                default_val = float(st.session_state.df[feat].median()) if st.session_state.df[feat].dtype in ['int64', 'float64'] else 0.0
                inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=default_val, step=1.0, format="%.2f")
                
                if 'age' in feat.lower(): default_age = inputs[feat]
                if 'sex' in feat.lower(): default_sex = inputs[feat]
                if 'chol' in feat.lower(): default_chol = inputs[feat]
                if 'trestbps' in feat.lower(): default_bp = inputs[feat]
                if 'thalach' in feat.lower(): default_thalach = inputs[feat]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        model_options = list(st.session_state.trainer.models.keys())
        selected_model = st.selectbox("Select Model", model_options, index=model_options.index(best_name))
        
        if st.button("🔍 Analyze Risk", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                df_input = pd.DataFrame([inputs])
                for feat in features:
                    if feat not in df_input.columns:
                        df_input[feat] = 0
                df_input = df_input[features]
                
                for col in df_input.columns:
                    if df_input[col].dtype == 'object':
                        df_input[col] = LabelEncoder().fit_transform(df_input[col].astype(str))
                df_input = df_input.fillna(0)
                
                pred, proba = st.session_state.trainer.predict(selected_model, df_input)
                
                if pred is not None:
                    risk_percent = proba[0][1] * 100 if proba is not None else 0
                    risk_level = "High Risk" if pred[0] == 1 else "Low Risk"
                    
                    db = Database()
                    db.save_prediction(st.session_state.user, risk_level, risk_percent, selected_model, inputs)
                    
                    st.session_state.history.insert(0, {
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Model": selected_model,
                        "Risk": risk_level,
                        "Probability": f"{risk_percent:.1f}%",
                        **inputs
                    })
                    
                    risk_color = "risk-high" if pred[0] == 1 else "risk-low"
                    st.markdown(f'<div class="risk-meter"><div class="risk-fill {risk_color}" style="width: {risk_percent}%"></div></div>', unsafe_allow_html=True)
                    
                    if pred[0] == 1:
                        st.markdown(f'<div class="error-alert"><h2>⚠️ High Risk</h2><p>Risk: {risk_percent:.1f}%</p><p>Consult a doctor immediately!</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-alert"><h2>✅ Low Risk</h2><p>Health: {100-risk_percent:.1f}%</p><p>Keep healthy habits!</p></div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Personalized Recommendations")
                    
                    recs = RecommendationEngine.get_recommendations(risk_percent, default_age, default_sex, default_chol, default_bp, default_thalach)
                    for rec in recs[:6]:
                        st.markdown(f'<div class="rec-card"><span style="font-size:1.2rem;">{rec["icon"]}</span> <strong>{rec["title"]}</strong><br><small>{rec["desc"]}</small></div>', unsafe_allow_html=True)

# ---------------------------
# COMPARISON PAGE
# ---------------------------
def comparison_page():
    st.markdown('<h1 class="gradient-text">Model Comparison</h1>', unsafe_allow_html=True)
    
    if st.session_state.results is not None:
        display_df = st.session_state.results.drop(columns=['Accuracy_Score'])
        st.dataframe(display_df, use_container_width=True)
        
        best = st.session_state.results.iloc[0]["Model"]
        st.markdown(f'<div class="best-model-card"><h3>🏆 Best: {best}</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(st.session_state.results, x="Model", y="Accuracy_Score", title="Accuracy Comparison", color="Model")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(st.session_state.results, x="Model", y="Accuracy_Score", title="Accuracy Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No models trained yet!")

# ---------------------------
# ANALYTICS PAGE
# ---------------------------
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
            st.metric("Total", len(df))
            st.metric("Positive", df[target].sum())
            st.metric("Negative", len(df) - df[target].sum())
        
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 1:
            st.subheader("Correlation Matrix")
            fig = px.imshow(numeric.corr(), text_auto=True, color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("No dataset loaded. Upload first!")

# ---------------------------
# HISTORY PAGE
# ---------------------------
def history_page():
    st.markdown('<h1 class="gradient-text">Prediction History</h1>', unsafe_allow_html=True)
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                st.markdown(f'<a href="data:file/csv;base64,{b64}" download="history.csv">✅ Download</a>', unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No predictions yet!")

# ---------------------------
# REPORTS PAGE
# ---------------------------
def reports_page():
    st.markdown('<h1 class="gradient-text">📄 Generate Reports</h1>', unsafe_allow_html=True)
    
    try:
        has_results = st.session_state.results is not None and isinstance(st.session_state.results, pd.DataFrame) and len(st.session_state.results) > 0
        
        if has_results:
            st.markdown('<div style="background:#10b981;padding:15px;border-radius:12px;margin-bottom:20px;"><h3 style="color:white;">✅ Models Trained Successfully!</h3></div>', unsafe_allow_html=True)
            
            dataset_info = {
                'samples': len(st.session_state.df) if st.session_state.df else 0,
                'features': len(st.session_state.selected_features) if st.session_state.selected_features else 0,
                'target': st.session_state.target_col or 'N/A',
                'positive': 0, 'negative': 0
            }
            
            if st.session_state.df and st.session_state.target_col and st.session_state.target_col in st.session_state.df.columns:
                try:
                    dataset_info['positive'] = int(st.session_state.df[st.session_state.target_col].sum())
                    dataset_info['negative'] = int(len(st.session_state.df) - dataset_info['positive'])
                except: pass
            
            feature_imp = pd.DataFrame({'Feature': [], 'Importance': []})
            if st.session_state.feature_importance is not None and len(st.session_state.feature_importance) > 0:
                feature_imp = st.session_state.feature_importance
            elif st.session_state.selected_features:
                weights = [1.0/len(st.session_state.selected_features)] * len(st.session_state.selected_features)
                feature_imp = pd.DataFrame({'Feature': st.session_state.selected_features, 'Importance': weights}).sort_values('Importance', ascending=False)
            
            best_model = st.session_state.results.iloc[0]["Model"] if len(st.session_state.results) > 0 else "N/A"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Samples", dataset_info['samples'])
            with col2: st.metric("Features", dataset_info['features'])
            with col3: st.metric("Positive Cases", dataset_info['positive'])
            with col4: st.metric("Negative Cases", dataset_info['negative'])
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                report_type = st.selectbox("Report Format", ["📊 HTML Report", "📈 CSV Report"])
            with col2:
                if st.button("🚀 Generate Report", use_container_width=True):
                    with st.spinner("Generating..."):
                        try:
                            report_gen = ReportGenerator()
                            if "HTML" in report_type:
                                html = report_gen.generate_html_report(st.session_state.results, dataset_info, feature_imp, best_model)
                                b64 = base64.b64encode(html.encode()).decode()
                                st.markdown(f'<div style="text-align:center;"><a href="data:text/html;base64,{b64}" download="report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" style="background:#10b981;color:white;padding:12px 30px;text-decoration:none;border-radius:8px;">📥 Download HTML Report</a></div>', unsafe_allow_html=True)
                                st.success("✅ Report generated!")
                            else:
                                csv_data = report_gen.generate_csv_report(dataset_info, feature_imp, st.session_state.results)
                                b64 = base64.b64encode(csv_data.encode()).decode()
                                st.markdown(f'<div style="text-align:center;"><a href="data:file/csv;base64,{b64}" download="report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="background:#3b82f6;color:white;padding:12px 30px;text-decoration:none;border-radius:8px;">📥 Download CSV Report</a></div>', unsafe_allow_html=True)
                                st.success("✅ Report generated!")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.markdown('<div style="background:#fef3c7;padding:30px;border-radius:20px;text-align:center;"><h2>⚠️ No Models Trained Yet!</h2><p>Complete the workflow first</p></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Feature Selection"): st.session_state.page = "Feature"; st.rerun()
            with col2:
                if st.button("🤖 Train Models"): st.session_state.page = "Train"; st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------
# SETTINGS PAGE
# ---------------------------
def settings_page():
    st.markdown('<h1 class="gradient-text">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🎨 Theme & Background", "✍️ Font & Color"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👤 Profile Settings")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Profile Picture")
            if st.session_state.profile_pic:
                st.image(st.session_state.profile_pic, width=150)
            else:
                st.markdown(f'<div style="text-align:center;"><div style="width:150px;height:150px;background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto;"><span style="font-size:60px;color:white;">{st.session_state.user[0].upper() if st.session_state.user else "👤"}</span></div></div>', unsafe_allow_html=True)
            
            uploaded_pic = st.file_uploader("Upload Photo", type=['jpg','png','jpeg'])
            if uploaded_pic:
                st.session_state.profile_pic = uploaded_pic
                st.success("✅ Profile picture updated!")
                st.rerun()
            if st.button("🗑️ Remove Picture", use_container_width=True):
                st.session_state.profile_pic = None
                st.rerun()
        
        with col2:
            st.markdown("### Personal Information")
            new_name = st.text_input("Display Name", value=st.session_state.profile_name or st.session_state.user)
            if new_name != st.session_state.profile_name:
                st.session_state.profile_name = new_name
            new_bio = st.text_area("Bio", value=st.session_state.profile_bio, height=80)
            if new_bio != st.session_state.profile_bio:
                st.session_state.profile_bio = new_bio
            
            db = Database()
            user_data = db.get_user_stats(st.session_state.user)
            if user_data:
                st.divider()
                st.write(f"**📧 Email:** {user_data[1] if len(user_data)>1 and user_data[1] else 'Not set'}")
                st.write(f"**🔑 Total Logins:** {user_data[2] if len(user_data)>2 else '0'}")
            if st.button("💾 Save Profile", use_container_width=True):
                st.success("✅ Profile updated!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎨 Theme & Background")
        
        st.markdown("### Gradient Themes")
        col1, col2 = st.columns(2)
        gradients = {
            "Default Dark": "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            "Ocean Blue": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "Sunset Orange": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
            "Forest Green": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
            "Royal Blue": "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)"
        }
        for i, (name, grad) in enumerate(gradients.items()):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"🎨 {name}", key=f"grad_{i}", use_container_width=True):
                    st.session_state.theme_bg = grad
                    st.success(f"✅ {name} theme applied!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Custom Color")
        custom_color = st.color_picker("Pick a color", "#0f172a")
        if st.button("Apply Custom Color", use_container_width=True):
            st.session_state.theme_bg = custom_color
            st.success("✅ Custom color applied!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Card Background")
        card_bg = st.color_picker("Card Background Color", "#1e293b")
        if st.button("Apply Card Color", use_container_width=True):
            st.session_state.theme_card_bg = card_bg
            st.success("✅ Card color applied!")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✍️ Font & Color")
        
        st.markdown("### Font Style")
        col1, col2 = st.columns(2)
        fonts = {
            "Default": "'Inter', sans-serif",
            "Modern": "'Poppins', 'Inter', sans-serif",
            "Classic": "'Georgia', 'Times New Roman', serif",
            "Minimal": "'Helvetica Neue', Arial, sans-serif"
        }
        for i, (name, font) in enumerate(fonts.items()):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"✍️ {name}", key=f"font_{i}", use_container_width=True):
                    st.session_state.theme_font = font
                    st.success(f"✅ {name} font applied!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Text Colors")
        col1, col2 = st.columns(2)
        with col1:
            text_color = st.color_picker("Text Color", "#ffffff")
            if st.button("Apply Text Color", key="text_btn", use_container_width=True):
                st.session_state.theme_text_color = text_color
                st.success("✅ Text color applied!")
                st.rerun()
        with col2:
            heading_color = st.color_picker("Heading Color", "#60a5fa")
            if st.button("Apply Heading Color", key="heading_btn", use_container_width=True):
                st.session_state.theme_heading_color = heading_color
                st.success("✅ Heading color applied!")
                st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reset All Settings", use_container_width=True):
            st.session_state.theme_bg = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
            st.session_state.theme_font = "'Inter', sans-serif"
            st.session_state.theme_text_color = "#ffffff"
            st.session_state.theme_heading_color = "#60a5fa"
            st.session_state.theme_card_bg = "rgba(30, 41, 59, 0.95)"
            st.success("✅ All settings reset!")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# API TEST PAGE
# ---------------------------
def api_page():
    st.markdown('<h1 class="gradient-text">🔌 API Integration</h1>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>📡 REST API Endpoints</h3><p>Your model is available as a REST API.</p></div>', unsafe_allow_html=True)
    
    api_url = st.text_input("API Base URL", value="http://localhost:8000")
    if st.button("🔌 Check API Health", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ API is healthy! Model loaded: {data.get('model_loaded', False)}")
            else:
                st.error(f"API returned status: {response.status_code}")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")
            st.info("Run: uvicorn api:app --reload --port 8000")
    
    st.markdown("---")
    st.subheader("💻 Code Examples")
    tabs = st.tabs(["Python", "JavaScript", "cURL"])
    with tabs[0]:
        st.code("""
import requests
response = requests.post("http://localhost:8000/predict", json={
    "age": 55, "sex": 1, "trestbps": 130, "chol": 220, "thalach": 138
})
print(response.json())
        """, language="python")
    with tabs[1]:
        st.code("""
fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({age:55,sex:1,trestbps:130,chol:220,thalach:138})
}).then(res=>res.json()).then(data=>console.log(data));
        """, language="javascript")
    with tabs[2]:
        st.code("""
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age":55,"sex":1,"trestbps":130,"chol":220,"thalach":138}'
        """, language="bash")

# ---------------------------
# MAIN
# ---------------------------
def main():
    load_css()
    init_session()
    
    if not st.session_state.auth:
        login_page()
        return
    
    with st.sidebar:
        if st.session_state.profile_pic:
            st.image(st.session_state.profile_pic, width=100)
        else:
            st.markdown(f'<div style="text-align:center;"><div style="width:80px;height:80px;background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto;"><span style="font-size:35px;color:white;">{st.session_state.user[0].upper() if st.session_state.user else "👤"}</span></div><h3>{st.session_state.profile_name or st.session_state.user}</h3><p style="font-size:12px;">{st.session_state.profile_bio}</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        pages = {
            "🏠 Dashboard": "Dashboard", "🔍 Feature Selection": "Feature", "🤖 Train Models": "Train",
            "❤️ Predict": "Predict", "📈 Comparison": "Comparison", "📊 Analytics": "Analytics",
            "📜 History": "History", "📄 Reports": "Reports", "🔌 API Test": "API", "⚙️ Settings": "Settings"
        }
        for display, page in pages.items():
            if st.button(display, use_container_width=True):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth = False
            st.rerun()
    
    page = st.session_state.page
    if page == "Dashboard": dashboard_page()
    elif page == "Feature": feature_page()
    elif page == "Train": train_page()
    elif page == "Predict": predict_page()
    elif page == "Comparison": comparison_page()
    elif page == "Analytics": analytics_page()
    elif page == "History": history_page()
    elif page == "Reports": reports_page()
    elif page == "API": api_page()
    elif page == "Settings": settings_page()

if __name__ == "__main__":
    main()
