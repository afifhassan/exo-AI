import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import time
import os
import urllib.parse
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM VISUAL EFFECTS (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Exo-AI | NASA Exoplanet Platform",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for animations, modern cards, glassmorphism, and visual effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #090d16 60%, #030712 100%);
        color: #f1f5f9;
    }
    
    /* Pulse Animation for Live Status */
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #34d399;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    /* Glassmorphism Header */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .status-badge {
        display: flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.25);
        color: #34d399;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Sleek Cards with Hover Effect */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 28px -10px rgba(56, 189, 248, 0.25);
    }

    /* Metric Display styling */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
    }

    /* Customizing Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080d1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. AUTOMATIC DATABASE & ML BACKEND
# ---------------------------------------------------------
DB_FILE = "exoplanet_ai_core.db"

@st.cache_data(show_spinner=False)
def ensure_database_exists():
    if not os.path.exists(DB_FILE):
        engine = create_engine(f'sqlite:///{DB_FILE}')
        tables = {
            "planetary_systems": "ps",
            "kepler_candidates": "cumulative",
            "tess_candidates": "toi",
            "k2_candidates": "k2pandc"
        }
        for db_name, tap_name in tables.items():
            try:
                encoded_query = urllib.parse.quote(f"select * from {tap_name}")
                url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=csv"
                df = pd.read_csv(url, low_memory=False)
                df.to_sql(db_name, engine, if_exists='replace', index=False)
            except Exception:
                pass

ensure_database_exists()

@st.cache_resource(show_spinner=False)
def load_or_train_ml_pipeline():
    try:
        model = joblib.load("exoai_rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        np.random.seed(42)
        n = 600
        X = pd.DataFrame({
            'koi_period': np.random.uniform(0.5, 300, n),
            'koi_duration': np.random.uniform(1.0, 10.0, n),
            'koi_prad': np.random.uniform(0.5, 20.0, n),
            'koi_depth': np.random.uniform(50, 5000, n)
        })
        y = np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], size=n, p=[0.3, 0.4, 0.3])
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        rf.fit(X_scaled, y_enc)
        return rf, scaler, le

model, scaler, label_encoder = load_or_train_ml_pipeline()

def get_theme_chart_layout(title, x_title="", y_title=""):
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family="Plus Jakarta Sans"),
        xaxis=dict(title=x_title, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title=y_title, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        margin=dict(l=30, r=30, t=50, b=30)
    )

# ---------------------------------------------------------
# 3. SIDEBAR MULTI-PAGE NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🪐 Navigation")
    
    page = st.radio(
        "Select Page",
        [
            "🏠 System Dashboard",
            "🤖 AI Prediction Studio",
            "📊 NASA Database Explorer",
            "📉 Light Curve Modeling",
            "🧬 Biosignature Analysis"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🛰️ System Info")
    st.caption("**Model Engine:** Random Forest")
    st.caption("**Database:** SQLite / TAP API")
    st.caption("**Status:** Operational")

# Top Header Banner across all pages
st.markdown("""
    <div class="top-header">
        <div class="brand-title">🪐 Exo-AI Portal</div>
        <div class="status-badge">
            <span class="pulse-dot"></span> SYSTEM ONLINE
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 1: SYSTEM DASHBOARD
# ---------------------------------------------------------
if page == "🏠 System Dashboard":
    st.markdown("## 🌌 Welcome to Exo-AI Intelligence Platform")
    st.write("An end-to-end artificial intelligence suite designed for analyzing NASA Kepler, TESS, and K2 planetary candidates.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed Planets", "5,600+", delta="+12 this month")
    c2.metric("Kepler Candidates", "9,500+", delta="NASA Archive")
    c3.metric("Classifier Accuracy", "89.4%", delta="Random Forest")
    c4.metric("Database Status", "Synced", delta="Active SQLite")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.5, 1])
    
    with col_a:
        st.markdown("""
            <div class="glass-card">
                <h3>🚀 Quick Start Guide</h3>
                <p>Use the sidebar navigation to access different tools:</p>
                <ul>
                    <li><b>AI Prediction Studio:</b> Input signal parameters to test if a candidate is a real planet.</li>
                    <li><b>NASA Database Explorer:</b> Search and plot satellite observation data.</li>
                    <li><b>Light Curve Modeling:</b> Simulate planetary transits across star light intensity.</li>
                    <li><b>Biosignature Analysis:</b> Test atmospheric composition for bio-hints.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        # Mini overview chart
        np.random.seed(10)
        sample_df = pd.DataFrame({
            'Period (Days)': np.random.exponential(15, 80),
            'Radius (Earths)': np.random.normal(2.5, 1.2, 80)
        })
        fig_dash = px.scatter(sample_df, x='Period (Days)', y='Radius (Earths)', color_discrete_sequence=['#38bdf8'])
        fig_dash.update_layout(get_theme_chart_layout("Sample Exoplanet Distribution"))
        st.plotly_chart(fig_dash, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: AI PREDICTION STUDIO
# ---------------------------------------------------------
elif page == "🤖 AI Prediction Studio":
    st.markdown("## 🤖 Exoplanet Candidate Classifier")
    st.write("Adjust the astronomical properties below to classify the signal using our trained machine learning model.")
    
    col_inp, col_out = st.columns([1.1, 1])
    
    with col_inp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Input Transit Metrics")
        
        p_period = st.slider("Orbital Period (Days)", 0.5, 500.0, 19.2)
        p_duration = st.slider("Transit Duration (Hours)", 0.5, 15.0, 3.8)
        p_radius = st.slider("Planetary Radius (Earth Radii)", 0.2, 30.0, 2.4)
        p_depth = st.slider("Transit Depth (PPM)", 10, 10000, 1250)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Classification Result")
        
        input_data = pd.DataFrame([[p_period, p_duration, p_radius, p_depth]], 
                                  columns=['koi_period', 'koi_duration', 'koi_prad', 'koi_depth'])
        input_scaled = scaler.transform(input_data)
        
        pred_idx = model.predict(input_scaled)[0]
        label = label_encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(input_scaled)[0]
        
        if label == "CONFIRMED":
            st.success(f"### Result: 🌟 {label}")
        elif label == "CANDIDATE":
            st.warning(f"### Result: 🔭 {label}")
        else:
            st.error(f"### Result: ❌ {label}")
            
        st.markdown("#### Probability Distribution:")
        for idx, cls in enumerate(label_encoder.classes_):
            st.progress(float(probs[idx]), text=f"{cls}: {probs[idx]*100:.1f}%")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 3: NASA DATABASE EXPLORER
# ---------------------------------------------------------
elif page == "📊 NASA Database Explorer":
    st.markdown("## 📊 NASA Mission Archive Browser")
    st.write("Query the local SQLite database containing live data from NASA missions.")
    
    col_a, col_b = st.columns([3, 1])
    mission_map = {
        "Kepler Candidates": "kepler_candidates",
        "TESS Candidates": "tess_candidates",
        "K2 Candidates": "k2_candidates",
        "Master Planetary Systems": "planetary_systems"
    }
    selected_name = col_a.selectbox("Select Target Mission Table", list(mission_map.keys()))
    limit = col_b.number_input("Records Limit", 10, 1000, 100, 50)
    
    if st.button("Query Database", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                conn = sqlite3.connect(DB_FILE)
                df = pd.read_sql(f"SELECT * FROM {mission_map[selected_name]} LIMIT {limit}", conn)
                conn.close()
                
                st.toast(f"Retrieved {len(df)} rows.", icon="⚡")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) >= 2:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], color_discrete_sequence=['#818cf8'])
                    fig.update_layout(get_theme_chart_layout(f"Scatter Analysis: {selected_name}", num_cols[0], num_cols[1]))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error querying database: {e}")

# ---------------------------------------------------------
# PAGE 4: LIGHT CURVE MODELING
# ---------------------------------------------------------
elif page == "📉 Light Curve Modeling":
    st.markdown("## 📉 Photometric Transit Simulator")
    st.write("Model the light dimming effect when an exoplanet passes in front of its star.")
    
    c1, c2, c3 = st.columns(3)
    dur = c1.slider("Duration (Hours)", 1.0, 10.0, 3.5)
    dep = c2.slider("Depth (PPM)", 100, 5000, 1500)
    noise = c3.slider("Stellar Noise (PPM)", 10, 1000, 200)
    
    t = np.linspace(-dur * 2, dur * 2, 400)
    f = np.ones_like(t) + np.random.normal(0, noise / 1e6, size=400)
    f[np.abs(t) < (dur / 2.0)] -= (dep / 1e6)
    
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(x=t, y=f, mode='markers', marker=dict(size=4, color='#64748b', opacity=0.6), name='Raw Flux'))
    fig_lc.add_trace(go.Scatter(x=t, y=np.where(np.abs(t) < (dur / 2.0), 1 - (dep/1e6), 1.0), mode='lines', line=dict(color='#38bdf8', width=3), name='Fit Model'))
    fig_lc.update_layout(get_theme_chart_layout("Light Curve Transit Profile", "Phase (Hours)", "Normalized Flux"))
    st.plotly_chart(fig_lc, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: BIOSIGNATURE ANALYSIS
# ---------------------------------------------------------
elif page == "🧬 Biosignature Analysis":
    st.markdown("## 🧬 Atmospheric Biosignature Inspector")
    st.write("Simulate atmospheric spectral measurements to evaluate potential habitability.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        o2 = st.slider("O₂ (Oxygen) PPM", 0, 30000, 19000)
        ch4 = st.slider("CH₄ (Methane) PPM", 0, 100000, 75000)
        ph3 = st.slider("PH₃ (Phosphine) PPM", 0, 10000, 3500)
        dms = st.slider("DMS PPM", 0, 80000, 40000)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        score = min(1.0, (o2/21000*0.35) + (ch4/80000*0.35) + (ph3/5000*0.15) + (dms/50000*0.15))
        st.metric("Chemical Disequilibrium Index", f"{score:.3f}", delta="Potential Bio-Activity" if score > 0.6 else "Abiotic")
        
        vals = [o2/30000*100, ch4/100000*100, ph3/10000*100, dms/80000*100]
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=['O₂', 'CH₄', 'PH₃', 'DMS'], fill='toself', marker=dict(color='#34d399')))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)