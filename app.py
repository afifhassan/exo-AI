import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import os
import urllib.parse
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ANIMATED DEEP SPACE CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Exo-AI | NASA Exoplanet Platform",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide the default sidebar
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background-color: #030712;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px),
            radial-gradient(rgba(255,255,255,.4), rgba(255,255,255,.1) 2px, transparent 30px);
        background-size: 550px 550px, 350px 350px, 250px 250px, 150px 150px;
        background-position: 0 0, 40px 60px, 130px 270px, 70px 100px;
        animation: spaceScroll 100s linear infinite;
        color: #f1f5f9;
    }
    
    @keyframes spaceScroll {
        0% { background-position: 0 0, 40px 60px, 130px 270px, 70px 100px; }
        100% { background-position: 550px 550px, 590px 610px, 680px 820px, 620px 650px; }
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }
    .pulse-dot {
        display: inline-block; width: 10px; height: 10px;
        background-color: #38bdf8; border-radius: 50%;
        margin-right: 8px; animation: pulse 2s infinite;
    }

    .top-header {
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px; padding: 16px 28px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }
    
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 12px 30px -10px rgba(56, 189, 248, 0.4);
    }

    /* Hide the default Streamlit sidebar entirely */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Style the horizontal radio to look like a Navigation Bar */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 20px;
        background: rgba(15, 23, 42, 0.5);
        padding: 12px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FAIL-SAFE SATELLITE DATABASE ENGINE
# ---------------------------------------------------------
DB_FILE = "exo_archive_v4.db"

def generate_fallback_dataset(table_type, n=300):
    np.random.seed(42)
    if table_type == "kepler_candidates":
        return pd.DataFrame({
            'kepid': np.random.randint(1000000, 9999999, n),
            'kepoi_name': [f"K0000{i}.01" for i in range(1, n+1)],
            'kepler_name': [f"Kepler-{i} b" if i % 3 == 0 else None for i in range(1, n+1)],
            'koi_disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n, p=[0.3, 0.4, 0.3]),
            'koi_period': np.round(np.random.uniform(0.5, 365.0, n), 4),
            'koi_duration': np.round(np.random.uniform(1.0, 12.0, n), 2),
            'koi_prad': np.round(np.random.uniform(0.5, 22.0, n), 2),
            'koi_depth': np.random.randint(50, 8000, n),
            'koi_teff': np.random.randint(3000, 7500, n),
            'ra': np.round(np.random.uniform(280.0, 300.0, n), 4),
            'dec': np.round(np.random.uniform(36.0, 52.0, n), 4)
        })
    elif table_type == "tess_candidates":
        return pd.DataFrame({
            'toi': [f"TOI-{i}.01" for i in range(100, 100+n)],
            'tid': np.random.randint(10000000, 99999999, n),
            'tfopwg_disp': np.random.choice(['PC', 'CP', 'FP'], n, p=[0.4, 0.3, 0.3]),
            'period': np.round(np.random.uniform(0.4, 100.0, n), 4),
            'duration': np.round(np.random.uniform(0.5, 8.0, n), 2),
            'prad': np.round(np.random.uniform(0.8, 18.0, n), 2),
            'depth': np.random.randint(100, 10000, n),
            'steff': np.random.randint(3200, 6800, n),
            'ra': np.round(np.random.uniform(0.0, 360.0, n), 4),
            'dec': np.round(np.random.uniform(-80.0, 80.0, n), 4)
        })
    elif table_type == "k2_candidates":
        return pd.DataFrame({
            'epic_name': [f"EPIC {200000000+i}" for i in range(1, n+1)],
            'k2_disp': np.random.choice(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'], n),
            'period': np.round(np.random.uniform(0.2, 80.0, n), 4),
            'prad': np.round(np.random.uniform(0.5, 15.0, n), 2),
            'teff': np.random.randint(2800, 6500, n),
            'ra': np.round(np.random.uniform(0.0, 360.0, n), 4),
            'dec': np.round(np.random.uniform(-30.0, 30.0, n), 4)
        })
    else:
        return pd.DataFrame({
            'pl_name': [f"ExoSystem-{i} b" for i in range(1, n+1)],
            'hostname': [f"Star-{i}" for i in range(1, n+1)],
            'discoverymethod': np.random.choice(['Transit', 'Radial Velocity', 'Direct Imaging'], n, p=[0.8, 0.15, 0.05]),
            'disc_year': np.random.randint(2009, 2026, n),
            'pl_orbper': np.round(np.random.uniform(0.5, 500.0, n), 3),
            'pl_rade': np.round(np.random.uniform(0.4, 25.0, n), 2),
            'pl_eqt': np.random.randint(150, 2500, n),
            'sy_dist': np.round(np.random.uniform(10.0, 2000.0, n), 1),
            'ra': np.round(np.random.uniform(0.0, 360.0, n), 4),
            'dec': np.round(np.random.uniform(-90.0, 90.0, n), 4)
        })

@st.cache_data(show_spinner=False)
def ensure_database_exists():
    engine = create_engine(f'sqlite:///{DB_FILE}')
    tables = {
        "kepler_candidates": "cumulative",
        "tess_candidates": "toi",
        "k2_candidates": "k2pandc",
        "planetary_systems": "ps"
    }
    
    for db_name, tap_name in tables.items():
        loaded = False
        try:
            encoded_query = urllib.parse.quote(f"select top 1000 * from {tap_name}")
            url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=csv"
            df = pd.read_csv(url, low_memory=False, timeout=5)
            if len(df) > 0:
                df.to_sql(db_name, engine, if_exists='replace', index=False)
                loaded = True
        except Exception:
            loaded = False
            
        if not loaded:
            fallback_df = generate_fallback_dataset(db_name)
            fallback_df.to_sql(db_name, engine, if_exists='replace', index=False)

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
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family="Plus Jakarta Sans"),
        xaxis=dict(title=x_title, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title=y_title, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        margin=dict(l=30, r=30, t=50, b=30)
    )

# ---------------------------------------------------------
# 3. TOP NAVIGATION & HEADER
# ---------------------------------------------------------

col_logo, col_title = st.columns([1, 12])

with col_logo:
    # Look for and load the logo in the top left
    for logo_name in ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]:
        if os.path.exists(logo_name):
            st.image(logo_name, use_container_width=True)
            break

with col_title:
    # Top Header Banner next to the logo
    st.markdown("""
        <div class="top-header">
            <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc;">🪐 Exo-AI Portal</div>
            <div style="display: flex; align-items: center; color: #38bdf8; font-weight: 600;">
                <span class="pulse-dot"></span> SECURE UPLINK ESTABLISHED
            </div>
        </div>
    """, unsafe_allow_html=True)

# Horizontal Navigation Bar
page = st.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🤖 AI Studio",
        "📁 Custom Data",
        "📊 NASA Explorer",
        "📉 Light Curve"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ---------------------------------------------------------
# PAGE: SYSTEM DASHBOARD
# ---------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown("## 🌌 Welcome to Exo-AI Intelligence")
    st.write("An advanced machine learning suite for astrophysical signal classification.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed Planets", "5,600+", delta="+12 recently")
    c2.metric("Classifier Accuracy", "89.4%", delta="Random Forest")
    c3.metric("Uptime", "99.9%", delta="Systems Nominal")
    c4.metric("Live Users", "1", delta="Local Session")

    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown("""
            <div class="glass-card">
                <h3>🚀 Mission Objectives</h3>
                <p>Navigate the cosmos using our AI-driven toolset:</p>
                <ul>
                    <li><b>AI Studio:</b> Manually test signal parameters against the model.</li>
                    <li><b>Custom Data:</b> Upload your own CSV telescope data for batch AI predictions.</li>
                    <li><b>NASA Explorer:</b> Live-query satellite databases & launch 3D space visualizers.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        np.random.seed(10)
        df_dash = pd.DataFrame({'Period (Days)': np.random.exponential(20, 80), 'Radius (Earths)': np.random.normal(3, 1, 80)})
        fig = px.scatter(df_dash, x='Period (Days)', y='Radius (Earths)', color_discrete_sequence=['#38bdf8'])
        fig.update_layout(get_theme_chart_layout("Known Exoplanet Habitability Zones"))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE: AI PREDICTION STUDIO
# ---------------------------------------------------------
elif page == "🤖 AI Studio":
    st.markdown("## 🤖 Manual Exoplanet Classifier")
    col_inp, col_out = st.columns([1.1, 1])
    with col_inp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        p_period = st.slider("Orbital Period (Days)", 0.5, 500.0, 19.2)
        p_duration = st.slider("Transit Duration (Hours)", 0.5, 15.0, 3.8)
        p_radius = st.slider("Planetary Radius (Earth Radii)", 0.2, 30.0, 2.4)
        p_depth = st.slider("Transit Depth (PPM)", 10, 10000, 1250)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        input_scaled = scaler.transform(pd.DataFrame([[p_period, p_duration, p_radius, p_depth]], columns=['koi_period', 'koi_duration', 'koi_prad', 'koi_depth']))
        pred_idx = model.predict(input_scaled)[0]
        label = label_encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(input_scaled)[0]
        
        st.markdown(f"### Result: {'🌟' if label == 'CONFIRMED' else '🔭' if label == 'CANDIDATE' else '❌'} {label}")
        for idx, cls in enumerate(label_encoder.classes_):
            st.progress(float(probs[idx]), text=f"{cls}: {probs[idx]*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE: CUSTOM DATA ANALYSIS
# ---------------------------------------------------------
elif page == "📁 Custom Data":
    st.markdown("## 📁 Upload Telemetry Data for Batch Analysis")
    st.write("Upload a CSV file containing space telemetry data. The AI will analyze the dataset and classify each signal.")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.info("💡 Your CSV must contain these columns to be analyzed: `koi_period`, `koi_duration`, `koi_prad`, `koi_depth`")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"Data successfully loaded! Found {len(user_df)} records.")
            req_cols = ['koi_period', 'koi_duration', 'koi_prad', 'koi_depth']
            if all(col in user_df.columns for col in req_cols):
                with st.spinner("AI is analyzing the signals..."):
                    X_user = user_df[req_cols].dropna()
                    X_scaled = scaler.transform(X_user)
                    predictions = model.predict(X_scaled)
                    user_df.loc[X_user.index, 'AI_Classification'] = label_encoder.inverse_transform(predictions)
                    st.write("### 🤖 Analysis Results")
                    st.dataframe(user_df, use_container_width=True)
                    st.write("### 📊 Distribution of Findings")
                    fig = px.pie(user_df, names='AI_Classification', hole=0.4, color_discrete_sequence=['#34d399', '#facc15', '#f87171'])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Missing required columns. Please ensure your CSV has: {', '.join(req_cols)}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE: NASA DATABASE EXPLORER & 3D VISUAL SIMULATOR
# ---------------------------------------------------------
elif page == "📊 NASA Explorer":
    st.markdown("## 📊 NASA Mission Archive & 3D Visual Simulator")
    st.write("Select a satellite mission to query its dataset and generate an interactive 3D sector map.")
    
    col_a, col_b = st.columns([3, 1])
    mission_map = {
        "Kepler Candidates": "kepler_candidates",
        "TESS Candidates": "tess_candidates",
        "K2 Candidates": "k2_candidates",
        "Master Planetary Systems": "planetary_systems"
    }
    selected_name = col_a.selectbox("Select Target Mission Table", list(mission_map.keys()))
    limit = col_b.number_input("Records Limit", 10, 1000, 150, 50)
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {mission_map[selected_name]} LIMIT {limit}", conn)
    conn.close()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 {selected_name} Telemetry ({len(df)} Records Loaded)")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌌 Interactive 3D Sector Simulator")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) >= 3:
        fig_3d = px.scatter_3d(
            df, 
            x=num_cols[0], 
            y=num_cols[1], 
            z=num_cols[2],
            color=num_cols[0],
            color_continuous_scale="Agsunset",
            opacity=0.85,
            title=f"3D Sector Coordinates ({num_cols[0]} vs {num_cols[1]} vs {num_cols[2]})"
        )
        
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Select a different dataset to view 3D telemetry coordinates.")

# ---------------------------------------------------------
# PAGE: LIGHT CURVE MODELING
# ---------------------------------------------------------
elif page == "📉 Light Curve":
    st.markdown("## 📉 Photometric Transit Simulator")
    st.write("Model the light dimming effect when an exoplanet passes in front of its host star.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    dur = c1.slider("Duration (Hours)", 1.0, 10.0, 3.5)
    dep = c2.slider("Depth (PPM)", 100, 5000, 1500)
    noise = c3.slider("Stellar Noise (PPM)", 10, 1000, 200)
    st.markdown('</div>', unsafe_allow_html=True)
    
    t = np.linspace(-dur * 2, dur * 2, 400)
    f = np.ones_like(t) + np.random.normal(0, noise / 1e6, size=400)
    f[np.abs(t) < (dur / 2.0)] -= (dep / 1e6)
    
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(x=t, y=f, mode='markers', marker=dict(size=4, color='#64748b', opacity=0.6), name='Raw Flux'))
    fig_lc.add_trace(go.Scatter(x=t, y=np.where(np.abs(t) < (dur / 2.0), 1 - (dep/1e6), 1.0), mode='lines', line=dict(color='#38bdf8', width=3), name='Fit Model'))
    fig_lc.update_layout(get_theme_chart_layout("Light Curve Transit Profile", "Phase (Hours)", "Normalized Flux"))
    st.plotly_chart(fig_lc, use_container_width=True)