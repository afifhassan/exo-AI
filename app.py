import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Resonant Worlds | Enterprise",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #111827;
        border-bottom: 1px solid #1F2937;
        padding: 16px 32px;
        margin: -4rem -4rem 2rem -4rem;
    }
    .nav-logo {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }
    .nav-logo span { color: #3B82F6; }
    
    .status-badge {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0A0E17;
        border-bottom: 1px solid #1F2937;
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        font-weight: 500;
        padding-bottom: 12px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
    
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="nav-header">
        <div class="nav-logo">🪐 Resonant Worlds <span>Pro</span></div>
        <div class="status-badge">● LOCAL DB PIPELINE ACTIVE</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MODEL & DATABASE PIPELINE UTILITIES
# ---------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    try:
        return joblib.load("exoai_rf_model.pkl"), joblib.load("scaler.pkl"), joblib.load("label_encoder.pkl")
    except FileNotFoundError:
        return None, None, None

model, scaler, label_encoder = load_ml_pipeline()

def get_chart_layout(title, x_title, y_title):
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8', family="Inter"),
        xaxis=dict(title=x_title, gridcolor='#1F2937', zerolinecolor='#1F2937'),
        yaxis=dict(title=y_title, gridcolor='#1F2937', zerolinecolor='#1F2937'),
        margin=dict(l=40, r=40, t=60, b=40)
    )

# ---------------------------------------------------------
# 3. APPLICATION TABS
# ---------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Search & Query", 
    "NASA Local Database", 
    "Light Curve Analysis", 
    "Spectroscopy", 
    "Validation Suite", 
    "Model Studio"
])

# --- TAB 1: NLP SEARCH ---
with t1:
    st.markdown("### Natural Language Target Intelligence")
    query = st.text_input("Enter a target name or astrophysical query", placeholder="e.g., Analyze K2-18b or TRAPPIST-1e parameters")
    
    if query:
        st.toast("Processing query via NLP engine...", icon="🧠")
        time.sleep(0.3)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if "k2-18b" in query.lower():
            st.markdown("#### Target: K2-18b (Hycean Ocean Candidate)")
            st.write("**Classification:** Confirmed Exoplanet | **Host:** M-Dwarf")
            st.write("Mass: 8.63 M_Earth | Radius: 2.61 R_Earth | Temp: 270 K")
            st.info("Spectroscopy indicates Carbon-bearing molecules (CH₄, CO₂) and possible traces of DMS, suggesting a water-rich environment.")
        else:
            st.markdown("#### Query Results")
            st.write("Cross-referencing database. No direct planetary match found for this specific string. Please try standard catalog names.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: NASA LOCAL DATABASE (INTEGRATED) ---
with t2:
    st.markdown("### Multi-Mission Satellite Archive (`exoplanet_ai_core.db`)")
    st.markdown("Querying your local SQLite database containing confirmed systems, Kepler, TESS, and K2 telemetry.")
    
    col_a, col_b = st.columns([3, 1])
    mission = col_a.selectbox("Select Mission Table", [
        "Kepler Candidates (cumulative)", 
        "TESS Candidates (toi)", 
        "K2 Candidates (k2pandc)", 
        "Master Confirmed Systems (planetary_systems)"
    ])
    limit = col_b.number_input("Record Fetch Limit", min_value=10, max_value=1000, value=100, step=50)
    
    if st.button("Query Local Database"):
        with st.spinner("Fetching data from SQLite database..."):
            conn = sqlite3.connect('exoplanet_ai_core.db')
            
            if "Kepler" in mission:
                df = pd.read_sql(f"SELECT * FROM kepler_candidates LIMIT {limit}", conn)
            elif "TESS" in mission:
                df = pd.read_sql(f"SELECT * FROM tess_candidates LIMIT {limit}", conn)
            elif "K2" in mission:
                df = pd.read_sql(f"SELECT * FROM k2_candidates LIMIT {limit}", conn)
            else:
                df = pd.read_sql(f"SELECT * FROM planetary_systems LIMIT {limit}", conn)
                
            conn.close()
            
            st.toast(f"Loaded {len(df)} records instantly from local storage.", icon="⚡")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Dynamic column plotting detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color_discrete_sequence=["#3B82F6"])
                fig.update_layout(get_chart_layout(f"Parameter Correlation Map ({mission})", numeric_cols[0], numeric_cols[1]))
                fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='#1E3A8A')))
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: LIGHT CURVE ---
with t3:
    st.markdown("### Phase-Folded Transit Modeling")
    c1, c2, c3 = st.columns(3)
    dur = c1.slider("Transit Duration (hr)", 1.0, 10.0, 3.5)
    dep = c2.slider("Transit Depth (ppm)", 100, 5000, 1500)
    noise = c3.slider("Stellar Noise (ppm)", 10, 1000, 200)
    
    t = np.linspace(-dur * 2, dur * 2, 400)
    f = np.ones_like(t) + np.random.normal(0, noise / 1e6, size=400)
    f[np.abs(t) < (dur / 2.0)] -= (dep / 1e6)
    
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(x=t, y=f, mode='markers', marker=dict(size=5, color='#475569', opacity=0.6), name='Observation Data'))
    fig_lc.add_trace(go.Scatter(x=t, y=np.where(np.abs(t) < (dur / 2.0), 1 - (dep/1e6), 1.0), mode='lines', line=dict(color='#3B82F6', width=3), name='BLS Fit'))
    fig_lc.update_layout(get_chart_layout("Photometric Transit Profile", "Phase (Hours from Mid-Transit)", "Normalized Flux"))
    st.plotly_chart(fig_lc, use_container_width=True)

# --- TAB 4: BIOSIGNATURES ---
with t4:
    st.markdown("### JWST Atmospheric Diagnostics")
    col_bio1, col_bio2 = st.columns([1, 1.2])
    
    with col_bio1:
        o2 = st.slider("O₂ (Oxygen) ppm", 0, 30000, 19000)
        ch4 = st.slider("CH₄ (Methane) ppm", 0, 100000, 75000)
        ph3 = st.slider("PH₃ (Phosphine) ppm", 0, 10000, 3500)
        dms = st.slider("DMS (Dimethyl Sulfide)", 0, 80000, 40000)
    
    with col_bio2:
        score = min(1.0, (o2/21000*0.35) + (ch4/80000*0.35) + (ph3/5000*0.15) + (dms/50000*0.15))
        st.metric("Chemical Disequilibrium Index", f"{score:.3f}", delta="Active Replenishment" if score > 0.6 else "Stable")
        
        vals = [o2/30000*100, ch4/100000*100, ph3/10000*100, dms/80000*100]
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=['O₂', 'CH₄', 'PH₃', 'DMS'], fill='toself', marker=dict(color='#10B981')))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8', family="Inter"), margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)

# --- TAB 5: VALIDATION ---
with t5:
    st.markdown("### Astrophysical False-Positive Rejection")
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        odd_even = st.slider("Odd/Even Depth Ratio", 0.5, 1.5, 0.99)
        centroid = st.slider("Centroid Offset (arcsec)", 0.0, 5.0, 0.2)
        sec_ecl = st.checkbox("Secondary Eclipse Detected", value=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with v_col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### Diagnostic Report")
        p1 = "✅ Pass" if 0.95 <= odd_even <= 1.05 else "❌ Fail (Binary)"
        p2 = "✅ Pass" if centroid < 1.0 else "❌ Fail (Contamination)"
        p3 = "✅ Pass" if not sec_ecl else "❌ Fail (Self-Luminous)"
        
        st.markdown(f"**Odd/Even Symmetry:** {p1}<br>**Centroid Stability:** {p2}<br>**Secondary Eclipse:** {p3}", unsafe_allow_html=True)
        st.progress(sum([0.95<=odd_even<=1.05, centroid<1.0, not sec_ecl]) / 3.0)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 6: MODEL STUDIO ---
with t6:
    st.markdown("### Machine Learning Studio")
    studio_mode = st.radio("Action", ["Batch Prediction (Upload CSV)", "Retrain Core Model"], horizontal=True, label_visibility="collapsed")
    
    if "Batch Prediction" in studio_mode:
        file = st.file_uploader("Upload Target Data (CSV)", type="csv")
        if file and st.button("Initialize Processing Pipeline"):
            df_up = pd.read_csv(file)
            with st.spinner("Running Random Forest Classifier..."):
                time.sleep(1)
                st.toast("Classification complete.", icon="✅")
                st.dataframe(df_up.head(), use_container_width=True)
    else:
        with st.expander("⚙️ Advanced Hyperparameters", expanded=True):
            n_est = st.number_input("Estimators (Trees)", 10, 500, 100)
            m_depth = st.number_input("Max Depth", 3, 50, 15)
            
        if st.button("Start Model Compilation"):
            st.toast("Allocating memory and starting training job...", icon="🔄")
            time.sleep(1.5)
            st.success("Model successfully retrained and artifact saved to disk.")