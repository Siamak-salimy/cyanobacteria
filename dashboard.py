# -*- coding: utf-8 -*-
"""
Professional Cyanobacteria Prediction Dashboard (Static / stlite compatible)
Run locally:  streamlit run dashboard.py
Run static:   python pack_app.py  →  docs/index.html
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config import DATA_FILE, SHEET_NAME, COL_MAP
from predict import predict_cyanobacteria, load_artifact

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Cyanobacteria Prediction Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Load CSS from external file (avoids JS escaping issues in stlite)
# ============================================================
_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

# ============================================================
# Hero header
# ============================================================
st.markdown('<div class="hero-title">Cyanobacteria Prediction Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Nonlinear ML ensemble · RandomForest · GPR · SVR · LOOCV-validated</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-line"></div>', unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource(show_spinner=False)
def _get_model():
    return load_artifact()

try:
    artifact = _get_model()
    MODEL_OK = True
except FileNotFoundError as e:
    MODEL_OK = False
    st.error(f"Model not found. Run `python train.py` first.\n\n{e}")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### Model Info")
    if MODEL_OK:
        st.markdown(f"""
        <div class="glass" style="font-size:0.92rem;line-height:1.9;">
            <b>Best model:</b> {artifact['best_name']}<br>
            <b>RMSE (real):</b> {artifact['rmse_real']:,.0f}<br>
            <b>Features:</b> {len(artifact['features'])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Model Comparison")
        if "results_df" in artifact:
            st.dataframe(
                artifact["results_df"].style.format({
                    "rmse_log": "{:.3f}", "mae": "{:.2f}",
                    "r2": "{:.3f}", "rmse_real": "{:,.0f}",
                }),
                use_container_width=True, height=230,
            )

    st.markdown("---")
    st.markdown("### Load Real Sample")
    if MODEL_OK:
        try:
            _df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).rename(columns=COL_MAP)
            sample_idx = st.selectbox(
                "Select a sample:",
                options=range(len(_df)),
                format_func=lambda i: f"Row {_df.iloc[i]['id']} - {_df.iloc[i]['cyanobacteria']:,.0f}",
                index=4,
            )
            if st.button("Load Sample Values"):
                row = _df.iloc[sample_idx]
                st.session_state["ec"]        = float(row["EC"])
                st.session_state["ph"]        = float(row["pH"])
                st.session_state["turbidity"] = float(row["turbidity"])
                st.session_state["sulfate"]   = float(row["sulfate"])
                st.session_state["nitrite"]   = float(row["nitrite"])
                st.session_state["nitrate"]   = float(row["nitrate"])
                st.session_state["phosphate"] = float(row["phosphate"])
                st.session_state["temp"]      = float(row["temperature"])
                st.session_state["true_val"]  = float(row["cyanobacteria"])
                st.rerun()
        except Exception as e:
            st.warning(f"Could not load sample: {e}")

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · scikit-learn")

# ============================================================
# Tabs
# ============================================================
tab_pred, tab_data = st.tabs(["Prediction", "Data"])

# ------------------------------------------------------------
# Tab 1: Prediction
# ------------------------------------------------------------
with tab_pred:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Environmental Parameters")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ec = st.number_input("Electrical Conductivity (uS/cm)", 0.0, 5000.0,
                             st.session_state.get("ec", 830.0), step=10.0)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 200.0,
                                    st.session_state.get("turbidity", 24.8), step=0.1)
    with c2:
        ph = st.number_input("pH", 4.0, 12.0,
                             st.session_state.get("ph", 8.36), step=0.01)
        sulfate = st.number_input("Sulfate (mg/L)", 0.0, 1000.0,
                                  st.session_state.get("sulfate", 123.19), step=0.1)
    with c3:
        nitrite = st.number_input("Nitrite (mg/L)", 0.0, 5.0,
                                  st.session_state.get("nitrite", 0.07), step=0.01)
        nitrate = st.number_input("Nitrate (mg/L)", 0.0, 50.0,
                                  st.session_state.get("nitrate", 7.68), step=0.1)
    with c4:
        phosphate = st.number_input("Phosphate (mg/L)", 0.0, 1.0,
                                    st.session_state.get("phosphate", 0.05), step=0.01)
        temperature = st.number_input("Temperature (C)", 0.0, 45.0,
                                      st.session_state.get("temp", 27.0), step=0.5)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    do_predict = st.button("Predict Cyanobacteria Count", use_container_width=True)

    if do_predict and MODEL_OK:
        with st.spinner("Analyzing..."):
            res = predict_cyanobacteria(
                ec, ph, turbidity, sulfate,
                nitrite, nitrate, phosphate, temperature,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # رنگ بر اساس سطح هشدار (کلیدها با کاراکترهای ساده)
        alert_key = res["alert"]
        if "کم" in alert_key:
            color, bg = "#22d3ee", "rgba(34,211,238,0.12)"
        elif "متوسط" in alert_key:
            color, bg = "#fbbf24", "rgba(251,191,36,0.14)"
        elif "بالا" in alert_key:
            color, bg = "#fb923c", "rgba(251,146,60,0.16)"
        else:
            color, bg = "#f87171", "rgba(248,113,113,0.18)"

        st.markdown(f"""
        <div class="result-card" style="background: {bg}; border: 1px solid {color}55;">
            <div class="result-value" style="color:{color};">{res['prediction']:,.0f}</div>
            <div class="result-label">Predicted cyanobacteria count (cells/mL)</div>
            <div style="margin-top:16px; color:#cbd5e1; font-size:0.95rem;">
                Confidence interval: <b style="color:{color}">{res['interval'][0]:,.0f}</b>
                - <b style="color:{color}">{res['interval'][1]:,.0f}</b>
            </div>
            <div class="alert-badge" style="background:{bg}; color:{color}; border:1px solid {color};">
                {res['alert']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "true_val" in st.session_state:
            true_val = st.session_state["true_val"]
            err = abs(true_val - res["prediction"])
            pct = err / true_val * 100 if true_val else 0
            st.info(
                f"Ground truth for this sample: **{true_val:,.0f}** - "
                f"Absolute error: **{err:,.0f}** ({pct:.1f}%)"
            )

        max_val = max(50000, res["prediction"] * 1.3)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["prediction"],
            number={"font": {"size": 36, "color": color, "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#64748b"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 1000],        "color": "rgba(34,211,238,0.18)"},
                    {"range": [1000, 10000],    "color": "rgba(251,191,36,0.18)"},
                    {"range": [10000, 30000],   "color": "rgba(251,146,60,0.18)"},
                    {"range": [30000, max_val], "color": "rgba(248,113,113,0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": res["prediction"],
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif", "color": "#e6edf3"},
            height=340, margin=dict(l=30, r=30, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------------------------------------
# Tab 2: Data
# ------------------------------------------------------------
with tab_data:
    st.markdown("#### Training Data")
    try:
        df_show = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).rename(columns=COL_MAP)
        st.dataframe(df_show, use_container_width=True, height=380)

        st.markdown("#### Cyanobacteria Distribution (log scale)")
        fig_hist = px.histogram(
            df_show, x="cyanobacteria", nbins=15,
            color_discrete_sequence=["#38bdf8"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font={"family": "Inter, sans-serif", "color": "#e6edf3"},
            xaxis_title="Cyanobacteria count",
            yaxis_title="Frequency",
            height=360,
        )
        fig_hist.update_xaxes(type="log", gridcolor="rgba(255,255,255,0.05)")
        fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Correlation Matrix")
        num_cols = ["EC", "pH", "turbidity", "sulfate",
                    "nitrite", "nitrate", "phosphate", "temperature", "cyanobacteria"]
        corr = df_show[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif", "color": "#e6edf3"},
            height=540,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load data: {e}")