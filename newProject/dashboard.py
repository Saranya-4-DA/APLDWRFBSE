"""
dashboard.py  —  Water Leak Detection Streamlit Dashboard
==========================================================
Run:  streamlit run dashboard.py
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Water Leak Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp { background: #0f1117; color: #e8eaf0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b26 !important;
    border-right: 1px solid #2a2f3d;
}
section[data-testid="stSidebar"] * { color: #c9cdd8 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 16px 20px !important;
}
div[data-testid="metric-container"] label { color: #7b8299 !important; font-size: 0.75rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }

/* Status banner */
.status-ok {
    background: linear-gradient(135deg, #0d2b1f 0%, #0a3326 100%);
    border: 1px solid #1a6641;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.status-leak {
    background: linear-gradient(135deg, #2b0d0d 0%, #330a0a 100%);
    border: 1px solid #6b1a1a;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.0); }
    50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0.12); }
}
.status-ok   h2 { color: #22c55e; margin: 0 0 4px; font-size: 1.4rem; }
.status-leak h2 { color: #ef4444; margin: 0 0 4px; font-size: 1.4rem; }
.status-ok   p  { color: #86efac; margin: 0; font-size: 0.88rem; }
.status-leak p  { color: #fca5a5; margin: 0; font-size: 0.88rem; }

/* Feature table */
.feat-table { width:100%; border-collapse: collapse; font-size: 0.82rem; }
.feat-table th {
    background: #1e2436;
    color: #7b8299;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 8px 12px;
    border-bottom: 1px solid #2a2f3d;
    text-align: left;
}
.feat-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #1e2436;
    color: #c9cdd8;
}
.feat-table tr:hover td { background: #1e2436; }

/* Slider label */
.vib-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #7b8299;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* Section headers */
.sec-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #4a90d9;
    margin: 0 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2f3d;
}

/* Zone pill */
.zone-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.zone-normal { background:#0d2b1f; color:#22c55e; border:1px solid #1a6641; }
.zone-minor  { background:#2b1d0d; color:#f59e0b; border:1px solid #6b461a; }
.zone-major  { background:#2b0d0d; color:#ef4444; border:1px solid #6b1a1a; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
HISTORY_LEN = 80

def _init_state():
    defaults = {
        "history_time"     : deque(maxlen=HISTORY_LEN),
        "history_rms"      : deque(maxlen=HISTORY_LEN),
        "history_leak"     : deque(maxlen=HISTORY_LEN),
        "history_size"     : deque(maxlen=HISTORY_LEN),
        "tick"             : 0,
        "auto_run"         : False,
        "last_result"      : None,
        "models_loaded"    : False,
        "model_error"      : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def get_models():

    import simulate_live

    st.write("DEBUG: simulate_live imported")

    print("DEBUG TERMINAL")
    print(simulate_live.__file__)

    return simulate_live.load_models()


# ── Inference wrapper ─────────────────────────────────────────────────────────
def run_inference(vibration_level: float) -> dict:
    from simulate_live import predict_from_level
    return predict_from_level(vibration_level, seed=int(time.time() * 1000) % 99999)


# ── Plotly helpers ────────────────────────────────────────────────────────────
PLOT_BG   = "#0f1117"
GRID_COL  = "#1e2436"
TEXT_COL  = "#7b8299"
GREEN_COL = "#22c55e"
RED_COL   = "#ef4444"
AMBER_COL = "#f59e0b"
BLUE_COL  = "#4a90d9"

def _base_layout(height=220):
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, size=11),
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(size=9)),
    )


def waveform_chart(signal, is_leak):
    colour = RED_COL if is_leak else GREEN_COL
    x = list(range(len(signal)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=signal, mode="lines",
        line=dict(color=colour, width=1.2),
        fill="tozeroy",
        fillcolor=colour.replace(")", ",0.08)").replace("rgb", "rgba"),
        name="Signal"
    ))
    fig.update_layout(
        **_base_layout(200),
        title=dict(text="Live Waveform", font=dict(size=12, color="#c9cdd8"), x=0.01),
        showlegend=False,
    )
    return fig


def history_chart(times, rms_vals, leak_flags):
    colours = [RED_COL if l else GREEN_COL for l in leak_flags]
    fig = go.Figure()

    # RMS line
    fig.add_trace(go.Scatter(
        x=list(times), y=list(rms_vals),
        mode="lines", line=dict(color=BLUE_COL, width=2),
        name="RMS", yaxis="y"
    ))

    # Leak markers
    leak_x = [t for t, l in zip(times, leak_flags) if l]
    leak_y = [r for r, l in zip(rms_vals, leak_flags) if l]
    if leak_x:
        fig.add_trace(go.Scatter(
            x=leak_x, y=leak_y,
            mode="markers",
            marker=dict(color=RED_COL, size=7, symbol="circle"),
            name="Leak", yaxis="y"
        ))

    fig.update_layout(
        **_base_layout(200),
        title=dict(text="RMS History", font=dict(size=12, color="#c9cdd8"), x=0.01),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )
    return fig


def gauge_chart(prob):
    colour = RED_COL if prob > 0.5 else GREEN_COL
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(size=28, color=colour)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9), tickcolor=TEXT_COL),
            bar=dict(color=colour, thickness=0.6),
            bgcolor=GRID_COL,
            bordercolor=GRID_COL,
            steps=[
                dict(range=[0, 50],  color="#0d2b1f"),
                dict(range=[50, 100], color="#2b0d0d"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=50),
        ),
        title=dict(text="Leak Probability", font=dict(size=11, color=TEXT_COL)),
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL),
    )
    return fig


def freq_bar(features):
    names = [c.replace("_", " ").title() for c in features.keys()]
    vals  = list(features.values())
    norm  = [(v - min(vals)) / (max(vals) - min(vals) + 1e-9) for v in vals]
    colours = [f"rgba(74,144,217,{0.35 + 0.65 * n})" for n in norm]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colours,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=8, color=TEXT_COL),
    ))
    fig.update_layout(
        **_base_layout(220),
        title=dict(text="Extracted Features", font=dict(size=12, color="#c9cdd8"), x=0.01),
        xaxis_tickangle=-30,
        bargap=0.3,
    )
    fig.update_yaxes(showticklabels=False)
    return fig


# ── Zone helper ───────────────────────────────────────────────────────────────
def zone_info(vl):
    if vl < 0.35:
        return "Normal", "zone-normal", "Pipe operating normally — no leak expected"
    elif vl < 0.65:
        return "0.2 mm Leak Zone", "zone-minor", "Vibration level suggests minor leak activity"
    else:
        return "2 mm Leak Zone", "zone-major", "High vibration — significant leak expected"


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💧 Leak Detector")
    st.markdown("<div style='height:4px;background:linear-gradient(90deg,#4a90d9,#22c55e);border-radius:2px;margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Model status ──────────────────────────────────────────────────────────
    try:
        get_models()
        st.markdown("**Model Status**")
        st.success("Models loaded", icon="✅")
        st.session_state["models_loaded"] = True
    except Exception as e:
        st.error(f"Models not found: {e}")
        st.warning("Run `train_cnn.py` and `train_rf.py` first.")
        st.session_state["model_error"] = str(e)
        st.stop()

    st.divider()

    # ── Vibration regulator ───────────────────────────────────────────────────
    st.markdown('<div class="vib-label">Vibration Intensity Regulator</div>', unsafe_allow_html=True)

    vib_level = st.slider(
        label="vibration_level_slider",
        min_value=0.0, max_value=1.0, value=0.05,
        step=0.01, label_visibility="collapsed"
    )

    # Zone pill
    zname, zcls, zdesc = zone_info(vib_level)
    st.markdown(
        f'<div style="margin:8px 0 4px">'
        f'<span class="zone-pill {zcls}">{zname}</span>'
        f'</div>'
        f'<div style="font-size:0.75rem;color:#7b8299;margin-bottom:12px">{zdesc}</div>',
        unsafe_allow_html=True
    )

    # Tick marks
    st.markdown("""
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;
                color:#4a4f63;margin-top:-4px;margin-bottom:16px">
        <span>0.0<br>Normal</span>
        <span style="text-align:center">0.35<br>0.2mm</span>
        <span style="text-align:center">0.65<br>2mm</span>
        <span style="text-align:right">1.0<br>Max</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("**Controls**")
    col_a, col_b = st.columns(2)
    with col_a:
        scan_btn = st.button("▶ Scan", use_container_width=True, type="primary")
    with col_b:
        clear_btn = st.button("⟳ Clear", use_container_width=True)

    auto = st.toggle("Auto-refresh (1 s)", value=st.session_state["auto_run"])
    st.session_state["auto_run"] = auto

    st.divider()

    # ── Info ──────────────────────────────────────────────────────────────────
    st.markdown("**Models**")
    st.markdown("""
    <div style="font-size:0.78rem;color:#7b8299;line-height:1.7">
    • <b style="color:#c9cdd8">1D CNN</b> — binary leak classifier<br>
    • <b style="color:#c9cdd8">Random Forest</b> — size regressor<br>
    • Window: 256 samples @ 1000 Hz<br>
    • Threshold: 0.50
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## Water Pipe Vibration — Leak Detection Dashboard")
st.markdown("<div style='height:3px;background:linear-gradient(90deg,#4a90d9 0%,#22c55e 50%,#ef4444 100%);border-radius:2px;margin-bottom:24px'></div>", unsafe_allow_html=True)

# ── Trigger inference ─────────────────────────────────────────────────────────
if clear_btn:
    st.session_state["history_time"].clear()
    st.session_state["history_rms"].clear()
    st.session_state["history_leak"].clear()
    st.session_state["history_size"].clear()
    st.session_state["last_result"] = None
    st.session_state["tick"] = 0

if scan_btn or st.session_state["auto_run"]:
    result = run_inference(vib_level)
    st.session_state["last_result"] = result
    st.session_state["tick"] += 1
    t = st.session_state["tick"]
    st.session_state["history_time"].append(t)
    st.session_state["history_rms"].append(result["features"]["rms"])
    st.session_state["history_leak"].append(result["is_leak"])
    st.session_state["history_size"].append(result["leak_size_mm"] or 0.0)


result = st.session_state.get("last_result")

# ── Status banner ──────────────────────────────────────────────────────────────
if result is None:
    st.info("Move the **Vibration Intensity Regulator** in the sidebar and press **▶ Scan** to start.")
else:
    is_leak  = result["is_leak"]
    prob     = result["probability"]
    size_mm  = result["leak_size_mm"]
    features = result["features"]

    if not is_leak:
        st.markdown("""
        <div class="status-ok">
          <h2>✔ &nbsp;NO LEAK DETECTED</h2>
          <p>Pipe vibration is within normal operating range — system is healthy.</p>
        </div>""", unsafe_allow_html=True)
    else:
        severity = "MINOR" if size_mm < 0.5 else "MAJOR"
        st.markdown(f"""
        <div class="status-leak">
          <h2>⚠ &nbsp;LEAK DETECTED &nbsp;·&nbsp; {severity}</h2>
          <p>Estimated leak size: <strong>{size_mm:.2f} mm</strong>
             &nbsp;|&nbsp; Confidence: <strong>{prob*100:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Top metrics row ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("RMS Amplitude",
                  f"{features['rms']:.4f}",
                  delta=None)
    with m2:
        st.metric("Peak Value",
                  f"{features['peak_value']:.4f}")
    with m3:
        st.metric("Dominant Freq",
                  f"{features['dominant_frequency']:.1f} Hz")
    with m4:
        size_disp = f"{size_mm:.2f} mm" if is_leak else "—"
        st.metric("Leak Size", size_disp)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Charts row 1: waveform + gauge ────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(waveform_chart(result["signal"], is_leak),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(gauge_chart(prob),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Charts row 2: history + feature bars ──────────────────────────────────
    c3, c4 = st.columns([2, 2])
    with c3:
        if st.session_state["history_time"]:
            st.plotly_chart(
                history_chart(
                    st.session_state["history_time"],
                    st.session_state["history_rms"],
                    st.session_state["history_leak"],
                ),
                use_container_width=True, config={"displayModeBar": False}
            )
    with c4:
        st.plotly_chart(freq_bar(features),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Feature detail table ───────────────────────────────────────────────────
    with st.expander("📊 All Extracted Features", expanded=False):
        labels = {
            "mean_amplitude"    : "Mean Amplitude",
            "rms"               : "RMS",
            "peak_value"        : "Peak Value",
            "std_dev"           : "Std Deviation",
            "dominant_frequency": "Dominant Freq (Hz)",
            "spectral_entropy"  : "Spectral Entropy",
            "zero_crossing_rate": "Zero Crossing Rate",
            "kurtosis"          : "Kurtosis",
            "skewness"          : "Skewness",
        }
        rows = "".join(
            f"<tr><td>{labels[k]}</td><td style='text-align:right;font-family:monospace'>{v:.6f}</td></tr>"
            for k, v in features.items()
        )
        st.markdown(
            f"<table class='feat-table'><thead><tr><th>Feature</th><th style='text-align:right'>Value</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True
        )


# ── Auto-refresh ───────────────────────────────────────────────────────────────
if st.session_state["auto_run"]:
    time.sleep(1.0)
    st.rerun()
