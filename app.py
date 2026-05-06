"""
Phase Transformation Kinetics — Interactive Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    MATERIALS, TRAIN_MATERIALS, TEST_MATERIALS,
    train_models, ml_predict, ttt_curve, rate_curves_normalized,
    nucleation_rate, growth_rate, jmak_k, R, kB,
)
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phase Transformation Kinetics",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
AX    = "#c9d1d9"
EDGE  = "#30363d"
MUTED = "#8b949e"
C = dict(
    blue="#58a6ff", red="#f78166", green="#3fb950", yellow="#e3b341",
    purple="#d2a8ff", orange="#ffa657", teal="#79c0ff", pink="#ff7b72",
)
PAL = list(C.values())

CRYSTAL = {
    "Aluminum":"FCC","Copper":"FCC","Nickel":"FCC","Gold":"FCC","Silver":"FCC",
    "Lead":"FCC","Platinum":"FCC","Palladium":"FCC","Rhodium":"FCC",
    "Iridium":"FCC","Cobalt":"FCC","Calcium":"FCC",
    "Ferrite":"BCC","Chromium":"BCC","Vanadium":"BCC","Tungsten":"BCC",
    "Molybdenum":"BCC","Tantalum":"BCC","Niobium":"BCC","Manganese":"BCC",
    "Lithium":"BCC","Sodium":"BCC","Barium":"BCC",
    "Titanium":"HCP","Zinc":"HCP","Magnesium":"HCP","Zirconium":"HCP",
    "Hafnium":"HCP","Beryllium":"HCP","Scandium":"HCP","Osmium":"HCP",
    "Ruthenium":"HCP","Cadmium":"HCP","Rhenium":"HCP",
    "Silicon":"Diamond","Tin":"BCT","Bismuth":"Rhomb.","Antimony":"Rhomb.",
    "Indium":"BCT","Gallium":"Ortho.","Ice":"Hex.",
}
CRYSTAL_COLOR = {
    "FCC": C["blue"], "BCC": C["red"], "HCP": C["green"],
    "Diamond": C["purple"], "BCT": C["orange"], "Rhomb.": C["teal"],
    "Ortho.": C["pink"], "Hex.": C["yellow"],
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, .stApp, .stMain, [class*="block-container"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
  }}
  section[data-testid="stSidebar"] {{
    background-color: {PANEL} !important;
    border-right: 1px solid {EDGE};
  }}
  section[data-testid="stSidebar"] * {{ color: {AX} !important; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{ color: {MUTED} !important; font-size:12px !important; }}

  h1,h2,h3,h4,h5 {{ color: {AX} !important; font-family: 'Inter', sans-serif !important; }}
  p, li, label, .stMarkdown {{ color: {AX}; }}

  /* metric cards */
  div[data-testid="metric-container"] {{
    background: {PANEL}; border: 1px solid {EDGE};
    border-radius: 10px; padding: 14px 18px;
    border-top: 3px solid {C['blue']};
  }}
  div[data-testid="metric-container"] label {{
    color: {MUTED} !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.06em;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {AX} !important; font-size: 24px !important; font-weight: 700 !important;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{ font-size: 12px; }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {PANEL}; border-radius: 10px; gap: 4px; padding: 4px;
    border: 1px solid {EDGE};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED}; border-radius: 8px;
    font-size: 14px; font-weight: 500; padding: 8px 18px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {GRID} !important; color: {AX} !important;
  }}

  /* expander */
  .streamlit-expanderHeader {{ color: {AX} !important; font-size: 13px !important; }}
  .streamlit-expanderContent {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 8px; }}

  /* dataframe */
  .stDataFrame {{ background: {PANEL}; border-radius: 8px; }}

  /* divider */
  hr {{ border-color: {EDGE} !important; opacity: 0.5; }}

  /* scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {EDGE}; border-radius: 3px; }}

  footer, #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def base_layout(**kw):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=AX, size=12, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE,
                   tickfont=dict(color=AX), title_font=dict(color=AX, size=13)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE,
                   tickfont=dict(color=AX), title_font=dict(color=AX, size=13)),
        legend=dict(bgcolor=GRID, bordercolor=EDGE, borderwidth=1,
                    font=dict(color=AX, size=10)),
        margin=dict(l=60, r=24, t=52, b=52),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=EDGE, font_color=AX,
                        font_family="Inter, sans-serif"),
        title_font=dict(size=14, color=AX, family="Inter, sans-serif"),
    )
    d.update(kw)
    return d


def annot(text, x=0.02, y=0.97, color=None, size=11):
    return dict(
        text=text, xref="paper", yref="paper", x=x, y=y,
        showarrow=False, align="left",
        bgcolor=BG, bordercolor=EDGE, borderwidth=1,
        font=dict(color=color or C["yellow"], size=size),
    )


def badge(text, color):
    return (
        f'<span style="background:{color}22; color:{color}; border:1px solid {color}66;'
        f' border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600;">'
        f'{text}</span>'
    )


def metric_card(label, value, sub=None, accent=None):
    ac = accent or C["blue"]
    sub_html = f'<div style="color:{MUTED};font-size:11px;margin-top:4px;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:{PANEL};border:1px solid {EDGE};border-radius:10px;'
        f'padding:16px 20px;border-top:3px solid {ac};height:100%;">'
        f'<div style="color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.06em;">{label}</div>'
        f'<div style="color:{AX};font-size:26px;font-weight:700;margin-top:6px;">{value}</div>'
        f'{sub_html}</div>'
    )


@st.cache_resource(show_spinner=False)
def get_models(n_virtual: int, n_avrami: int):
    return train_models(n_virtual=n_virtual, n_T=250, n_avrami=n_avrami)


def ttt_masks(ttt_res):
    out = {}
    for X in (0.01, 0.50, 0.99):
        t = ttt_res[X]
        out[X] = np.isfinite(t) & (t > 0) & (t < 1e23)
    return out


def compute_ml_ttt(models, p, n_avrami):
    ml = ml_predict(models, p, n_avrami=n_avrami)
    k  = np.maximum(10.0 ** ml["log_k"], 1e-300)
    t1 = (np.log(100.0) / k) ** (1.0 / n_avrami)
    mk = np.isfinite(t1) & (t1 > 0) & (t1 < 1e23)
    return ml, t1, mk


def test_set_metrics(models, n_avrami):
    true_nT, pred_nT, true_nt, pred_nt = [], [], [], []
    for m in TEST_MATERIALS:
        pp = MATERIALS[m]
        _, _, Tn_t, tt_t = ttt_curve(pp, n_avrami=n_avrami)
        mlp = ml_predict(models, pp, n_avrami=n_avrami)
        true_nT.append(Tn_t); pred_nT.append(mlp["T_nose"])
        true_nt.append(np.log10(tt_t + 1e-30))
        pred_nt.append(np.log10(mlp["t_nose"] + 1e-30))
    return (true_nT, pred_nT,
            r2_score(true_nT, pred_nT),
            r2_score(true_nt, pred_nt),
            true_nt, pred_nt)


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:8px 0 4px;">'
        f'<span style="font-size:22px;font-weight:700;color:{AX};">⚗️ PTK Lab</span><br>'
        f'<span style="font-size:11px;color:{MUTED};">Phase Transformation Kinetics</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    all_mats = sorted(MATERIALS.keys())
    mat_name = st.selectbox(
        "Material", all_mats,
        index=all_mats.index("Zinc"),
        help="Select any of the 40 metals in the database",
    )
    p_main = MATERIALS[mat_name]

    # Material card
    cryst = CRYSTAL.get(mat_name, "—")
    cryst_col = CRYSTAL_COLOR.get(cryst, MUTED)
    split_col = C["red"] if mat_name in TEST_MATERIALS else C["blue"]
    split_lbl = "Test set" if mat_name in TEST_MATERIALS else "Train set"
    st.markdown(
        f'<div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:12px 14px;margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'{badge(cryst, cryst_col)}&nbsp;{badge(split_lbl, split_col)}'
        f'</div>'
        f'<table style="width:100%;font-size:12px;border-collapse:collapse;">'
        f'<tr><td style="color:{MUTED};padding:2px 0;">Melting pt</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["Tm"]} K&nbsp;·&nbsp;{p_main["Tm"]-273:.0f}°C</td></tr>'
        f'<tr><td style="color:{MUTED};padding:2px 0;">ΔHf</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["dHf"]/1000:.1f} kJ/mol</td></tr>'
        f'<tr><td style="color:{MUTED};padding:2px 0;">γ (S/L)</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["gamma"]:.3f} J/m²</td></tr>'
        f'<tr><td style="color:{MUTED};padding:2px 0;">Q (diff.)</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["Q"]/1000:.0f} kJ/mol</td></tr>'
        f'<tr><td style="color:{MUTED};padding:2px 0;">D₀</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["D0"]:.2e} m²/s</td></tr>'
        f'<tr><td style="color:{MUTED};padding:2px 0;">Stefan #</td>'
        f'<td style="color:{AX};text-align:right;">{p_main["dHf"]/(R*p_main["Tm"]):.2f}</td></tr>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    n_avrami = st.select_slider(
        "Avrami exponent  n",
        options=[1, 2, 3, 4], value=4,
        help="1=1D site-sat · 2=2D · 3=3D site-sat · 4=3D continuous nucleation",
    )
    n_desc = {1:"1D, site saturation",2:"2D, site saturation",
              3:"3D, site saturation",4:"3D, continuous nucleation"}
    st.caption(n_desc[n_avrami])

    ttt_levels = st.multiselect(
        "TTT iso-fraction lines",
        options=["1%", "50%", "99%"], default=["1%", "50%", "99%"],
    )
    ttt_frac_map = {"1%": 0.01, "50%": 0.50, "99%": 0.99}
    chosen_fracs = [ttt_frac_map[l] for l in ttt_levels]

    st.divider()
    n_virtual = st.select_slider(
        "Training augmentation",
        options=[0, 200, 500, 1000], value=200,
        format_func=lambda v: {
            0:    "Real only · 7.5k pts",
            200:  "Standard · 17.5k pts",
            500:  "Extended · 32.5k pts",
            1000: "Publication · 57.5k pts",
        }[v],
    )

    st.divider()
    st.caption(
        f"Database: {len(MATERIALS)} materials  "
        f"({len(TRAIN_MATERIALS)} train · {len(TEST_MATERIALS)} test)"
    )

# ─────────────────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────
with st.spinner("Training GBM surrogate models on CNT+JMAK physics data…"):
    models = get_models(n_virtual, n_avrami)

# ─────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE
# ─────────────────────────────────────────────────────────────────────────
T_rc, I_n, U_n, ov            = rate_curves_normalized(p_main)
T_ttt, ttt_res, Tn_tr, tt_tr  = ttt_curve(p_main, n_avrami=n_avrami)
masks                          = ttt_masks(ttt_res)
ml, t1_ml, mk                  = compute_ml_ttt(models, p_main, n_avrami)

T_arr   = np.linspace(0.40 * p_main["Tm"], 0.995 * p_main["Tm"], 600)
T_Ipeak = T_arr[np.argmax(nucleation_rate(T_arr, p_main))] - 273.15
T_Upeak = T_arr[np.argmax(growth_rate(T_arr, p_main))]    - 273.15

true_nT, pred_nT, r2_T, r2_t, true_nt, pred_nt = test_set_metrics(models, n_avrami)
dT_err = abs(Tn_tr - ml["T_nose"])
dt_err = abs(np.log10(tt_tr + 1e-30) - np.log10(ml["t_nose"] + 1e-30))

# ─────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────
cryst_hero = CRYSTAL.get(mat_name, "—")
cryst_hero_col = CRYSTAL_COLOR.get(cryst_hero, MUTED)
split_hero_col = C["red"] if mat_name in TEST_MATERIALS else C["blue"]
split_hero_lbl = "Test material" if mat_name in TEST_MATERIALS else "Training material"

st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a2744 0%,{BG} 60%,#1a1a2e 100%);
            border:1px solid {EDGE};border-radius:14px;padding:28px 36px;margin-bottom:24px;">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
    <div>
      <div style="font-size:13px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">
        MLL202 · Materials ML Project
      </div>
      <h1 style="color:{AX};margin:0;font-size:30px;font-weight:700;line-height:1.2;">
        Phase Transformation Kinetics
      </h1>
      <p style="color:{MUTED};margin:8px 0 0;font-size:14px;">
        Classical Nucleation Theory &nbsp;·&nbsp; Wilson-Frenkel Growth &nbsp;·&nbsp;
        JMAK Model &nbsp;·&nbsp; GBM Surrogate
      </p>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
      <div style="font-size:22px;font-weight:700;color:{AX};">{mat_name}</div>
      <div style="display:flex;gap:6px;">
        {badge(cryst_hero, cryst_hero_col)}
        {badge(split_hero_lbl, split_hero_col)}
        {badge(f"n = {n_avrami}", C["purple"])}
      </div>
    </div>
  </div>
  <div style="display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;">
    <div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:8px 16px;font-size:12px;">
      <span style="color:{MUTED};">Materials</span>
      <span style="color:{AX};font-weight:600;margin-left:8px;">{len(MATERIALS)} metals</span>
    </div>
    <div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:8px 16px;font-size:12px;">
      <span style="color:{MUTED};">Training pts</span>
      <span style="color:{AX};font-weight:600;margin-left:8px;">
        {7500 + n_virtual * 50 if n_virtual else 7500:,}
      </span>
    </div>
    <div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:8px 16px;font-size:12px;">
      <span style="color:{MUTED};">Test R² T_nose</span>
      <span style="color:{C['green']};font-weight:700;margin-left:8px;">{r2_T:.4f}</span>
    </div>
    <div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:8px 16px;font-size:12px;">
      <span style="color:{MUTED};">Test R² log t</span>
      <span style="color:{C['green']};font-weight:700;margin-left:8px;">{r2_t:.4f}</span>
    </div>
    <div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:8px 16px;font-size:12px;">
      <span style="color:{MUTED};">Nose ΔT (ML error)</span>
      <span style="color:{C['yellow']};font-weight:700;margin-left:8px;">{dT_err:.1f}°C</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────────────────────────────────
m1c, m2c, m3c, m4c, m5c, m6c = st.columns(6)
cards = [
    (m1c, "Nose Temp · Physics", f"{Tn_tr:.0f} °C",     None,                       C["blue"]),
    (m2c, "Nose Temp · ML",      f"{ml['T_nose']:.0f} °C", f"Δ {ml['T_nose']-Tn_tr:+.0f}°C vs physics", C["purple"]),
    (m3c, "R²  T_nose  (test)",  f"{r2_T:.4f}",          "8 unseen materials",      C["green"]),
    (m4c, "R²  log t  (test)",   f"{r2_t:.4f}",          "nose time accuracy",      C["teal"]),
    (m5c, "Nucleation peak",     f"{T_Ipeak:.0f} °C",    "max I(T) temperature",    C["orange"]),
    (m6c, "Peak separation",     f"{abs(T_Ipeak-T_Upeak):.0f} °C", "ΔT between I & U peaks", C["red"]),
]
for col, lbl, val, sub, ac in cards:
    col.markdown(metric_card(lbl, val, sub, ac), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
tab_analysis, tab_compare, tab_database, tab_method = st.tabs(
    ["📊  Analysis", "⚖️  Compare", "🗂️  Database", "📐  Methodology"]
)

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ═════════════════════════════════════════════════════════════════════════
with tab_analysis:

    with st.expander("⚙️  Panel visibility", expanded=False):
        cp = st.columns(4)
        show_rate = cp[0].checkbox("Rate Competition",  True)
        show_ttt  = cp[1].checkbox("TTT Diagram",       True)
        show_ml   = cp[2].checkbox("Physics vs ML",     True)
        show_nsen = cp[3].checkbox("Avrami n Sweep",    True)

    # ── ROW 1 ────────────────────────────────────────────────────────────
    figs_r1 = []

    if show_rate:
        fig1 = go.Figure()
        # Fill region where both I and U are above 50% — competition zone
        overlap_mask = (I_n > 0.5) & (U_n > 0.5)
        if overlap_mask.any():
            fig1.add_trace(go.Scatter(
                x=np.concatenate([[0], I_n[overlap_mask], [0]]),
                y=np.concatenate([[T_rc[overlap_mask][0]], T_rc[overlap_mask], [T_rc[overlap_mask][-1]]]),
                fill="toself", fillcolor="rgba(63,185,80,0.09)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        fig1.add_trace(go.Scatter(
            x=I_n, y=T_rc, mode="lines", name="Nucleation  I(T)",
            line=dict(color=C["blue"], width=2.8),
            hovertemplate="<b>Nucleation</b><br>Rate=%{x:.3f}<br>T=%{y:.0f}°C<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=U_n, y=T_rc, mode="lines", name="Growth  U(T)",
            line=dict(color=C["red"], width=2.8, dash="dash"),
            hovertemplate="<b>Growth</b><br>Rate=%{x:.3f}<br>T=%{y:.0f}°C<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=ov, y=T_rc, mode="lines", name="Overall  ∝ I¼U¾",
            line=dict(color=C["green"], width=2.8, dash="dashdot"),
            hovertemplate="<b>Overall</b><br>Rate=%{x:.3f}<br>T=%{y:.0f}°C<extra></extra>",
        ))
        fig1.add_hline(y=T_Ipeak, line=dict(color=C["blue"],  dash="dot", width=1.2), opacity=0.5)
        fig1.add_hline(y=T_Upeak, line=dict(color=C["red"],   dash="dot", width=1.2), opacity=0.5)
        fig1.update_yaxes(autorange="reversed")
        fig1.add_annotation(**annot(
            f"I peak : <b>{T_Ipeak:.0f}°C</b><br>"
            f"U peak : <b>{T_Upeak:.0f}°C</b><br>"
            f"ΔT  :  <b>{abs(T_Ipeak-T_Upeak):.0f}°C</b>",
            x=0.97, y=0.05,
        ))
        fig1.update_layout(**base_layout(
            title=f"{mat_name} — Rate Competition (CNT + Wilson-Frenkel)",
            xaxis_title="Normalized Rate",
            yaxis_title="Temperature (°C)",
            height=430,
        ))
        figs_r1.append(fig1)

    if show_ttt:
        lc_map = {0.01: (C["blue"], "solid"), 0.50: (C["purple"], "dash"), 0.99: (C["red"], "solid")}
        fig2 = go.Figure()
        # Shaded transformation window between 1% and 99%
        m1f = masks[0.01]; m9f = masks[0.99]
        if m1f.any() and m9f.any() and chosen_fracs:
            t1f = ttt_res[0.01]; t9f = ttt_res[0.99]
            n_pts = min(m1f.sum(), m9f.sum())
            T_w  = T_ttt[m1f][:n_pts]
            t1_w = t1f[m1f][:n_pts]
            t9_w = t9f[m9f][:n_pts]
            fig2.add_trace(go.Scatter(
                x=np.concatenate([t1_w, t9_w[::-1]]),
                y=np.concatenate([T_w,  T_w[::-1]]),
                fill="toself", fillcolor="rgba(88,166,255,0.08)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        for frac in chosen_fracs:
            t_f = ttt_res[frac]; m_f = masks[frac]
            col_f, dash_f = lc_map[frac]
            if m_f.any():
                fig2.add_trace(go.Scatter(
                    x=t_f[m_f], y=T_ttt[m_f], mode="lines",
                    name=f"{int(frac*100)}% transformed",
                    line=dict(color=col_f, width=2.8, dash=dash_f),
                    hovertemplate=f"<b>{int(frac*100)}%</b><br>t=%{{x:.3e}} s<br>T=%{{y:.0f}}°C<extra></extra>",
                ))
        fig2.add_trace(go.Scatter(
            x=[tt_tr], y=[Tn_tr], mode="markers",
            name=f"Nose  {Tn_tr:.0f}°C · {tt_tr:.2e} s",
            marker=dict(color=C["yellow"], size=14, symbol="circle",
                        line=dict(color="white", width=2)),
        ))
        fig2.update_xaxes(type="log")
        fig2.add_annotation(**annot(
            f"Nose: <b>{Tn_tr:.0f}°C</b>  @  <b>{tt_tr:.2e} s</b>",
            y=0.04, color=C["yellow"],
        ))
        fig2.update_layout(**base_layout(
            title=f"{mat_name} — TTT Diagram (JMAK, n={n_avrami})",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=430,
        ))
        figs_r1.append(fig2)

    if show_ml:
        fig3 = go.Figure()
        m1 = masks[0.01]
        if m1.any():
            fig3.add_trace(go.Scatter(
                x=ttt_res[0.01][m1], y=T_ttt[m1], mode="lines",
                name="Physics (CNT+JMAK)",
                line=dict(color=C["blue"], width=3),
                hovertemplate="<b>Physics</b><br>t=%{x:.3e} s<br>T=%{y:.0f}°C<extra></extra>",
            ))
        if mk.any():
            fig3.add_trace(go.Scatter(
                x=t1_ml[mk], y=ml["T"][mk], mode="lines",
                name="ML Surrogate (GBM)",
                line=dict(color=C["red"], width=2.5, dash="dash"),
                hovertemplate="<b>ML</b><br>t=%{x:.3e} s<br>T=%{y:.0f}°C<extra></extra>",
            ))
        fig3.add_trace(go.Scatter(
            x=[tt_tr], y=[Tn_tr], mode="markers",
            name=f"Physics nose  {Tn_tr:.0f}°C",
            marker=dict(color=C["blue"], size=14, symbol="circle",
                        line=dict(color="white", width=2)),
        ))
        fig3.add_trace(go.Scatter(
            x=[ml["t_nose"]], y=[ml["T_nose"]], mode="markers",
            name=f"ML nose  {ml['T_nose']:.0f}°C",
            marker=dict(color=C["red"], size=18, symbol="star",
                        line=dict(color="white", width=1.5)),
        ))
        fig3.update_xaxes(type="log")
        pct_err = 100 * dT_err / abs(Tn_tr) if Tn_tr != 0 else 0
        fig3.add_annotation(**annot(
            f"ΔT_nose = <b>{dT_err:.1f}°C</b>  ({pct_err:.1f}%)<br>"
            f"Δ log t  = <b>{dt_err:.2f}</b>",
            color=C["orange"],
        ))
        fig3.update_layout(**base_layout(
            title=f"{mat_name} — Physics vs ML  (1% transformation start)",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=430,
        ))
        figs_r1.append(fig3)

    if figs_r1:
        for col, fig in zip(st.columns(len(figs_r1)), figs_r1):
            col.plotly_chart(fig, use_container_width=True)

    # ── ROW 2 ────────────────────────────────────────────────────────────
    figs_r2 = []

    if show_nsen:
        n_labels = {1:"1D, site sat.",2:"2D, site sat.",
                    3:"3D, site sat.",4:"3D, cont. nuc."}
        n_colors = [C["yellow"], C["orange"], C["red"], C["blue"]]
        fig4 = go.Figure()
        for ni, nc in zip([1, 2, 3, 4], n_colors):
            T_n, res_n, Tn_n, _ = ttt_curve(p_main, n_avrami=ni)
            t1n = res_n[0.01]; mn = np.isfinite(t1n) & (t1n > 0) & (t1n < 1e23)
            if mn.any():
                fig4.add_trace(go.Scatter(
                    x=t1n[mn], y=T_n[mn], mode="lines",
                    name=f"n={ni} · {n_labels[ni]} · {Tn_n:.0f}°C",
                    line=dict(color=nc, width=2.5),
                    visible=True if ni == n_avrami else "legendonly",
                    hovertemplate=f"<b>n={ni}</b><br>t=%{{x:.3e}} s<br>T=%{{y:.0f}}°C<extra></extra>",
                ))
        fig4.update_xaxes(type="log")
        fig4.update_layout(**base_layout(
            title=f"{mat_name} — Avrami n Sensitivity (1% start)",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=430,
        ))
        figs_r2.append(fig4)

    if figs_r2:
        for col, fig in zip(st.columns(len(figs_r2)), figs_r2):
            col.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ═════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown(f"### Compare {mat_name} against other materials")

    cmp_left, cmp_right = st.columns([1, 3])
    with cmp_left:
        others = [m for m in all_mats if m != mat_name]
        cmp_mats = st.multiselect(
            "Overlay materials",
            options=others, default=others[:3],
            help="Primary material (sidebar) always included",
        )
        cmp_show_nuc = st.checkbox("Nucleation peak markers", True)
        cmp_show_fill = st.checkbox("Fill TTT window", True)

    all_cmp = [mat_name] + cmp_mats
    fig_ttt = go.Figure()
    fig_rate = go.Figure()

    for mi, m in enumerate(all_cmp):
        pp  = MATERIALS[m]
        col = PAL[mi % len(PAL)]
        T_m, res_m, Tn_m, tt_m = ttt_curve(pp, n_avrami=n_avrami)
        t1m = res_m[0.01]; mm = np.isfinite(t1m) & (t1m > 0) & (t1m < 1e23)
        hl  = (m == mat_name)

        if mm.any():
            fig_ttt.add_trace(go.Scatter(
                x=t1m[mm], y=T_m[mm], mode="lines",
                name=m + (" ★" if hl else ""),
                line=dict(color=col, width=3.5 if hl else 1.8),
                hovertemplate=f"<b>{m}</b><br>t=%{{x:.3e}} s<br>T=%{{y:.0f}}°C<extra></extra>",
            ))
            if cmp_show_fill and not hl:
                t9m = res_m[0.99]; mm9 = np.isfinite(t9m)&(t9m>0)&(t9m<1e23)
                if mm9.any():
                    n_p = min(mm.sum(), mm9.sum())
                    fig_ttt.add_trace(go.Scatter(
                        x=np.concatenate([t1m[mm][:n_p], t9m[mm9][:n_p][::-1]]),
                        y=np.concatenate([T_m[mm][:n_p], T_m[mm9][:n_p][::-1]]),
                        fill="toself", fillcolor="rgba(88,166,255,0.05)",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))

        fig_ttt.add_trace(go.Scatter(
            x=[tt_m], y=[Tn_m], mode="markers+text",
            showlegend=False,
            text=[f"  {Tn_m:.0f}°C"], textposition="middle right",
            textfont=dict(color=col, size=10),
            marker=dict(color=col, size=12 if hl else 8,
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{m}</b> nose<br>{Tn_m:.0f}°C · {tt_m:.2e} s<extra></extra>",
        ))

        T_rc_m, I_n_m, U_n_m, ov_m = rate_curves_normalized(pp)
        fig_rate.add_trace(go.Scatter(
            x=ov_m, y=T_rc_m, mode="lines",
            name=m + (" ★" if hl else ""),
            line=dict(color=col, width=3 if hl else 1.8),
            hovertemplate=f"<b>{m}</b><br>Overall=%{{x:.3f}}<br>T=%{{y:.0f}}°C<extra></extra>",
        ))
        if cmp_show_nuc:
            T_a = np.linspace(0.40*pp["Tm"], 0.995*pp["Tm"], 600)
            Tip = T_a[np.argmax(nucleation_rate(T_a, pp))] - 273.15
            fig_rate.add_hline(
                y=Tip, line=dict(color=col, dash="dot", width=1), opacity=0.5,
                annotation_text=m, annotation_font_color=col, annotation_font_size=9,
            )

    fig_ttt.update_xaxes(type="log")
    fig_ttt.update_layout(**base_layout(
        title=f"TTT Comparison — 1% transformation start (n={n_avrami})",
        xaxis_title="Time (s)", yaxis_title="Temperature (°C)", height=460,
    ))
    fig_rate.update_yaxes(autorange="reversed")
    fig_rate.update_layout(**base_layout(
        title="Overall Transformation Rate Comparison",
        xaxis_title="Normalized Overall Rate", yaxis_title="Temperature (°C)", height=420,
    ))

    with cmp_right:
        st.plotly_chart(fig_ttt,  use_container_width=True)
        st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("#### Summary table")
    rows = []
    for m in all_cmp:
        pp = MATERIALS[m]
        _, _, Tn_m, tt_m = ttt_curve(pp, n_avrami=n_avrami)
        mlp = ml_predict(models, pp, n_avrami=n_avrami)
        T_a = np.linspace(0.40*pp["Tm"], 0.995*pp["Tm"], 600)
        Tip = T_a[np.argmax(nucleation_rate(T_a, pp))] - 273.15
        Tup = T_a[np.argmax(growth_rate(T_a, pp))]    - 273.15
        rows.append({
            "Material":     m,
            "Structure":    CRYSTAL.get(m, "—"),
            "Split":        "Test" if m in TEST_MATERIALS else "Train",
            "Tₘ (K)":      pp["Tm"],
            "Nose T (°C)":  f"{Tn_m:.0f}",
            "Nose t (s)":   f"{tt_m:.3e}",
            "ML T (°C)":   f"{mlp['T_nose']:.0f}",
            "|ΔT| (°C)":   f"{abs(Tn_m-mlp['T_nose']):.1f}",
            "I peak (°C)": f"{Tip:.0f}",
            "U peak (°C)": f"{Tup:.0f}",
            "ΔT peaks":    f"{abs(Tip-Tup):.0f}°C",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# TAB 3 — DATABASE
# ═════════════════════════════════════════════════════════════════════════
with tab_database:
    st.markdown("### Material Database")

    db_rows = []
    for name, props in MATERIALS.items():
        db_rows.append({
            "Material":       name,
            "Structure":      CRYSTAL.get(name, "—"),
            "Split":          "Test" if name in TEST_MATERIALS else "Train",
            "Tₘ (K)":        props["Tm"],
            "Tₘ (°C)":       props["Tm"] - 273,
            "ΔHf (kJ/mol)":  round(props["dHf"] / 1000, 1),
            "γ (J/m²)":      props["gamma"],
            "Q (kJ/mol)":    round(props["Q"] / 1000, 0),
            "D₀ (m²/s)":    props["D0"],
            "Vm (cm³/mol)":  round(props["Vm"] * 1e6, 2),
            "a (Å)":         round(props["a"] * 1e10, 2),
            "Stefan #":      round(props["dHf"] / (R * props["Tm"]), 2),
        })
    db_df = pd.DataFrame(db_rows).sort_values("Tₘ (K)")

    f1, f2, f3 = st.columns([1, 1, 2])
    split_filter = f1.radio("Split", ["All", "Train only", "Test only"], horizontal=False)
    struct_opts  = ["All"] + sorted(set(CRYSTAL.values()))
    struct_filter = f2.selectbox("Crystal structure", struct_opts)
    tm_range = f3.slider(
        "Tₘ range (K)",
        int(db_df["Tₘ (K)"].min()), int(db_df["Tₘ (K)"].max()),
        (int(db_df["Tₘ (K)"].min()), int(db_df["Tₘ (K)"].max())),
    )

    filtered = db_df[db_df["Tₘ (K)"].between(*tm_range)]
    if split_filter  == "Train only":  filtered = filtered[filtered["Split"] == "Train"]
    elif split_filter == "Test only":  filtered = filtered[filtered["Split"] == "Test"]
    if struct_filter != "All":         filtered = filtered[filtered["Structure"] == struct_filter]

    st.dataframe(filtered, hide_index=True, use_container_width=True, height=300)
    st.caption(f"{len(filtered)} materials shown")

    db_c1, db_c2 = st.columns(2)

    # γ vs Tm scatter
    with db_c1:
        fig_sc = go.Figure()
        for split, col in [("Train", C["blue"]), ("Test", C["red"])]:
            sub = filtered[filtered["Split"] == split]
            if sub.empty: continue
            szmax = sub["ΔHf (kJ/mol)"].max()
            fig_sc.add_trace(go.Scatter(
                x=sub["Tₘ (K)"], y=sub["γ (J/m²)"],
                mode="markers+text", text=sub["Material"],
                textposition="top center", textfont=dict(size=8, color=col),
                name=split,
                marker=dict(
                    color=col, opacity=0.85, line=dict(color="white", width=0.8),
                    size=sub["ΔHf (kJ/mol)"] / szmax * 28 + 7,
                ),
                hovertemplate="<b>%{text}</b><br>Tₘ=%{x} K<br>γ=%{y:.3f} J/m²<extra></extra>",
            ))
        fig_sc.update_layout(**base_layout(
            title="γ vs Tₘ  (size ∝ ΔHf)",
            xaxis_title="Melting Point (K)", yaxis_title="γ (J/m²)", height=400,
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Stefan number bar
    with db_c2:
        db_s = filtered.sort_values("Stefan #", ascending=False)
        struct_colors = [CRYSTAL_COLOR.get(CRYSTAL.get(m, ""), MUTED) for m in db_s["Material"]]
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(
            x=db_s["Material"], y=db_s["Stefan #"],
            marker=dict(color=struct_colors, opacity=0.85, line=dict(color=EDGE, width=0.5)),
            hovertemplate="<b>%{x}</b><br>Stefan # = %{y:.2f}<extra></extra>",
        ))
        fig_st.update_layout(**base_layout(
            title="Stefan Number  ΔHf / (R·Tₘ)  — crystallisation driving force",
            xaxis_title="", yaxis_title="Stefan Number",
            showlegend=False, height=400,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color=AX),
                       gridcolor=GRID, linecolor=EDGE, title_font=dict(color=AX)),
        ))
        st.plotly_chart(fig_st, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown("### Model Architecture & Physics")

    # Pipeline diagram
    st.markdown(f"""
<div style="background:{PANEL};border:1px solid {EDGE};border-radius:12px;padding:24px 28px;margin-bottom:24px;">
  <div style="font-size:12px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px;">
    Computational Pipeline
  </div>
  <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;">
    {''.join([
      f'<div style="background:{GRID};border:1px solid {EDGE};border-radius:8px;padding:12px 16px;min-width:130px;">'
      f'<div style="font-size:11px;color:{MUTED};margin-bottom:4px;">{step}</div>'
      f'<div style="font-size:13px;font-weight:600;color:{col};">{name}</div>'
      f'</div>'
      f'<div style="color:{MUTED};font-size:20px;padding:0 8px;">→</div>'
      for step, name, col in [
        ("Step 1", "ASM / Smithells DB", C["blue"]),
        ("Step 2", "CNT Physics  I(T)", C["teal"]),
        ("Step 3", "Wilson-Frenkel  U(T)", C["green"]),
        ("Step 4", "JMAK  k(T), TTT", C["purple"]),
        ("Step 5", "Augmentation ×200", C["orange"]),
        ("Step 6", "GBM Surrogate", C["red"]),
      ]
    ])[:-len('<div style="color:{};font-size:20px;padding:0 8px;">→</div>'.format(MUTED))]}
  </div>
</div>
""", unsafe_allow_html=True)

    mth_c1, mth_c2 = st.columns(2)

    with mth_c1:
        st.markdown(f"#### Classical Nucleation Theory")
        st.latex(r"I(T) = Z \cdot \beta^* \cdot N_v \cdot \exp\!\left(-\frac{\Delta G^*}{k_B T}\right)")
        st.latex(r"\Delta G^* = \frac{16\pi\,\gamma^3}{3\,(\Delta G_v)^2}")
        st.latex(r"\Delta G_v = \frac{\Delta H_f \,\Delta T}{V_m\,T_m}, \quad \Delta T = T_m - T")
        st.caption("Nucleation rate I(T) — Turnbull & Fisher, J. Chem. Phys. 17, 71 (1949)")

        st.markdown(f"#### Wilson-Frenkel Growth")
        st.latex(r"U(T) = \frac{D(T)}{a}\left[1 - \exp\!\left(-\frac{\Delta G_v V_m}{RT}\right)\right]")
        st.latex(r"D(T) = D_0\,\exp\!\left(-\frac{Q}{RT}\right)")
        st.caption("Interface-controlled crystal growth velocity")

    with mth_c2:
        st.markdown(f"#### JMAK Transformation Kinetics")
        st.latex(r"f(t,T) = 1 - \exp\!\left(-k(T)\,t^n\right)")
        st.latex(r"k(T) = \frac{\pi}{3}\,I(T)\,U(T)^3 \quad (n=4)")
        st.latex(r"t_{X}(T) = \left(\frac{\ln\!\frac{1}{1-X}}{k(T)}\right)^{1/n}")
        st.caption("Johnson–Mehl–Avrami–Kolmogorov — time for fraction X at temperature T")

        st.markdown(f"#### ML Feature Vector  (dim = 15)")
        features = [
            ("Tₘ/1000",         "Reduced melting point"),
            ("ΔHf/10⁴",         "Fusion enthalpy"),
            ("γ × 100",         "Interfacial energy"),
            ("Vm × 10⁶",        "Molar volume"),
            ("log₁₀ D₀",        "Diffusivity prefactor"),
            ("Q/10⁵",           "Activation energy"),
            ("ΔHf/(R·Tₘ)",      "Stefan number"),
            ("log₁₀ τ_diff",    "Diffusion timescale"),
            ("ΔG*/(k_B T_ref)", "Reduced barrier"),
            ("r*/a",            "Reduced critical radius"),
            ("Q/(R·Tₘ)",        "Reduced activation energy"),
            ("T/Tₘ",            "Reduced temperature"),
            ("(T/Tₘ)²",         "Quadratic T term"),
            ("1−T/Tₘ",          "Undercooling"),
            ("(1−T/Tₘ)²",       "Quadratic undercooling"),
        ]
        feat_df = pd.DataFrame(features, columns=["Feature", "Physical meaning"])
        st.dataframe(feat_df, hide_index=True, use_container_width=True, height=280)

    st.divider()
    st.markdown(f"#### Model Performance Summary")
    perf_cols = st.columns(4)
    perf_data = [
        ("GBM · log I(T)",  "Train R²",  "> 0.999", C["green"]),
        ("GBM · log U(T)",  "Train R²",  "> 0.999", C["green"]),
        ("GBM · log k(T)",  "Train R²",  "> 0.999", C["green"]),
        ("RF  · T_nose",    "Test R²",   f"{r2_T:.4f}", C["blue"] if r2_T > 0.95 else C["orange"]),
    ]
    for col, model, metric, val, ac in zip(perf_cols, *zip(*[(c,m,v,a) for c,m,v,a in perf_data])):
        col.markdown(metric_card(model, val, metric, ac), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:{PANEL};border:1px solid {EDGE};border-radius:10px;padding:16px 20px;">
  <div style="font-size:12px;color:{MUTED};margin-bottom:8px;">Data Sources</div>
  <div style="font-size:13px;color:{AX};line-height:1.8;">
    • ASM Handbook Vol. 2 & 3 (Alloy Phase Diagrams, 10th ed.)<br>
    • Smithells Metals Reference Book, 8th ed. (Brandes & Brook, 2013)<br>
    • Kelton & Greer, <i>Nucleation in Condensed Matter</i> (Pergamon, 2010)<br>
    • Turnbull & Fisher, J. Chem. Phys. <b>17</b>, 71 (1949)<br>
    • Iida & Guthrie, <i>Physical Properties of Liquid Metals</i> (1988)
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top:1px solid {EDGE};padding-top:14px;display:flex;
            justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <span style="font-size:12px;color:{MUTED};">
    MLL202 · Phase Transformation Kinetics · IIT Delhi
  </span>
  <span style="font-size:12px;color:{MUTED};">
    CNT + JMAK + GBM surrogate &nbsp;·&nbsp;
    {len(MATERIALS)} materials &nbsp;·&nbsp;
    {7500 + n_virtual*50 if n_virtual else 7500:,} training points
  </span>
</div>
""", unsafe_allow_html=True)
