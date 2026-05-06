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

st.markdown(f"""
<style>
  .stApp, .stMain {{ background-color:{BG}; }}
  section[data-testid="stSidebar"] {{ background-color:{PANEL}; }}
  section[data-testid="stSidebar"] * {{ color:{AX} !important; }}
  h1,h2,h3,h4,h5 {{ color:{AX} !important; }}
  p, li, label {{ color:{AX}; }}
  div[data-testid="metric-container"] {{
    background:{PANEL}; border:1px solid {EDGE};
    border-radius:8px; padding:12px 16px;
  }}
  div[data-testid="metric-container"] label {{ color:{MUTED} !important; font-size:13px; }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color:{C['blue']} !important; font-size:22px; font-weight:700;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{
    font-size:12px;
  }}
  .stTabs [data-baseweb="tab-list"] {{ background:{PANEL}; border-radius:8px; gap:4px; }}
  .stTabs [data-baseweb="tab"] {{ background:{PANEL}; color:{MUTED}; border-radius:6px; }}
  .stTabs [aria-selected="true"] {{ background:{GRID}; color:{AX} !important; }}
  .stDataFrame {{ background:{PANEL}; }}
  footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def base_layout(**kw):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=AX, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE,
                   tickfont=dict(color=AX), title_font=dict(color=AX)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=EDGE,
                   tickfont=dict(color=AX), title_font=dict(color=AX)),
        legend=dict(bgcolor=GRID, bordercolor=EDGE, borderwidth=1,
                    font=dict(color=AX, size=10)),
        margin=dict(l=60, r=20, t=48, b=48),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=EDGE, font_color=AX),
    )
    d.update(kw)
    return d


def annotation(text, x=0.02, y=0.97, color=None):
    return dict(
        text=text, xref="paper", yref="paper", x=x, y=y,
        showarrow=False, align="left",
        bgcolor=BG, bordercolor=EDGE, borderwidth=1,
        font=dict(color=color or C["yellow"], size=11),
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
        true_nT.append(Tn_t)
        pred_nT.append(mlp["T_nose"])
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
    st.markdown(f"## ⚗️ PTK Dashboard")
    st.caption("Phase Transformation Kinetics · CNT + JMAK + ML")
    st.divider()

    st.subheader("🎯 Material")
    all_mats = sorted(MATERIALS.keys())
    mat_name = st.selectbox(
        "Primary material", all_mats,
        index=all_mats.index("Zinc"),
        help="Select any of the 40 metals in the database",
    )
    p_main = MATERIALS[mat_name]
    split_tag = "🧪 Test set" if mat_name in TEST_MATERIALS else "🔬 Training set"
    st.caption(split_tag)

    st.divider()
    st.subheader("⚙️ Physics")
    n_avrami = st.select_slider(
        "Avrami exponent n",
        options=[1, 2, 3, 4],
        value=4,
        help="1=1D site-sat | 2=2D site-sat | 3=3D site-sat | 4=3D continuous nucleation",
    )
    ttt_levels = st.multiselect(
        "TTT iso-fraction lines",
        options=["1%", "50%", "99%"],
        default=["1%", "50%", "99%"],
    )
    ttt_frac_map = {"1%": 0.01, "50%": 0.50, "99%": 0.99}
    chosen_fracs = [ttt_frac_map[l] for l in ttt_levels]

    st.divider()
    st.subheader("🤖 ML Training")
    n_virtual = st.select_slider(
        "Augmentation level",
        options=[0, 200, 500, 1000],
        value=200,
        format_func=lambda v: {
            0: "Real only (7.5k pts)",
            200: "Standard (17.5k pts)",
            500: "Extended (32.5k pts)",
            1000: "Publication (57.5k pts)",
        }[v],
    )

    st.divider()
    st.subheader("📋 Properties")
    props_df = pd.DataFrame({
        "Parameter": ["Tₘ", "ΔHf", "γ", "Q", "D₀", "Vm", "a"],
        "Value": [
            f"{p_main['Tm']} K  ({p_main['Tm']-273:.0f}°C)",
            f"{p_main['dHf']/1000:.1f} kJ/mol",
            f"{p_main['gamma']:.3f} J/m²",
            f"{p_main['Q']/1000:.0f} kJ/mol",
            f"{p_main['D0']:.2e} m²/s",
            f"{p_main['Vm']*1e6:.2f} cm³/mol",
            f"{p_main['a']*1e10:.2f} Å",
        ],
    })
    st.dataframe(props_df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────
with st.spinner("Training ML surrogate models…"):
    models = get_models(n_virtual, n_avrami)

# ─────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE
# ─────────────────────────────────────────────────────────────────────────
T_rc, I_n, U_n, ov         = rate_curves_normalized(p_main)
T_ttt, ttt_res, Tn_tr, tt_tr = ttt_curve(p_main, n_avrami=n_avrami)
masks                        = ttt_masks(ttt_res)
ml, t1_ml, mk               = compute_ml_ttt(models, p_main, n_avrami)

T_arr = np.linspace(0.40 * p_main["Tm"], 0.995 * p_main["Tm"], 600)
T_Ipeak = T_arr[np.argmax(nucleation_rate(T_arr, p_main))] - 273.15
T_Upeak = T_arr[np.argmax(growth_rate(T_arr, p_main))]    - 273.15

true_nT, pred_nT, r2_T, r2_t, true_nt, pred_nt = test_set_metrics(models, n_avrami)
dT_err = abs(Tn_tr - ml["T_nose"])
dt_err = abs(np.log10(tt_tr + 1e-30) - np.log10(ml["t_nose"] + 1e-30))

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    f"# Phase Transformation Kinetics"
)
st.caption(
    f"CNT · Wilson-Frenkel · JMAK · GBM Surrogate  "
    f"| **{mat_name}** | Avrami n = {n_avrami} "
    f"| {len(TRAIN_MATERIALS)} train + {len(TEST_MATERIALS)} test materials"
)

# ── Top metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Nose Temp · Physics", f"{Tn_tr:.0f} °C")
c2.metric("Nose Temp · ML",      f"{ml['T_nose']:.0f} °C",
          delta=f"{ml['T_nose']-Tn_tr:+.0f} °C")
c3.metric("R² T_nose (test)",    f"{r2_T:.4f}")
c4.metric("R² log t (test)",     f"{r2_t:.4f}")
c5.metric("I peak",              f"{T_Ipeak:.0f} °C")
c6.metric("Peak Separation",     f"{abs(T_Ipeak-T_Upeak):.0f} °C")

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
tab_analysis, tab_compare, tab_database = st.tabs(
    ["📊 Analysis", "⚖️ Compare Materials", "🗂️ Database"]
)

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ═════════════════════════════════════════════════════════════════════════
with tab_analysis:

    # ── Panel toggles ────────────────────────────────────────────────────
    with st.expander("Panel visibility", expanded=False):
        col_p = st.columns(6)
        show_rate  = col_p[0].checkbox("Rate Competition",   True)
        show_ttt   = col_p[1].checkbox("TTT Diagram",        True)
        show_ml    = col_p[2].checkbox("Physics vs ML",       True)
        show_nsen  = col_p[3].checkbox("Avrami Sensitivity",  True)
        show_all   = col_p[4].checkbox("All Materials TTT",   True)
        show_gen   = col_p[5].checkbox("ML Generalisation",   True)

    # ── ROW 1 ─────────────────────────────────────────────────────────────
    figs_r1 = []

    # Panel 1 — Rate Competition
    if show_rate:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=I_n, y=T_rc, mode="lines", name="Nucleation  I(T)",
            line=dict(color=C["blue"], width=3),
            hovertemplate="Rate=%{x:.3f}<br>T=%{y:.0f}°C",
        ))
        fig1.add_trace(go.Scatter(
            x=U_n, y=T_rc, mode="lines", name="Growth  U(T)",
            line=dict(color=C["red"], width=3, dash="dash"),
            hovertemplate="Rate=%{x:.3f}<br>T=%{y:.0f}°C",
        ))
        fig1.add_trace(go.Scatter(
            x=ov, y=T_rc, mode="lines", name="Overall  ∝ I¼U¾",
            line=dict(color=C["green"], width=3, dash="dashdot"),
            hovertemplate="Rate=%{x:.3f}<br>T=%{y:.0f}°C",
        ))
        fig1.add_hline(y=T_Ipeak, line=dict(color=C["blue"],  dash="dot", width=1), opacity=0.5)
        fig1.add_hline(y=T_Upeak, line=dict(color=C["red"],   dash="dot", width=1), opacity=0.5)
        fig1.update_yaxes(autorange="reversed")
        fig1.add_annotation(**annotation(
            f"I peak : {T_Ipeak:.0f}°C<br>U peak : {T_Upeak:.0f}°C<br>ΔT : {abs(T_Ipeak-T_Upeak):.0f}°C",
            x=0.97, y=0.05,
        ))
        fig1.update_layout(**base_layout(
            title=f"{mat_name} — Rate Competition",
            xaxis_title="Normalized Rate",
            yaxis_title="Temperature (°C)",
            height=420,
        ))
        figs_r1.append(fig1)

    # Panel 2 — TTT (physics)
    if show_ttt:
        lc_map = {0.01: (C["blue"], "solid"), 0.50: (C["purple"], "dash"), 0.99: (C["red"], "solid")}
        fig2 = go.Figure()
        for frac in chosen_fracs:
            t_f = ttt_res[frac]; m_f = masks[frac]
            col_f, dash_f = lc_map[frac]
            if m_f.any():
                fig2.add_trace(go.Scatter(
                    x=t_f[m_f], y=T_ttt[m_f], mode="lines",
                    name=f"{int(frac*100)}%",
                    line=dict(color=col_f, width=2.8, dash=dash_f),
                    hovertemplate="t=%{x:.3e}s<br>T=%{y:.0f}°C",
                ))
        fig2.add_trace(go.Scatter(
            x=[tt_tr], y=[Tn_tr], mode="markers",
            name=f"Nose  {Tn_tr:.0f}°C  {tt_tr:.2e}s",
            marker=dict(color=C["yellow"], size=14, symbol="circle",
                        line=dict(color="white", width=1.5)),
        ))
        fig2.update_xaxes(type="log", minor=dict(showgrid=True))
        fig2.update_layout(**base_layout(
            title=f"{mat_name} — TTT Diagram (JMAK n={n_avrami})",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=420,
        ))
        figs_r1.append(fig2)

    # Panel 3 — Physics vs ML
    if show_ml:
        fig3 = go.Figure()
        m1 = masks[0.01]
        if m1.any():
            fig3.add_trace(go.Scatter(
                x=ttt_res[0.01][m1], y=T_ttt[m1], mode="lines",
                name="Physics  (CNT+JMAK)", line=dict(color=C["blue"], width=3),
                hovertemplate="t=%{x:.3e}s<br>T=%{y:.0f}°C",
            ))
        if mk.any():
            fig3.add_trace(go.Scatter(
                x=t1_ml[mk], y=ml["T"][mk], mode="lines",
                name="ML  (GBM)", line=dict(color=C["red"], width=2.5, dash="dash"),
                hovertemplate="t=%{x:.3e}s<br>T=%{y:.0f}°C",
            ))
        fig3.add_trace(go.Scatter(
            x=[tt_tr], y=[Tn_tr], mode="markers",
            name=f"True nose  {Tn_tr:.0f}°C",
            marker=dict(color=C["blue"], size=14, symbol="circle",
                        line=dict(color="white", width=1.5)),
        ))
        fig3.add_trace(go.Scatter(
            x=[ml["t_nose"]], y=[ml["T_nose"]], mode="markers",
            name=f"ML nose  {ml['T_nose']:.0f}°C",
            marker=dict(color=C["red"], size=16, symbol="star",
                        line=dict(color="white", width=1.5)),
        ))
        fig3.update_xaxes(type="log")
        fig3.add_annotation(**annotation(
            f"ΔT_nose = {dT_err:.1f}°C<br>Δlog t = {dt_err:.2f}",
            color=C["orange"],
        ))
        fig3.update_layout(**base_layout(
            title=f"{mat_name} — Physics vs ML  (1% start)",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=420,
        ))
        figs_r1.append(fig3)

    if figs_r1:
        cols = st.columns(len(figs_r1))
        for col, fig in zip(cols, figs_r1):
            col.plotly_chart(fig, use_container_width=True)

    # ── ROW 2 ─────────────────────────────────────────────────────────────
    figs_r2 = []

    # Panel 4 — Avrami n sensitivity
    if show_nsen:
        n_labels  = {1: "1D site-sat", 2: "2D site-sat",
                     3: "3D site-sat", 4: "3D cont. nuc."}
        n_colors  = [C["yellow"], C["orange"], C["red"], C["blue"]]
        fig4 = go.Figure()
        for ni, nc in zip([1, 2, 3, 4], n_colors):
            T_n, res_n, Tn_n, _ = ttt_curve(p_main, n_avrami=ni)
            t1n = res_n[0.01]; mn = np.isfinite(t1n) & (t1n > 0) & (t1n < 1e23)
            if mn.any():
                fig4.add_trace(go.Scatter(
                    x=t1n[mn], y=T_n[mn], mode="lines",
                    name=f"n={ni}  {n_labels[ni]}  ({Tn_n:.0f}°C)",
                    line=dict(color=nc, width=2.5),
                    visible=True if ni == n_avrami else "legendonly",
                    hovertemplate="t=%{x:.3e}s<br>T=%{y:.0f}°C",
                ))
        fig4.update_xaxes(type="log")
        fig4.update_layout(**base_layout(
            title=f"{mat_name} — Avrami n Sensitivity",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=420,
        ))
        figs_r2.append(fig4)

    # Panel 5 — All training materials
    if show_all:
        fig5 = go.Figure()
        for mi, m in enumerate(TRAIN_MATERIALS):
            pp  = MATERIALS[m]
            col = PAL[mi % len(PAL)]
            T_m, res_m, Tn_m, tt_m = ttt_curve(pp, n_avrami=n_avrami)
            t1m = res_m[0.01]; mm = np.isfinite(t1m) & (t1m > 0) & (t1m < 1e23)
            hl  = (m == mat_name)
            if mm.any():
                fig5.add_trace(go.Scatter(
                    x=t1m[mm], y=T_m[mm], mode="lines", name=m,
                    line=dict(color=col, width=3.5 if hl else 1.4),
                    opacity=1.0 if hl else 0.65,
                    hovertemplate=f"<b>{m}</b><br>t=%{{x:.3e}}s<br>T=%{{y:.0f}}°C",
                ))
            fig5.add_trace(go.Scatter(
                x=[tt_m], y=[Tn_m], mode="markers", showlegend=False,
                marker=dict(color=col, size=9 if hl else 5,
                            line=dict(color="white", width=1) if hl else dict()),
                hovertemplate=f"<b>{m}</b> nose<br>{Tn_m:.0f}°C  {tt_m:.2e}s",
            ))
        fig5.update_xaxes(type="log")
        fig5.update_layout(**base_layout(
            title="All Training Materials — TTT (1% start)",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=420,
            legend=dict(bgcolor=GRID, bordercolor=EDGE, font=dict(color=AX, size=9)),
        ))
        figs_r2.append(fig5)

    # Panel 6 — ML generalisation
    if show_gen:
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=TEST_MATERIALS, y=true_nT, name="Physics (true)",
            marker=dict(color=C["blue"], opacity=0.85,
                        line=dict(color="white", width=0.5)),
            width=0.35, offset=-0.18,
            hovertemplate="<b>%{x}</b><br>True T_nose=%{y:.0f}°C",
        ))
        fig6.add_trace(go.Bar(
            x=TEST_MATERIALS, y=pred_nT, name="ML predicted",
            marker=dict(color=C["red"], opacity=0.85,
                        line=dict(color="white", width=0.5)),
            width=0.35, offset=0.18,
            hovertemplate="<b>%{x}</b><br>ML T_nose=%{y:.0f}°C",
        ))
        fig6.add_annotation(**annotation(
            f"R²  T_nose  = {r2_T:.4f}<br>R²  log t   = {r2_t:.4f}",
        ))
        fig6.update_layout(**base_layout(
            title=f"ML Generalisation — {len(TEST_MATERIALS)} Unseen Test Materials",
            xaxis_title="Material",
            yaxis_title="Nose Temperature (°C)",
            barmode="overlay",
            height=420,
        ))
        figs_r2.append(fig6)

    if figs_r2:
        cols = st.columns(len(figs_r2))
        for col, fig in zip(cols, figs_r2):
            col.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ═════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare materials side-by-side")

    cmp_col1, cmp_col2 = st.columns([1, 3])
    with cmp_col1:
        others = [m for m in all_mats if m != mat_name]
        cmp_mats = st.multiselect(
            "Select materials to overlay",
            options=others,
            default=others[:3],
            help="Primary material is always included",
        )
        cmp_show_nuc = st.checkbox("Show nucleation peak lines", True)

    all_cmp = [mat_name] + cmp_mats

    # Overlaid TTT
    fig_cmp_ttt = go.Figure()
    fig_cmp_rate = go.Figure()

    for mi, m in enumerate(all_cmp):
        pp  = MATERIALS[m]
        col = PAL[mi % len(PAL)]
        T_m, res_m, Tn_m, tt_m = ttt_curve(pp, n_avrami=n_avrami)
        t1m = res_m[0.01]; mm = np.isfinite(t1m) & (t1m > 0) & (t1m < 1e23)
        hl  = (m == mat_name)

        if mm.any():
            fig_cmp_ttt.add_trace(go.Scatter(
                x=t1m[mm], y=T_m[mm], mode="lines",
                name=m + (" ★" if hl else ""),
                line=dict(color=col, width=3 if hl else 2),
                hovertemplate=f"<b>{m}</b><br>t=%{{x:.3e}}s<br>T=%{{y:.0f}}°C",
            ))
        fig_cmp_ttt.add_trace(go.Scatter(
            x=[tt_m], y=[Tn_m], mode="markers+text",
            showlegend=False,
            text=[f"  {Tn_m:.0f}°C"],
            textposition="middle right",
            textfont=dict(color=col, size=10),
            marker=dict(color=col, size=10, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{m}</b> nose<br>{Tn_m:.0f}°C  {tt_m:.2e}s",
        ))

        # Rate curves
        T_rc_m, I_n_m, U_n_m, ov_m = rate_curves_normalized(pp)
        fig_cmp_rate.add_trace(go.Scatter(
            x=ov_m, y=T_rc_m, mode="lines",
            name=m + (" ★" if hl else ""),
            line=dict(color=col, width=3 if hl else 1.8),
            hovertemplate=f"<b>{m}</b><br>Overall=%{{x:.3f}}<br>T=%{{y:.0f}}°C",
        ))
        if cmp_show_nuc:
            T_m_arr = np.linspace(0.40*pp["Tm"], 0.995*pp["Tm"], 600)
            Tip = T_m_arr[np.argmax(nucleation_rate(T_m_arr, pp))] - 273.15
            fig_cmp_rate.add_hline(
                y=Tip, line=dict(color=col, dash="dot", width=1), opacity=0.55,
                annotation_text=m, annotation_font_color=col, annotation_font_size=9,
            )

    fig_cmp_ttt.update_xaxes(type="log")
    fig_cmp_ttt.update_yaxes(autorange=None)
    fig_cmp_ttt.update_layout(**base_layout(
        title=f"TTT Comparison — 1% start  (n={n_avrami})",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        height=480,
    ))

    fig_cmp_rate.update_yaxes(autorange="reversed")
    fig_cmp_rate.update_layout(**base_layout(
        title="Overall Transformation Rate Comparison",
        xaxis_title="Normalized Overall Rate",
        yaxis_title="Temperature (°C)",
        height=480,
    ))

    with cmp_col2:
        st.plotly_chart(fig_cmp_ttt, use_container_width=True)
        st.plotly_chart(fig_cmp_rate, use_container_width=True)

    # Summary table
    st.subheader("Nose comparison table")
    rows = []
    for m in all_cmp:
        pp = MATERIALS[m]
        _, _, Tn_m, tt_m = ttt_curve(pp, n_avrami=n_avrami)
        mlp = ml_predict(models, pp, n_avrami=n_avrami)
        T_a = np.linspace(0.40*pp["Tm"], 0.995*pp["Tm"], 600)
        Tip = T_a[np.argmax(nucleation_rate(T_a, pp))] - 273.15
        Tup = T_a[np.argmax(growth_rate(T_a, pp))]    - 273.15
        rows.append({
            "Material":   m,
            "Split":      "Test" if m in TEST_MATERIALS else "Train",
            "Tₘ (K)":    pp["Tm"],
            "Nose T (°C)": f"{Tn_m:.0f}",
            "Nose t (s)":  f"{tt_m:.3e}",
            "ML T (°C)":  f"{mlp['T_nose']:.0f}",
            "ΔT_nose":    f"{abs(Tn_m - mlp['T_nose']):.1f}°C",
            "I peak (°C)": f"{Tip:.0f}",
            "U peak (°C)": f"{Tup:.0f}",
            "ΔT peaks":   f"{abs(Tip-Tup):.0f}°C",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# TAB 3 — DATABASE
# ═════════════════════════════════════════════════════════════════════════
with tab_database:
    st.subheader("Material Database Browser")

    # Build full dataframe
    db_rows = []
    for name, props in MATERIALS.items():
        db_rows.append({
            "Material":    name,
            "Split":       "Test" if name in TEST_MATERIALS else "Train",
            "Tₘ (K)":     props["Tm"],
            "Tₘ (°C)":    props["Tm"] - 273,
            "ΔHf (kJ/mol)": round(props["dHf"] / 1000, 1),
            "γ (J/m²)":    props["gamma"],
            "Q (kJ/mol)":  round(props["Q"] / 1000, 0),
            "D₀ (m²/s)":  props["D0"],
            "Vm (cm³/mol)": round(props["Vm"] * 1e6, 2),
            "a (Å)":       round(props["a"] * 1e10, 2),
            "Stefan #":    round(props["dHf"] / (8.314 * props["Tm"]), 2),
        })
    db_df = pd.DataFrame(db_rows).sort_values("Tₘ (K)")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        split_filter = st.radio("Show", ["All", "Train only", "Test only"], horizontal=True)
    with col_f2:
        tm_range = st.slider(
            "Tₘ range (K)",
            int(db_df["Tₘ (K)"].min()), int(db_df["Tₘ (K)"].max()),
            (int(db_df["Tₘ (K)"].min()), int(db_df["Tₘ (K)"].max())),
        )

    filtered = db_df[db_df["Tₘ (K)"].between(*tm_range)]
    if split_filter == "Train only":
        filtered = filtered[filtered["Split"] == "Train"]
    elif split_filter == "Test only":
        filtered = filtered[filtered["Split"] == "Test"]

    st.dataframe(filtered, hide_index=True, use_container_width=True, height=320)

    # Scatter: γ vs Tm, sized by ΔHf
    fig_scatter = go.Figure()
    for split, col in [("Train", C["blue"]), ("Test", C["red"])]:
        sub = filtered[filtered["Split"] == split]
        fig_scatter.add_trace(go.Scatter(
            x=sub["Tₘ (K)"], y=sub["γ (J/m²)"],
            mode="markers+text",
            text=sub["Material"],
            textposition="top center",
            textfont=dict(size=9, color=col),
            name=split,
            marker=dict(
                color=col, opacity=0.85,
                size=sub["ΔHf (kJ/mol)"] / sub["ΔHf (kJ/mol)"].max() * 30 + 6,
                line=dict(color="white", width=0.8),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Tₘ = %{x} K<br>γ = %{y:.3f} J/m²"
            ),
        ))
    fig_scatter.update_layout(**base_layout(
        title="Interfacial Energy γ vs Melting Point  (marker size ∝ ΔHf)",
        xaxis_title="Melting Point Tₘ (K)",
        yaxis_title="γ (J/m²)",
        height=440,
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Bar: Stefan number
    fig_stefan = go.Figure()
    db_s = filtered.sort_values("Stefan #", ascending=False)
    colors_s = [C["red"] if s == "Test" else C["blue"] for s in db_s["Split"]]
    fig_stefan.add_trace(go.Bar(
        x=db_s["Material"], y=db_s["Stefan #"],
        marker=dict(color=colors_s, opacity=0.85, line=dict(color=EDGE, width=0.5)),
        hovertemplate="<b>%{x}</b><br>Stefan # = %{y:.2f}",
        name="Stefan number",
    ))
    fig_stefan.update_layout(**base_layout(
        title="Stefan Number  ΔHf / (R · Tₘ)  — driving force for crystallisation",
        xaxis_title="Material",
        yaxis_title="Stefan Number",
        height=360,
        showlegend=False,
        xaxis=dict(tickangle=-40, tickfont=dict(size=9, color=AX),
                   gridcolor=GRID, linecolor=EDGE, title_font=dict(color=AX)),
    ))
    st.plotly_chart(fig_stefan, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: ASM Handbook · Smithells · Kelton & Greer (2010) · Turnbull & Fisher (1949)  |  "
    "Model: GBM (sklearn) trained on CNT+JMAK physics engine  |  "
    "MLL202 Project"
)
