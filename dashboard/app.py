"""M7: FlowCast Smart Traffic Flow Prediction — Interactive Analytics Dashboard.

Implements all 9 required PRD Section 14.1 views and 4 supporting modules (14.2)
backed entirely by real persisted pipeline outputs and trained models.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.ml_models import NumPyLinearRegressionGD  # noqa: F401
from src.utils import load_config

# =============================================================================
# PAGE CONFIG & FUTURISTIC DARK GLASSMORPHIC THEME
# =============================================================================
st.set_page_config(
    page_title="FlowCast AI | Northline Corridor Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 12% 10%, #0f172a 0%, #030712 100%);
    color: #e2e8f0;
}
.fc-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.78) 0%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), inset 0 0 18px rgba(56, 189, 248, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.fc-title {
    font-size: 2.0rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 55%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.fc-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
}
.fc-status {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #34d399;
    display: flex;
    align-items: center;
    gap: 8px;
}
.fc-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}
.kpi-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    padding: 16px 18px;
    transition: all 0.25s ease;
}
.kpi-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-pill {
    display: inline-block;
    margin-top: 6px;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
}
.pill-cyan { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.pill-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.pill-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
.pill-red { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    padding-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px;
    padding: 8px 18px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22) 0%, rgba(99, 102, 241, 0.22) 100%);
    border-color: #38bdf8 !important;
    color: #f8fafc !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.22);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Consistent PRD Section 14.3 severity color palette
SEVERITY_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
SEVERITY_COLORS = {
    "Free-flow": "#10b981",
    "Moderate": "#f59e0b",
    "Heavy": "#f97316",
    "Severe": "#ef4444",
}
SEVERITY_SCORE = {"Free-flow": 0, "Moderate": 1, "Heavy": 2, "Severe": 3}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.45)",
    plot_bgcolor="rgba(15, 23, 42, 0.45)",
    font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
    margin=dict(l=20, r=20, t=44, b=24),
    xaxis=dict(gridcolor="rgba(148, 163, 184, 0.12)"),
    yaxis=dict(gridcolor="rgba(148, 163, 184, 0.12)"),
)


# =============================================================================
# CACHED DATA & ARTIFACT LOADERS
# =============================================================================
@st.cache_data(show_spinner="Loading FlowCast corridor telemetry & predictions...")
def load_dashboard_data():
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]

    clean_df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["cleaned_merged"])
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)

    c_preds = pd.read_parquet(models_dir / "classical_test_predictions.parquet")
    c_preds["timestamp"] = pd.to_datetime(c_preds["timestamp"], utc=True)

    dl_preds = pd.read_parquet(models_dir / "dl_test_predictions.parquet")
    dl_preds["timestamp"] = pd.to_datetime(dl_preds["timestamp"], utc=True)

    merged_preds = c_preds.merge(
        dl_preds.drop(columns=["actual_volume"]),
        on=["road_id", "timestamp"],
        how="left",
    )

    classical_sb = pd.read_csv(models_dir / "classical_scoreboard.csv")
    bench_df = pd.read_csv(models_dir / "classical_vs_dl_benchmark.csv")
    dl_hist = pd.read_csv(models_dir / "dl_training_history.csv")
    gd_hist = pd.read_csv(models_dir / "numpy_gd_loss_history.csv")
    cm_df = pd.read_csv(models_dir / "congestion_confusion_matrix.csv", index_col=0)
    roc_df = pd.read_csv(models_dir / "accident_roc_curve.csv")

    fi_map = {
        "Traffic Volume": pd.read_csv(models_dir / "volume_feature_importance.csv"),
        "Travel Time": pd.read_csv(models_dir / "travel_time_feature_importance.csv"),
        "Congestion (4-Class)": pd.read_csv(models_dir / "congestion_feature_importance.csv"),
        "Accident Risk": pd.read_csv(models_dir / "accident_feature_importance.csv"),
    }

    with open(models_dir / "error_analysis.json", "r", encoding="utf-8") as f:
        err_analysis = json.load(f)

    with open(models_dir / "inference_contract.json", "r", encoding="utf-8") as f:
        contract = json.load(f)

    cards = {}
    for card_file in sorted((models_dir / "model_cards").glob("*.json")):
        with open(card_file, "r", encoding="utf-8") as f:
            cards[card_file.stem] = json.load(f)

    dq_path = PROJECT_ROOT / cfg["paths"]["data_quality_report"]
    dq_text = dq_path.read_text(encoding="utf-8") if dq_path.exists() else "Report not generated."

    return {
        "config": cfg,
        "clean_df": clean_df,
        "preds_df": merged_preds,
        "classical_sb": classical_sb,
        "bench_df": bench_df,
        "dl_hist": dl_hist,
        "gd_hist": gd_hist,
        "cm_df": cm_df,
        "roc_df": roc_df,
        "fi_map": fi_map,
        "err_analysis": err_analysis,
        "contract": contract,
        "model_cards": cards,
        "dq_text": dq_text,
    }


@st.cache_resource(show_spinner="Loading trained inference models...")
def load_inference_models():
    models_dir = PROJECT_ROOT / "models"
    return {
        "vol_model": joblib.load(models_dir / "volume_best_model.joblib"),
        "tt_model": joblib.load(models_dir / "travel_time_best_model.joblib"),
        "cong_model": joblib.load(models_dir / "congestion_best_model.joblib"),
        "acc_model": joblib.load(models_dir / "accident_best_model.joblib"),
        "scaler_s1": joblib.load(models_dir / "feature_scaler.joblib"),
        "scaler_s2": joblib.load(models_dir / "feature_scaler_s2.joblib"),
        "scaler_risk": joblib.load(models_dir / "feature_scaler_risk.joblib"),
        "slot_baseline": pd.read_parquet(models_dir / "slot_baseline.parquet"),
        "road_acc_baseline": pd.read_parquet(models_dir / "road_acc_baseline.parquet"),
        "feat_tail": pd.read_parquet(
            PROJECT_ROOT / "data/processed/flowcast_features.parquet"
        )
        .sort_values(["road_id", "timestamp"])
        .groupby("road_id")
        .tail(1)
        .reset_index(drop=True),
    }


def format_arrow_safe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed numeric/missing benchmark table cleanly to strings for Arrow."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = df[col].apply(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
        else:
            out[col] = df[col].fillna("—").astype(str)
    return out


def build_corridor_gps_map(snap: pd.DataFrame, ts_label: str, mode: str) -> go.Figure:
    """Build interactive corridor spatial map (100% reliable GPS schematic or OSM tiles)."""
    snap_sorted = snap.sort_values("road_id").copy()

    if mode == "OpenStreetMap Tiles":
        fig = px.scatter_map(
            snap_sorted,
            lat="latitude",
            lon="longitude",
            color="display_cong",
            size="display_vol",
            size_max=22,
            color_discrete_map=SEVERITY_COLORS,
            category_orders={"display_cong": SEVERITY_ORDER},
            hover_name="road_name",
            hover_data={
                "road_id": True,
                "display_vol": ":.0f",
                "pred_travel_time": ":.2f",
                "pred_accident_prob": ":.3f",
                "display_conf": ":.1%",
                "latitude": ":.4f",
                "longitude": ":.4f",
            },
            center={
                "lat": float(snap_sorted["latitude"].mean()),
                "lon": float(snap_sorted["longitude"].mean()),
            },
            zoom=11.0,
            height=430,
            title=f"Northline Corridor Street Map — {ts_label}",
        )
        fig.update_layout(
            map_style="open-street-map",
            paper_bgcolor="rgba(15, 23, 42, 0.45)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=10, r=10, t=44, b=10),
            legend=dict(title="Congestion", orientation="h", y=-0.08, x=0.0),
        )
        return fig

    # Default: High-contrast GPS Spatial Corridor Radar (zero external tile dependencies)
    fig = go.Figure()

    # Arterial network backbone connecting sequential corridor sensors
    fig.add_trace(
        go.Scatter(
            x=snap_sorted["longitude"],
            y=snap_sorted["latitude"],
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.28)", width=2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    max_vol = max(float(snap_sorted["display_vol"].max()), 1.0)
    for level in SEVERITY_ORDER:
        sub = snap_sorted[snap_sorted["display_cong"] == level]
        if sub.empty:
            continue
        sizes = 14 + (sub["display_vol"] / max_vol) * 20
        hover_texts = [
            (
                f"<b>{r.road_id} — {r.road_name}</b><br>"
                f"Congestion: <b>{r.display_cong}</b> ({r.display_conf*100:.1f}% conf)<br>"
                f"Pred Volume: <b>{r.display_vol:.0f}</b> veh / 30m<br>"
                f"Travel Time: <b>{r.pred_travel_time:.2f} min</b><br>"
                f"Risk Score: <b>{r.pred_accident_prob*100:.1f} / 100</b><br>"
                f"GPS: ({r.latitude:.4f}, {r.longitude:.4f})"
            )
            for r in sub.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=sub["longitude"],
                y=sub["latitude"],
                mode="markers+text",
                name=level,
                text=sub["road_id"],
                textposition="top center",
                textfont=dict(size=9, color="#e2e8f0", family="JetBrains Mono"),
                marker=dict(
                    size=sizes,
                    color=SEVERITY_COLORS[level],
                    opacity=0.90,
                    line=dict(color="#f8fafc", width=1.4),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    lon_pad = max((snap_sorted["longitude"].max() - snap_sorted["longitude"].min()) * 0.14, 0.01)
    lat_pad = max((snap_sorted["latitude"].max() - snap_sorted["latitude"].min()) * 0.16, 0.01)

    fig.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.55)",
        plot_bgcolor="rgba(7, 13, 28, 0.92)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
        title=dict(text=f"Corridor GPS Spatial Map — {ts_label}", y=0.96),
        xaxis=dict(
            title="Longitude (°E)",
            range=[snap_sorted["longitude"].min() - lon_pad, snap_sorted["longitude"].max() + lon_pad],
            gridcolor="rgba(56, 189, 248, 0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude (°N)",
            range=[snap_sorted["latitude"].min() - lat_pad, snap_sorted["latitude"].max() + lat_pad],
            gridcolor="rgba(56, 189, 248, 0.12)",
            zeroline=False,
        ),
        height=430,
        margin=dict(l=18, r=18, t=48, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
    )
    return fig


def predict_from_feature_frame(feat_rows: pd.DataFrame, contract: dict, artifacts: dict) -> pd.DataFrame:
    """Run real two-stage classical ML inference on a feature dataframe."""
    s1_cols = contract["stage1_features"]
    s2_cols = contract["stage2_features"]
    risk_cols = contract["risk_features"]

    X_s1 = feat_rows[s1_cols].to_numpy(dtype=np.float64)
    X_s1_sc = artifacts["scaler_s1"].transform(X_s1)

    p_vol = np.clip(
        artifacts["vol_model"].predict(X_s1_sc if contract["volume_uses_scaled"] else X_s1),
        0.0,
        None,
    )
    cap_half = feat_rows["road_capacity"].to_numpy(dtype=np.float64) / 2.0
    p_vc = p_vol / cap_half

    extra = np.column_stack(
        [
            p_vol,
            p_vc,
            cap_half - p_vol,
            feat_rows["is_rain"].to_numpy(dtype=np.float64) * p_vc,
            feat_rows["is_fog"].to_numpy(dtype=np.float64) * p_vc,
        ]
    )
    X_s2 = np.hstack([X_s1, extra])
    X_s2_sc = artifacts["scaler_s2"].transform(X_s2)

    p_tt = np.clip(
        artifacts["tt_model"].predict(X_s2_sc if contract["travel_time_uses_scaled"] else X_s2),
        0.1,
        None,
    )

    c_probs = artifacts["cong_model"].predict_proba(
        X_s2_sc if contract["congestion_uses_scaled"] else X_s2
    )
    mult = np.asarray(contract["congestion_multipliers"], dtype=np.float64)
    c_ids = np.argmax(c_probs * mult, axis=1)
    c_conf = np.max(c_probs, axis=1)
    idx_to_cong = {0: "Free-flow", 1: "Moderate", 2: "Heavy", 3: "Severe"}

    risk_idx = [s2_cols.index(c) for c in risk_cols]
    X_r = X_s2[:, risk_idx]
    X_r_sc = artifacts["scaler_risk"].transform(X_r)
    acc_prob = artifacts["acc_model"].predict_proba(
        X_r_sc if contract["accident_uses_scaled"] else X_r
    )[:, 1]

    hw = float(contract["volume_q90_halfwidth"])
    return pd.DataFrame(
        {
            "pred_volume": np.round(p_vol, 1),
            "vol_lower_90": np.round(np.clip(p_vol - hw, 0.0, None), 1),
            "vol_upper_90": np.round(p_vol + hw, 1),
            "pred_vc_ratio": np.round(p_vc, 3),
            "pred_travel_time_min": np.round(p_tt, 2),
            "pred_congestion": [idx_to_cong[i] for i in c_ids],
            "congestion_confidence": np.round(c_conf * 100, 1),
            "risk_score_100": np.round(acc_prob * 100, 1),
        }
    )


# =============================================================================
# LOAD DATA & RENDER HEADER
# =============================================================================
data = load_dashboard_data()
clean_df = data["clean_df"]
preds_df = data["preds_df"]
bench_df = data["bench_df"]
contract = data["contract"]

st.markdown(
    """
<div class="fc-header">
    <div>
        <div class="fc-title">🚦 FLOWCAST // NORTHLINE CORRIDOR INTELLIGENCE</div>
        <div class="fc-sub">Meridian Mobility Systems · Real-Time Short-Horizon Volume, Congestion, Travel Time & Accident Risk Analytics</div>
    </div>
    <div class="fc-status">
        <div class="fc-dot"></div>
        <span>25 LOOP/RADAR SEGMENTS ONLINE · v1.0</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Pick latest active peak-hour window inside last 5 days as default snapshot
avail_ts_5d = sorted(preds_df["timestamp"].unique())[-240:]
recent_5d = preds_df[preds_df["timestamp"].isin(avail_ts_5d)]
ts_activity = (
    recent_5d.groupby("timestamp")
    .agg(
        alerts=("pred_congestion", lambda s: s.isin(["Heavy", "Severe"]).sum()),
        avg_vol=("pred_volume", "mean"),
    )
    .reset_index()
    .sort_values(["alerts", "avg_vol"], ascending=[False, False])
)
default_peak_ts = ts_activity.iloc[0]["timestamp"]
peak_snap = preds_df[preds_df["timestamp"] == default_peak_ts].copy()

best_vol_row = bench_df[bench_df["Task"] == "Traffic Volume"].sort_values("RMSE").iloc[0]
best_dl_cong = (
    bench_df[
        (bench_df["Family"] == "Deep Learning (PyTorch)")
        & (bench_df["Task"] == "Congestion (4-Class)")
    ]
    .sort_values("Macro_F1", ascending=False)
    .iloc[0]
)
severe_heavy_cnt = int(peak_snap["pred_congestion"].isin(["Heavy", "Severe"]).sum())

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">Corridor Telemetry</div>
        <div class="kpi-val">{len(clean_df):,}</div>
        <span class="kpi-pill pill-cyan">25 Segments · 30-Min Grain</span></div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">Best Volume MAPE</div>
        <div class="kpi-val">{best_vol_row['MAPE']:.2f}%</div>
        <span class="kpi-pill pill-green">Target ≤ 12% · R² {best_vol_row['R2']:.3f}</span></div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">Best Congestion F1</div>
        <div class="kpi-val">{best_dl_cong['Macro_F1']:.4f}</div>
        <span class="kpi-pill pill-cyan">LSTM Acc {best_dl_cong['Accuracy']*100:.1f}%</span></div>""",
        unsafe_allow_html=True,
    )
with k4:
    avg_vol_peak = peak_snap["pred_volume"].mean()
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">Peak Window Avg Flow</div>
        <div class="kpi-val">{avg_vol_peak:.0f}</div>
        <span class="kpi-pill pill-amber">{pd.Timestamp(default_peak_ts).strftime('%d %b %H:%M')} UTC</span></div>""",
        unsafe_allow_html=True,
    )
with k5:
    pill_cls = (
        "pill-red" if severe_heavy_cnt >= 5 else ("pill-amber" if severe_heavy_cnt > 0 else "pill-green")
    )
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">Peak Congestion Alerts</div>
        <div class="kpi-val">{severe_heavy_cnt} / 25</div>
        <span class="kpi-pill {pill_cls}">Heavy / Severe Segments</span></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Flat Navigation Tabs (All 9 Required Views + 4 Supporting Modules in <= 2 clicks)
tab_ops, tab_heat, tab_hist, tab_perf, tab_tools = st.tabs(
    [
        "🛰️ Live Predictions & Forecast",
        "🔥 Congestion Heatmap & Road Comparison",
        "📈 Historical Trends & Weather Impact",
        "🧠 Model Performance & Feature Importance",
        "⚙️ Upload, Horizon Predictor & Reports",
    ]
)

# =============================================================================
# TAB 1: LIVE PREDICTIONS (VIEW 1), FORECAST VIS (VIEW 7), CONFIDENCE (VIEW 8)
# =============================================================================
with tab_ops:
    st.subheader("1. Live Corridor Snapshot & Ranked Risk Monitor (View 1)")

    sel_col1, sel_col2, sel_col3 = st.columns([1.8, 1.1, 1.1])
    with sel_col1:
        selected_ts = st.select_slider(
            "Select Corridor Forecast Window (Held-Out Test Timeline):",
            options=avail_ts_5d,
            value=default_peak_ts,
            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M UTC"),
        )
    with sel_col2:
        model_choice = st.radio(
            "Active Forecast Engine:",
            ["Classical ML (Stacked)", "Deep Learning (PyTorch)"],
            horizontal=True,
        )
    with sel_col3:
        map_mode = st.radio(
            "Map Display Mode:",
            ["Corridor GPS Radar", "OpenStreetMap Tiles"],
            horizontal=True,
        )

    snap = preds_df[preds_df["timestamp"] == selected_ts].copy()
    if model_choice.startswith("Deep"):
        snap["display_vol"] = snap["dl_pred_volume"]
        snap["display_low"] = snap["dl_vol_lower_90"]
        snap["display_high"] = snap["dl_vol_upper_90"]
        snap["display_cong"] = snap["dl_pred_congestion"]
        snap["display_conf"] = snap["dl_congestion_confidence"]
    else:
        snap["display_vol"] = snap["pred_volume"]
        snap["display_low"] = snap["vol_lower_90"]
        snap["display_high"] = snap["vol_upper_90"]
        snap["display_cong"] = snap["pred_congestion"]
        snap["display_conf"] = snap["congestion_confidence"]

    col_map, col_tbl = st.columns([1.2, 1.3])
    with col_map:
        ts_str = pd.Timestamp(selected_ts).strftime("%d %b %H:%M UTC")
        fig_map = build_corridor_gps_map(snap, ts_str, map_mode)
        st.plotly_chart(fig_map, width="stretch")

    with col_tbl:
        snap_show = snap.sort_values(
            ["pred_accident_prob", "display_vol"], ascending=[False, False]
        )[
            [
                "road_id",
                "road_name",
                "display_vol",
                "display_cong",
                "display_conf",
                "pred_travel_time",
                "pred_accident_prob",
            ]
        ].copy()
        snap_show.columns = [
            "Road ID",
            "Segment",
            "Pred Vol",
            "Congestion",
            "Confidence",
            "Travel (m)",
            "Risk Score",
        ]
        snap_show["Pred Vol"] = snap_show["Pred Vol"].round(0).astype(int)
        snap_show["Confidence"] = (snap_show["Confidence"] * 100).round(1).astype(str) + "%"
        snap_show["Risk Score"] = (snap_show["Risk Score"] * 100).round(1).astype(str) + " / 100"
        st.dataframe(snap_show, width="stretch", height=430, hide_index=True)

    st.markdown("---")
    st.subheader("2. Forecast Visualisation & 90% Prediction Confidence Bands (Views 7 & 8)")

    f_col1, f_col2 = st.columns([1.2, 1.8])
    with f_col1:
        road_options = sorted(preds_df["road_id"].unique().tolist())
        road_names = dict(zip(preds_df["road_id"], preds_df["road_name"]))
        sel_road = st.selectbox(
            "Select Road Segment:",
            road_options,
            format_func=lambda r: f"{r} — {road_names[r]}",
        )
    with f_col2:
        horizon_days = st.slider("Display Window (Last N Days of Test Set):", 2, 14, 5)

    road_ts_df = preds_df[preds_df["road_id"] == sel_road].sort_values("timestamp").tail(horizon_days * 48)

    fig_fc = go.Figure()
    upper_col = "dl_vol_upper_90" if model_choice.startswith("Deep") else "vol_upper_90"
    lower_col = "dl_vol_lower_90" if model_choice.startswith("Deep") else "vol_lower_90"
    fig_fc.add_trace(
        go.Scatter(
            x=pd.concat([road_ts_df["timestamp"], road_ts_df["timestamp"][::-1]]),
            y=pd.concat([road_ts_df[upper_col], road_ts_df[lower_col][::-1]]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Prediction Interval",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=road_ts_df["timestamp"],
            y=road_ts_df["actual_volume"],
            mode="lines",
            name="Actual Observed Volume",
            line=dict(color="#f8fafc", width=2),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=road_ts_df["timestamp"],
            y=road_ts_df["pred_volume"],
            mode="lines",
            name="Classical ML Forecast",
            line=dict(color="#38bdf8", width=2, dash="dot"),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=road_ts_df["timestamp"],
            y=road_ts_df["bilstm_pred_volume"],
            mode="lines",
            name="PyTorch BiLSTM Forecast",
            line=dict(color="#c084fc", width=1.8),
        )
    )
    fig_fc.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.45)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
        xaxis=dict(title="Timestamp (UTC)", gridcolor="rgba(148, 163, 184, 0.12)"),
        yaxis=dict(title="Vehicles / 30-Min Window", gridcolor="rgba(148, 163, 184, 0.12)"),
        title=dict(
            text=f"{sel_road} ({road_names[sel_road]}) — Actual vs Classical & BiLSTM Forecasts (90% Confidence Band)",
            y=0.96,
        ),
        margin=dict(l=20, r=20, t=52, b=68),
        height=440,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_fc, width="stretch")

    in_band = (
        (road_ts_df["actual_volume"] >= road_ts_df[lower_col])
        & (road_ts_df["actual_volume"] <= road_ts_df[upper_col])
    ).mean() * 100
    st.info(
        f"**Prediction Confidence Audit ({sel_road}):** Empirical 90% uncertainty band achieves **{in_band:.1f}% actual coverage** "
        f"over the selected {horizon_days}-day test horizon. Mean congestion classification confidence: "
        f"**{road_ts_df['congestion_confidence'].mean()*100:.1f}%**."
    )

# =============================================================================
# TAB 2: CONGESTION HEATMAP (VIEW 3) & ROAD COMPARISON (VIEW 4)
# =============================================================================
with tab_heat:
    st.subheader("3. Corridor Congestion Severity Heatmap — Segment × Hour of Day (View 3)")

    hm_day_filter = st.radio(
        "Day Type Filter:", ["All Days", "Weekdays (Mon–Fri)", "Weekends (Sat–Sun)"], horizontal=True
    )
    hm_src = preds_df.copy()
    if hm_day_filter.startswith("Weekdays"):
        hm_src = hm_src[~hm_src["day_of_week"].isin(["Saturday", "Sunday"])]
    elif hm_day_filter.startswith("Weekends"):
        hm_src = hm_src[hm_src["day_of_week"].isin(["Saturday", "Sunday"])]

    hm_src["sev_num"] = hm_src["pred_congestion"].map(SEVERITY_SCORE)
    hm_pivot = hm_src.pivot_table(
        index="road_id", columns="hour", values="sev_num", aggfunc="mean"
    ).sort_index()

    fig_hm = px.imshow(
        hm_pivot,
        color_continuous_scale=[
            [0.0, "#10b981"],
            [0.33, "#f59e0b"],
            [0.66, "#f97316"],
            [1.0, "#ef4444"],
        ],
        zmin=0,
        zmax=3,
        aspect="auto",
        labels=dict(x="Hour of Day (0–23)", y="Road Segment", color="Avg Severity (0–3)"),
        title=f"Predicted Congestion Severity Matrix ({hm_day_filter}) — 0=Free-flow · 1=Moderate · 2=Heavy · 3=Severe",
    )
    fig_hm.update_layout(**PLOT_LAYOUT, height=540)
    st.plotly_chart(fig_hm, width="stretch")

    st.markdown("---")
    st.subheader("4. Multi-Segment Road Comparison & Reliability Benchmark (View 4)")

    road_list = sorted(clean_df["road_id"].unique().tolist())
    sel_roads_cmp = st.multiselect(
        "Select Road Segments to Compare Side-by-Side:",
        options=road_list,
        default=road_list[:6],
    )
    if sel_roads_cmp:
        cmp_df = clean_df[clean_df["road_id"].isin(sel_roads_cmp)]
        road_summary = (
            cmp_df.groupby(["road_id", "road_name"])
            .agg(
                Avg_Volume=("traffic_volume", "mean"),
                Peak_Volume=("traffic_volume", lambda s: np.percentile(s, 95)),
                Avg_Speed=("avg_speed", "mean"),
                Avg_Travel_Time=("travel_time", "mean"),
                TT_Reliability_P95=("travel_time", lambda s: np.percentile(s, 95)),
                Severe_Heavy_Pct=("congestion_level", lambda s: s.isin(["Heavy", "Severe"]).mean() * 100),
                Accident_Rate_Pct=("accident_count", lambda s: (s > 0).mean() * 100),
            )
            .reset_index()
            .round(2)
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_rc1 = px.bar(
                road_summary,
                x="road_id",
                y=["Avg_Volume", "Peak_Volume"],
                barmode="group",
                color_discrete_sequence=["#38bdf8", "#818cf8"],
                title="Average vs 95th-Percentile Peak Volume by Segment",
                labels={"value": "Vehicles / 30-Min", "variable": "Metric"},
            )
            fig_rc1.update_layout(**PLOT_LAYOUT, height=370)
            st.plotly_chart(fig_rc1, width="stretch")

        with rc2:
            fig_rc2 = px.bar(
                road_summary,
                x="road_id",
                y=["Avg_Speed", "Severe_Heavy_Pct"],
                barmode="group",
                color_discrete_sequence=["#34d399", "#ef4444"],
                title="Mean Speed (km/h) vs Heavy/Severe Congestion Share (%)",
                labels={"value": "Speed / Congestion %", "variable": "Metric"},
            )
            fig_rc2.update_layout(**PLOT_LAYOUT, height=370)
            st.plotly_chart(fig_rc2, width="stretch")

        st.dataframe(road_summary, width="stretch", hide_index=True)

# =============================================================================
# TAB 3: HISTORICAL TRENDS (VIEW 2) & WEATHER VS TRAFFIC (VIEW 9)
# =============================================================================
with tab_hist:
    st.subheader("5. Historical Corridor Trends & Peak-Hour Patterns (View 2)")

    hist_road = st.selectbox(
        "Inspect Segment Historical Trend:",
        ["All Corridor (Average)"] + sorted(clean_df["road_id"].unique().tolist()),
    )
    hist_src = clean_df if hist_road.startswith("All") else clean_df[clean_df["road_id"] == hist_road]
    hist_src = hist_src.copy()
    hist_src["hour"] = hist_src["timestamp"].dt.hour
    hist_src["day_type"] = np.where(
        hist_src["day_of_week"].isin(["Saturday", "Sunday"]), "Weekend", "Weekday"
    )

    daily = (
        hist_src.groupby("date")
        .agg(avg_volume=("traffic_volume", "mean"), avg_speed=("avg_speed", "mean"))
        .reset_index()
    )
    hourly = (
        hist_src.groupby(["hour", "day_type"])
        .agg(avg_volume=("traffic_volume", "mean"), avg_speed=("avg_speed", "mean"))
        .reset_index()
    )

    ht1, ht2 = st.columns(2)
    with ht1:
        fig_daily = px.line(
            daily,
            x="date",
            y="avg_volume",
            title=f"Daily Mean 30-Min Traffic Volume ({hist_road})",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_daily.update_layout(**PLOT_LAYOUT, height=370, yaxis_title="Vehicles / 30-Min")
        st.plotly_chart(fig_daily, width="stretch")

    with ht2:
        fig_peak = px.line(
            hourly,
            x="hour",
            y="avg_volume",
            color="day_type",
            markers=True,
            color_discrete_map={"Weekday": "#f97316", "Weekend": "#34d399"},
            title=f"Diurnal Peak-Hour Profile: Weekday vs Weekend ({hist_road})",
        )
        fig_peak.update_layout(
            **PLOT_LAYOUT, height=370, xaxis_title="Hour of Day", yaxis_title="Avg Volume"
        )
        st.plotly_chart(fig_peak, width="stretch")

    st.markdown("---")
    st.subheader("6. Weather vs Traffic Flow, Speed & Incident Risk (View 9)")

    w_stats = (
        clean_df.groupby("weather_condition")
        .agg(
            Windows=("traffic_volume", "count"),
            Avg_Volume=("traffic_volume", "mean"),
            Avg_Speed=("avg_speed", "mean"),
            Avg_Travel_Time=("travel_time", "mean"),
            Accident_Rate_Pct=("accident_count", lambda s: (s > 0).mean() * 100),
            Avg_Visibility_m=("visibility", "mean"),
        )
        .reset_index()
        .round(2)
    )

    wc1, wc2 = st.columns(2)
    with wc1:
        fig_w1 = px.bar(
            w_stats,
            x="weather_condition",
            y=["Avg_Speed", "Accident_Rate_Pct"],
            barmode="group",
            color_discrete_sequence=["#38bdf8", "#ef4444"],
            title="Weather Impact: Mean Speed (km/h) vs Accident Rate (%)",
        )
        fig_w1.update_layout(**PLOT_LAYOUT, height=370)
        st.plotly_chart(fig_w1, width="stretch")

    with wc2:
        rain_sample = preds_df[preds_df["rainfall"] > 0].sample(
            min(1500, int((preds_df["rainfall"] > 0).sum())), random_state=42
        )
        fig_w2 = px.scatter(
            rain_sample,
            x="rainfall",
            y="actual_volume",
            color="pred_congestion",
            color_discrete_map=SEVERITY_COLORS,
            opacity=0.65,
            title="Observed Volume vs Hourly Rainfall Intensity (Rain Windows)",
            labels={"rainfall": "Rainfall (mm/hr)", "actual_volume": "Observed Volume"},
        )
        fig_w2.update_layout(**PLOT_LAYOUT, height=370)
        st.plotly_chart(fig_w2, width="stretch")

    st.dataframe(w_stats, width="stretch", hide_index=True)

# =============================================================================
# TAB 4: MODEL PERFORMANCE (VIEW 5) & FEATURE IMPORTANCE (VIEW 6)
# =============================================================================
with tab_perf:
    st.subheader("7. Classical ML vs PyTorch Deep Learning Benchmark (View 5)")

    st.markdown("**Head-to-Head Test Set Comparison (Exact Same Chronological Test Split: 25,925 Windows)**")
    st.dataframe(format_arrow_safe_table(bench_df), width="stretch", hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        vol_b = bench_df[bench_df["Task"] == "Traffic Volume"].sort_values("RMSE")
        fig_vb = px.bar(
            vol_b,
            x="RMSE",
            y="Model",
            color="Family",
            orientation="h",
            text="RMSE",
            color_discrete_map={
                "Classical ML": "#38bdf8",
                "Deep Learning (PyTorch)": "#c084fc",
            },
            title="Volume Regression Benchmark — Test RMSE (Lower is Better)",
        )
        fig_vb.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_vb.update_layout(**PLOT_LAYOUT, height=380)
        st.plotly_chart(fig_vb, width="stretch")

    with p2:
        dl_h = data["dl_hist"]
        fig_loss = px.line(
            dl_h,
            x="epoch",
            y=["train_loss", "val_loss"],
            facet_col="model",
            markers=True,
            color_discrete_map={"train_loss": "#38bdf8", "val_loss": "#f59e0b"},
            title="PyTorch Sequence Models — Train vs Val Loss Curves",
        )
        fig_loss.update_layout(**PLOT_LAYOUT, height=380)
        st.plotly_chart(fig_loss, width="stretch")

    d1, d2, d3 = st.columns(3)
    with d1:
        fig_cm = px.imshow(
            data["cm_df"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Congestion Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(**PLOT_LAYOUT, height=340)
        st.plotly_chart(fig_cm, width="stretch")

    with d2:
        roc_df = data["roc_df"]
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                name="Accident Risk ROC",
                line=dict(color="#f43f5e", width=2.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Baseline",
                line=dict(color="#64748b", dash="dash"),
            )
        )
        fig_roc.update_layout(
            **PLOT_LAYOUT,
            height=340,
            title="Accident Risk ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, width="stretch")

    with d3:
        gd_h = data["gd_hist"]
        fig_gd = px.line(
            gd_h,
            x="epoch",
            y="train_mse",
            color_discrete_sequence=["#34d399"],
            title="NumPy GD MSE Convergence",
        )
        fig_gd.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Epoch", yaxis_title="Train MSE")
        st.plotly_chart(fig_gd, width="stretch")

    st.markdown("**Error Analysis by Hour of Day & Weather Condition (PRD Section 13.3)**")
    ea1, ea2 = st.columns(2)
    with ea1:
        err_hr = pd.DataFrame(data["err_analysis"]["by_hour"])
        fig_eh = px.bar(
            err_hr,
            x="hour",
            y="rmse",
            color="mae",
            color_continuous_scale="Plasma",
            title="Volume Forecast RMSE by Hour of Day",
        )
        fig_eh.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig_eh, width="stretch")
    with ea2:
        err_w = pd.DataFrame(data["err_analysis"]["by_weather"])
        fig_ew = px.bar(
            err_w,
            x="weather_condition",
            y="rmse",
            text="mae",
            color="weather_condition",
            title="Volume Forecast RMSE by Weather Condition",
        )
        fig_ew.update_traces(texttemplate="MAE: %{text:.1f}", textposition="outside")
        fig_ew.update_layout(**PLOT_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_ew, width="stretch")

    st.markdown("---")
    st.subheader("8. Ranked Feature Importance Drivers (View 6)")
    fi_task = st.selectbox("Select Prediction Task:", list(data["fi_map"].keys()))
    fi_df = data["fi_map"][fi_task].head(15).sort_values("importance", ascending=True)
    fig_fi = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
        title=f"Top 15 Predictive Drivers — {fi_task}",
    )
    fig_fi.update_layout(**PLOT_LAYOUT, height=430, yaxis_title="")
    st.plotly_chart(fig_fi, width="stretch")

# =============================================================================
# TAB 5: SUPPORTING MODULES (PRD 14.2: PREDICTOR, UPLOAD, RETRAIN, REPORTS)
# =============================================================================
with tab_tools:
    st.subheader("Interactive Multi-Horizon Forecast Simulator (30–120 Min Ahead)")
    artifacts = load_inference_models()
    tail_df = artifacts["feat_tail"]

    sim1, sim2, sim3, sim4 = st.columns(4)
    with sim1:
        sim_road = st.selectbox("Segment:", sorted(tail_df["road_id"].unique().tolist()), key="sim_r")
    with sim2:
        sim_horizon = st.selectbox(
            "Forecast Horizon:",
            [1, 2, 3, 4],
            format_func=lambda h: f"+{h * 30} min ({h} window{'s' if h > 1 else ''})",
        )
    with sim3:
        sim_weather = st.selectbox("Scenario Weather:", ["Clear", "Cloudy", "Rain", "Fog"])
    with sim4:
        sim_rain = st.slider("Rainfall (mm/hr):", 0.0, 10.0, 3.5 if sim_weather == "Rain" else 0.0, 0.5)

    sim5, sim6, sim7 = st.columns(3)
    with sim5:
        sim_vis = st.slider(
            "Visibility (m):",
            300,
            10000,
            1200 if sim_weather == "Fog" else (3500 if sim_weather == "Rain" else 10000),
            100,
        )
    with sim6:
        sim_event = st.checkbox("Major Local Event Active", value=False)
    with sim7:
        sim_holiday = st.checkbox("Public Holiday", value=False)

    if st.button("Run Real-Time Multi-Horizon Forecast", type="primary"):
        base_row = tail_df[tail_df["road_id"] == sim_road].iloc[[0]].copy()
        rows_out = []
        curr_row = base_row.copy()
        for step in range(1, sim_horizon + 1):
            curr_row["timestamp"] = pd.to_datetime(curr_row["timestamp"], utc=True) + pd.Timedelta(minutes=30)
            ts = curr_row["timestamp"].iloc[0]
            curr_row["hour"] = ts.hour
            curr_row["minute"] = ts.minute
            curr_row["time_slot"] = ts.hour * 2 + (ts.minute // 30)
            h_frac = ts.hour + ts.minute / 60.0
            curr_row["hour_sin"] = np.sin(2.0 * np.pi * h_frac / 24.0)
            curr_row["hour_cos"] = np.cos(2.0 * np.pi * h_frac / 24.0)
            curr_row["is_morning_peak"] = int(ts.hour in [7, 8, 9, 10])
            curr_row["is_evening_peak"] = int(ts.hour in [17, 18, 19, 20])
            curr_row["is_peak"] = int(
                curr_row["is_morning_peak"].iloc[0] or curr_row["is_evening_peak"].iloc[0]
            )
            curr_row["rainfall"] = sim_rain
            curr_row["visibility"] = sim_vis
            curr_row["is_rain"] = int(sim_weather == "Rain" or sim_rain > 0)
            curr_row["is_fog"] = int(sim_weather == "Fog")
            curr_row["is_low_visibility"] = int(sim_vis < 2000)
            for cond in ["Clear", "Cloudy", "Fog", "Rain"]:
                curr_row[f"weather_{cond.lower()}"] = int(sim_weather == cond)
            curr_row["public_holiday"] = int(sim_holiday)
            curr_row["event_flag"] = int(sim_event)
            curr_row["holiday_x_peak"] = int(sim_holiday and curr_row["is_peak"].iloc[0])
            curr_row["event_x_peak"] = int(sim_event and curr_row["is_peak"].iloc[0])
            curr_row["rain_x_peak"] = int(curr_row["is_rain"].iloc[0] and curr_row["is_peak"].iloc[0])
            curr_row["low_vis_x_peak"] = int(
                curr_row["is_low_visibility"].iloc[0] and curr_row["is_peak"].iloc[0]
            )

            res_step = predict_from_feature_frame(curr_row, contract, artifacts)
            res_step.insert(0, "Horizon", f"+{step * 30} min")
            res_step.insert(1, "Forecast Time", ts.strftime("%Y-%m-%d %H:%M"))
            rows_out.append(res_step)

            p_v = float(res_step["pred_volume"].iloc[0])
            curr_row["vol_lag_4"] = curr_row["vol_lag_3"]
            curr_row["vol_lag_3"] = curr_row["vol_lag_2"]
            curr_row["vol_lag_2"] = curr_row["vol_lag_1"]
            curr_row["vol_lag_1"] = p_v
            curr_row["vol_roll_mean_4"] = np.mean(
                [
                    curr_row["vol_lag_1"].iloc[0],
                    curr_row["vol_lag_2"].iloc[0],
                    curr_row["vol_lag_3"].iloc[0],
                    curr_row["vol_lag_4"].iloc[0],
                ]
            )
            curr_row["vc_ratio_lag_1"] = p_v / (curr_row["road_capacity"].iloc[0] / 2.0)

        st.dataframe(pd.concat(rows_out, ignore_index=True), width="stretch", hide_index=True)

    st.markdown("---")
    col_up, col_rep = st.columns(2)

    with col_up:
        st.subheader("Data-Upload & Batch Prediction Module (FR-17)")
        st.caption(
            "Upload a CSV containing engineered FlowCast feature rows (or download a sample template below) to score with persisted production models."
        )
        sample_csv = tail_df[["road_id", "timestamp"] + contract["stage1_features"]].head(10).to_csv(index=False)
        st.download_button(
            "⬇️ Download Sample Inference CSV (10 Segments)",
            data=sample_csv,
            file_name="flowcast_sample_upload.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            missing_cols = [c for c in contract["stage1_features"] if c not in up_df.columns]
            if missing_cols:
                st.error(
                    f"Schema validation failed! Missing {len(missing_cols)} required feature columns: {missing_cols[:8]}"
                )
            else:
                scored = predict_from_feature_frame(up_df, contract, artifacts)
                out_df = pd.concat(
                    [
                        up_df[[c for c in ["road_id", "timestamp"] if c in up_df.columns]].reset_index(
                            drop=True
                        ),
                        scored,
                    ],
                    axis=1,
                )
                st.success(f"Validated & scored {len(out_df):,} records successfully!")
                st.dataframe(out_df.head(25), width="stretch", hide_index=True)
                st.download_button(
                    "⬇️ Download Scored Predictions CSV",
                    data=out_df.to_csv(index=False),
                    file_name="flowcast_batch_predictions.csv",
                    mime="text/csv",
                )

    with col_rep:
        st.subheader("Reports & Insights Exporter (FR-18)")
        min_d = preds_df["timestamp"].dt.date.min()
        max_d = preds_df["timestamp"].dt.date.max()
        date_range = st.date_input("Select Report Date Range:", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            r_df = preds_df[
                (preds_df["timestamp"].dt.date >= date_range[0])
                & (preds_df["timestamp"].dt.date <= date_range[1])
            ]
        else:
            r_df = preds_df

        top_risk = (
            r_df.groupby(["road_id", "road_name"])["pred_accident_prob"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        top_cong = (
            r_df.groupby(["road_id", "road_name"])["pred_congestion"]
            .apply(lambda s: s.isin(["Heavy", "Severe"]).mean() * 100)
            .sort_values(ascending=False)
            .head(5)
        )

        report_md = "\n".join(
            [
                "# FlowCast Corridor Operational Insights Summary",
                f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"- **Date Range:** {date_range[0] if isinstance(date_range, tuple) else min_d} to {date_range[-1] if isinstance(date_range, tuple) else max_d}",
                f"- **Total Evaluated Windows:** {len(r_df):,}",
                f"- **Mean Predicted Volume:** {r_df['pred_volume'].mean():.1f} veh/30-min",
                f"- **Mean Predicted Travel Time:** {r_df['pred_travel_time'].mean():.2f} min",
                f"- **Heavy/Severe Congestion Share:** {r_df['pred_congestion'].isin(['Heavy', 'Severe']).mean()*100:.2f}%",
                "",
                "## Top 5 High-Congestion Segments",
                *[f"- `{rid}` ({rname}): {val:.1f}% Heavy/Severe windows" for (rid, rname), val in top_cong.items()],
                "",
                "## Top 5 Elevated Accident-Risk Segments",
                *[f"- `{rid}` ({rname}): {val*100:.1f} / 100 avg risk score" for (rid, rname), val in top_risk.items()],
            ]
        )
        st.download_button(
            "📄 Export Operational Insights Report (.md)",
            data=report_md,
            file_name="flowcast_insights_summary.md",
            mime="text/markdown",
        )
        st.download_button(
            "📊 Export Filtered Predictions (.csv)",
            data=r_df.to_csv(index=False),
            file_name="flowcast_filtered_predictions.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Model Governance Cards, Pipeline Retraining & Data-Quality Audit")
    g1, g2 = st.columns([1.2, 1.3])
    with g1:
        card_choice = st.selectbox("Inspect Model Card (PRD Section 13.4):", list(data["model_cards"].keys()))
        st.json(data["model_cards"][card_choice], expanded=False)

        if st.button("🔄 Trigger Data Ingestion + Feature Refresh (Quick Check)"):
            with st.spinner("Running M1 Schema & Quarantine Validation..."):
                res = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "src/ingest.py")],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )
            if res.returncode == 0:
                st.success(
                    "M1 Schema & Quarantine Check passed! To retrain full ML/DL pipeline, run `python run_pipeline.py`."
                )
            else:
                st.error(res.stderr)
    with g2:
        with st.expander("📋 View Full Data-Quality & Cleaning Audit Report (reports/data_quality.md)", expanded=True):
            st.markdown(data["dq_text"])