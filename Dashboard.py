"""
app.py — Streamlit Dashboard for Autonomous Network Operations.
Displays live telemetry, agent traces, human-in-the-loop approval,
and a LIVE view of network_config.json (the Digital Twin).
Hardcoded 10s auto-refresh.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import time
import json
from datetime import datetime, timedelta
from streamlit_agraph import agraph, Node, Edge, Config

# ─────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
REFRESH_SEC = 5

st.set_page_config(page_title="Axios", page_icon="N", layout="wide", initial_sidebar_state="expanded")

if "last_approval_action" not in st.session_state:
    st.session_state.last_approval_action = None

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    .stApp { font-family: 'Inter', sans-serif; background-color: #131314 !important; }
    section[data-testid="stSidebar"] { background-color: #1E1F20 !important; }
    header[data-testid="stHeader"] { background-color: #131314 !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span, .stApp label, .stApp li { color: #E3E3E3; }
    .stApp .stCaption, .stApp small { color: #C4C7C5 !important; }
    .stMarkdown hr { border-color: #444746 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1E1F20; }
    ::-webkit-scrollbar-thumb { background: #444746; border-radius: 3px; }

    /* ── Header ── */
    .main-header {
        background: #1E1F20; border: 1px solid #444746;
        padding: 1.2rem 1.8rem; border-radius: 10px; margin-bottom: 1.2rem; color: #E3E3E3;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; color: #E3E3E3; }
    .main-header p  { margin: 0.3rem 0 0 0; color: #C4C7C5; font-size: 0.85rem; }

    /* ── Agent Trace Steps ── */
    .trace-step {
        background: #1E1F20; border-left: 3px solid #444746; padding: 0.7rem 1rem; margin-bottom: 0.4rem;
        border-radius: 0 8px 8px 0; font-family: 'Inter', monospace; font-size: 0.8rem; color: #C4C7C5;
    }
    .trace-step.observe     { border-left-color: #A8C7FA; }
    .trace-step.retrieve    { border-left-color: #8AB4F8; }
    .trace-step.investigate { border-left-color: #A8C7FA; }
    .trace-step.reason      { border-left-color: #FDE293; }
    .trace-step.human       { border-left-color: #F28B82; }
    .trace-step.act         { border-left-color: #81C995; }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: #1E1F20; border: 1px solid #444746; border-radius: 10px; padding: 1rem;
    }
    div[data-testid="stMetric"] label { color: #C4C7C5 !important; font-size: 0.8rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #E3E3E3 !important; }

    /* ── Approval Box ── */
    .approval-box {
        background: rgba(242, 139, 130, 0.05); border: 1px solid rgba(242, 139, 130, 0.3);
        border-radius: 10px; padding: 1.5rem; margin: 1rem 0; color: #E3E3E3;
    }
    .approval-box h3 { color: #F28B82; margin: 0 0 0.8rem 0; font-size: 1rem; font-weight: 600; }
    .approval-box p { color: #C4C7C5; }
    .approval-box code { background: #131314; color: #A8C7FA; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }

    /* ── Config Viewer ── */
    .config-viewer {
        background: #131314; border: 1px solid #444746; border-radius: 8px; padding: 1rem;
        font-family: 'Inter', monospace; font-size: 0.75rem; color: #C4C7C5;
        max-height: 350px; overflow-y: auto;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #131314 !important; border: 1px solid #444746 !important;
        color: #A8C7FA !important; border-radius: 20px !important; font-weight: 500;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1E1F20 !important; }
    .stButton > button[kind="primary"] {
        background: #A8C7FA !important; color: #000000 !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover { background: #8AB4F8 !important; }

    /* ── Expander ── */
    details { background: #1E1F20 !important; border: 1px solid #444746 !important; border-radius: 8px !important; }
    details summary { color: #E3E3E3 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { border-bottom-color: #444746; }
    .stTabs [data-baseweb="tab"] { color: #C4C7C5; }
    .stTabs [aria-selected="true"] { color: #A8C7FA !important; border-bottom-color: #A8C7FA !important; }

    /* ── Agraph Topology ── */
    iframe[title="streamlit_agraph.agraph"] { background: #131314 !important; }
    [data-testid="stCustomComponentV1"] { background: #131314 !important; border-radius: 10px; }
    .react-graph-vis { background: #131314 !important; }
    canvas { background: #131314 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fetch_api(endpoint, method="GET", payload=None):
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "POST":
            return requests.post(url, json=payload, timeout=30).json()
        return requests.get(url, timeout=5).json()
    except Exception as e:
        return {"error": str(e)}

def trace_class(line):
    u = line.upper()
    if "OBSERVE" in u: return "observe"
    if "RETRIEV" in u: return "retrieve"
    if "INVESTIGAT" in u: return "investigate"
    if "REASON" in u: return "reason"
    if "HUMAN" in u or "APPROVAL" in u: return "human"
    if "EXECUTOR" in u or "ACT" in u or "ACTION" in u: return "act"
    if "ERROR" in u: return "human"
    return ""


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Network Controllers")
    st.markdown("🔮 **Predictive Forecasting**: Live Linear Regression active")

    # ─── LIVE ROUTER CONFIG (Digital Twin view) ───
    config_data = fetch_api("/network-config")
    if config_data and "error" not in config_data:
        for router_name, router_state in config_data.items():
            status = router_state.get("status", "online")
            route = router_state.get("current_route", "?")
            flags = [k for k, v in router_state.items() if isinstance(v, bool) and v]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{router_name}**")
                st.caption(f"Status: {status} | Route: {route}")
                if flags:
                    st.caption(f"Flags: {', '.join(flags)}")
            with c2:
                if st.button("Reset", key=f"hr_{router_name}"):
                    fetch_api("/api/resolve/hard_reset", "POST", {"router_name": router_name})
                    st.rerun()
    else:
        st.warning("Cannot read config.")

    st.markdown("---")

    # Raw JSON view
    with st.expander("Raw JSON"):
        if config_data and "error" not in config_data:
            st.markdown(f'<div class="config-viewer"><pre>{json.dumps(config_data, indent=2)}</pre></div>', unsafe_allow_html=True)

    st.caption(f"Auto-refresh: {REFRESH_SEC}s")


# ─────────────────────────────────────────────
# Header (Custom Typography)
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding-top: 2rem; padding-bottom: 2rem; font-family: 'Inter', 'Segoe UI', sans-serif;">
    <h1 style="font-weight: 800; font-size: 4.5rem; letter-spacing: 2px; margin: 0; line-height: 1; color: #8AB4F8;">
        ΛXIOS
    </h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Live Topology Map (Streamlit Agraph)
# ─────────────────────────────────────────────
st.markdown("### Live Network Topology")
config_data = fetch_api("/network-config")

if config_data and "error" not in config_data:
    nodes = []
    edges = []
    
    # Add Internet Node
    nodes.append(Node(id="Internet", label="Internet / Core-Cloud", size=25, color="#A8C7FA"))
    
    for router_name, r_state in config_data.items():
        status = r_state.get("status", "online")
        flags = [k for k, v in r_state.items() if isinstance(v, bool) and v]
        
        if status == "rebooting":
            color = "#444746"
        elif status == "offline" or flags:
            color = "#FF0000"
        elif "Edge" in router_name:
            color = "#81C995"
        else:
            color = "#8AB4F8"
            
        nodes.append(Node(id=router_name, label=router_name, size=20, color=color))
        
        route = r_state.get("current_route", "Primary-Link-A")
        
        if "-via-" in route:
            target = route.split("-via-")[1]
            edge_label = "BACKUP" if "Backup" in route else "PRIMARY"
            edge_color = "#81C995" if "Backup" in route else ("#FDE293" if r_state.get("is_congested") else "#444746")
            edge_width = 2 if "Backup" in route else (3 if r_state.get("is_congested") else 2)
            edges.append(Edge(source=router_name, target=target, label=edge_label, color=edge_color, width=edge_width))
        else:
            edge_color = "#FDE293" if r_state.get("is_congested") else "#444746"
            edge_width = 3 if r_state.get("is_congested") else 2
            if "Edge" in router_name:
                fallback_target = "Core-Router-Delhi" if "North" in router_name else "Core-Router-Chennai"
                edges.append(Edge(source=router_name, target=fallback_target, label="PRIMARY", color=edge_color, width=edge_width))
            else:
                edges.append(Edge(source=router_name, target="Internet", label="PRIMARY", color=edge_color, width=edge_width))
                
    config = Config(height=450, width="100%", directed=True, nodeHighlightBehavior=True, highlightColor="#A8C7FA", collapsible=False, interaction={"zoomView": False, "dragView": False}, physics={"enabled": False})
    
    agraph(nodes=nodes, edges=edges, config=config)
else:
    st.warning("Cannot load topology configuration.")


# ─────────────────────────────────────────────
# Human-in-the-Loop Approval Panel
# ─────────────────────────────────────────────
pending = fetch_api("/api/pending-approvals")
if pending and pending.get("count", 0) > 0:
    for item in pending.get("pending", []):
        tid = item.get("thread_id", "?")
        st.markdown(f"""<div class="approval-box">
            <h3>HIGH-RISK ACTION — Awaiting NOC Approval</h3>
            <p><strong>Action:</strong> {item.get('action', 'N/A')}</p>
            <p><strong>Args:</strong> <code>{item.get('action_args', '{}')}</code></p>
            <p><strong>Router:</strong> {item.get('anomaly', {}).get('router', 'N/A')}</p>
            <p><strong>Metric:</strong> {item.get('anomaly', {}).get('metric', 'N/A')} = {item.get('anomaly', {}).get('value', 'N/A')}</p>
            <p><strong>Time:</strong> {item.get('timestamp', 'N/A')}</p>
        </div>""", unsafe_allow_html=True)

        for log_line in item.get("logs", []):
            cls = trace_class(log_line)
            st.markdown(f'<div class="trace-step {cls}">{log_line}</div>', unsafe_allow_html=True)

        ca, cr = st.columns(2)
        with ca:
            if st.button("APPROVE", key=f"a_{tid}", type="primary", use_container_width=True):
                fetch_api("/api/approve", "POST", {"thread_id": tid})
                st.session_state.last_approval_action = f"Approved: {tid}"
                st.rerun()
        with cr:
            if st.button("REJECT", key=f"r_{tid}", use_container_width=True):
                fetch_api("/api/reject", "POST", {"thread_id": tid})
                st.session_state.last_approval_action = f"Rejected: {tid}"
                st.rerun()

if st.session_state.last_approval_action:
    if "Approved" in st.session_state.last_approval_action:
        st.success(f"{st.session_state.last_approval_action}")
    else:
        st.warning(f"{st.session_state.last_approval_action}")
    st.session_state.last_approval_action = None


# ─────────────────────────────────────────────
# Metrics + Chart
# ─────────────────────────────────────────────
st.markdown("### Network Health")
st.markdown("**Anomaly Detection:** Powered by Custom Random Forest ML Model")

full_data = fetch_api("/telemetry?limit=100")

if full_data and "data" in full_data and full_data["data"]:
    pts = full_data["data"]
    latest = pts[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{latest.get('latency_ms', 0):.1f} ms")
    c2.metric("Pkt Loss", f"{latest.get('packet_loss_pct', 0):.1f} %")
    c3.metric("CPU", f"{latest.get('cpu_utilization_pct', 0):.1f} %")
    c4.metric("BGP Flaps", f"{latest.get('bgp_flaps_per_min', 0)}")

    st.markdown("### Live Telemetry (IST)")

    # Filter to last 1 hour
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    filtered_pts = []
    for p in pts:
        try:
            raw = p["timestamp"].replace(" IST", "").strip()
            ts = datetime.fromisoformat(raw)
            if ts >= one_hour_ago:
                filtered_pts.append(p)
        except (ValueError, TypeError):
            filtered_pts.append(p)

    chart_pts = filtered_pts if filtered_pts else pts
    # Extract time-only strings for x-axis
    time_labels = []
    for p in chart_pts:
        try:
            raw = p["timestamp"].replace(" IST", "").strip()
            ts = datetime.fromisoformat(raw)
            time_labels.append(ts.strftime("%H:%M:%S"))
        except (ValueError, TypeError):
            time_labels.append(p["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_labels, y=[p["latency_ms"] for p in chart_pts],
                             mode="lines", name="Latency (ms)", line=dict(color="#A8C7FA", width=2)))
    
    if any("predicted_latency" in p for p in chart_pts):
        fig.add_trace(go.Scatter(x=time_labels, y=[p.get("predicted_latency", p.get("latency_ms", 0)) for p in chart_pts],
                                 mode="lines", name="Predicted (t+5)", line=dict(color="#FFFFFF", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=time_labels, y=[p["packet_loss_pct"] for p in chart_pts],
                             mode="lines", name="Pkt Loss (%)", line=dict(color="#F28B82", width=2)))
    fig.add_trace(go.Scatter(x=time_labels, y=[p.get("cpu_utilization_pct", 0) for p in chart_pts],
                             mode="lines", name="CPU (%)", line=dict(color="#FDE293", width=2)))

    # Current date annotation in top-right
    today_str = now.strftime("%d %b %Y")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#C4C7C5"),
        xaxis=dict(title="Time (IST)", tickangle=-45, nticks=12, gridcolor="rgba(255,255,255,0.05)", gridwidth=1, showgrid=False, linecolor="#444746", tickfont=dict(color="#C4C7C5", size=11), type="category"),
        yaxis=dict(title="Value", gridcolor="rgba(255,255,255,0.05)", gridwidth=1, showgrid=True, linecolor="#444746", tickfont=dict(color="#C4C7C5", size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#C4C7C5", size=11)),
        annotations=[dict(
            text=today_str, xref="paper", yref="paper", x=1, y=1.12,
            showarrow=False, font=dict(size=11, color="#C4C7C5"),
            xanchor="right", yanchor="bottom"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for telemetry... Start the FastAPI server first.")


# ─────────────────────────────────────────────
# Agent Action Log
# ─────────────────────────────────────────────
st.markdown("### Agent Action Log")
logs_data = fetch_api("/agent-logs")
if logs_data and logs_data.get("logs"):
    for i, entry in enumerate(reversed(logs_data["logs"][-5:])):
        ts = entry.get("timestamp", "")
        trigger = entry.get("trigger", "unknown")
        icons = {"auto_detection": "[auto]", "human_approved": "[approved]", "human_rejected": "[rejected]"}
        icon = icons.get(trigger, "[event]")

        with st.expander(f"{icon} {trigger} — {ts}", expanded=(i == 0)):
            if "error" in entry:
                st.error(f"```\n{entry['error']}\n```")
                continue

            result = entry.get("result", {})
            if result.get("error"):
                st.error(f"```\n{result['error']}\n```")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Action", result.get("recommended_action", "N/A"))
            mc2.metric("Risk", (result.get("risk_level") or "N/A").upper())
            mc3.metric("Status", result.get("status", "N/A"))

            for line in result.get("logs", []):
                cls = trace_class(line)
                st.markdown(f'<div class="trace-step {cls}">{line}</div>', unsafe_allow_html=True)

            ar = result.get("action_result", "")
            if ar:
                st.markdown(f'<div class="trace-step act"><strong>Result:</strong> {ar}</div>', unsafe_allow_html=True)
else:
    st.info("No agent runs yet. Inject an anomaly to trigger the agent.")


# ─────────────────────────────────────────────
# Hardcoded 10s auto-refresh
# ─────────────────────────────────────────────
time.sleep(REFRESH_SEC)
st.rerun()
