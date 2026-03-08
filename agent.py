"""
agent.py — LangGraph Network Operations Agent.
Tools write DIRECTLY to network_config.json (no HTTP middleman).
Human-in-the-loop: graph halts at human_approval for HIGH risk actions.
All timestamps IST. Full error tracebacks.
"""

import os
import json
import operator
import uuid
import traceback
import asyncio
import threading
import time as _time
from typing import Annotated, List, Optional, TypedDict
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")
def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S IST")

# ─────────────────────────────────────────────
# Config File I/O (same as in main.py — shared logic)
# ─────────────────────────────────────────────
from pathlib import Path
CONFIG_FILE = Path("network_config.json")
_config_lock = threading.Lock()

def read_config() -> dict:
    with _config_lock:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

def write_config(config: dict):
    with _config_lock:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

# ─────────────────────────────────────────────
# 1. LLM
# ─────────────────────────────────────────────
ACTION_CACHE = {}

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

# ─────────────────────────────────────────────
# 2. ChromaDB RAG
# ─────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "network_sops"
_vectorstore = None
_retriever = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        _vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _vectorstore

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
    return _retriever

# ─────────────────────────────────────────────
# 3a. Investigative Tools — Partial Observability & Blast Radius
# ─────────────────────────────────────────────

TOPOLOGY_FILE = Path("data/topology.json")

@tool
def run_device_diagnostics(router_name: str):
    """
    Runs diagnostics on a specific router. Opens network_config.json and extracts
    the status and anomaly flags ONLY for the given router_name.
    The agent MUST call this tool first under partial observability to discover
    the root cause before attempting any fixes.
    Args:
        router_name: The router to diagnose (e.g., 'Core-Router-Mumbai').
    Returns:
        JSON string with the router's current status and all anomaly flags.
    """
    try:
        config = read_config()
        if router_name not in config:
            return json.dumps({"error": f"Router '{router_name}' not found in network_config.json"})
        router_state = config[router_name]
        result = {
            "router": router_name,
            "status": router_state.get("status", "unknown"),
            "current_route": router_state.get("current_route", "unknown"),
            "flags": {
                "is_congested": router_state.get("is_congested", False),
                "bgp_down": router_state.get("bgp_down", False),
                "cpu_spiking": router_state.get("cpu_spiking", False),
                "interface_flapping": router_state.get("interface_flapping", False),
            }
        }
        active_flags = [k for k, v in result["flags"].items() if v]
        result["active_anomalies"] = active_flags if active_flags else ["none"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Diagnostics failed: {e}"})

@tool
def calculate_blast_radius(router_name: str):
    """
    Calculates the blast radius (downstream impact) of an issue on the given router.
    Reads data/topology.json to determine how many downstream nodes and links
    depend on this router. The agent MUST call this after diagnostics and BEFORE
    choosing a mitigation action.
    Args:
        router_name: The router to assess impact for (e.g., 'Core-Router-Mumbai').
    Returns:
        A string describing the impact level and affected downstream infrastructure.
    """
    try:
        with open(TOPOLOGY_FILE, "r", encoding="utf-8") as f:
            topology = json.load(f)
    except Exception:
        topology = {"routers": [], "links": []}

    # Find router type
    router_type = "unknown"
    for r in topology.get("routers", []):
        if r["name"] == router_name:
            router_type = r.get("type", "unknown")
            break

    # Count downstream nodes (targets where this router is the source)
    downstream_nodes = set()
    downstream_links = []
    for link in topology.get("links", []):
        if link["source"] == router_name:
            downstream_nodes.add(link["target"])
            downstream_links.append(link["name"])
        elif link["target"] == router_name:
            downstream_nodes.add(link["source"])
            downstream_links.append(link["name"])

    # Classify edge vs core downstream
    edge_count = sum(1 for n in downstream_nodes if "Edge" in n)
    core_count = sum(1 for n in downstream_nodes if "Core" in n)
    peering_count = sum(1 for n in downstream_nodes if "Internet" in n or "Peering" in n)

    # Impact classification
    if router_type == "core" and len(downstream_nodes) >= 3:
        impact = "CRITICAL"
    elif router_type == "core":
        impact = "HIGH"
    elif router_type == "edge":
        impact = "MODERATE"
    else:
        impact = "LOW"

    summary_parts = []
    if core_count:
        summary_parts.append(f"{core_count} core router(s)")
    if edge_count:
        summary_parts.append(f"{edge_count} edge router(s)")
    if peering_count:
        summary_parts.append(f"{peering_count} peering link(s)")

    downstream_desc = ", ".join(summary_parts) if summary_parts else "no directly connected nodes"

    return (
        f"Blast Radius Assessment [{now_ist()}]:\n"
        f"  Router: {router_name} (type: {router_type})\n"
        f"  Impact Level: {impact}\n"
        f"  Connected Infrastructure: {downstream_desc}\n"
        f"  Affected Links: {', '.join(downstream_links) if downstream_links else 'none'}\n"
        f"  Downstream Nodes: {', '.join(sorted(downstream_nodes)) if downstream_nodes else 'none'}"
    )

# ─────────────────────────────────────────────
# 3b. Mitigation Tools — DIRECT FILE WRITES to network_config.json
# ─────────────────────────────────────────────

@tool
def reroute_traffic(source_router: str, target_router: str):
    """
    Reroutes traffic from a congested router to a backup path.
    WRITES to network_config.json: changes current_route to Backup-Link-B, clears is_congested.
    Use this for: congestion, high latency.
    Args:
        source_router: The failing router (e.g., 'Core-Router-Mumbai').
        target_router: The backup router (e.g., 'Core-Router-Bangalore').
    """
    try:
        config = read_config()
        if source_router not in config:
            return f"ACTION FAILED [{now_ist()}]: Router '{source_router}' not found in network_config.json"
            
        valid_routers = list(config.keys())
        if target_router not in valid_routers:
            return f"ACTION FAILED [{now_ist()}]: target '{target_router}' does not exist in the network. Valid routers are: {valid_routers}"
        
        backup_route = f"Backup-via-{target_router}"
        config[source_router]["current_route"] = backup_route
        config[source_router]["is_congested"] = False
        config[source_router]["interface_flapping"] = False
        config[source_router]["status"] = "online"
        write_config(config)
        
        return f"ACTION SUCCESS [{now_ist()}]: network_config.json updated — {source_router} route changed to '{backup_route}'. Congestion cleared."
    except Exception as e:
        return f"ACTION FAILED [{now_ist()}]: {e}"

@tool
def restart_interface(router: str, interface: str):
    """
    Restarts a router's interface. Sets status='rebooting' in network_config.json, 
    waits 5 seconds, then sets status='online' and clears ALL anomaly flags.
    Use this for: cpu_spike, interface_flapping, hardware degradation.
    Args:
        router: Router name (e.g., 'Core-Router-Mumbai').
        interface: Interface ID (e.g., 'Gi0/1').
    """
    try:
        config = read_config()
        if router not in config:
            return f"ACTION FAILED [{now_ist()}]: Router '{router}' not found in network_config.json"
        
        # Phase 1: Mark as rebooting
        config[router]["status"] = "rebooting"
        write_config(config)
        
        # Phase 2: Simulate boot time
        _time.sleep(5)
        
        # Phase 3: Come back online, clear everything
        config = read_config()
        config[router]["status"] = "online"
        config[router]["is_congested"] = False
        config[router]["bgp_down"] = False
        config[router]["cpu_spiking"] = False
        config[router]["interface_flapping"] = False
        config[router]["current_route"] = "Primary-Link-A"
        write_config(config)
        
        return f"ACTION SUCCESS [{now_ist()}]: network_config.json updated — {router} rebooted. Status: online. All flags cleared."
    except Exception as e:
        return f"ACTION FAILED [{now_ist()}]: {e}"

@tool
def adjust_qos(router: str, policy: str):
    """
    Applies a QoS policy to manage congestion or DDoS.
    WRITES to network_config.json: clears is_congested flag.
    Use this for: packet loss on edge routers, DDoS mitigation.
    Args:
        router: Router name (e.g., 'Edge-Router-Delhi').
        policy: Policy name (e.g., 'EDGE_PROTECT', 'VOIP_PRIORITY').
    """
    try:
        config = read_config()
        if router not in config:
            return f"ACTION FAILED [{now_ist()}]: Router '{router}' not found in network_config.json"
        
        config[router]["is_congested"] = False
        config[router]["status"] = "online"
        write_config(config)
        
        return f"ACTION SUCCESS [{now_ist()}]: network_config.json updated — QoS '{policy}' applied on {router}. Congestion cleared."
    except Exception as e:
        return f"ACTION FAILED [{now_ist()}]: {e}"

@tool
def reset_bgp_session(router: str, peer: str = "upstream"):
    """
    Resets a BGP session. WRITES to network_config.json: clears bgp_down flag.
    Use this for: bgp_down, routing flaps, 100% packet loss with BGP flaps.
    THIS IS HIGH RISK — requires human approval before execution.
    Args:
        router: Router name (e.g., 'Core-Router-Mumbai').
        peer: The BGP peer to reset.
    """
    try:
        config = read_config()
        if router not in config:
            return f"ACTION FAILED [{now_ist()}]: Router '{router}' not found in network_config.json"
        
        config[router]["bgp_down"] = False
        config[router]["status"] = "online"
        write_config(config)
        
        return f"ACTION SUCCESS [{now_ist()}]: network_config.json updated — {router} BGP session reset. bgp_down cleared."
    except Exception as e:
        return f"ACTION FAILED [{now_ist()}]: {e}"

@tool
def escalate_to_noc(issue_summary: str, router: str = "Unknown"):
    """
    Escalates to NOC team. Does NOT modify network_config.json.
    THIS IS HIGH RISK — requires human approval.
    Args:
        issue_summary: Description of the issue.
        router: The impacted router.
    """
    return f"ESCALATION [{now_ist()}]: Issue escalated to NOC for {router}. Summary: {issue_summary}"

TOOL_MAP = {
    "reroute_traffic": reroute_traffic,
    "restart_interface": restart_interface,
    "adjust_qos": adjust_qos,
    "reset_bgp_session": reset_bgp_session,
    "escalate_to_noc": escalate_to_noc,
}
ALL_TOOLS = list(TOOL_MAP.values())

# Investigative tools (not in TOOL_MAP / ALL_TOOLS — called deterministically, not by LLM)
INVESTIGATIVE_TOOLS = {
    "run_device_diagnostics": run_device_diagnostics,
    "calculate_blast_radius": calculate_blast_radius,
}

# ─────────────────────────────────────────────
# 4. State
# ─────────────────────────────────────────────
class NetworkAgentState(TypedDict):
    anomaly_payload: dict
    retrieved_context: str
    diagnostic_result: str          # from run_device_diagnostics
    blast_radius_result: str        # from calculate_blast_radius
    llm_reasoning: str
    recommended_action: str
    action_args: Optional[str]
    action_result: str
    risk_level: str
    human_approved: bool
    needs_rollback: bool
    reasoning_log: Annotated[List[str], operator.add]
    action_history: Annotated[List[str], operator.add]

# ─────────────────────────────────────────────
# 5. Nodes
# ─────────────────────────────────────────────
def observe_node(state: NetworkAgentState):
    """PARTIAL OBSERVABILITY: Only emits the symptom alert. No config flags are revealed."""
    p = state.get("anomaly_payload", {})
    # DO NOT read network_config.json here — the agent must investigate first
    return {"reasoning_log": [
        f"Observer [{now_ist()}]: ⚠️ ALERT — ML Model detected anomaly on {p.get('router')}. "
        f"Symptom: {p.get('metric')} = {p.get('value')} (threshold: {p.get('threshold')}). "
        f"Root cause is UNKNOWN — agent must investigate."
    ]}

def retrieve_node(state: NetworkAgentState):
    p = state.get("anomaly_payload", {})
    query = f"{p.get('metric')} issue on {p.get('router')}"
    try:
        docs = get_retriever().invoke(query)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        return {"retrieved_context": context, "reasoning_log": [f"Retriever [{now_ist()}]: Found {len(docs)} SOPs for '{query}'"]}
    except Exception as e:
        return {"retrieved_context": f"RAG failed: {e}", "reasoning_log": [f"Retriever [{now_ist()}]: RAG error — {e}"]}

def investigate_node(state: NetworkAgentState):
    """Runs investigative tools deterministically: diagnostics + blast radius.
    This gives the agent the information it was blind to in observe_node."""
    p = state.get("anomaly_payload", {})
    router = p.get("router", "Unknown")

    # 1. Run device diagnostics
    diag_result = run_device_diagnostics.invoke({"router_name": router})

    # 2. Calculate blast radius
    blast_result = calculate_blast_radius.invoke({"router_name": router})

    return {
        "diagnostic_result": diag_result,
        "blast_radius_result": blast_result,
        "reasoning_log": [
            f"Investigator [{now_ist()}]: 🔍 Diagnostics for {router}: {diag_result}",
            f"Investigator [{now_ist()}]: 💥 {blast_result}",
        ],
    }

def reason_and_decide_node(state: NetworkAgentState):
    p = state.get("anomaly_payload", {})
    ctx = state.get("retrieved_context", "No context.")
    diag = state.get("diagnostic_result", "No diagnostics available.")
    blast = state.get("blast_radius_result", "No blast radius data.")

    state_signature = f"{p.get('router')}_{diag}"
    if state_signature in ACTION_CACHE:
        cached = ACTION_CACHE[state_signature]
        action = cached.get('recommended_action')
        args = cached.get('action_args')
        risk_level = cached.get('risk_level')
        fast_path_log = "⚡ [FAST-PATH] Semantic Cache Hit! Bypassing LLM API to save latency."
        return {
            "llm_reasoning": fast_path_log,
            "recommended_action": action,
            "action_args": args,
            "risk_level": risk_level,
            "reasoning_log": [f"Reasoner [{now_ist()}]: {fast_path_log}\nAction='{action}', Risk={risk_level.upper()}, Args={args}."],
        }

    prompt = f"""
    You are an Autonomous Network Operations AI for IndiaNet ISP.
    You operate under PARTIAL OBSERVABILITY — you were initially BLIND to the root cause.
    The investigation tools have now revealed the device state and blast radius below.
    Your mitigation tools write DIRECTLY to network_config.json, which is the source of truth.

    ═══════════════════════════════════════
    INITIAL SYMPTOM (from ML Model Alert):
    ═══════════════════════════════════════
    - Router: {p.get('router')}
    - Metric: {p.get('metric')}
    - Value: {p.get('value')}
    - Threshold: {p.get('threshold')}

    ═══════════════════════════════════════
    INVESTIGATION RESULTS (from diagnostics tool):
    ═══════════════════════════════════════
    {diag}

    ═══════════════════════════════════════
    BLAST RADIUS ASSESSMENT:
    ═══════════════════════════════════════
    {blast}

    ═══════════════════════════════════════
    SOPs & KNOWLEDGE BASE:
    ═══════════════════════════════════════
    {ctx[:3000]}

    ═══════════════════════════════════════
    TOOL SELECTION RULES (follow exactly):
    ═══════════════════════════════════════
    | Root Cause (from diagnostics)  | Tool to call             | Risk  |
    |-------------------------------|--------------------------|-------|
    | is_congested = true           | reroute_traffic          | LOW   |
    | interface_flapping = true     | restart_interface        | LOW   |
    | cpu_spiking = true            | restart_interface        | LOW   |
    | bgp_down = true               | reset_bgp_session        | HIGH  |
    | packet_loss (no BGP issue)    | adjust_qos               | LOW   |

    AVAILABLE MITIGATION TOOLS:
    1. reroute_traffic(source_router, target_router) — use for congestion. Target a valid core router.
    2. restart_interface(router, interface) — use for cpu_spike, interface_flap. Interface is always "Gi0/1".
    3. adjust_qos(router, policy) — use for DDoS/packet loss. Policy is "EDGE_PROTECT" or "VOIP_PRIORITY".
    4. reset_bgp_session(router, peer) — use for bgp_down ONLY. HIGH RISK.
    5. escalate_to_noc(issue_summary, router) — escalate. HIGH RISK.

    INSTRUCTIONS:
    1. Read the INVESTIGATION RESULTS above — use the active anomaly flags to determine the root cause.
    2. Consider the BLAST RADIUS — if impact is CRITICAL or HIGH, note this in your reasoning.
    3. Match the root cause flag to the correct tool from the table above.
    4. If bgp_down is true, you MUST call reset_bgp_session.
    5. If interface_flapping is true, you MUST call restart_interface.
    6. If cpu_spiking is true, you MUST call restart_interface.
    7. If is_congested is true, you MUST call reroute_traffic.
    8. Call exactly ONE tool.
    9. In your reasoning, ALWAYS mention the blast radius impact level.
    """

    try:
        config = read_config()
        valid_routers = list(config.keys())
        system_msg = (
            "You are an expert network ops AI operating under partial observability. "
            "You have just received investigation results and blast radius data. "
            "Use the diagnosed root cause to select the correct mitigation tool. "
            "Always reference the blast radius in your reasoning. Call exactly one tool.\n\n"
            f"CRITICAL TOPOLOGY CONSTRAINT: The ONLY valid routers in this network are: {valid_routers}. "
            f"Do not attempt to route traffic to any other names."
        )
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        return {
            "llm_reasoning": f"LLM ERROR: {e}",
            "recommended_action": "escalate_to_noc",
            "action_args": json.dumps({"issue_summary": f"LLM failed: {e}", "router": p.get("router", "Unknown")}),
            "risk_level": "high",
            "reasoning_log": [f"Reasoner [{now_ist()}]: LLM FAILED — {e}\n{traceback.format_exc()}"],
        }

    risk_level = "high"
    action = "escalate_to_noc"
    args = json.dumps({"issue_summary": "Fallback.", "router": p.get("router", "Unknown")})
    reasoning = response.content or ""

    if response.tool_calls:
        tc = response.tool_calls[0]
        action = tc["name"]
        args = json.dumps(tc["args"])
        risk_level = "high" if action in ("reset_bgp_session", "escalate_to_noc") else "low"

    return {
        "llm_reasoning": reasoning,
        "recommended_action": action,
        "action_args": args,
        "risk_level": risk_level,
        "reasoning_log": [f"Reasoner [{now_ist()}]: Action='{action}', Risk={risk_level.upper()}, Args={args}. Blast radius considered."],
    }

def human_approval_node(state: NetworkAgentState):
    return {"human_approved": True, "reasoning_log": [f"Human Approval [{now_ist()}]: APPROVED by NOC."]}

def act_node(state: NetworkAgentState):
    try:
        import requests
        requests.post("http://127.0.0.1:8000/api/config/backup")
    except:
        pass

    tool_name = state.get("recommended_action", "")
    try:
        args = json.loads(state.get("action_args", "{}"))
    except Exception:
        args = {}
    try:
        result = str(TOOL_MAP[tool_name].invoke(args)) if tool_name in TOOL_MAP else f"Error: Tool '{tool_name}' not found."
    except Exception as e:
        result = f"TOOL ERROR [{now_ist()}]: {e}\n{traceback.format_exc()}"
    return {
        "action_result": result,
        "reasoning_log": [f"Executor [{now_ist()}]: {result}"],
        "action_history": [f"{tool_name} | {args} | {result}"],
    }

def verify_node(state: NetworkAgentState):
    p = state.get("anomaly_payload", {})
    router = p.get("router", "Unknown")
    _time.sleep(3)
    needs_rollback = False
    log = f"Verifier [{now_ist()}]: Health verified."
    try:
        import requests
        res = requests.get(f"http://127.0.0.1:8000/api/config/verify_health?router_name={router}").json()
        if res.get("status") == "success":
            is_healthy = res.get("is_healthy", False)
            flags = res.get("flags", [])
            state_status = res.get("state_status", "unknown")
            if not is_healthy or state_status in ["offline", "rebooting"]:
                needs_rollback = True
                log = f"Verifier [{now_ist()}]: Health check FAILED. Active flags: {flags}, Status: {state_status}"
            else:
                log = f"Verifier [{now_ist()}]: Health check PASSED."
    except Exception as e:
        log = f"Verifier [{now_ist()}]: API error: {e}"
        
    return {
        "needs_rollback": needs_rollback,
        "reasoning_log": [log]
    }

def rollback_node(state: NetworkAgentState):
    log = f"Rollback [{now_ist()}]: Action failed to resolve anomaly. Configuration rolled back."
    try:
        import requests
        requests.post("http://127.0.0.1:8000/api/config/rollback")
    except Exception as e:
        log = f"Rollback [{now_ist()}]: Rollback failed: {e}"
        
    return {
        "action_result": "FAILED - ROLLED BACK",
        "reasoning_log": [log],
        "recommended_action": "escalate_to_noc",
        "action_args": json.dumps({"issue_summary": "Rollback triggered."}),
        "risk_level": "high"
    }

def learn_node(state: NetworkAgentState):
    p = state.get("anomaly_payload", {})
    action = state.get("recommended_action", "Unknown")
    result = state.get("action_result", "Unknown")
    router = p.get("router", "Unknown")
    metric = p.get("metric", "Unknown")
    value = p.get("value", "Unknown")
    diag = state.get("diagnostic_result", "No diagnostics available.")

    state_signature = f"{router}_{diag}"
    ACTION_CACHE[state_signature] = {
        "recommended_action": action,
        "action_args": state.get("action_args", "{}"),
        "risk_level": state.get("risk_level", "low")
    }

    post_mortem = f"Incident on {router}: {metric} spiked to {value}. Action taken: {action}. Result: {result}. Date: {now_ist()}"

    history_file = os.path.join("data", "incident_history.md")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    try:
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"- {post_mortem}\n")
    except Exception:
        pass

    try:
        get_vectorstore().add_texts([post_mortem])
        learn_log = f"Learner [{now_ist()}]: Post-mortem saved to incident_history.md and embedded into ChromaDB."
    except Exception as e:
        learn_log = f"Learner [{now_ist()}]: Failed to embed to ChromaDB: {e}"

    return {
        "reasoning_log": [learn_log]
    }

# ─────────────────────────────────────────────
# 6. Graph — with investigative loop for partial observability
# ─────────────────────────────────────────────
workflow = StateGraph(NetworkAgentState)
workflow.add_node("observe", observe_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("investigate", investigate_node)       # NEW: diagnostic + blast radius
workflow.add_node("reason_and_decide", reason_and_decide_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("act", act_node)
workflow.add_node("verify", verify_node)
workflow.add_node("rollback", rollback_node)
workflow.add_node("learn", learn_node)

# Flow: observe → retrieve → investigate → reason_and_decide → ...
workflow.set_entry_point("observe")
workflow.add_edge("observe", "retrieve")
workflow.add_edge("retrieve", "investigate")              # NEW edge
workflow.add_edge("investigate", "reason_and_decide")     # NEW edge

def route_decision(state):
    return "human_approval" if state.get("risk_level") == "high" else "act"

def verify_decision(state):
    return "rollback" if state.get("needs_rollback") else "learn"

workflow.add_conditional_edges("reason_and_decide", route_decision)
workflow.add_edge("human_approval", "act")
workflow.add_edge("act", "verify")
workflow.add_conditional_edges("verify", verify_decision)
workflow.add_edge("rollback", "human_approval")
workflow.add_edge("learn", END)

checkpointer = MemorySaver()
agent_app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])

# ─────────────────────────────────────────────
# 7. Public API
# ─────────────────────────────────────────────
def _stream_logs(stream) -> list:
    logs = []
    for event in stream:
        for node, update in event.items():
            if "reasoning_log" in update:
                logs.append(f"[{node.upper()}] {update['reasoning_log'][-1]}")
    return logs

def start_agent(anomaly_payload: dict) -> dict:
    thread_id = f"anomaly_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        logs = _stream_logs(agent_app.stream(
            {"anomaly_payload": anomaly_payload, "reasoning_log": [], "action_history": []}, config=config))
    except Exception as e:
        return {"status": "error", "thread_id": thread_id, "logs": [f"[ERROR] {e}\n{traceback.format_exc()}"],
                "recommended_action": "none", "action_result": f"Agent failed: {e}", "risk_level": "unknown", "error": str(e)}

    try:
        snapshot = agent_app.get_state(config)
    except Exception as e:
        return {"status": "error", "thread_id": thread_id, "logs": logs + [f"[ERROR] State: {e}"],
                "recommended_action": "none", "action_result": f"State failed: {e}", "risk_level": "unknown"}

    if snapshot.next and "human_approval" in snapshot.next:
        vals = snapshot.values
        return {"status": "pending_approval", "thread_id": thread_id, "logs": logs,
                "recommended_action": vals.get("recommended_action", "unknown"),
                "action_args": vals.get("action_args", "{}"), "risk_level": "high",
                "action_result": "⏳ Awaiting human approval..."}

    vals = agent_app.get_state(config).values
    return {"status": "completed", "thread_id": thread_id, "logs": logs,
            "recommended_action": vals.get("recommended_action", "none"),
            "action_result": vals.get("action_result", "No action"), "risk_level": vals.get("risk_level", "low")}

def resume_agent(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        logs = _stream_logs(agent_app.stream(None, config=config))
        vals = agent_app.get_state(config).values
        return {"status": "completed", "thread_id": thread_id, "logs": logs,
                "recommended_action": vals.get("recommended_action", "none"),
                "action_result": vals.get("action_result", "No action"), "risk_level": vals.get("risk_level", "high")}
    except Exception as e:
        return {"status": "error", "thread_id": thread_id, "logs": [f"[ERROR] {e}\n{traceback.format_exc()}"],
                "recommended_action": "none", "action_result": f"Resume failed: {e}", "risk_level": "high"}
