"""
main.py — FastAPI Backend: Digital Twin Network Simulator.
Reads network_config.json as the single source of truth.
Agent tools write directly to the config file (no HTTP resolution needed).
FastAPI handles: telemetry generation, anomaly injection, human-in-the-loop, and live logging.
"""

import os
import json
import random
import asyncio
import math
import uuid
import traceback
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# ML Model Configuration
ML_MODEL_PATH = Path("models/telecom_anomaly_model.pkl")
ML_MODEL = None

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ─────────────────────────────────────────────
# IST
# ─────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S IST")

def now_ist_dt() -> datetime:
    return datetime.now(IST)

# ─────────────────────────────────────────────
# Config File I/O
# ─────────────────────────────────────────────
CONFIG_FILE = Path("network_config.json")
_config_lock = threading.Lock()

DEFAULT_CONFIG = {
    "Core-Router-Mumbai": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
    "Core-Router-Delhi": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
    "Core-Router-Bangalore": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
    "Core-Router-Chennai": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
    "Edge-Router-North": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
    "Edge-Router-South": {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False},
}

def read_config() -> dict:
    with _config_lock:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            write_config_unsafe(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

def write_config(config: dict):
    with _config_lock:
        write_config_unsafe(config)

def write_config_unsafe(config: dict):
    """Write without acquiring lock (for use inside already-locked contexts)."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

if not CONFIG_FILE.exists():
    write_config(DEFAULT_CONFIG)

# ─────────────────────────────────────────────
# In-Memory Storage
# ─────────────────────────────────────────────
TELEMETRY_BUFFER = deque(maxlen=500)
AGENT_LOGS = []
LATENCY_HISTORY = deque(maxlen=50)
PENDING_APPROVALS = {}
ACTION_HISTORY = {}  # Track all actions: action_id -> {action_data}
LOG_FILE = Path("live_network_logs.jsonl")
AUDIT_LOG_FILE = Path("logs/audit_trail.jsonl")

# Ensure logs directory exists
AUDIT_LOG_FILE.parent.mkdir(exist_ok=True)

# Load topology
TOPOLOGY_FILE = os.path.join("data", "topology.json")
try:
    with open(TOPOLOGY_FILE, "r") as f:
        TOPOLOGY = json.load(f)
    ROUTERS = [r["name"] for r in TOPOLOGY["routers"]]
except Exception:
    TOPOLOGY = {"routers": [], "links": []}
    ROUTERS = list(DEFAULT_CONFIG.keys())

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class AnomalyRequest(BaseModel):
    anomaly_type: str
    router_name: str

class ApprovalAction(BaseModel):
    thread_id: str

class BulkInjectRequest(BaseModel):
    injections: list[dict]  # [{"anomaly_type": ..., "router_name": ...}, ...]

class StressScenarioRequest(BaseModel):
    scenario: str  # "cascade_failure", "random_chaos", "full_meltdown"

class HardResetRequest(BaseModel):
    router_name: str

# ─────────────────────────────────────────────
# Telemetry from Digital Twin
# ─────────────────────────────────────────────
def calculate_zscore(value: float, history: deque) -> float:
    if len(history) < 10:
        return 0.0
    values = list(history)
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    return (value - mean) / std

def generate_telemetry_point(force_router: str | None = None):
    """Reads network_config.json LIVE on every call."""
    config = read_config()
    if not ROUTERS:
        return {}
    router = force_router if force_router else random.choice(ROUTERS[:min(4, len(ROUTERS))])
    state = config.get(router, {})

    status = state.get("status", "online")
    route = state.get("current_route", "Primary-Link-A")

    # Rebooting / Offline → downtime
    if status in ["rebooting", "offline"]:
        return {
            "timestamp": now_ist(), "router": router,
            "latency_ms": 0, "packet_loss_pct": 100.0, "cpu_utilization_pct": 0,
            "bgp_flaps_per_min": 0, "interface": "Gi0/1", "current_route": route, "status": status,
        }

    # Healthy baseline
    latency = max(5, random.gauss(20, 5))
    packet_loss = random.uniform(0.0, 0.3)
    cpu_util = random.uniform(10, 30)
    bgp_flaps = 0
    is_anomalous = False

    # Backup Route Telemetry Override (Healthy Priority)
    if status == "online" and "Backup" in route:
        return {
            "timestamp": now_ist(), "router": router,
            "latency_ms": round(max(5, random.gauss(20, 5)), 2), 
            "packet_loss_pct": round(random.uniform(0.0, 0.5), 3),
            "cpu_utilization_pct": round(random.uniform(10, 30), 1), 
            "bgp_flaps_per_min": 0,
            "interface": "Gi0/1", "current_route": route,
            "status": "normal",
        }

    # Congestion on primary link → bad. On backup → healthy (agent fixed it!)
    if state.get("is_congested") and "Primary" in route:
        latency = random.uniform(250, 400)
        packet_loss = random.uniform(5.0, 15.0)
        is_anomalous = True
    elif state.get("is_congested") and "Primary" not in route:
        latency = max(5, random.gauss(25, 5))
        packet_loss = random.uniform(0.0, 0.5)

    if state.get("bgp_down"):
        packet_loss = 100.0
        bgp_flaps = random.randint(1, 5)
        is_anomalous = True

    if state.get("cpu_spiking"):
        cpu_util = random.uniform(90, 99)
        latency += random.uniform(50, 100)
        is_anomalous = True

    if state.get("interface_flapping"):
        packet_loss = random.uniform(20.0, 50.0)
        latency += random.uniform(20, 50)
        is_anomalous = True

    return {
        "timestamp": now_ist(), "router": router,
        "latency_ms": round(max(0, latency), 2), "packet_loss_pct": round(packet_loss, 3),
        "cpu_utilization_pct": round(cpu_util, 1), "bgp_flaps_per_min": bgp_flaps,
        "interface": "Gi0/1", "current_route": route,
        "status": "anomaly" if is_anomalous else "normal",
    }

def write_audit_log(action_record: dict):
    """Write audit trail entry to logs/audit_trail.jsonl"""
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(action_record) + "\n")
    except Exception as e:
        print(f"Failed to write audit log: {e}")

def write_jsonl_log(point: dict):
    config = read_config()
    router = point["router"]
    r_state = config.get(router, {})
    is_active = any([
        r_state.get("is_congested") and r_state.get("current_route") == "Primary-Link-A",
        r_state.get("bgp_down"), r_state.get("cpu_spiking"), r_state.get("interface_flapping"),
        r_state.get("status") == "rebooting",
    ])
    entry = {
        "timestamp": point["timestamp"], "router_name": router,
        "current_traffic_route": r_state.get("current_route", "Primary-Link-A"),
        "traffic_amount_mbps": round(random.uniform(50, 200) if is_active else random.uniform(400, 900), 1),
        "anomaly_status": "active" if is_active else "resolved",
        "latency_ms": point["latency_ms"], "packet_loss_pct": point["packet_loss_pct"],
        "cpu_utilization_pct": point["cpu_utilization_pct"],
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# ─────────────────────────────────────────────
# Background Task
# ─────────────────────────────────────────────
_last_agent_trigger: dict[str, datetime] = {}
AGENT_COOLDOWN_SECONDS = 25

async def telemetry_background_task():
    global _last_agent_trigger
    while True:
        try:
            point = generate_telemetry_point()
            if not point:
                await asyncio.sleep(2)
                continue
            TELEMETRY_BUFFER.append(point)
            LATENCY_HISTORY.append(point.get("latency_ms", 0))

            try:
                router_name = point.get("router")
                recent_latency = [p.get("latency_ms", 0) for p in TELEMETRY_BUFFER if p.get("router") == router_name][-15:]
                if len(recent_latency) > 1:
                    X = np.arange(len(recent_latency)).reshape(-1, 1)
                    y = np.array(recent_latency)
                    model = LinearRegression()
                    model.fit(X, y)
                    next_t = np.array([[len(recent_latency) + 4]])
                    pred_latency = max(0, float(model.predict(next_t)[0]))
                else:
                    pred_latency = point.get("latency_ms", 0)
            except Exception:
                pred_latency = point.get("latency_ms", 0)
            
            point["predicted_latency"] = round(pred_latency, 2)

            write_jsonl_log(point)

            is_bad = False
            
            if ML_MODEL is not None:
                import pandas as pd
                try:
                    features = pd.DataFrame([{
                        "latency_ms": point.get("latency_ms", 0),
                        "packet_loss_pct": point.get("packet_loss_pct", 0),
                        "cpu_utilization": point.get("cpu_utilization_pct", 0),
                        "bgp_flaps": point.get("bgp_flaps_per_min", 0)
                    }])
                    prediction = ML_MODEL.predict(features)
                    is_bad = bool(prediction[0] == 1)
                except Exception as e:
                    print(f"ML Model predict error: {e}")
                    
            # Fallback to rules engine if model fails or isn't loaded
            if ML_MODEL is None and not is_bad:
                is_bad = (
                    point.get("packet_loss_pct", 0) > 10.0
                    or point.get("cpu_utilization_pct", 0) > 90.0
                    or abs(calculate_zscore(point.get("latency_ms", 0), LATENCY_HISTORY)) > 3.0
                )

            if is_bad and point.get("status") != "rebooting":
                router = point.get("router")
                now = now_ist_dt()
                last_trigger = _last_agent_trigger.get(router)
                if last_trigger and (now - last_trigger).total_seconds() < AGENT_COOLDOWN_SECONDS:
                    await asyncio.sleep(2)
                    continue

                _last_agent_trigger[router] = now

                metric = "latency"
                value = point.get("latency_ms", 0)
                threshold = 100
                if point.get("packet_loss_pct", 0) > 10.0:
                    metric, value, threshold = "packet_loss", point.get("packet_loss_pct", 0), 5.0
                elif point.get("cpu_utilization_pct", 0) > 90.0:
                    metric, value, threshold = "cpu_utilization", point.get("cpu_utilization_pct", 0), 80.0

                anomaly_payload = {
                    "router": router, "metric": metric, "value": value,
                    "threshold": threshold, "timestamp": point.get("timestamp"),
                }

                try:
                    from agent import start_agent
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, start_agent, anomaly_payload)

                    AGENT_LOGS.append({
                        "id": str(uuid.uuid4())[:8], "timestamp": now_ist(),
                        "trigger": "auto_detection", "anomaly": anomaly_payload, "result": result,
                    })

                    if result.get("status") == "pending_approval":
                        PENDING_APPROVALS[result["thread_id"]] = {
                            "thread_id": result["thread_id"],
                            "action": result.get("recommended_action"),
                            "action_args": result.get("action_args"),
                            "anomaly": anomaly_payload,
                            "logs": result.get("logs", []),
                            "timestamp": now_ist(),
                        }
                except Exception as e:
                    AGENT_LOGS.append({
                        "id": str(uuid.uuid4())[:8], "timestamp": now_ist(),
                        "trigger": "auto_detection", "error": f"{e}\n{traceback.format_exc()}",
                    })

        except Exception as e:
            print(f"Backend loop error: {e}")
        await asyncio.sleep(2)

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    global ML_MODEL
    try:
        if ML_MODEL_PATH.exists():
            ML_MODEL = joblib.load(ML_MODEL_PATH)
            print(f"Loaded Custom ML Model securely from {ML_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load ML model: {e}")
        
    task = asyncio.create_task(telemetry_background_task())
    yield
    task.cancel()

app = FastAPI(title="NetOps Digital Twin API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "NetOps Digital Twin API online", "time": now_ist()}

@app.get("/telemetry")
def get_telemetry(limit: int = 100):
    return {"data": list(TELEMETRY_BUFFER)[-limit:], "network_state": read_config()}

@app.get("/topology")
def get_topology():
    return TOPOLOGY

@app.get("/agent-logs")
def get_agent_logs():
    return {"logs": AGENT_LOGS, "count": len(AGENT_LOGS)}

@app.get("/network-config")
def get_network_config():
    return read_config()

# ─────────────────────────────────────────────
# Anomaly Injection (writes to config file)
# ─────────────────────────────────────────────
@app.post("/api/simulate-anomaly")
def simulate_anomaly(req: AnomalyRequest):
    config = read_config()
    if req.router_name not in config:
        config[req.router_name] = {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False}

    type_map = {"congestion": "is_congested", "bgp_down": "bgp_down", "cpu_spike": "cpu_spiking", "interface_flap": "interface_flapping"}
    flag = type_map.get(req.anomaly_type)
    if not flag:
        raise HTTPException(400, "Unknown anomaly_type")

    config[req.router_name][flag] = True
    config[req.router_name]["status"] = "offline"
    config[req.router_name]["current_route"] = "Primary-Link-A"  # Reset route so congestion is visible
    write_config(config)

    _last_agent_trigger.pop(req.router_name, None)

    for _ in range(3):
        p = generate_telemetry_point(force_router=req.router_name)
        TELEMETRY_BUFFER.append(p)
        LATENCY_HISTORY.append(p["latency_ms"])
        write_jsonl_log(p)

    return {"status": "success", "message": f"network_config.json updated: {req.anomaly_type}=true on {req.router_name}", "timestamp": now_ist()}

# ─────────────────────────────────────────────
# Backup / Rollback & Hard Reset
# ─────────────────────────────────────────────
_config_backup = {}

@app.post("/api/config/backup")
def backup_config():
    global _config_backup
    _config_backup = read_config()
    return {"status": "success", "message": "Configuration backed up"}

@app.post("/api/config/rollback")
def rollback_config():
    global _config_backup
    if not _config_backup:
        raise HTTPException(400, "No backup available")

    # Log rollback to audit trail
    action_id = str(uuid.uuid4())[:8]
    audit_record = {
        "action_id": action_id,
        "timestamp": now_ist(),
        "event_type": "auto_rollback",
        "reason": "verification_failed_or_manual_trigger",
        "original_state": read_config(),
        "rollback_state": _config_backup.copy()
    }
    write_audit_log(audit_record)
    ACTION_HISTORY[action_id] = audit_record

    write_config(_config_backup)
    return {"status": "success", "message": "Configuration rolled back", "action_id": action_id}

@app.get("/api/config/verify_health")
def verify_health(router_name: str):
    config = read_config()
    if router_name not in config:
        raise HTTPException(404, "Router not found")
    state = config[router_name]
    flags = [k for k, v in state.items() if isinstance(v, bool) and v]
    # Healthy if no flags AND status is online
    is_healthy = (len(flags) == 0) and (state.get("status") == "online")
    return {"status": "success", "is_healthy": is_healthy, "flags": flags}

@app.post("/api/resolve/hard_reset")
def hard_reset(req: HardResetRequest, bg_tasks: BackgroundTasks):
    config = read_config()
    if req.router_name not in config:
        raise HTTPException(404, "Router not found")

    # Set to rebooting immediately
    config[req.router_name]["status"] = "rebooting"
    write_config(config)

    bg_tasks.add_task(reboot_sequence_sync, req.router_name)
    
    return {"status": "success", "message": f"{req.router_name} is rebooting for 5 seconds."}

def reboot_sequence_sync(router):
    import time
    time.sleep(5)
    cfg = read_config()
    if router in cfg:
        cfg[router]["status"] = "online"
        cfg[router]["current_route"] = "Primary-Link-A"
        cfg[router]["is_congested"] = False
        cfg[router]["bgp_down"] = False
        cfg[router]["cpu_spiking"] = False
        cfg[router]["interface_flapping"] = False
        write_config(cfg)

# ─────────────────────────────────────────────
# Stress Test Page & Advanced Endpoints
# ─────────────────────────────────────────────
STRESS_HTML = Path("static/stress_test.html")

@app.get("/stress-test", response_class=HTMLResponse)
def serve_stress_test():
    """Serve the standalone stress test HTML page."""
    try:
        return HTMLResponse(content=STRESS_HTML.read_text(encoding="utf-8"), status_code=200)
    except FileNotFoundError:
        raise HTTPException(404, "stress_test.html not found")

@app.post("/api/reset-all")
def reset_all_routers():
    """Reset every router to healthy defaults."""
    write_config(DEFAULT_CONFIG.copy())
    _last_agent_trigger.clear()
    return {"status": "success", "message": "All routers reset to healthy defaults", "timestamp": now_ist()}

@app.post("/api/bulk-inject")
def bulk_inject(req: BulkInjectRequest):
    """Inject multiple anomalies at once."""
    config = read_config()
    type_map = {"congestion": "is_congested", "bgp_down": "bgp_down", "cpu_spike": "cpu_spiking", "interface_flap": "interface_flapping"}
    count = 0

    for item in req.injections:
        rname = item.get("router_name", "")
        atype = item.get("anomaly_type", "")
        flag = type_map.get(atype)
        if not flag:
            continue
        if rname not in config:
            config[rname] = {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False}
        config[rname][flag] = True
        config[rname]["status"] = "offline"
        config[rname]["current_route"] = "Primary-Link-A"
        _last_agent_trigger.pop(rname, None)
        count += 1

    write_config(config)

    # Generate a few telemetry points to surface the anomalies
    affected = {item.get("router_name") for item in req.injections if item.get("router_name") in config}
    for r in affected:
        for _ in range(2):
            p = generate_telemetry_point(force_router=r)
            TELEMETRY_BUFFER.append(p)
            LATENCY_HISTORY.append(p["latency_ms"])
            write_jsonl_log(p)

    return {"status": "success", "injected": count, "timestamp": now_ist()}

@app.post("/api/stress-scenario")
def stress_scenario(req: StressScenarioRequest):
    """Run a pre-built stress scenario."""
    config = read_config()
    type_map = {"congestion": "is_congested", "bgp_down": "bgp_down", "cpu_spike": "cpu_spiking", "interface_flap": "interface_flapping"}
    injections = []

    if req.scenario == "cascade_failure":
        # Sequential failures across core routers
        targets = [("Core-Router-Mumbai", "congestion"), ("Core-Router-Delhi", "bgp_down"), ("Core-Router-Bangalore", "cpu_spike")]
        for rname, atype in targets:
            injections.append({"router_name": rname, "anomaly_type": atype})

    elif req.scenario == "random_chaos":
        import random as _rnd
        all_routers = list(config.keys())
        all_types = list(type_map.keys())
        n = _rnd.randint(3, 5)
        for _ in range(n):
            injections.append({"router_name": _rnd.choice(all_routers), "anomaly_type": _rnd.choice(all_types)})

    elif req.scenario == "full_meltdown":
        for rname in config:
            for atype in type_map:
                injections.append({"router_name": rname, "anomaly_type": atype})

    else:
        raise HTTPException(400, f"Unknown scenario: {req.scenario}")

    # Apply all injections
    for item in injections:
        rname = item["router_name"]
        flag = type_map.get(item["anomaly_type"])
        if not flag:
            continue
        if rname not in config:
            config[rname] = {"status": "online", "current_route": "Primary-Link-A", "is_congested": False, "bgp_down": False, "cpu_spiking": False, "interface_flapping": False}
        config[rname][flag] = True
        config[rname]["status"] = "offline"
        config[rname]["current_route"] = "Primary-Link-A"
        _last_agent_trigger.pop(rname, None)

    write_config(config)

    # Generate telemetry points
    affected = {item["router_name"] for item in injections if item["router_name"] in config}
    for r in affected:
        for _ in range(2):
            p = generate_telemetry_point(force_router=r)
            TELEMETRY_BUFFER.append(p)
            LATENCY_HISTORY.append(p["latency_ms"])
            write_jsonl_log(p)

    return {"status": "success", "scenario": req.scenario, "injections_count": len(injections), "timestamp": now_ist()}

# ─────────────────────────────────────────────
# Human-in-the-Loop
# ─────────────────────────────────────────────
@app.get("/api/pending-approvals")
def get_pending_approvals():
    return {"pending": list(PENDING_APPROVALS.values()), "count": len(PENDING_APPROVALS)}

@app.post("/api/approve")
def approve_action(req: ApprovalAction):
    if req.thread_id not in PENDING_APPROVALS:
        raise HTTPException(404, "No pending approval")
    try:
        from agent import resume_agent
        result = resume_agent(req.thread_id)
        info = PENDING_APPROVALS.pop(req.thread_id, {})

        # Log approval + execution to audit trail
        action_id = str(uuid.uuid4())[:8]
        audit_record = {
            "action_id": action_id,
            "timestamp": now_ist(),
            "event_type": "human_approval_and_execution",
            "thread_id": req.thread_id,
            "anomaly": info.get("anomaly", {}),
            "recommended_action": info.get("action", "N/A"),
            "action_args": info.get("action_args", {}),
            "outcome": result.get("status", "unknown"),
            "agent_result": result
        }
        write_audit_log(audit_record)
        ACTION_HISTORY[action_id] = audit_record

        AGENT_LOGS.append({"id": str(uuid.uuid4())[:8], "timestamp": now_ist(), "trigger": "human_approved", "anomaly": info.get("anomaly", {}), "result": result})
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": f"{e}\n{traceback.format_exc()}"}

@app.post("/api/reject")
def reject_action(req: ApprovalAction):
    if req.thread_id not in PENDING_APPROVALS:
        raise HTTPException(404, "No pending approval")
    info = PENDING_APPROVALS.pop(req.thread_id, {})

    # Log rejection to audit trail
    action_id = str(uuid.uuid4())[:8]
    audit_record = {
        "action_id": action_id,
        "timestamp": now_ist(),
        "event_type": "human_rejection",
        "thread_id": req.thread_id,
        "anomaly": info.get("anomaly", {}),
        "recommended_action": info.get("action", "N/A"),
        "outcome": "rejected"
    }
    write_audit_log(audit_record)
    ACTION_HISTORY[action_id] = audit_record

    AGENT_LOGS.append({
        "id": str(uuid.uuid4())[:8], "timestamp": now_ist(), "trigger": "human_rejected",
        "anomaly": info.get("anomaly", {}),
        "result": {"logs": [f"[HUMAN_APPROVAL] REJECTED [{now_ist()}]"], "action_result": "Rejected.", "recommended_action": info.get("action", "N/A"), "risk_level": "high"},
    })
    return {"status": "rejected", "message": "Rejected by NOC."}

# ─────────────────────────────────────────────
# New Endpoints: Audit & Observability
# ─────────────────────────────────────────────

@app.get("/api/audit-log")
def get_audit_log(limit: int = 100):
    """Fetch recent audit trail entries"""
    try:
        # Read the last `limit` lines from audit log
        if not AUDIT_LOG_FILE.exists():
            return {"audit_entries": [], "count": 0}

        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            recent_lines = lines[-limit:]
            entries = [json.loads(line) for line in recent_lines if line.strip()]
        return {"audit_entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": str(e), "audit_entries": [], "count": 0}

@app.get("/api/action/{action_id}/rollback-status")
def get_rollback_status(action_id: str):
    """Get rollback status for a specific action"""
    if action_id not in ACTION_HISTORY:
        raise HTTPException(404, f"Action {action_id} not found")

    action = ACTION_HISTORY[action_id]
    return {
        "action_id": action_id,
        "action": action.get("action_executed", {}).get("tool", "N/A"),
        "timestamp": action.get("timestamp"),
        "outcome": action.get("outcome", "unknown"),
        "rollback_triggered": action.get("rollback_triggered", False),
        "rollback_timestamp": action.get("rollback_timestamp"),
        "verification_result": action.get("verification_result"),
    }

@app.get("/api/agent/observability-state")
def get_observability_state():
    """Get current observability state of the agent and network"""
    config = read_config()

    # Count active anomalies
    active_anomalies = {}
    for router, state in config.items():
        anomalies = [k for k, v in state.items() if isinstance(v, bool) and v]
        if anomalies:
            active_anomalies[router] = anomalies

    # Get pending approvals
    pending_approvals_summary = [
        {
            "thread_id": p.get("thread_id"),
            "action": p.get("action"),
            "anomaly": p.get("anomaly"),
            "timestamp": p.get("timestamp")
        }
        for p in PENDING_APPROVALS.values()
    ]

    # Telemetry statistics
    telemetry_stats = {
        "total_points_collected": len(TELEMETRY_BUFFER),
        "total_agent_runs": len(AGENT_LOGS),
        "recent_telemetry": list(TELEMETRY_BUFFER)[-10:] if TELEMETRY_BUFFER else []
    }

    return {
        "timestamp": now_ist(),
        "active_anomalies_by_router": active_anomalies,
        "active_anomaly_count": sum(len(a) for a in active_anomalies.values()),
        "pending_approvals": pending_approvals_summary,
        "pending_approvals_count": len(PENDING_APPROVALS),
        "telemetry_stats": telemetry_stats,
        "agent_action_log_size": len(AGENT_LOGS),
        "action_history_size": len(ACTION_HISTORY),
        "network_config": config
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
