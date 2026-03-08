---

```html
<div align="center">
  <h1 style="font-size: 3.5em; font-weight: 800; letter-spacing: 2px; margin-bottom: 0;">ΛXIOS</h1>
  <h3 style="font-weight: 400; color: #64748B; margin-top: 5px; letter-spacing: 1px;">Automated Network Intelligence</h3>
  <br>
  <p>
    <i>An AI-native autonomous network operations platform for a simulated ISP.</i>
  </p>
</div>

<p align="center">
  The system detects network anomalies in real-time using a custom-trained <b>Random Forest ML model</b>, reasons about root causes using an LLM operating under <b>partial observability</b>, assesses blast radius before taking action, and automatically resolves issues via a control plane. To optimize latency and API costs, it utilizes a <b>Semantic NLP Fast-Path Cache</b> to bypass the LLM for known issues, keeping human operators in the loop only for high-risk actions.
</p>

---

## 🏗️ Key Architecture Features

| Feature | Description |
|---|---|
| **Semantic NLP Fast-Path Cache** | 🧠 Uses `scikit-learn` (TF-IDF & Cosine Similarity) to vectorize incoming anomalies. If an anomaly is >90% similar to a previously solved incident, the agent bypasses the LLM API and executes the cached fix instantly in milliseconds. |
| **Partial Observability** | The agent is *blind* to the root cause at alert time — it only receives a symptom (e.g., "latency spike on Core-Router"). It must actively call diagnostic tools to discover actual anomaly flags before acting. |
| **Blast Radius Assessment** | Before mitigation, the agent computes how many downstream nodes/links are affected based on the live topology, classifying impact as CRITICAL / HIGH / MODERATE / LOW. |
| **Custom ML Anomaly Detection** | A Random Forest model (`models/telecom_anomaly_model.pkl`) trained on ISP telemetry data detects anomalies from latency, packet loss, CPU utilization, and BGP flap features. |
| **Human-in-the-Loop** | High-risk actions (e.g., BGP resets) halt the LangGraph pipeline at an interrupt gate, requiring explicit NOC approval via the Streamlit dashboard before execution. |
| **Digital Twin (Config-as-State)** | `network_config.json` is the single source of truth. Agent tools write directly to this file. The telemetry engine reads it live to reflect the current network state. |
| **RAG-Powered Reasoning** | SOPs and past incident reports are embedded in ChromaDB and retrieved at reasoning time, giving the LLM context-aware decision support for novel issues. |

---

## 📁 Project Structure

```text
AXIOS/
├── main.py                # FastAPI backend — telemetry, ML detection, control plane
├── agent.py               # LangGraph AI agent — observe → investigate → reason → act
├── app.py                 # Streamlit NOC dashboard — live Digital Twin UI
├── run_all_tests.py       # Automated integration test & QA suite
├── train_model.py         # ML model training script (Random Forest)
├── setup_db.py            # One-time ChromaDB knowledge base ingestion
├── network_config.json    # Digital Twin state file (Source of Truth)
├── requirements.txt       # Python dependencies
├── .env                   # Google API key for Gemini LLM & embeddings
├── data/
│   ├── topology.json      # Network topology (Fully meshed Core + Edge layout)
│   ├── sops.md            # SOPs & past incident reports
│   └── telecom_training_data.csv # Training data for the ML model
└── models/
    └── telecom_anomaly_model.pkl # Trained Random Forest model

```

---

## 📄 Core Services Overview

### 1. `main.py` — FastAPI Backend & Telemetry Engine

The central orchestrator that simulates the live ISP network. Built with heavy exception-handling to ensure chaos engineering tests never crash the server.

* **Telemetry Generation:** Generates synthetic network metrics every 2 seconds.
* **ML Anomaly Detection:** Runs telemetry through the Random Forest model. If anomalous, dynamically sets the router status to `"offline"`.
* **Agent Triggering:** Automatically calls the LangGraph agent in an async thread, passing a *symptom-only* payload.
* **Human-in-the-Loop:** Manages pending approvals via `/api/approve` and `/api/reject`.

### 2. `agent.py` — LangGraph Cognitive Engine

The autonomous reasoning engine.

* **Investigative Tools:** `run_device_diagnostics` (discovers root cause) and `calculate_blast_radius` (assesses downstream impact).
* **Semantic Routing Cache:** Extracts symptoms, vectorizes them via TF-IDF, and calculates Cosine Similarity against past fixes. Cache hits bypass the LLM entirely.
* **Mitigation Tools:** `reroute_traffic`, `restart_interface`, `adjust_qos`, `reset_bgp_session`. Tools execute by directly mutating `network_config.json` and restoring `"status": "online"`.

### 3. `app.py` — Streamlit NOC Dashboard

The real-time monitoring UI that acts as a Digital Twin.

* **Live Topology Map:** Interactive `agraph` network map. Dynamically parses `-via-` routing strings to draw redundant edge-to-core connections. Failing nodes instantly render as pulsing red.
* **Agent Action Log:** Real-time trace of the LangGraph state machine, highlighting Semantic Cache Hits and LLM reasoning.
* **Simulation Control:** Panel to inject simulated anomalies (CPU spikes, Congestion, BGP drops) directly into the state file.

---

## 🔗 The Agentic Loop

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  1. ML Model detects anomaly → Triggers Agent (Symptom Only)            │
│  2. OBSERVE: Logs symptom alert (blind to root cause)                   │
│  3. RETRIEVE: Queries ChromaDB for SOPs                                 │
│  4. INVESTIGATE: Discovers actual flags & calculates Blast Radius       │
│  5. SEMANTIC ROUTER:                                                    │
│       ├─▶ [TF-IDF Similarity > 90%] ──▶ FAST-PATH (Bypass LLM) ──┐      │
│       └─▶ [Similarity < 90%] ─────────▶ GEMINI LLM REASONING     │      │
│  6. [APPROVE]: Halts for NOC human-in-the-loop if HIGH risk      │      │
│  7. ACT: Executes mitigation tool directly on network_config.json│      │
│  8. VERIFY: Health check, rolls back if resolution failed     ◀──┘      │
│  9. LEARN: Caches successful resolution for future NLP matching         │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## 🚀 How to Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 4. Set up the vector knowledge base (one-time)
python setup_db.py

# 5. Start the FastAPI backend
python main.py

# 6. Start the Streamlit dashboard (in a separate terminal)
streamlit run app.py

```

> **Note:** Ensure your `GOOGLE_API_KEY` is set in the `.env` file before running. The backend runs on `http://127.0.0.1:8000` and the dashboard on `http://localhost:8501`.

```

```
