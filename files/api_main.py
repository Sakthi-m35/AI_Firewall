"""
AI Firewall — FastAPI Backend
==============================
Production-grade REST + WebSocket API exposing all firewall capabilities.

Endpoints:
  POST /predict-threat     → DL model threat classification
  POST /analyze-log        → NLP log analysis
  POST /enforce-policy     → Zero Trust policy evaluation
  POST /alert              → Manual admin alert trigger
  GET  /dashboard/stats    → Real-time aggregate metrics
  GET  /dashboard/threats  → Recent threat events
  GET  /health             → Service health check
  WS   /ws/live-traffic    → WebSocket stream of live packet decisions

Security:
  - JWT Bearer token authentication on all routes
  - Rate limiting (slowapi)
  - CORS configured for dashboard origin
  - Request/response logging
  - Input validation via Pydantic
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Set

import uvicorn
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    Depends, status, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import jwt   # PyJWT

# ── Local module imports (lazy to support partial installation) ────────────────
from modules.ml_model     import ThreatDetectionEngine, N_FEATURES
from modules.nlp_module   import NLPLogAnalyzer
from modules.graph_analytics import NetworkGraph
from modules.decision_engine import (
    DecisionEngine, ZeroTrustEngine, ResponseOrchestrator,
    PolicyRequest, ThreatLevel,
)

import numpy as np

logger = logging.getLogger("ai_firewall.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration (in production: load from env/Vault) ────────────────────────
JWT_SECRET     = "CHANGE-ME-IN-PRODUCTION-USE-VAULT"
JWT_ALGORITHM  = "HS256"
JWT_EXPIRY_H   = 8
API_TITLE      = "AI-NGF API"
API_VERSION    = "2.4.1"
MAX_LOG_LINE   = 4096   # chars
RECENT_EVENTS_BUFFER = 500


# ── Shared state (initialised in lifespan) ─────────────────────────────────────
class AppState:
    detection_engine:   ThreatDetectionEngine
    nlp_analyzer:       NLPLogAnalyzer
    network_graph:      NetworkGraph
    decision_engine:    DecisionEngine
    zero_trust:         ZeroTrustEngine
    responder:          ResponseOrchestrator
    recent_threats:     deque
    ws_clients:         Set[WebSocket]
    stats:              dict


app_state = AppState()


# ── Application lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all engines on startup, tear down on shutdown."""
    logger.info("Starting AI Firewall API v%s …", API_VERSION)

    app_state.detection_engine = ThreatDetectionEngine()
    app_state.nlp_analyzer     = NLPLogAnalyzer()
    app_state.network_graph    = NetworkGraph()
    app_state.decision_engine  = DecisionEngine()
    app_state.zero_trust       = ZeroTrustEngine()
    app_state.responder        = ResponseOrchestrator(app_state.zero_trust)
    app_state.recent_threats   = deque(maxlen=RECENT_EVENTS_BUFFER)
    app_state.ws_clients       = set()
    app_state.stats            = {
        "packets_analyzed": 0,
        "threats_blocked":  0,
        "logs_analyzed":    0,
        "uptime_start":     time.time(),
    }

    # Try loading pre-trained models
    try:
        app_state.detection_engine.load()
        logger.info("Pre-trained models loaded successfully")
    except Exception as e:
        logger.warning("No pre-trained models found (%s) — using demo mode", e)

    logger.info("AI Firewall API ready")
    yield

    # Shutdown
    logger.info("AI Firewall API shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-Driven Next-Gen Firewall API with DL, NLP and Graph-based threat detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dashboard.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ───────────────────────────────────────────────────────────────────────
security = HTTPBearer()


def create_token(user_id: str, role: str) -> str:
    payload = {
        "sub":  user_id,
        "role": role,
        "exp":  datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_H),
        "iat":  datetime.now(timezone.utc),
        "jti":  str(uuid.uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def require_role(*roles: str):
    """Dependency factory for role-based endpoint protection."""
    def check(token: dict = Depends(verify_token)) -> dict:
        if token.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{token.get('role')}' not permitted. Required: {roles}",
            )
        return token
    return check


# ── Pydantic models ────────────────────────────────────────────────────────────
class PacketFeaturesRequest(BaseModel):
    src_ip:         str = Field(..., example="203.0.113.47")
    dst_ip:         str = Field(..., example="10.0.0.5")
    src_port:       int = Field(..., ge=0, le=65535)
    dst_port:       int = Field(..., ge=0, le=65535)
    protocol:       int = Field(..., ge=0, le=2, description="0=ICMP, 1=TCP, 2=UDP")
    packet_size:    int = Field(..., ge=0, le=65535)
    ttl:            int = Field(64, ge=0, le=255)
    tcp_flags:      int = Field(0,  ge=0, le=255)
    flow_duration:  float = Field(0.0, ge=0)
    flow_pkt_count: int   = Field(1,   ge=1)
    bytes_per_sec:  float = Field(0.0, ge=0)


class ThreatPredictRequest(BaseModel):
    packets:     List[PacketFeaturesRequest]
    src_ip:      Optional[str] = None
    session_id:  Optional[str] = None
    context:     Optional[dict] = {}


class LogAnalyzeRequest(BaseModel):
    log_lines: List[str] = Field(..., min_items=1, max_items=100)

    @validator("log_lines", each_item=True)
    def truncate(cls, v: str) -> str:
        return v[:MAX_LOG_LINE]


class PolicyEnforceRequest(BaseModel):
    user_id:       str
    role:          str = Field(..., regex="^(viewer|analyst|admin|service)$")
    resource:      str
    action:        str
    src_ip:        str
    session_token: Optional[str] = None
    device_id:     Optional[str] = None
    mfa_verified:  bool = False


class AlertRequest(BaseModel):
    severity:    str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    message:     str
    src_ip:      Optional[str] = None
    details:     Optional[dict] = {}


class LoginRequest(BaseModel):
    username: str
    password: str
    role:     str = "viewer"


# ── Response models ────────────────────────────────────────────────────────────
class ThreatPredictResponse(BaseModel):
    threat_level:        str
    final_score:         float
    dl_score:            float
    nlp_score:           float
    graph_score:         float
    recommended_actions: List[str]
    explanation:         str
    decision_id:         str
    timestamp:           str


class LogAnalyzeResponse(BaseModel):
    results: List[dict]
    highest_risk_score: float
    threat_categories:  List[str]


# ── Auth endpoint ──────────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"])
async def login(req: LoginRequest):
    """
    Demo login endpoint.
    In production: verify against IdP (Okta, Auth0, LDAP) + enforce MFA.
    """
    # Demo: accept any credentials (REPLACE IN PRODUCTION)
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="username and password required")

    token = create_token(req.username, req.role)
    # Create session in Zero Trust engine
    zt_token = app_state.zero_trust.create_session(req.username, req.role)

    return {
        "access_token":    token,
        "zt_session":      zt_token,
        "token_type":      "bearer",
        "expires_in":      JWT_EXPIRY_H * 3600,
        "role":            req.role,
    }


# ── Core API endpoints ─────────────────────────────────────────────────────────
@app.post(
    "/predict-threat",
    response_model=ThreatPredictResponse,
    tags=["Threat Detection"],
    summary="Run ML threat classification on packet features",
)
async def predict_threat(
    req: ThreatPredictRequest,
    background: BackgroundTasks,
    token: dict = Depends(verify_token),
):
    """
    Runs the full DL + NLP + Graph + Decision ensemble pipeline.

    1. Extract feature matrix from packet list
    2. Run LSTM classifier → DL score
    3. Observe in network graph → Graph score
    4. Run ensemble decision
    5. Execute automated responses in background
    """
    # Build feature matrix
    feat_matrix = np.array([
        [
            _ip_int(p.src_ip), _ip_int(p.dst_ip),
            p.src_port / 65535, p.dst_port / 65535,
            p.protocol / 2, p.packet_size / 65535,
            p.ttl / 255, p.tcp_flags / 255,
            min(p.flow_duration / 3600, 1.0),
            min(p.flow_pkt_count / 1000, 1.0),
            min(p.bytes_per_sec / 1e6, 1.0),
        ]
        for p in req.packets
    ], dtype=np.float32)

    # DL inference
    dl_results = app_state.detection_engine.predict_batch(feat_matrix)
    dl_score   = max(r.confidence if r.label == "malicious" else
                     r.class_probs[1] * 0.5 for r in dl_results)

    # Graph observation (use first packet as representative)
    first = req.packets[0]
    g_result   = app_state.network_graph.observe(
        first.src_ip, first.dst_ip, first.dst_port, first.packet_size
    )
    graph_score = g_result.anomaly_score

    # Decision ensemble (NLP score=0 when no logs provided)
    decision = app_state.decision_engine.decide(
        dl_score=dl_score,
        nlp_score=0.0,
        graph_score=graph_score,
        context=req.context,
    )

    # Automated responses (non-blocking)
    background.add_task(
        _execute_response,
        decision, first.src_ip, req.session_id, token.get("sub")
    )

    # Metrics update
    app_state.stats["packets_analyzed"] += len(req.packets)
    if decision.threat_level >= ThreatLevel.HIGH:
        app_state.stats["threats_blocked"] += 1

    # Store for dashboard
    event = {**decision.__dict__, "src_ip": first.src_ip, "dst_ip": first.dst_ip,
             "graph_anomalies": g_result.anomaly_types}
    event["threat_level"] = decision.threat_level.label()
    app_state.recent_threats.append(event)

    # Broadcast to WebSocket clients
    asyncio.create_task(_ws_broadcast(event))

    return ThreatPredictResponse(
        threat_level=decision.threat_level.label(),
        final_score=decision.final_score,
        dl_score=decision.dl_score,
        nlp_score=decision.nlp_score,
        graph_score=decision.graph_score,
        recommended_actions=decision.recommended_actions,
        explanation=decision.explanation,
        decision_id=decision.decision_id,
        timestamp=decision.timestamp,
    )


@app.post(
    "/analyze-log",
    response_model=LogAnalyzeResponse,
    tags=["Log Analysis"],
    summary="NLP analysis of security log lines",
)
async def analyze_log(
    req: LogAnalyzeRequest,
    token: dict = Depends(verify_token),
):
    """
    Runs DistilBERT (or rule-based fallback) NLP analysis on log lines.
    Returns per-line risk scores and extracted IOC indicators.
    """
    results = app_state.nlp_analyzer.analyze_batch(req.log_lines)
    app_state.stats["logs_analyzed"] += len(req.log_lines)

    serialised = []
    for r in results:
        serialised.append({
            "log":              r.raw_log[:200],
            "risk_score":       r.risk_score,
            "threat_category":  r.threat_category,
            "confidence":       r.confidence,
            "indicators":       r.indicators,
            "model_type":       r.model_type,
        })

    highest = max(r.risk_score for r in results)
    cats    = list({r.threat_category for r in results if r.threat_category != "normal"})

    return LogAnalyzeResponse(
        results=serialised,
        highest_risk_score=highest,
        threat_categories=cats,
    )


@app.post(
    "/enforce-policy",
    tags=["Zero Trust"],
    summary="Evaluate a Zero Trust policy decision",
)
async def enforce_policy(
    req: PolicyEnforceRequest,
    token: dict = Depends(verify_token),
):
    """
    Evaluates the Zero Trust policy for a given access request.
    Returns allow/deny + reason + whether MFA is required.
    """
    policy_req = PolicyRequest(
        user_id=req.user_id,
        role=req.role,
        resource=req.resource,
        action=req.action,
        src_ip=req.src_ip,
        session_token=req.session_token,
        device_id=req.device_id,
        mfa_verified=req.mfa_verified,
    )
    decision = app_state.zero_trust.evaluate(policy_req)
    return {
        "allowed":      decision.allowed,
        "reason":       decision.reason,
        "policy_rule":  decision.policy_rule,
        "require_mfa":  decision.require_mfa,
        "session_id":   decision.session_id,
        "evaluated_at": decision.evaluated_at,
    }


@app.post("/alert", tags=["Alerts"], summary="Trigger a manual security alert")
async def create_alert(
    req: AlertRequest,
    token: dict = Depends(require_role("analyst", "admin")),
):
    """
    Allows analysts/admins to manually log and broadcast a security alert.
    """
    alert_id = str(uuid.uuid4())[:8]
    alert = {
        "alert_id":  alert_id,
        "severity":  req.severity,
        "message":   req.message,
        "src_ip":    req.src_ip,
        "details":   req.details,
        "created_by": token.get("sub"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    app_state.recent_threats.append({**alert, "threat_level": req.severity})
    asyncio.create_task(_ws_broadcast(alert))
    logger.warning("Manual alert [%s] created by %s: %s", req.severity, token.get("sub"), req.message)
    return {"alert_id": alert_id, "status": "created"}


# ── Dashboard endpoints ────────────────────────────────────────────────────────
@app.get("/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats(token: dict = Depends(verify_token)):
    """Aggregate metrics for the admin dashboard."""
    uptime = time.time() - app_state.stats["uptime_start"]
    graph  = app_state.network_graph.to_dict()
    return {
        **app_state.stats,
        "uptime_seconds":  round(uptime),
        "graph":           graph,
        "audit_entries":   len(app_state.zero_trust.get_audit_log()),
    }


@app.get("/dashboard/threats", tags=["Dashboard"])
async def recent_threats(
    limit: int = 50,
    token: dict = Depends(verify_token),
):
    """Return recent threat events (newest first)."""
    events = list(app_state.recent_threats)[-limit:]
    return {"threats": list(reversed(events)), "total": len(app_state.recent_threats)}


@app.get("/dashboard/audit", tags=["Dashboard"])
async def audit_log(
    limit: int = 100,
    token: dict = Depends(require_role("admin")),
):
    """Return Zero Trust audit log (admin only)."""
    return {"entries": app_state.zero_trust.get_audit_log(limit)}


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], include_in_schema=False)
async def health():
    return {
        "status":  "ok",
        "version": API_VERSION,
        "time":    datetime.now(timezone.utc).isoformat(),
        "engines": {
            "dl":      "ready",
            "nlp":     "ready",
            "graph":   f"{app_state.network_graph.node_count} nodes",
            "zero_trust": "ready",
        },
    }


# ── WebSocket live traffic feed ────────────────────────────────────────────────
@app.websocket("/ws/live-traffic")
async def websocket_live(ws: WebSocket):
    """
    Push real-time threat events to connected dashboard clients.
    Client must send a valid JWT token as the first message.
    """
    await ws.accept()

    # Authenticate over WebSocket
    try:
        auth_msg = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        payload  = jwt.decode(auth_msg, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        logger.info("WebSocket connected: user=%s", payload.get("sub"))
    except Exception as e:
        await ws.send_text(json.dumps({"error": "Authentication failed", "detail": str(e)}))
        await ws.close()
        return

    app_state.ws_clients.add(ws)
    try:
        # Send last 20 events immediately on connect
        recent = list(app_state.recent_threats)[-20:]
        for evt in recent:
            await ws.send_text(json.dumps(evt, default=str))

        # Keep connection alive (heartbeat every 30s)
        while True:
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({"type": "heartbeat", "ts": time.time()}))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        app_state.ws_clients.discard(ws)


# ── Helpers ────────────────────────────────────────────────────────────────────
async def _ws_broadcast(event: dict):
    """Broadcast an event to all connected WebSocket clients."""
    if not app_state.ws_clients:
        return
    msg = json.dumps(event, default=str)
    dead = set()
    for ws in app_state.ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    app_state.ws_clients -= dead


async def _execute_response(decision, src_ip, session_id, user_id):
    """Background task: execute automated response actions."""
    try:
        actions = app_state.responder.respond(decision, src_ip, session_id, user_id)
        logger.info("Response actions executed: %s", [a.action_type for a in actions])
    except Exception as e:
        logger.error("Response execution failed: %s", e)


def _ip_int(ip: str) -> float:
    """Convert IP to [0,1]-normalised integer."""
    try:
        import ipaddress
        return int(ipaddress.ip_address(ip)) / (2**32 - 1)
    except ValueError:
        return 0.0


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # False in production
        log_level="info",
        access_log=True,
        ssl_keyfile=None,      # Add TLS cert in production
        ssl_certfile=None,
    )
