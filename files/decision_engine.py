"""
Decision Engine + Zero Trust Policy Engine + Automated Response
===============================================================
Three tightly coupled components:

  DecisionEngine    — Combines DL, NLP, and Graph scores into final threat level
  ZeroTrustEngine   — OPA-style policy evaluation (deny-by-default, RBAC, MFA)
  ResponseOrchestrator — Maps threat level → automated actions

Final threat levels: CLEAR | LOW | MEDIUM | HIGH | CRITICAL
"""

import time
import uuid
import logging
import hashlib
import hmac
import json
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ── Enumerations ───────────────────────────────────────────────────────────────
class ThreatLevel(IntEnum):
    CLEAR    = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def label(self) -> str:
        return self.name


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ThreatDecision:
    threat_level:   ThreatLevel
    final_score:    float                    # weighted ensemble [0, 1]
    dl_score:       float                    # deep-learning component
    nlp_score:      float                    # NLP log analysis component
    graph_score:    float                    # graph anomaly component
    recommended_actions: List[str]
    explanation:    str
    decision_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PolicyRequest:
    user_id:        str
    role:           str           # viewer | analyst | admin | service
    resource:       str           # e.g. "/api/data", "/dashboard"
    action:         str           # read | write | delete | execute
    src_ip:         str
    device_id:      Optional[str] = None
    session_token:  Optional[str] = None
    mfa_verified:   bool = False
    threat_score:   float = 0.0   # from DecisionEngine
    user_agent:     str = ""
    timestamp:      float = field(default_factory=time.time)


@dataclass
class PolicyDecision:
    allowed:        bool
    reason:         str
    policy_rule:    str           # which rule triggered
    require_mfa:    bool = False
    session_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    evaluated_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ResponseAction:
    action_type:    str           # block_ip | terminate_session | trigger_mfa | alert_admin | rate_limit
    target:         str           # IP, session_id, user_id
    severity:       str
    message:        str
    audit_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    executed:       bool = False


# ── Decision Engine ────────────────────────────────────────────────────────────
class DecisionEngine:
    """
    Combines three model outputs into a final threat decision using
    configurable weighted ensemble scoring.

    Default weights (sum = 1.0):
      DL model  : 0.50  (most reliable — trained on labelled data)
      NLP module: 0.30  (log analysis, catches payload-level threats)
      Graph     : 0.20  (topology-level, good for lateral movement)
    """

    WEIGHTS = {"dl": 0.50, "nlp": 0.30, "graph": 0.20}

    # Score thresholds → ThreatLevel
    THRESHOLDS = [
        (0.85, ThreatLevel.CRITICAL),
        (0.70, ThreatLevel.HIGH),
        (0.50, ThreatLevel.MEDIUM),
        (0.25, ThreatLevel.LOW),
        (0.00, ThreatLevel.CLEAR),
    ]

    def decide(
        self,
        dl_score:    float,
        nlp_score:   float,
        graph_score: float,
        context:     Optional[Dict[str, Any]] = None,
    ) -> ThreatDecision:
        """
        Compute weighted ensemble score and map to ThreatLevel.

        dl_score    : P(malicious) from LSTM [0,1]
        nlp_score   : risk_score from NLP module [0,1]
        graph_score : anomaly_score from graph engine [0,1]
        context     : optional extra signals (e.g. {"is_internal": True})
        """
        w = self.WEIGHTS
        final_score = (
            w["dl"]    * dl_score
            + w["nlp"]   * nlp_score
            + w["graph"] * graph_score
        )

        # Context-based adjustments
        if context:
            # Internal traffic: slight reduction (legitimate lateral is common)
            if context.get("is_internal") and graph_score < 0.5:
                final_score *= 0.90
            # Known bad actor IP boosts score
            if context.get("blacklisted_ip"):
                final_score = min(final_score + 0.25, 1.0)
            # Threat intel enrichment
            if context.get("threat_intel_match"):
                final_score = min(final_score + 0.20, 1.0)

        final_score = round(min(max(final_score, 0.0), 1.0), 4)
        level       = self._score_to_level(final_score)
        actions     = self._recommend_actions(level)
        explanation = self._explain(dl_score, nlp_score, graph_score, final_score, level)

        return ThreatDecision(
            threat_level=level,
            final_score=final_score,
            dl_score=round(dl_score, 4),
            nlp_score=round(nlp_score, 4),
            graph_score=round(graph_score, 4),
            recommended_actions=actions,
            explanation=explanation,
        )

    def _score_to_level(self, score: float) -> ThreatLevel:
        for threshold, level in self.THRESHOLDS:
            if score >= threshold:
                return level
        return ThreatLevel.CLEAR

    def _recommend_actions(self, level: ThreatLevel) -> List[str]:
        action_map = {
            ThreatLevel.CRITICAL: ["block_ip", "terminate_session", "alert_admin", "trigger_mfa"],
            ThreatLevel.HIGH:     ["block_ip", "alert_admin", "trigger_mfa"],
            ThreatLevel.MEDIUM:   ["rate_limit", "alert_admin"],
            ThreatLevel.LOW:      ["monitor", "log_event"],
            ThreatLevel.CLEAR:    ["log_event"],
        }
        return action_map.get(level, ["log_event"])

    def _explain(
        self,
        dl: float, nlp: float, graph: float,
        final: float, level: ThreatLevel,
    ) -> str:
        parts = [
            f"DL={dl:.3f}×{self.WEIGHTS['dl']} "
            f"NLP={nlp:.3f}×{self.WEIGHTS['nlp']} "
            f"Graph={graph:.3f}×{self.WEIGHTS['graph']} "
            f"→ Ensemble={final:.4f} → {level.label()}"
        ]
        if dl > 0.8:
            parts.append("LSTM detected malicious packet sequence pattern")
        if nlp > 0.7:
            parts.append("NLP identified high-risk log indicators")
        if graph > 0.7:
            parts.append("Graph engine detected unusual communication topology")
        return ". ".join(parts)


# ── Zero Trust Policy Engine ───────────────────────────────────────────────────
# Role-permission matrix (RBAC)
RBAC: Dict[str, List[str]] = {
    "viewer":   ["read"],
    "analyst":  ["read", "analyze", "export_logs"],
    "admin":    ["read", "write", "delete", "execute", "manage_policies", "export_logs"],
    "service":  ["read", "write", "execute"],   # service accounts
}

# Resource-permission requirements
RESOURCE_PERMS: Dict[str, str] = {
    "/dashboard":          "read",
    "/api/threats":        "read",
    "/api/threats/block":  "write",
    "/api/policies":       "manage_policies",
    "/api/logs":           "export_logs",
    "/api/predict":        "read",
    "/api/admin":          "manage_policies",
    "/api/retrain":        "execute",
}


class ZeroTrustEngine:
    """
    Policy enforcement engine implementing Zero Trust principles:
      • Deny by default
      • Least privilege (RBAC)
      • Continuous re-authentication
      • Device trust verification
      • Behaviour-based lockout (threat score)
      • MFA requirement for sensitive operations

    No dependency on an external OPA server — implements Rego-equivalent
    logic in pure Python for embedded deployment.
    (For production, this class can delegate to OPA via HTTP.)
    """

    # Sensitive actions always require MFA
    MFA_REQUIRED_PERMS = {"write", "delete", "execute", "manage_policies"}
    # Threshold above which a session is terminated
    THREAT_LOCKOUT_SCORE = 0.75
    # Maximum session age in seconds (15 minutes)
    SESSION_TTL = 900

    def __init__(self):
        self._sessions:  Dict[str, dict]       = {}
        self._blocklist: set                   = set()
        self._devices:   set                   = {"device-corp-001", "device-corp-002"}
        self._audit_log: List[dict]            = []

    # ── Main evaluation ────────────────────────────────────────────────────────

    def evaluate(self, req: PolicyRequest) -> PolicyDecision:
        """
        Evaluate a policy request and return allow/deny + reasoning.
        Evaluations are logged to the internal audit trail.
        """
        result = self._run_policy(req)
        self._audit(req, result)
        return result

    def _run_policy(self, req: PolicyRequest) -> PolicyDecision:
        """
        Execute all policy rules in priority order.
        First matching deny rule short-circuits.
        """

        # ── Rule 1: IP blocklist ───────────────────────────────────────────────
        if req.src_ip in self._blocklist:
            return PolicyDecision(
                allowed=False,
                reason=f"Source IP {req.src_ip} is blocklisted",
                policy_rule="ip_blocklist",
            )

        # ── Rule 2: Threat score lockout ───────────────────────────────────────
        if req.threat_score >= self.THREAT_LOCKOUT_SCORE:
            return PolicyDecision(
                allowed=False,
                reason=f"Session threat score {req.threat_score:.2f} exceeds lockout threshold {self.THREAT_LOCKOUT_SCORE}",
                policy_rule="threat_score_lockout",
            )

        # ── Rule 3: Token/session validation ──────────────────────────────────
        if not self._validate_session(req):
            return PolicyDecision(
                allowed=False,
                reason="Session token invalid or expired",
                policy_rule="session_validation",
                require_mfa=True,
            )

        # ── Rule 4: Device trust ───────────────────────────────────────────────
        if req.device_id and req.device_id not in self._devices:
            return PolicyDecision(
                allowed=False,
                reason=f"Device {req.device_id} is not in the trusted device registry",
                policy_rule="device_trust",
                require_mfa=True,
            )

        # ── Rule 5: RBAC permission check ─────────────────────────────────────
        required_perm = RESOURCE_PERMS.get(req.resource, "read")
        role_perms    = RBAC.get(req.role, [])
        if required_perm not in role_perms:
            return PolicyDecision(
                allowed=False,
                reason=f"Role '{req.role}' lacks permission '{required_perm}' for {req.resource}",
                policy_rule="rbac_permission",
            )

        # ── Rule 6: MFA enforcement for sensitive operations ───────────────────
        if required_perm in self.MFA_REQUIRED_PERMS and not req.mfa_verified:
            return PolicyDecision(
                allowed=False,
                reason=f"Action '{req.action}' on {req.resource} requires MFA verification",
                policy_rule="mfa_required",
                require_mfa=True,
            )

        # ── Rule 7: Behavioural anomaly check ─────────────────────────────────
        if req.threat_score > 0.40:
            return PolicyDecision(
                allowed=False,
                reason=f"Elevated threat score ({req.threat_score:.2f}) — step-up auth required",
                policy_rule="behavioural_anomaly",
                require_mfa=True,
            )

        # ── Allow ──────────────────────────────────────────────────────────────
        return PolicyDecision(
            allowed=True,
            reason=f"All policies satisfied for {req.role}/{req.user_id} → {req.resource}",
            policy_rule="allow",
        )

    # ── Session management ─────────────────────────────────────────────────────

    def _validate_session(self, req: PolicyRequest) -> bool:
        """Verify session token exists, is unexpired, and matches user."""
        if not req.session_token:
            return False
        session = self._sessions.get(req.session_token)
        if not session:
            return False
        age = time.time() - session["created_at"]
        if age > self.SESSION_TTL:
            del self._sessions[req.session_token]
            return False
        return session.get("user_id") == req.user_id

    def create_session(self, user_id: str, role: str) -> str:
        """Create a new session token (call after successful OAuth2/MFA)."""
        token = hmac.new(
            key=b"secret-key-rotate-in-prod",
            msg=f"{user_id}{time.time()}".encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()
        self._sessions[token] = {
            "user_id":    user_id,
            "role":       role,
            "created_at": time.time(),
        }
        return token

    def block_ip(self, ip: str):
        self._blocklist.add(ip)
        logger.info("ZeroTrust: blocked IP %s", ip)

    def unblock_ip(self, ip: str):
        self._blocklist.discard(ip)

    def register_device(self, device_id: str):
        self._devices.add(device_id)

    # ── Audit trail ────────────────────────────────────────────────────────────

    def _audit(self, req: PolicyRequest, decision: PolicyDecision):
        entry = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "user":     req.user_id,
            "role":     req.role,
            "resource": req.resource,
            "action":   req.action,
            "src_ip":   req.src_ip,
            "allowed":  decision.allowed,
            "rule":     decision.policy_rule,
            "reason":   decision.reason,
        }
        self._audit_log.append(entry)
        lvl = logging.INFO if decision.allowed else logging.WARNING
        logger.log(lvl, "ZeroTrust | %s | %s | %s→%s | %s",
                   "ALLOW" if decision.allowed else "DENY",
                   req.user_id, req.src_ip, req.resource, decision.reason)

    def get_audit_log(self, limit: int = 100) -> List[dict]:
        return self._audit_log[-limit:]


# ── Automated Response Orchestrator ───────────────────────────────────────────
class ResponseOrchestrator:
    """
    Translates threat decisions into concrete security actions.
    All actions are logged; execution adapters are pluggable.
    """

    def __init__(
        self,
        policy_engine: ZeroTrustEngine,
        alert_callback=None,    # async callable(ResponseAction)
    ):
        self.policy       = policy_engine
        self._alert_cb    = alert_callback or self._default_alert
        self._action_log: List[ResponseAction] = []

    def respond(
        self,
        decision: ThreatDecision,
        src_ip:   str,
        session_id: Optional[str] = None,
        user_id:    Optional[str] = None,
    ) -> List[ResponseAction]:
        """
        Execute all recommended actions for a ThreatDecision.
        Returns list of ResponseAction records.
        """
        actions_taken: List[ResponseAction] = []

        for action_type in decision.recommended_actions:
            action = self._execute(
                action_type, decision, src_ip, session_id, user_id
            )
            if action:
                actions_taken.append(action)
                self._action_log.append(action)

        return actions_taken

    def _execute(
        self,
        action_type:  str,
        decision:     ThreatDecision,
        src_ip:       str,
        session_id:   Optional[str],
        user_id:      Optional[str],
    ) -> Optional[ResponseAction]:
        """Dispatch to the appropriate handler."""
        handlers = {
            "block_ip":          self._block_ip,
            "terminate_session": self._terminate_session,
            "trigger_mfa":       self._trigger_mfa,
            "alert_admin":       self._alert_admin,
            "rate_limit":        self._rate_limit,
            "monitor":           self._monitor,
            "log_event":         self._log_event,
        }
        handler = handlers.get(action_type)
        if not handler:
            logger.warning("Unknown action type: %s", action_type)
            return None
        return handler(decision, src_ip, session_id, user_id)

    # ── Action handlers ────────────────────────────────────────────────────────

    def _block_ip(self, d: ThreatDecision, ip, _sid, _uid) -> ResponseAction:
        self.policy.block_ip(ip)
        action = ResponseAction(
            action_type="block_ip",
            target=ip,
            severity=d.threat_level.label(),
            message=f"IP {ip} blocked. Score={d.final_score:.3f}. {d.explanation}",
            executed=True,
        )
        logger.critical("RESPONSE: Blocked IP %s (score=%.3f)", ip, d.final_score)
        return action

    def _terminate_session(self, d, ip, sid, uid) -> Optional[ResponseAction]:
        if not sid:
            return None
        # Remove from active sessions
        if sid in self.policy._sessions:
            del self.policy._sessions[sid]
        action = ResponseAction(
            action_type="terminate_session",
            target=sid or "unknown",
            severity=d.threat_level.label(),
            message=f"Session terminated. User={uid}. Score={d.final_score:.3f}",
            executed=True,
        )
        logger.critical("RESPONSE: Terminated session %s for user %s", sid, uid)
        return action

    def _trigger_mfa(self, d, ip, sid, uid) -> ResponseAction:
        otp = str(uuid.uuid4().int)[:6]   # In production: use TOTP/HOTP
        action = ResponseAction(
            action_type="trigger_mfa",
            target=uid or ip,
            severity=d.threat_level.label(),
            message=f"MFA OTP generated for {uid or ip}. OTP={otp} (demo only — use real TOTP)",
            executed=True,
        )
        logger.warning("RESPONSE: MFA triggered for user=%s ip=%s", uid, ip)
        self._alert_cb(action)
        return action

    def _alert_admin(self, d, ip, sid, uid) -> ResponseAction:
        action = ResponseAction(
            action_type="alert_admin",
            target=ip,
            severity=d.threat_level.label(),
            message=(
                f"[{d.threat_level.label()}] Threat detected from {ip}. "
                f"Score={d.final_score:.3f}. {d.explanation}"
            ),
            executed=True,
        )
        self._alert_cb(action)
        return action

    def _rate_limit(self, d, ip, sid, uid) -> ResponseAction:
        # In production: integrate with nginx/iptables rate limiting
        return ResponseAction(
            action_type="rate_limit",
            target=ip,
            severity=d.threat_level.label(),
            message=f"Rate limit applied to {ip} (100 req/min). Score={d.final_score:.3f}",
            executed=True,
        )

    def _monitor(self, d, ip, sid, uid) -> ResponseAction:
        return ResponseAction(
            action_type="monitor",
            target=ip,
            severity=d.threat_level.label(),
            message=f"Enhanced monitoring enabled for {ip}. Score={d.final_score:.3f}",
            executed=True,
        )

    def _log_event(self, d, ip, sid, uid) -> ResponseAction:
        return ResponseAction(
            action_type="log_event",
            target=ip,
            severity=d.threat_level.label(),
            message=f"Event logged. {d.explanation}",
            executed=True,
        )

    @staticmethod
    def _default_alert(action: ResponseAction):
        logger.warning("ALERT: [%s] %s → %s",
                       action.severity, action.action_type, action.message)

    def get_action_log(self, limit: int = 200) -> List[dict]:
        return [asdict(a) for a in self._action_log[-limit:]]


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Decision + Policy + Response — Smoke Test ===\n")

    engine     = DecisionEngine()
    zero_trust = ZeroTrustEngine()
    responder  = ResponseOrchestrator(zero_trust)

    # ── Test 1: Critical SQL injection ────────────────────────────────────────
    d = engine.decide(dl_score=0.97, nlp_score=0.91, graph_score=0.84)
    print(f"[SQL Injection] → {d.threat_level.label()} score={d.final_score}")
    print(f"  Actions: {d.recommended_actions}")
    print(f"  {d.explanation}\n")
    actions = responder.respond(d, src_ip="203.0.113.47")
    for a in actions:
        print(f"  ✓ {a.action_type}: {a.message[:70]}")

    # ── Test 2: Normal traffic ─────────────────────────────────────────────────
    print()
    d2 = engine.decide(dl_score=0.04, nlp_score=0.02, graph_score=0.01)
    print(f"[Normal]       → {d2.threat_level.label()} score={d2.final_score}")

    # ── Test 3: Zero Trust policy ──────────────────────────────────────────────
    print("\n--- Zero Trust Evaluations ---")
    token = zero_trust.create_session("alice", "admin")

    req = PolicyRequest(
        user_id="alice", role="admin",
        resource="/api/threats/block", action="write",
        src_ip="10.0.0.5", session_token=token,
        mfa_verified=True, threat_score=0.02,
    )
    pd_result = zero_trust.evaluate(req)
    print(f"Admin write (MFA=True):  {'ALLOW' if pd_result.allowed else 'DENY'} — {pd_result.reason}")

    req2 = PolicyRequest(
        user_id="bob", role="viewer",
        resource="/api/policies", action="write",
        src_ip="10.0.0.6", session_token="bad-token",
        mfa_verified=False, threat_score=0.1,
    )
    pd2 = zero_trust.evaluate(req2)
    print(f"Viewer write (no token): {'ALLOW' if pd2.allowed else 'DENY'} — {pd2.reason}")
