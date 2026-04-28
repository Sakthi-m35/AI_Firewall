"""
Graph-Based Anomaly Detection Module
=====================================
Represents the network as a directed weighted graph:
  Nodes  = devices / users  (IP addresses or hostnames)
  Edges  = communication events (weighted by frequency + recency)

Detects:
  • Lateral movement   (unusual internal traversal paths)
  • New edges          (devices communicating for the first time)
  • Hub anomalies      (sudden degree explosion)
  • Community breaks   (traffic between isolated network segments)
  • Port scan patterns (one node touching many ports on a target)

Output per query: GraphAnomalyResult with score [0,1] and explanations.
"""

import time
import math
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EDGE_DECAY_HALF_LIFE   = 3600.0    # seconds — recent traffic is weighted higher
NEW_EDGE_RISK          = 0.65      # base risk for first-seen communication
LATERAL_MOVEMENT_RISK  = 0.85      # risk for detected lateral movement path
HUB_EXPLOSION_RISK     = 0.75      # risk for sudden degree spike
COMMUNITY_BREACH_RISK  = 0.80      # risk for cross-community traffic
PORT_SCAN_RISK         = 0.78      # risk for port-scan-like behaviour
PORT_SCAN_THRESHOLD    = 15        # distinct ports in SCAN_WINDOW → port scan
SCAN_WINDOW_SECONDS    = 60
GRAPH_REBUILD_INTERVAL = 300       # rebuild community structure every 5 min
MAX_GRAPH_NODES        = 50_000    # hard cap to bound memory


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class GraphAnomalyResult:
    anomaly_score:   float                  # 0 = normal, 1 = highly anomalous
    anomaly_types:   List[str]              # e.g. ["lateral_movement", "new_edge"]
    path_found:      Optional[List[str]]    # traversal path if lateral movement
    src_node:        str
    dst_node:        str
    edge_weight:     float                  # historical edge weight (0 if new)
    explanation:     str                    # human-readable reason


# ── Edge metadata ──────────────────────────────────────────────────────────────
@dataclass
class EdgeData:
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)
    pkt_count:   int   = 0
    byte_count:  int   = 0
    ports_seen:  Set[int] = field(default_factory=set)

    def weight(self, now: Optional[float] = None) -> float:
        """Exponentially decayed weight favouring recent traffic."""
        now = now or time.time()
        age    = now - self.last_seen
        decay  = math.exp(-age / EDGE_DECAY_HALF_LIFE)
        volume = math.log1p(self.pkt_count)
        return decay * volume


# ── Port-scan tracker ──────────────────────────────────────────────────────────
class PortScanTracker:
    """Per-source rolling window of (dst_ip, dst_port) pairs."""

    def __init__(self, window: float = SCAN_WINDOW_SECONDS):
        self._window   = window
        self._events: Dict[str, deque] = defaultdict(deque)
        self._lock     = threading.Lock()

    def record(self, src: str, dst: str, port: int) -> bool:
        """Return True if this event tips src into port-scan territory."""
        now = time.time()
        with self._lock:
            q = self._events[src]
            q.append((now, dst, port))
            # Expire old events
            while q and now - q[0][0] > self._window:
                q.popleft()
            distinct_ports = len({p for _, _, p in q})
            return distinct_ports >= PORT_SCAN_THRESHOLD


# ── Network graph ──────────────────────────────────────────────────────────────
class NetworkGraph:
    """
    Live network communication graph with anomaly detection.
    Thread-safe via a read-write lock (rwlock via threading.Lock).
    """

    def __init__(self):
        self.G                   = nx.DiGraph()
        self._lock               = threading.Lock()
        self._communities: Dict[str, int] = {}    # node → community id
        self._last_rebuild       = 0.0
        self._port_scanner       = PortScanTracker()
        self._baseline: Dict[Tuple[str,str], float] = {}  # edge → historical mean weight

        # Background community rebuild
        self._rebuild_thread = threading.Thread(
            target=self._rebuild_loop, daemon=True
        )
        self._rebuild_thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(
        self,
        src: str, dst: str, port: int,
        pkt_size: int = 0,
    ) -> GraphAnomalyResult:
        """
        Record a communication event and return an anomaly assessment.
        This is the hot path — must be fast.
        """
        now = time.time()
        anomaly_types: List[str] = []
        score = 0.0
        path_found = None

        with self._lock:
            is_new_edge = not self.G.has_edge(src, dst)

            # Ensure nodes exist
            if src not in self.G:
                self.G.add_node(src, first_seen=now, degree_history=[])
            if dst not in self.G:
                self.G.add_node(dst, first_seen=now, degree_history=[])

            # Update edge
            if is_new_edge:
                self.G.add_edge(src, dst, data=EdgeData())
            edata: EdgeData = self.G[src][dst]["data"]
            edata.last_seen = now
            edata.pkt_count += 1
            edata.byte_count += pkt_size
            edata.ports_seen.add(port)

            edge_weight = edata.weight(now)

            # ── Anomaly checks ─────────────────────────────────────────────────

            # 1. New edge (first-time communication)
            if is_new_edge:
                anomaly_types.append("new_edge")
                score = max(score, NEW_EDGE_RISK)

            # 2. Community breach (src and dst in different communities)
            if self._is_community_breach(src, dst):
                anomaly_types.append("community_breach")
                score = max(score, COMMUNITY_BREACH_RISK)

            # 3. Lateral movement detection (internal→internal unusual path)
            if self._is_internal(src) and self._is_internal(dst):
                lm_path = self._detect_lateral_movement(src, dst)
                if lm_path:
                    anomaly_types.append("lateral_movement")
                    score = max(score, LATERAL_MOVEMENT_RISK)
                    path_found = lm_path

            # 4. Hub explosion (sudden large degree increase)
            if self._is_hub_explosion(src):
                anomaly_types.append("hub_explosion")
                score = max(score, HUB_EXPLOSION_RISK)

        # 5. Port scan (checked outside lock, has its own lock)
        if self._port_scanner.record(src, dst, port):
            anomaly_types.append("port_scan")
            score = max(score, PORT_SCAN_RISK)

        explanation = self._explain(src, dst, anomaly_types, score, path_found)

        return GraphAnomalyResult(
            anomaly_score=round(min(score, 1.0), 4),
            anomaly_types=anomaly_types,
            path_found=path_found,
            src_node=src,
            dst_node=dst,
            edge_weight=round(edge_weight, 4) if not is_new_edge else 0.0,
            explanation=explanation,
        )

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def get_risk_nodes(self, top_n: int = 20) -> List[dict]:
        """Return the top-N nodes by betweenness centrality (potential pivots)."""
        with self._lock:
            if self.G.number_of_nodes() < 3:
                return []
            try:
                bc = nx.betweenness_centrality(self.G, normalized=True)
                sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_n]
                return [{"node": n, "centrality": round(c, 4)} for n, c in sorted_bc]
            except Exception:
                return []

    def to_dict(self) -> dict:
        """Serialise graph stats (for dashboard API)."""
        with self._lock:
            return {
                "nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "communities": len(set(self._communities.values())),
                "risk_nodes": self.get_risk_nodes(5),
            }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_internal(ip: str) -> bool:
        """Quick RFC-1918 check."""
        return (
            ip.startswith("10.")
            or ip.startswith("192.168.")
            or ip.startswith("172.16.")
            or ip.startswith("172.17.")
        )

    def _is_community_breach(self, src: str, dst: str) -> bool:
        c_src = self._communities.get(src, -1)
        c_dst = self._communities.get(dst, -2)
        return c_src != c_dst and c_src != -1 and c_dst != -2

    def _detect_lateral_movement(
        self, src: str, dst: str
    ) -> Optional[List[str]]:
        """
        Lateral movement heuristic: src reached dst via intermediate hops
        (i.e., the direct edge is new but multi-hop path exists through
        nodes that were already compromised/suspicious).
        Uses shortest-path length as a signal.
        """
        try:
            # If there's a path ≥2 hops through internal nodes, flag it
            if nx.has_path(self.G, src, dst):
                path = nx.shortest_path(self.G, src, dst)
                # Only flag if intermediate hops exist (path length > 2)
                if len(path) >= 3:
                    return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None

    def _is_hub_explosion(self, node: str, threshold_factor: float = 3.0) -> bool:
        """Flag if out-degree has grown > threshold_factor × historical mean."""
        current_degree = self.G.out_degree(node)
        history = self.G.nodes[node].get("degree_history", [])
        if len(history) < 5:
            # Record and don't flag yet
            history.append(current_degree)
            self.G.nodes[node]["degree_history"] = history[-100:]
            return False
        mean_degree = np.mean(history)
        self.G.nodes[node]["degree_history"] = (history + [current_degree])[-100:]
        return current_degree > threshold_factor * mean_degree and current_degree > 10

    def _rebuild_communities(self):
        """
        Rebuild Louvain community structure for the undirected projection.
        Runs in background to avoid blocking the hot path.
        """
        try:
            if self.G.number_of_nodes() < 4:
                return
            undirected = self.G.to_undirected()
            # Use greedy modularity (available without community extras)
            communities_gen = nx.community.greedy_modularity_communities(undirected)
            new_map = {}
            for cid, community in enumerate(communities_gen):
                for node in community:
                    new_map[node] = cid
            with self._lock:
                self._communities = new_map
            logger.debug("Rebuilt communities: %d groups, %d nodes",
                         len(set(new_map.values())), len(new_map))
        except Exception as exc:
            logger.warning("Community rebuild failed: %s", exc)

    def _rebuild_loop(self):
        while True:
            time.sleep(GRAPH_REBUILD_INTERVAL)
            self._rebuild_communities()

    @staticmethod
    def _explain(
        src: str, dst: str,
        anomaly_types: List[str],
        score: float,
        path: Optional[List[str]],
    ) -> str:
        if not anomaly_types:
            return f"Normal communication: {src} → {dst}"
        reasons = []
        if "new_edge" in anomaly_types:
            reasons.append(f"First-ever communication between {src} and {dst}")
        if "community_breach" in anomaly_types:
            reasons.append("Traffic crosses network segment boundary")
        if "lateral_movement" in anomaly_types and path:
            reasons.append(f"Lateral movement path detected: {' → '.join(path)}")
        if "hub_explosion" in anomaly_types:
            reasons.append(f"{src} contacted an unusually high number of nodes")
        if "port_scan" in anomaly_types:
            reasons.append(f"{src} contacted ≥{PORT_SCAN_THRESHOLD} distinct ports on {dst}")
        return "; ".join(reasons)


# ── Prebuilt graph loader (for testing with sample topology) ───────────────────
def build_sample_graph() -> NetworkGraph:
    """
    Create a small synthetic network topology for testing:
      - 3 network segments (office, DMZ, server)
      - Pre-seed with ~200 normal edges
    """
    import random
    graph = NetworkGraph()

    office_hosts = [f"10.0.1.{i}" for i in range(1, 30)]
    dmz_hosts    = [f"10.0.2.{i}" for i in range(1, 10)]
    server_hosts = [f"10.0.3.{i}" for i in range(1, 5)]

    normal_pairs = (
        [(s, d) for s in office_hosts for d in server_hosts[:3]]
        + [(s, d) for s in office_hosts for d in dmz_hosts[:3]]
        + [(s, "8.8.8.8") for s in office_hosts[:5]]
    )

    rng = random.Random(42)
    for src, dst in rng.sample(normal_pairs, min(200, len(normal_pairs))):
        for _ in range(rng.randint(1, 20)):
            graph.observe(src, dst, rng.choice([80, 443, 8080, 53]))

    logger.info("Sample graph: %d nodes, %d edges", graph.node_count, graph.edge_count)
    return graph


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Graph Anomaly Detection — Smoke Test ===\n")
    graph = build_sample_graph()

    test_events = [
        # Normal traffic
        ("10.0.1.5", "10.0.3.1", 443, "Expected web→DB"),
        # New edge
        ("10.0.1.22", "10.0.3.4", 3306, "New office→DB direct connection"),
        # Port scan
        *[("10.0.1.99", "10.0.3.1", p, "Port scan") for p in range(20, 40)],
        # Lateral movement seed
        ("10.0.2.1", "10.0.1.5", 445, "DMZ→office (lateral)"),
        ("10.0.1.5", "10.0.3.4", 445, "office→server (lateral)"),
    ]

    for src, dst, port, desc in test_events[:5]:
        result = graph.observe(src, dst, port)
        print(f"[{desc}]")
        print(f"  {src} → {dst}:{port}")
        print(f"  score={result.anomaly_score:.3f}  types={result.anomaly_types}")
        print(f"  {result.explanation}\n")

    print(f"Graph state: {graph.to_dict()}")
