"""
Traffic Capture Module
======================
Captures live network packets using Scapy, extracts ML-ready features,
and streams them to the threat detection pipeline via a queue.

Extracted features per packet/flow:
  src_ip, dst_ip, src_port, dst_port, protocol,
  packet_size, ttl, flags, session_duration, pkt_count
"""

import time
import queue
import threading
import ipaddress
import logging
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional, Generator

import numpy as np

# Scapy import with graceful fallback for environments without raw socket access
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
    conf.verb = 0  # suppress Scapy output
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Running in simulation mode.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
FLOW_TIMEOUT_SECONDS = 60          # Expire a flow if idle for this long
MAX_QUEUE_SIZE       = 10_000      # Back-pressure limit on the packet queue
FEATURE_NAMES = [
    "src_ip_int", "dst_ip_int", "src_port", "dst_port",
    "protocol",   "packet_size", "ttl",     "tcp_flags",
    "flow_duration", "flow_pkt_count", "bytes_per_sec",
]

# Protocol numeric encoding
PROTO_MAP = {1: 0, 6: 1, 17: 2}   # ICMP=0, TCP=1, UDP=2


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class FlowKey:
    """Identifies a bidirectional network flow."""
    src_ip:   str
    dst_ip:   str
    src_port: int
    dst_port: int
    protocol: int

    def __hash__(self):
        # Make bidirectional: (A→B) == (B→A)
        ep1 = (self.src_ip, self.src_port)
        ep2 = (self.dst_ip, self.dst_port)
        return hash((min(ep1, ep2), max(ep1, ep2), self.protocol))

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class FlowRecord:
    """Aggregated statistics for a single network flow."""
    key:          FlowKey
    start_time:   float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)
    pkt_count:    int   = 0
    total_bytes:  int   = 0
    flags_seen:   int   = 0   # bitmask of all TCP flags observed

    def update(self, pkt_size: int, flags: int = 0):
        self.last_seen   = time.time()
        self.pkt_count  += 1
        self.total_bytes += pkt_size
        self.flags_seen |= flags

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.start_time, 1e-6)

    @property
    def bytes_per_sec(self) -> float:
        return self.total_bytes / self.duration


@dataclass
class PacketFeatures:
    """ML-ready feature vector extracted from a packet + its flow context."""
    src_ip:        str
    dst_ip:        str
    src_port:      int
    dst_port:      int
    protocol:      int   # 0=ICMP, 1=TCP, 2=UDP
    packet_size:   int
    ttl:           int
    tcp_flags:     int
    flow_duration: float
    flow_pkt_count:int
    bytes_per_sec: float
    timestamp:     float = field(default_factory=time.time)
    raw_label:     Optional[str] = None  # for labelled datasets

    def to_numpy(self) -> np.ndarray:
        """Convert to normalized float32 numpy vector for the ML model."""
        return np.array([
            _ip_to_int(self.src_ip),
            _ip_to_int(self.dst_ip),
            self.src_port / 65535.0,
            self.dst_port / 65535.0,
            self.protocol / 2.0,
            self.packet_size / 65535.0,
            self.ttl / 255.0,
            self.tcp_flags / 255.0,
            min(self.flow_duration / 3600.0, 1.0),
            min(self.flow_pkt_count / 1000.0, 1.0),
            min(self.bytes_per_sec / 1e6, 1.0),
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ip_to_int(ip: str) -> float:
    """Convert dotted-decimal IP to [0,1]-normalised integer."""
    try:
        return int(ipaddress.ip_address(ip)) / (2**32 - 1)
    except ValueError:
        return 0.0


# ── Flow tracker ───────────────────────────────────────────────────────────────
class FlowTracker:
    """
    Maintains a table of active flows and aggregates per-flow statistics.
    Thread-safe via a lock.
    """

    def __init__(self, timeout: float = FLOW_TIMEOUT_SECONDS):
        self._flows: dict[FlowKey, FlowRecord] = {}
        self._lock   = threading.Lock()
        self._timeout = timeout

        # Background thread to reap expired flows
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def update(self, key: FlowKey, pkt_size: int, flags: int = 0) -> FlowRecord:
        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowRecord(key=key)
            record = self._flows[key]
            record.update(pkt_size, flags)
            return record

    def _reap_loop(self):
        while True:
            time.sleep(10)
            now = time.time()
            with self._lock:
                expired = [k for k, v in self._flows.items()
                           if now - v.last_seen > self._timeout]
                for k in expired:
                    del self._flows[k]
            if expired:
                logger.debug("Reaped %d expired flows", len(expired))

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._flows)


# ── Packet processor ───────────────────────────────────────────────────────────
class PacketProcessor:
    """
    Converts raw Scapy packets → PacketFeatures and pushes to a queue.
    """

    def __init__(self, output_queue: queue.Queue, tracker: FlowTracker):
        self.queue   = output_queue
        self.tracker = tracker
        self.stats   = defaultdict(int)

    def process(self, pkt) -> Optional[PacketFeatures]:
        """Parse a Scapy packet and return a PacketFeatures object."""
        if IP not in pkt:
            return None

        ip_layer = pkt[IP]
        src_ip   = ip_layer.src
        dst_ip   = ip_layer.dst
        ttl      = ip_layer.ttl
        pkt_size = len(pkt)
        proto_raw = ip_layer.proto

        src_port = dst_port = tcp_flags = 0

        if TCP in pkt:
            src_port  = pkt[TCP].sport
            dst_port  = pkt[TCP].dport
            tcp_flags = int(pkt[TCP].flags)
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        elif ICMP in pkt:
            pass  # ICMP has no ports

        protocol = PROTO_MAP.get(proto_raw, 2)

        # Update flow record
        fk     = FlowKey(src_ip, dst_ip, src_port, dst_port, proto_raw)
        record = self.tracker.update(fk, pkt_size, tcp_flags)

        features = PacketFeatures(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            packet_size=pkt_size,
            ttl=ttl,
            tcp_flags=tcp_flags,
            flow_duration=record.duration,
            flow_pkt_count=record.pkt_count,
            bytes_per_sec=record.bytes_per_sec,
        )

        self.stats["processed"] += 1
        return features

    def handle(self, pkt):
        """Scapy callback — process and enqueue; drop if queue is full."""
        features = self.process(pkt)
        if features is None:
            return
        try:
            self.queue.put_nowait(features)
            self.stats["enqueued"] += 1
        except queue.Full:
            self.stats["dropped"] += 1
            logger.warning("Packet queue full — dropping packet")


# ── Traffic capture engine ─────────────────────────────────────────────────────
class TrafficCapture:
    """
    Main capture engine.  Call start() to begin sniffing; read from
    the public `queue` attribute to consume PacketFeatures objects.
    """

    def __init__(
        self,
        interface:  Optional[str] = None,
        bpf_filter: str = "ip",
        max_queue:  int = MAX_QUEUE_SIZE,
    ):
        self.interface  = interface       # None = all interfaces
        self.bpf_filter = bpf_filter
        self.queue      = queue.Queue(maxsize=max_queue)
        self._tracker   = FlowTracker()
        self._processor = PacketProcessor(self.queue, self._tracker)
        self._stop_evt  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start packet capture in a background thread."""
        if not SCAPY_AVAILABLE:
            logger.warning("Scapy unavailable — starting simulated capture")
            self._thread = threading.Thread(target=self._simulate, daemon=True)
        else:
            self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.info("Traffic capture started on interface=%s filter='%s'",
                    self.interface or "all", self.bpf_filter)

    def stop(self):
        self._stop_evt.set()
        logger.info("Traffic capture stopping…")

    def stream(self) -> Generator[PacketFeatures, None, None]:
        """Yield PacketFeatures objects as they arrive (blocking generator)."""
        while not self._stop_evt.is_set():
            try:
                yield self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

    @property
    def stats(self) -> dict:
        return {
            **self._processor.stats,
            "active_flows": self._tracker.active_count,
            "queue_size":   self.queue.qsize(),
        }

    # ── Private methods ────────────────────────────────────────────────────────

    def _capture(self):
        """Real Scapy capture loop."""
        sniff(
            iface=self.interface,
            filter=self.bpf_filter,
            prn=self._processor.handle,
            store=False,
            stop_filter=lambda _: self._stop_evt.is_set(),
        )

    def _simulate(self):
        """
        Synthetic packet generator for testing without root/Scapy.
        Produces a realistic mix of normal and attack traffic.
        """
        import random
        ATTACK_SCENARIOS = [
            # (src_ip, dst_ip, dport, proto, label)
            ("203.0.113.47", "10.0.0.5",  3306, 6,  "sql_injection"),
            ("198.51.100.9", "10.0.0.1",  22,   6,  "ssh_brute"),
            ("10.0.0.201",   "10.0.0.5",  445,  6,  "lateral_move"),
        ]
        seq = 0
        while not self._stop_evt.is_set():
            is_attack = random.random() < 0.04   # 4% attack traffic
            if is_attack:
                sc = random.choice(ATTACK_SCENARIOS)
                src_ip, dst_ip, dport, proto, label = sc
                sport = random.randint(1024, 65535)
                pkt_size = random.randint(40, 200)
                flags = 2  # SYN
            else:
                src_ip   = f"10.0.{random.randint(0,10)}.{random.randint(1,254)}"
                dst_ip   = f"10.0.0.{random.randint(1,20)}"
                sport    = random.randint(1024, 65535)
                dport    = random.choice([80, 443, 8080, 53])
                proto    = random.choice([6, 17])
                pkt_size = random.randint(64, 1500)
                flags    = random.choice([16, 24])   # ACK, PSH+ACK
                label    = "normal"

            fk     = FlowKey(src_ip, dst_ip, sport, dport, proto)
            record = self._tracker.update(fk, pkt_size, flags)

            feat = PacketFeatures(
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=sport, dst_port=dport,
                protocol=PROTO_MAP.get(proto, 2),
                packet_size=pkt_size, ttl=64,
                tcp_flags=flags,
                flow_duration=record.duration,
                flow_pkt_count=record.pkt_count,
                bytes_per_sec=record.bytes_per_sec,
                raw_label=label,
            )
            try:
                self.queue.put_nowait(feat)
                self._processor.stats["enqueued"] += 1
            except queue.Full:
                self._processor.stats["dropped"] += 1

            seq += 1
            time.sleep(random.uniform(0.001, 0.01))   # ~100-1000 pps


# ── CLI entrypoint (for standalone testing) ────────────────────────────────────
if __name__ == "__main__":
    cap = TrafficCapture()
    cap.start()
    print("Streaming packets (Ctrl-C to stop)…\n")
    try:
        for i, feat in enumerate(cap.stream()):
            print(f"[{i:05d}] {feat.src_ip}:{feat.src_port} → "
                  f"{feat.dst_ip}:{feat.dst_port} "
                  f"proto={feat.protocol} size={feat.packet_size}B "
                  f"flows={cap.stats['active_flows']}")
            if i >= 99:
                break
    except KeyboardInterrupt:
        cap.stop()
    print("\nStats:", cap.stats)
