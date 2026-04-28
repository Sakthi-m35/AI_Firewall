# AI-Driven Next-Gen Firewall (AI-NGF)
## Complete Technical Reference

---

## 📁 Folder Structure

```
ai_firewall/
├── modules/
│   ├── __init__.py
│   ├── traffic_capture.py      # Scapy packet capture + flow tracker
│   ├── ml_model.py             # LSTM classifier + Autoencoder
│   ├── nlp_module.py           # DistilBERT log analysis
│   ├── graph_analytics.py      # NetworkX anomaly detection
│   └── decision_engine.py      # Ensemble engine + Zero Trust + Response
│
├── api_main.py                 # FastAPI backend (all endpoints)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
│
├── dashboard/                  # React.js frontend
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Overview.jsx         # Metrics + live charts
│   │   │   ├── ThreatTable.jsx      # Active threats with XAI
│   │   │   ├── LogViewer.jsx        # NLP log analysis display
│   │   │   ├── PolicyManager.jsx    # Zero Trust policy editor
│   │   │   └── ModelStatus.jsx      # AI model health
│   │   ├── hooks/
│   │   │   └── useLiveTraffic.js    # WebSocket hook
│   │   └── api/
│   │       └── client.js            # Axios API client
│   ├── package.json
│   └── Dockerfile.dashboard
│
├── models/                     # Saved model weights (git-ignored)
│   ├── lstm_classifier.keras
│   ├── autoencoder.keras
│   ├── scaler.pkl
│   ├── ae_scaler.pkl
│   └── nlp_classifier/         # Fine-tuned DistilBERT
│
├── db/
│   └── init.sql                # PostgreSQL schema
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboards/
│
├── tests/
│   ├── test_traffic_capture.py
│   ├── test_ml_model.py
│   ├── test_nlp_module.py
│   ├── test_graph_analytics.py
│   ├── test_decision_engine.py
│   └── test_api.py
│
└── scripts/
    ├── train_models.py         # Offline model training script
    ├── ingest_threat_intel.py  # Pull IOC feeds (AbuseIPDB, etc.)
    └── generate_synthetic.py   # GAN-based synthetic attack data
```

---

## 🚀 Deployment Guide

### Prerequisites
- Docker ≥ 24.0 + Docker Compose v2
- Python 3.11+ (for local dev)
- 8 GB RAM minimum (16 GB for ML training)
- Root/sudo for packet capture (Scapy)

### Step 1 — Clone & configure
```bash
git clone https://github.com/yourorg/ai-ngf.git
cd ai-ngf

# Copy and edit environment file
cp .env.example .env
# Edit .env: set JWT_SECRET, DB_PASS, GRAFANA_PASS
```

### Step 2 — (Optional) Train models offline
```bash
# Download CICIDS 2017 dataset from https://www.unb.ca/cic/datasets/ids-2017.html
# Place in data/

pip install -r requirements.txt
python scripts/train_models.py \
  --dataset data/MachineLearningCSV/MachineLearningCVE/ \
  --type cicids \
  --epochs 30
# Trained models saved to models/
```

### Step 3 — Start all services
```bash
docker compose up -d --build

# Watch logs
docker compose logs -f api

# Verify health
curl http://localhost:8000/health
```

### Step 4 — Access interfaces
| Service     | URL                       | Default creds  |
|-------------|---------------------------|----------------|
| API Docs    | http://localhost:8000/docs | —              |
| Dashboard   | http://localhost:3000      | admin/admin123 |
| MLflow      | http://localhost:5000      | —              |
| Grafana     | http://localhost:3001      | admin/admin123 |
| Prometheus  | http://localhost:9090      | —              |

### Step 5 — First API call
```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123","role":"admin"}' \
  | jq -r .access_token)

# Predict threat
curl -X POST http://localhost:8000/predict-threat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "packets": [{
      "src_ip": "203.0.113.47",
      "dst_ip": "10.0.0.5",
      "src_port": 45123,
      "dst_port": 3306,
      "protocol": 1,
      "packet_size": 120,
      "flow_duration": 2.5,
      "flow_pkt_count": 47,
      "bytes_per_sec": 4800
    }]
  }'
```

---

## 🧪 Testing Strategy

### Unit tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v --tb=short

# Run specific module
pytest tests/test_ml_model.py -v
```

### API integration tests (pytest)
```python
# tests/test_api.py (example)
from fastapi.testclient import TestClient
from api_main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_threat_unauthenticated():
    r = client.post("/predict-threat", json={})
    assert r.status_code == 403

def test_predict_threat():
    # Get token
    login = client.post("/auth/login",
        json={"username": "admin", "password": "admin", "role": "admin"})
    token = login.json()["access_token"]

    r = client.post("/predict-threat",
        headers={"Authorization": f"Bearer {token}"},
        json={"packets": [{
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 1234, "dst_port": 80, "protocol": 1,
            "packet_size": 512, "flow_pkt_count": 1
        }]}
    )
    assert r.status_code == 200
    body = r.json()
    assert "threat_level" in body
    assert "final_score" in body
    assert 0.0 <= body["final_score"] <= 1.0
```

### Load testing (Locust)
```python
# locustfile.py
from locust import HttpUser, task, between

class FirewallUser(HttpUser):
    wait_time = between(0.1, 0.5)
    token = None

    def on_start(self):
        r = self.client.post("/auth/login",
            json={"username":"load","password":"test","role":"viewer"})
        self.token = r.json().get("access_token", "")

    @task(3)
    def predict(self):
        self.client.post("/predict-threat",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"packets": [{"src_ip":"10.0.0.1","dst_ip":"10.0.0.2",
                "src_port":1234,"dst_port":80,"protocol":1,"packet_size":512,"flow_pkt_count":1}]})

    @task(1)
    def analyze_log(self):
        self.client.post("/analyze-log",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"log_lines":["Normal login from 10.0.0.5"]})

# Run: locust -f locustfile.py --host http://localhost:8000 -u 100 -r 10
```

---

## ⚠️ Limitations & Production Improvements

### Current Limitations
| Area | Limitation | Production Fix |
|------|------------|----------------|
| Training | Demo uses simulated data | Train on real CICIDS/NSL-KDD/UNSW-NB15 |
| NLP | Falls back to rules without GPU | Deploy on GPU instance with CUDA |
| Scapy capture | Requires root privileges | Use `CAP_NET_RAW` + non-root user in Docker |
| Authentication | JWT secret is hardcoded | Use HashiCorp Vault / AWS Secrets Manager |
| Storage | Threat events in memory deque | Persist to MongoDB + PostgreSQL |
| Rate limiting | Not implemented in API | Add `slowapi` rate limiter |
| GNN | Graph uses NetworkX, not GNN | Add PyG/DGL for GNN inference |
| Encryption | API uses HTTP in dev | Terminate TLS at nginx/load balancer |
| MFA | OTP is a random UUID | Integrate TOTP (pyotp) or SMS gateway |

### Recommended Production Stack
```
Internet → CloudFlare WAF
         → AWS ALB (TLS termination)
         → nginx (rate limiting, static files)
         → API (FastAPI, 4+ workers, gunicorn)
         → Redis (session store, rate limit counters)
         → PostgreSQL RDS (policies, users)
         → MongoDB Atlas (logs, events)
         → S3 (model artifacts)
         → SageMaker (model training, inference endpoints)
         → EKS (Kubernetes orchestration)
         → Datadog / Grafana Cloud (observability)
```

### Security Hardening Checklist
- [ ] Rotate JWT_SECRET weekly via Vault
- [ ] Enable PostgreSQL SSL
- [ ] Encrypt MongoDB at rest (WiredTiger)
- [ ] TLS 1.3 only on all endpoints
- [ ] SIEM integration (Splunk, Elastic SIEM)
- [ ] Log anonymisation (hash IPs in audit log)
- [ ] Regular pen testing (quarterly)
- [ ] Model adversarial robustness evaluation
- [ ] Kubernetes NetworkPolicy (pod isolation)
- [ ] Pod Security Standards (restricted profile)

---

## 📊 Sample API Response: /predict-threat

```json
{
  "threat_level": "CRITICAL",
  "final_score": 0.9312,
  "dl_score": 0.97,
  "nlp_score": 0.91,
  "graph_score": 0.84,
  "recommended_actions": [
    "block_ip",
    "terminate_session",
    "alert_admin",
    "trigger_mfa"
  ],
  "explanation": "DL=0.970×0.5 NLP=0.910×0.3 Graph=0.840×0.2 → Ensemble=0.9312 → CRITICAL. LSTM detected malicious packet sequence pattern. NLP identified high-risk log indicators. Graph engine detected unusual communication topology.",
  "decision_id": "a3f2b1c4",
  "timestamp": "2024-11-15T14:32:07.412Z"
}
```

---

## 🔄 Continuous Learning Pipeline

```python
# scripts/train_models.py (simplified)

import schedule, time
from modules.ml_model import ThreatDetectionEngine
from modules.nlp_module import NLPLogAnalyzer, generate_synthetic_logs

def retrain():
    print("Starting scheduled retraining...")
    
    # 1. Pull labelled feedback from MongoDB
    # feedback = mongo_client.ngf_logs.feedback.find({"labelled": True})
    
    # 2. Augment with synthetic data (optional GAN)
    texts, labels = generate_synthetic_logs(n_per_class=500)
    
    # 3. Retrain NLP module
    nlp = NLPLogAnalyzer()
    nlp.fine_tune(texts, labels, epochs=2)
    
    # 4. Retrain DL model on new network traffic
    engine = ThreatDetectionEngine()
    # engine.train_from_file("new_data.csv", dataset="cicids")
    
    print("Retraining complete. New models deployed.")

schedule.every().day.at("21:00").do(retrain)

while True:
    schedule.run_pending()
    time.sleep(60)
```
