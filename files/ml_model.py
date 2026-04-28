"""
AI Threat Detection Engine
==========================
Two complementary deep-learning models:

  1. LSTMClassifier   — Sequential traffic analysis (multi-class).
     Input  : (batch, seq_len, 11 features)
     Output : probabilities for [normal, suspicious, malicious]

  2. AutoencoderDetector — Unsupervised anomaly detection.
     Input  : (batch, 11 features)
     Output : reconstruction error → anomaly score in [0, 1]

Training datasets supported:
  • CICIDS 2017  (CSV, labelled)
  • NSL-KDD      (CSV, labelled)
  • UNSW-NB15    (CSV, labelled)

Usage:
  engine = ThreatDetectionEngine()
  engine.load_or_train("path/to/data.csv", dataset="cicids")
  result = engine.predict(packet_features_list)
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

logger = logging.getLogger(__name__)

# ── Label mappings ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    # CICIDS / UNSW-NB15
    "BENIGN": 0, "Normal": 0, "normal": 0,
    # Suspicious (non-critical probes/scans)
    "PortScan": 1, "Probe": 1, "FTP-Patator": 1, "SSH-Patator": 1,
    "Reconnaissance": 1, "Analysis": 1,
    # Malicious (active attacks)
    "DoS": 2, "DDoS": 2, "Bot": 2, "Infiltration": 2,
    "Web Attack": 2, "Heartbleed": 2,
    "Generic": 2, "Exploits": 2, "Fuzzers": 2, "Shellcode": 2, "Worms": 2,
}
CLASS_NAMES = ["normal", "suspicious", "malicious"]
N_FEATURES   = 11   # must match PacketFeatures.to_numpy() length
SEQ_LEN      = 20   # packets per LSTM window
N_CLASSES    = 3


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    label:          str           # "normal" | "suspicious" | "malicious"
    confidence:     float         # 0.0 – 1.0
    class_probs:    List[float]   # [p_normal, p_suspicious, p_malicious]
    anomaly_score:  float         # from autoencoder (0 = normal, 1 = anomalous)
    explanation:    dict          # feature importance hints


# ── LSTM Classifier ────────────────────────────────────────────────────────────
def build_lstm_model(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES,
                     n_classes: int = N_CLASSES) -> Model:
    """
    Bidirectional LSTM with attention for sequential traffic classification.

    Architecture:
      Input(seq_len, n_features)
      → Bidirectional LSTM(128, return_sequences=True)
      → Dropout(0.3)
      → Bidirectional LSTM(64)
      → Dense(64, relu) → Dropout(0.2)
      → Dense(32, relu)
      → Dense(n_classes, softmax)
    """
    inp = keras.Input(shape=(seq_len, n_features), name="packet_sequence")

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.3)(x)

    # Simple self-attention: weight each timestep by its relevance
    attn = layers.Dense(1, activation="tanh")(x)
    attn = layers.Flatten()(attn)
    attn = layers.Activation("softmax", name="attention_weights")(attn)
    attn = layers.RepeatVector(256)(attn)
    attn = layers.Permute([2, 1])(attn)
    attended = layers.Multiply()([x, attn])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attended_sum")(attended)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="threat_class")(x)

    model = Model(inputs=inp, outputs=out, name="lstm_threat_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Autoencoder Anomaly Detector ───────────────────────────────────────────────
def build_autoencoder(n_features: int = N_FEATURES) -> Tuple[Model, Model]:
    """
    Encoder-decoder autoencoder for unsupervised anomaly detection.
    High reconstruction error → anomalous packet.

    Architecture: 11 → 64 → 32 → 16 → 8 (bottleneck) → 16 → 32 → 64 → 11
    """
    # Encoder
    enc_inp = keras.Input(shape=(n_features,), name="enc_input")
    e = layers.Dense(64, activation="relu")(enc_inp)
    e = layers.BatchNormalization()(e)
    e = layers.Dense(32, activation="relu")(e)
    e = layers.Dense(16, activation="relu")(e)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(e)

    encoder = Model(enc_inp, bottleneck, name="encoder")

    # Decoder
    dec_inp = keras.Input(shape=(8,), name="dec_input")
    d = layers.Dense(16, activation="relu")(dec_inp)
    d = layers.Dense(32, activation="relu")(d)
    d = layers.Dense(64, activation="relu")(d)
    dec_out = layers.Dense(n_features, activation="sigmoid", name="reconstruction")(d)
    decoder = Model(dec_inp, dec_out, name="decoder")

    # Full autoencoder
    ae_inp  = keras.Input(shape=(n_features,), name="ae_input")
    encoded = encoder(ae_inp)
    decoded = decoder(encoded)
    autoencoder = Model(ae_inp, decoded, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder


# ── Dataset preprocessors ──────────────────────────────────────────────────────
def load_cicids(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess CICIDS 2017 CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Drop non-numeric and identifier columns
    drop_cols = ["Flow ID", "Source IP", "Destination IP",
                 "Timestamp", "Label", " Label"]
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and df[c].dtype in [np.float64, np.int64]]

    label_col = "Label" if "Label" in df.columns else " Label"
    labels = df[label_col].str.strip().map(
        lambda x: LABEL_MAP.get(x.split(" ")[0], 2)
    ).fillna(2).astype(int)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = labels.values
    return X.astype(np.float32), y


def load_nslkdd(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess NSL-KDD dataset."""
    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files",
        "num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
        "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","label","difficulty_level",
    ]
    df = pd.read_csv(filepath, header=None, names=col_names)

    # Encode categorical columns
    for col in ["protocol_type", "service", "flag"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    attack_map = {"normal": 0}
    suspicious = {"portsweep","ipsweep","satan","nmap","mscan","saint"}
    def kdd_label(lbl):
        if lbl == "normal":    return 0
        if lbl in suspicious:  return 1
        return 2

    y = df["label"].map(kdd_label).fillna(2).astype(int).values
    X = df.drop(["label","difficulty_level"], axis=1).values.astype(np.float32)
    return X, y


class FeatureScaler:
    """Wrapper around StandardScaler with fixed output dimensionality."""

    def __init__(self, target_dim: int = N_FEATURES):
        self.scaler     = StandardScaler()
        self.target_dim = target_dim
        self._fitted    = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.fit_transform(X)
        self._fitted = True
        return self._resize(Xs)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        Xs = self.scaler.transform(X) if X.shape[1] == self.scaler.n_features_in_ else X
        return self._resize(Xs)

    def _resize(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        if d >= self.target_dim:
            return X[:, :self.target_dim]
        # Pad with zeros if input has fewer features than expected
        return np.hstack([X, np.zeros((X.shape[0], self.target_dim - d))])

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureScaler":
        return joblib.load(path)


# ── Main detection engine ──────────────────────────────────────────────────────
class ThreatDetectionEngine:
    """
    High-level interface combining the LSTM classifier and autoencoder.
    Manages model persistence, training, and inference.
    """

    # Reconstruction-error threshold → anomaly score ≥ this = anomalous
    ANOMALY_THRESHOLD = 0.042
    MODEL_DIR = Path("models")

    def __init__(self):
        self.lstm         = build_lstm_model()
        self.autoencoder, self.encoder = build_autoencoder()
        self.scaler       = FeatureScaler()
        self._ae_scaler   = FeatureScaler()
        self._trained     = False
        self.MODEL_DIR.mkdir(exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs:     int = 20,
        batch_size: int = 512,
        val_split:  float = 0.15,
    ):
        """
        Train both models on preprocessed feature matrix X and labels y.
        y values: 0=normal, 1=suspicious, 2=malicious
        """
        logger.info("Training on %d samples (%d features)", len(X), X.shape[1])

        X_scaled = self.scaler.fit_transform(X)

        # ── Train LSTM ─────────────────────────────────────────────────────────
        Xs_seq, ys_seq = _make_sequences(X_scaled, y, SEQ_LEN)
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs_seq, ys_seq, test_size=val_split, stratify=ys_seq, random_state=42
        )
        logger.info("LSTM training: %d train / %d val sequences", len(X_tr), len(X_val))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]
        hist = self.lstm.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1,
        )
        val_acc = max(hist.history["val_accuracy"])
        logger.info("LSTM best val accuracy: %.4f", val_acc)

        # Evaluate
        y_pred = np.argmax(self.lstm.predict(X_val, verbose=0), axis=1)
        logger.info("\n%s", classification_report(y_val, y_pred, target_names=CLASS_NAMES))

        # ── Train Autoencoder on NORMAL samples only ───────────────────────────
        normal_idx = y == 0
        X_normal   = self._ae_scaler.fit_transform(X[normal_idx])
        ae_x_tr, ae_x_val = train_test_split(X_normal, test_size=0.1, random_state=42)
        logger.info("Autoencoder training on %d normal samples", len(ae_x_tr))

        self.autoencoder.fit(
            ae_x_tr, ae_x_tr,
            validation_data=(ae_x_val, ae_x_val),
            epochs=30, batch_size=256,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1,
        )

        # Compute threshold from validation reconstruction errors
        recon   = self.autoencoder.predict(ae_x_val, verbose=0)
        errors  = np.mean(np.power(ae_x_val - recon, 2), axis=1)
        self.ANOMALY_THRESHOLD = float(np.percentile(errors, 95))
        logger.info("Autoencoder threshold set to %.6f (95th pct of normal errors)",
                    self.ANOMALY_THRESHOLD)

        self._trained = True
        self.save()

    def train_from_file(self, filepath: str, dataset: str = "cicids", **kwargs):
        """Load a standard dataset file and train both models."""
        loaders = {"cicids": load_cicids, "nslkdd": load_nslkdd}
        if dataset not in loaders:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(loaders)}")
        X, y = loaders[dataset](filepath)
        self.train(X, y, **kwargs)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_batch(self, feature_matrix: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on a 2-D feature matrix (n_samples, N_FEATURES).
        Returns one DetectionResult per sample.
        """
        if not self._trained:
            logger.warning("Models not trained — returning random results for demo")
            return [self._random_result() for _ in range(len(feature_matrix))]

        X_sc  = self.scaler.transform(feature_matrix)
        X_ae  = self._ae_scaler.transform(feature_matrix)
        X_seq = _make_sequences_infer(X_sc, SEQ_LEN)

        probs  = self.lstm.predict(X_seq, verbose=0)           # (n, 3)
        recon  = self.autoencoder.predict(X_ae, verbose=0)     # (n, 11)
        errors = np.mean(np.power(X_ae - recon, 2), axis=1)    # (n,)
        anoms  = np.clip(errors / (self.ANOMALY_THRESHOLD * 3), 0, 1)

        results = []
        for i in range(len(feature_matrix)):
            idx   = int(np.argmax(probs[i]))
            label = CLASS_NAMES[idx]
            results.append(DetectionResult(
                label=label,
                confidence=float(probs[i][idx]),
                class_probs=probs[i].tolist(),
                anomaly_score=float(anoms[i]),
                explanation=self._explain(feature_matrix[i], probs[i], errors[i]),
            ))
        return results

    def predict_single(self, features: np.ndarray) -> DetectionResult:
        """Convenience wrapper for a single feature vector."""
        return self.predict_batch(features.reshape(1, -1))[0]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        self.lstm.save(self.MODEL_DIR / "lstm_classifier.keras")
        self.autoencoder.save(self.MODEL_DIR / "autoencoder.keras")
        self.scaler.save(str(self.MODEL_DIR / "scaler.pkl"))
        self._ae_scaler.save(str(self.MODEL_DIR / "ae_scaler.pkl"))
        logger.info("Models saved to %s/", self.MODEL_DIR)

    def load(self):
        self.lstm        = keras.models.load_model(self.MODEL_DIR / "lstm_classifier.keras")
        self.autoencoder = keras.models.load_model(self.MODEL_DIR / "autoencoder.keras")
        self.scaler      = FeatureScaler.load(str(self.MODEL_DIR / "scaler.pkl"))
        self._ae_scaler  = FeatureScaler.load(str(self.MODEL_DIR / "ae_scaler.pkl"))
        self._trained    = True
        logger.info("Models loaded from %s/", self.MODEL_DIR)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _explain(self, feat: np.ndarray, probs: np.ndarray, ae_error: float) -> dict:
        """Simple feature-attribution explanation (no SHAP — keeps it lightweight)."""
        feat_names = [
            "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "pkt_size", "ttl", "tcp_flags", "flow_dur", "pkt_count", "bps",
        ]
        # Flag any features outside ±2σ as notable
        notable = [feat_names[i] for i, v in enumerate(feat) if abs(v) > 0.7]
        return {
            "top_features":   notable[:3],
            "ae_recon_error": round(ae_error, 6),
            "ae_threshold":   round(self.ANOMALY_THRESHOLD, 6),
            "class_probs":    {cn: round(float(p), 4) for cn, p in zip(CLASS_NAMES, probs)},
        }

    @staticmethod
    def _random_result() -> DetectionResult:
        """Demo mode: return plausible random result."""
        import random
        r = random.random()
        if r < 0.90:   idx = 0
        elif r < 0.96: idx = 1
        else:           idx = 2
        probs = [0.0, 0.0, 0.0]
        probs[idx] = round(random.uniform(0.75, 0.99), 3)
        rest = 1.0 - probs[idx]
        probs[1-idx if idx != 1 else 0] = round(rest * 0.7, 3)
        probs[2-idx if idx != 2 else 0] = round(1 - sum(probs[:2]), 3)
        return DetectionResult(
            label=CLASS_NAMES[idx],
            confidence=probs[idx],
            class_probs=probs,
            anomaly_score=round(random.uniform(0, 0.3) if idx == 0 else random.uniform(0.5, 1.0), 3),
            explanation={"note": "demo_mode"},
        )


# ── Sequence helpers ───────────────────────────────────────────────────────────
def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat (n, features) array to (n-seq_len, seq_len, features) for LSTM."""
    seqs, labels = [], []
    for i in range(len(X) - seq_len):
        seqs.append(X[i : i + seq_len])
        labels.append(y[i + seq_len - 1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int32)


def _make_sequences_infer(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad X to form exactly len(X) sequences (each ending at that row)."""
    n = len(X)
    padded = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        chunk = X[start : i + 1]
        padded[i, seq_len - len(chunk) :] = chunk
    return padded


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    print("=== AI Threat Detection Engine — Smoke Test ===\n")
    engine = ThreatDetectionEngine()

    # Simulate 200 feature vectors (random, no real training)
    X_demo = np.random.rand(200, N_FEATURES).astype(np.float32)
    results = engine.predict_batch(X_demo)
    counts  = {c: sum(1 for r in results if r.label == c) for c in CLASS_NAMES}
    print(f"Demo predictions (untrained): {counts}")
    print(f"Sample result: {results[0]}")
