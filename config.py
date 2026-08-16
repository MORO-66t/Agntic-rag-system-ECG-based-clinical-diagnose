"""
Configuration loader — reads all credentials and settings from .env file.

Usage:
    from config import (
        DB_CONFIG, KAFKA_CONFIG, GROQ_API_KEYS, GROQ_MODEL,
        LLM_MODE, HF_TOKEN, HF_TOKENS, PDF_DOCUMENT_NAME,
    )

All modules should import from here instead of reading .env directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=True)

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS: list[str] = [
    k.strip()
    for k in os.getenv("GROQ_API_KEYS", "").split(",")
    if k.strip()
]
# Fallback: if GROQ_API_KEYS is not set, use GROQ_API_KEY as single key
if not GROQ_API_KEYS and GROQ_API_KEY:
    GROQ_API_KEYS = [GROQ_API_KEY]
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ── HuggingFace (legacy — used by token_manager.py for rotation) ───────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_TOKENS: list[str] = [
    token.strip()
    for token in os.getenv("HF_TOKENS", "").split(",")
    if token.strip()
]
# Fallback: if HF_TOKENS is not set, use HF_TOKEN as a single entry
if not HF_TOKENS and HF_TOKEN:
    HF_TOKENS = [HF_TOKEN]

# ── LLM Mode ─────────────────────────────────────────────────────────────────
LLM_MODE: str = os.getenv("LLM_MODE", "groq").strip().lower()

# ── Local model path (for test/offline mode) ─────────────────────────────────
LOCAL_LLM_MODEL_PATH: str = os.getenv("LOCAL_LLM_MODEL_PATH", "") or None

# ── Database ─────────────────────────────────────────────────────────────────
DB_CONFIG: dict = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "ecg_agent_data"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ── PDF Knowledge Base ───────────────────────────────────────────────────────
PDF_DOCUMENT_NAME: str = os.getenv(
    "PDF_DOCUMENT_NAME",
    "Cardiac_Diagnostics_Comprehensive_KB.pdf",
)
PDF_DEFAULT_PATH: str = os.getenv(
    "PDF_DEFAULT_PATH",
    "Cardiac_Diagnostics_Comprehensive_KB.pdf",
)

# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_CONFIG: dict = {
    "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    "consumer_group": os.getenv("KAFKA_CONSUMER_GROUP", "ecg-processor"),
    "raw_input_topic": os.getenv("KAFKA_RAW_INPUT_TOPIC", "ecg.raw.signal"),
    "temporal_events_topic": os.getenv("KAFKA_TEMPORAL_EVENTS_TOPIC", "ecg.events.temporal"),
    "clinical_results_topic": os.getenv("KAFKA_CLINICAL_RESULTS_TOPIC", "ecg.results.clinical"),
    "patient_registration_topic": os.getenv("KAFKA_PATIENT_REGISTRATION_TOPIC", "ecg.patient.registration"),
    "max_poll_records": int(os.getenv("KAFKA_MAX_POLL_RECORDS", "500")),
    "session_timeout_ms": int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000")),
    "enable_auto_commit": os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "false").lower() == "true",
    "compression": os.getenv("KAFKA_COMPRESSION", "gzip"),
}