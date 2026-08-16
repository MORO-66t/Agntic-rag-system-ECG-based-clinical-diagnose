"""
kafka_producer.py
==================
Kafka producer for ECG processing results.
Publishes to Stream 2 (temporal events) and Stream 3 (clinical results).

Architecture
------------
    ECGPipeline / temporal_analysis
         │
         ├──► ECGEventProducer.publish_temporal_event()
         │       → ecg.events.temporal (Stream 2)
         │       → Non-agent events only (patterns, rates, pauses)
         │
         └──► ECGEventProducer.publish_clinical_result()
                 → ecg.results.clinical (Stream 3)
                 → Agent-triggered disease detections + AI assessment
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import KAFKA_CONFIG

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Event types that should go to Stream 2 (temporal, non-agent)
# These are the events where trigger_agent=False in Event_Manager.py
# ─────────────────────────────────────────────────────────────────────────────
TEMPORAL_EVENT_TYPES = {
    # Ectopy patterns
    "BIGEMINY", "TRIGEMINY", "QUADRIGEMINY", "COUPLET",
    "ATRIAL_BIGEMINY", "ATRIAL_TRIGEMINY", "ATRIAL_QUADRIGEMINY",
    "ATRIAL_COUPLET", "ATRIAL_TRIPLET",
    # Rate (non-extreme)
    "BRADYCARDIA", "TACHYCARDIA",
    # Burden
    "HIGH_PVC_BURDEN",
    # Pauses (non-asystole)
    "PAUSE_DETECTED",
    # Escape beats
    "VENTRICULAR_ESCAPE_BEAT", "JUNCTIONAL_ESCAPE_BEAT", "ATRIAL_ESCAPE_BEAT",
    # Conduction (first degree only — higher degrees go to Stream 3)
    "FIRST_DEGREE_AV_BLOCK",
    # Quality
    "LOW_SIGNAL_QUALITY",
    # RR irregularity (non-diagnostic flag)
    "RR_IRREGULARITY_SUGGESTIVE",
}

# Event types that should go to Stream 3 (clinical results)
# These are agent-triggered or clinically significant findings
CLINICAL_EVENT_TYPES = {
    "AFIB_DETECTED", "AFLUTTER_SUSPECTED", "VT_RUN",
    "SVT_SUSPECTED",
    "EXTREME_TACHYCARDIA", "EXTREME_BRADYCARDIA",
    "PROLONGED_ASYSTOLE",
    "POSSIBLE_LONG_QT", "POSSIBLE_ISCHEMIC_PATTERN",
    "POSSIBLE_HEART_FAILURE_PATTERN",
    "MOBITZ_I_AV_BLOCK", "MOBITZ_II_AV_BLOCK",
    "SECOND_DEGREE_AV_BLOCK_2TO1", "HIGH_GRADE_AV_BLOCK",
    "THIRD_DEGREE_AV_BLOCK",
    "DISEASE_WPW_PREEXCITATION", "DISEASE_BRUGADA_SYNDROME",
    "DISEASE_HCM", "DISEASE_ARVC",
    "DISEASE_PULMONARY_EMBOLISM", "DISEASE_PERICARDITIS",
    "DISEASE_CARDIAC_TAMPONADE", "DISEASE_LVH_HYPERTENSION",
    "DISEASE_PULMONARY_HYPERTENSION", "DISEASE_CARDIAC_AMYLOIDOSIS",
    "DISEASE_VENTRICULAR_FIBRILLATION", "DISEASE_DETECTED",
}


class ECGEventProducer:
    """
    Produces Kafka messages for ECG events and clinical results.

    This class is designed to be injected into ECGPipeline and
    temporal_analysis so they can publish results without knowing
    about Kafka internals.

    Usage:
        producer = ECGEventProducer()
        producer.publish_temporal_event(event, session_metadata)
        producer.publish_clinical_result(event, agent_response, features)
    """

    def __init__(self, kafka_config: Optional[Dict[str, Any]] = None):
        self.config = kafka_config or KAFKA_CONFIG
        self._producer = None  # Lazy-initialized

    def _get_producer(self):
        """Lazy-initialize the Kafka producer."""
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                self._producer = KafkaProducer(
                    bootstrap_servers=self.config["bootstrap_servers"],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: str(k).encode("utf-8"),
                    compression_type=self.config.get("compression", "gzip"),
                    acks="all",
                    retries=3,
                    max_in_flight_requests_per_connection=5,
                    enable_idempotence=True,
                )
                logger.info("Kafka producer initialized (brokers=%s)",
                            self.config["bootstrap_servers"])
            except ImportError:
                logger.warning("kafka-python not installed — producer disabled")
                self._producer = None
            except Exception as e:
                logger.error("Failed to initialize Kafka producer: %s", e)
                self._producer = None
        return self._producer

    def _make_message_id(self) -> str:
        return str(uuid.uuid4())

    def _safe_send(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        """Send a message with error handling. Returns True on success."""
        producer = self._get_producer()
        if producer is None:
            logger.debug("Producer not available — message dropped (topic=%s)", topic)
            return False
        try:
            future = producer.send(topic, key=key, value=value)
            future.get(timeout=5)  # Block until acknowledged
            return True
        except Exception as e:
            logger.error("Failed to send to %s: %s", topic, e)
            return False

    # ── Stream 2: Temporal Events ───────────────────────────────────────

    def publish_temporal_event(
        self,
        event: Dict[str, Any],
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a non-agent temporal event to Stream 2.

        Args:
            event: The event dict from analyze_temporal_window().
                   Must contain event_type, severity, metadata_json, etc.
            session_metadata: Optional dict with session_id, patient_id, etc.

        Returns:
            True if published successfully.
        """
        event_type = event.get("event_type", "")
        if event_type not in TEMPORAL_EVENT_TYPES:
            return False  # Not a temporal event

        session_meta = session_metadata or {}

        message = {
            "schema_version": "1.0",
            "message_id": self._make_message_id(),
            "session_id": session_meta.get("session_id", event.get("session_id", "")),
            "patient_id": session_meta.get("patient_id", ""),
            "event_type": event_type,
            "severity": event.get("severity", "info"),
            "timestamp": event.get("timestamp", datetime.utcnow().timestamp()),
            "beat_index": event.get("beat_index", 0),

            # Episode information (from episode_manager)
            "episode": {
                "episode_id": event.get("episode_id"),
                "episode_action": event.get("episode_action"),
                "storage_status": event.get("storage_status"),
                "start_beat_index": (event.get("episode") or {}).get("start_beat_index"),
                "last_beat_index": (event.get("episode") or {}).get("last_beat_index"),
                "start_timestamp": (event.get("episode") or {}).get("start_timestamp"),
                "last_timestamp": (event.get("episode") or {}).get("last_timestamp"),
            },

            # Key metadata (extracted, not raw)
            "metadata": self._extract_temporal_metadata(event_type, event.get("metadata_json", {})),
        }

        key = f"{message['session_id']}:{event_type}"
        topic = self.config.get("temporal_events_topic", "ecg.events.temporal")
        return self._safe_send(topic, key, message)

    def _extract_temporal_metadata(
        self,
        event_type: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant metadata fields for temporal events.

        Always carries `reason` and `ecg_findings` through when the upstream
        detector populated them, so the frontend can display why an event
        fired and what ECG evidence supports it.
        """
        result: Dict[str, Any] = {}

        # Always pass through reason / ecg_findings when present
        if metadata.get("reason"):
            result["reason"] = metadata["reason"]
        if metadata.get("ecg_findings"):
            ecg = metadata["ecg_findings"]
            if isinstance(ecg, list):
                result["ecg_findings"] = [str(item) for item in ecg]
            else:
                result["ecg_findings"] = ecg

        if event_type in ("BIGEMINY", "TRIGEMINY", "QUADRIGEMINY", "COUPLET",
                          "ATRIAL_BIGEMINY", "ATRIAL_TRIGEMINY", "ATRIAL_QUADRIGEMINY",
                          "ATRIAL_COUPLET", "ATRIAL_TRIPLET"):
            result["pattern_confidence"] = metadata.get("pattern_confidence")
            result["total_ventricular_beats"] = metadata.get("total_ventricular_beats")
            result["total_supraventricular_beats"] = metadata.get("total_supraventricular_beats")
            result["total_evaluated_beats"] = metadata.get("total_evaluated_beats")

        elif event_type == "HIGH_PVC_BURDEN":
            result["pvc_burden"] = metadata.get("pvc_burden")
            result["apc_burden"] = metadata.get("apc_burden")

        elif event_type in ("BRADYCARDIA", "TACHYCARDIA",
                            "EXTREME_BRADYCARDIA", "EXTREME_TACHYCARDIA"):
            result["average_hr"] = metadata.get("average_hr")
            result["observed_duration_sec"] = metadata.get("observed_duration_sec")

        elif event_type == "PAUSE_DETECTED" or event_type == "PROLONGED_ASYSTOLE":
            result["longest_pause_ms"] = metadata.get("longest_pause_ms")
            result["pause_count"] = metadata.get("pause_count")

        elif "ESCAPE_BEAT" in event_type:
            result["escape_type"] = metadata.get("escape_type")
            result["escape_rr"] = metadata.get("escape_rr")

        elif event_type == "FIRST_DEGREE_AV_BLOCK":
            result["pr_interval_ms"] = metadata.get("pr_interval_ms")

        elif event_type == "LOW_SIGNAL_QUALITY":
            result["avg_signal_quality"] = metadata.get("avg_signal_quality")

        elif event_type == "RR_IRREGULARITY_SUGGESTIVE":
            result["rr_cv"] = metadata.get("rr_cv")
            result["rmssd_sec"] = metadata.get("rmssd_sec")

        return result

    # ── Episode Closed Notifications ────────────────────────────────────
    def publish_closed_event(
        self,
        event: Dict[str, Any],
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish an episode-closed notification to the temporal topic.

        Closed events are not new findings — they carry the full episode
        summary (duration, start/end beat, start/end timestamp, reason,
        findings) for a previously-published event that has now ended.
        They go to Stream 2 (ecg.events.temporal) regardless of whether
        the event type was temporal or clinical, because they are status
        updates, not new clinical results.
        """
        session_meta = session_metadata or {}
        ep = event.get("episode") or {}

        message = {
            "schema_version": "1.0",
            "message_id": self._make_message_id(),
            "session_id": session_meta.get("session_id", event.get("session_id", "")),
            "patient_id": session_meta.get("patient_id", ""),
            "event_type": event.get("event_type", ""),
            "severity": event.get("severity", "info"),
            "timestamp": event.get("timestamp", datetime.utcnow().timestamp()),
            "beat_index": event.get("beat_index", 0),
            "storage_status": "closed",
            "episode_action": "closed",
            "episode": {
                "episode_id": event.get("episode_id"),
                "start_beat_index": ep.get("start_beat_index"),
                "last_beat_index": ep.get("last_beat_index"),
                "start_timestamp": ep.get("start_timestamp"),
                "last_timestamp": ep.get("last_timestamp"),
                "duration_beats": ep.get("duration_beats", 0),
                "duration_sec": ep.get("duration_sec", 0.0),
                "confidence_evolution": ep.get("confidence_evolution"),
                "ended_reason": ep.get("ended_reason", "condition_no_longer_true"),
            },
            "metadata": {
                "reason": (ep.get("latest_detector_metadata") or {}).get("reason", ""),
                "ecg_findings": (ep.get("latest_detector_metadata") or {}).get("ecg_findings", []),
                "recurrence_count": ep.get("recurrence_count", ep.get("evaluations_count", 0)),
            },
        }

        key = f"{message['session_id']}:{message['event_type']}:closed"
        topic = self.config.get("temporal_events_topic", "ecg.events.temporal")
        return self._safe_send(topic, key, message)

    # ── Stream 3: Clinical Results ──────────────────────────────────────

    def publish_clinical_result(
        self,
        event: Dict[str, Any],
        agent_response: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a clinical disease detection result to Stream 3.

        Args:
            event: The event dict from analyze_temporal_window().
            agent_response: The response from PDFSemanticECGAgent.analyze() (if triggered).
            features: The beat feature dict (for key ECG measurements).
            session_metadata: Session/patient metadata.

        Returns:
            True if published successfully.
        """
        event_type = event.get("event_type", "")
        if event_type not in CLINICAL_EVENT_TYPES:
            return False  # Not a clinical event

        session_meta = session_metadata or {}
        meta = event.get("metadata_json", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        # Build the clinical result message
        message = {
            "schema_version": "1.0",
            "message_id": self._make_message_id(),
            "session_id": session_meta.get("session_id", event.get("session_id", "")),
            "patient_id": session_meta.get("patient_id", ""),
            "event_type": event_type,
            "severity": event.get("severity", "info"),
            "timestamp": event.get("timestamp", datetime.utcnow().timestamp()),
            "beat_index": event.get("beat_index", 0),

            # Episode information
            "episode": {
                "episode_id": event.get("episode_id"),
                "episode_action": event.get("episode_action"),
                "storage_status": event.get("storage_status"),
                "start_beat_index": (event.get("episode") or {}).get("start_beat_index"),
                "last_beat_index": (event.get("episode") or {}).get("last_beat_index"),
                "start_timestamp": (event.get("episode") or {}).get("start_timestamp"),
                "last_timestamp": (event.get("episode") or {}).get("last_timestamp"),
            },

            # Disease-specific fields
            "disease": meta.get("disease", event_type),
            "confidence": meta.get("confidence", 0.0),
            "reason": meta.get("reason", ""),

            # Key ECG findings (from features or metadata)
            "ecg_findings": self._extract_ecg_findings(features, meta),

            # Risk and recommendations
            "risk_level": self._infer_risk_level(event.get("severity", "info")),
            "recommendation": "",  # Filled by agent if available

            # Agent assessment (if triggered)
            "agent_assessment": self._build_agent_assessment(agent_response),
        }

        key = f"{message['session_id']}:{event_type}"
        topic = self.config.get("clinical_results_topic", "ecg.results.clinical")
        return self._safe_send(topic, key, message)

    def _extract_ecg_findings(
        self,
        features: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract key ECG measurements for clinical output."""
        findings: Dict[str, Any] = {}

        if features:
            findings["heart_rate_bpm"] = features.get("heart_rate")
            findings["rr_interval_ms"] = features.get("rr_interval")
            findings["qrs_duration_ms"] = features.get("qrs_width")
            findings["qtc_ms"] = features.get("qtc_fridericia") or features.get("qtc")
            findings["pr_interval_ms"] = features.get("pr_interval_ms")
            findings["p_wave_detected"] = features.get("p_wave_detected")
            findings["st_deviation_mv"] = features.get("st_deviation")
            findings["t_wave_inverted"] = features.get("t_wave_inverted")

        # Add disease-specific findings from metadata
        if "ecg_findings" in metadata:
            ecg = metadata["ecg_findings"]
            if isinstance(ecg, dict):
                findings.update(ecg)
            elif isinstance(ecg, list):
                # Disease detector stores ecg_findings as a list of strings
                # Store as a JSON-compatible list under a single key
                findings["details"] = [str(item) for item in ecg]

        return findings

    def _infer_risk_level(self, severity: str) -> str:
        mapping = {
            "critical": "critical",
            "high": "elevated",
            "moderate": "moderate",
            "low": "low",
            "info": "informational",
        }
        return mapping.get(severity, "unknown")

    def _build_agent_assessment(
        self,
        agent_response: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract relevant fields from the agent response."""
        if not agent_response:
            return None

        return {
            "triggered": True,
            "assessment_id": agent_response.get("assessment_id"),
            "llm_confidence": agent_response.get("confidence"),
            "summary": agent_response.get("summary", ""),
            "icd10_codes": agent_response.get("icd10_codes", []),
            "differential_diagnoses": agent_response.get("differential_diagnoses", []),
            "recommendations": agent_response.get("recommendations", []),
        }

    def close(self) -> None:
        """Flush and close the producer."""
        if self._producer is not None:
            try:
                self._producer.flush()
                self._producer.close()
                logger.info("Kafka producer closed.")
            except Exception as e:
                logger.error("Error closing Kafka producer: %s", e)