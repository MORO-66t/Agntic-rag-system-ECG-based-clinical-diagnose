"""
ecg_kafka_service.py
======================
Main entry point for the Kafka-based ECG processing service.

Architecture
------------
    Kafka (ecg.raw.signal) ──► RawECGStreamProcessor
                                    │
                                    ▼
                            ECGPipeline.process_beat()
                                    │
                                    ├──► ECGEventProducer → ecg.events.temporal
                                    └──► ECGEventProducer → ecg.results.clinical

    Kafka (ecg.patient.registration) ──► PatientMetadataCache
                                              │
                                              ▼
                                      Joined with beats by session_id

This service is designed to run as a standalone process.
It does NOT replace realtime_stream.py — that file still works for
testing with WFDB files.

Usage:
    python ecg_kafka_service.py
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import uuid
from typing import Any, Dict, Optional

from config import DB_CONFIG, KAFKA_CONFIG
from database import ECGDatabase
from ecg_pipeline import ECGPipeline
from kafka_producer import ECGEventProducer
from kafka_raw_consumer import RawECGStreamProcessor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Patient Metadata Cache
# ─────────────────────────────────────────────────────────────────────────────

class PatientMetadataCache:
    """
    Caches patient metadata from Stream 4 (ecg.patient.registration).
    
    In production, this would consume from a Kafka topic. For now,
    it provides an interface that can be populated from the database
    or from a direct API call.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get cached patient metadata."""
        return self._cache.get(patient_id)

    def set(self, patient_id: str, metadata: Dict[str, Any]) -> None:
        """Cache patient metadata."""
        self._cache[patient_id] = metadata

    def get_or_fetch(self, patient_id: str, db: Optional[ECGDatabase] = None) -> Dict[str, Any]:
        """Get from cache, or fetch from database if available."""
        cached = self.get(patient_id)
        if cached is not None:
            return cached

        if db is not None:
            try:
                metadata = db.get_patient_metadata(patient_id)
                if metadata:
                    self.set(patient_id, metadata)
                    return metadata
            except Exception:
                pass

        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Poll Function
# ─────────────────────────────────────────────────────────────────────────────

class KafkaMessagePoller:
    """
    Wraps a Kafka consumer to provide a simple poll() callable
    for RawECGStreamProcessor.iter_beats().
    """

    def __init__(self, kafka_config: Dict[str, Any]):
        self.config = kafka_config
        self._consumer = None
        self._buffer: list = []
        self._buffer_index = 0

    def _get_consumer(self):
        if self._consumer is None:
            try:
                from kafka import KafkaConsumer
                self._consumer = KafkaConsumer(
                    self.config["raw_input_topic"],
                    bootstrap_servers=self.config["bootstrap_servers"],
                    group_id=self.config["consumer_group"],
                    enable_auto_commit=self.config.get("enable_auto_commit", False),
                    max_poll_records=self.config.get("max_poll_records", 500),
                    session_timeout_ms=120000,  # 2 min (NK2 can block >30s)
                    heartbeat_interval_ms=30000,  # heartbeat every 30s
                    max_poll_interval_ms=180000,  # max time between polls: 3 min
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    auto_offset_reset="latest",
                )
                logger.info("Kafka consumer initialized (topic=%s, group=%s)",
                            self.config["raw_input_topic"], self.config["consumer_group"])
            except ImportError:
                logger.warning("kafka-python not installed — using simulated poll")
                self._consumer = None
            except Exception as e:
                logger.error("Failed to initialize Kafka consumer: %s", e)
                self._consumer = None
        return self._consumer

    def poll(self) -> Optional[Dict[str, Any]]:
        """
        Returns the next message value, or None if no message available.
        Compatible with RawECGStreamProcessor.iter_beats() poll_function.
        """
        # Drain buffer first
        if self._buffer_index < len(self._buffer):
            msg = self._buffer[self._buffer_index]
            self._buffer_index += 1
            return msg

        consumer = self._get_consumer()
        if consumer is None:
            time.sleep(0.1)
            return None

        try:
            raw_msgs = consumer.poll(timeout_ms=1000, max_records=10)
            if not raw_msgs:
                return None

            all_msgs = []
            for tp, msgs in raw_msgs.items():
                for msg in msgs:
                    all_msgs.append(msg.value)

            if not all_msgs:
                return None

            self._buffer = all_msgs
            self._buffer_index = 1
            return all_msgs[0]

        except Exception as e:
            logger.error("Kafka poll error: %s", e)
            return None

    def close(self):
        if self._consumer is not None:
            try:
                self._consumer.close()
                logger.info("Kafka consumer closed.")
            except Exception as e:
                logger.error("Error closing Kafka consumer: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main Service
# ─────────────────────────────────────────────────────────────────────────────

class ECGKafkaService:
    """
    Main ECG processing service that connects Kafka → Pipeline → Kafka.

    Usage:
        service = ECGKafkaService()
        service.run()
    """

    def __init__(
        self,
        kafka_config: Optional[Dict[str, Any]] = None,
        db_config: Optional[Dict[str, str]] = None,
        enable_agent: bool = True,
        temporal_window_size: int = 300,
        debug_morphology: bool = False,
    ):
        self.kafka_config = kafka_config or KAFKA_CONFIG
        self.db_config = db_config or DB_CONFIG
        self.enable_agent = enable_agent
        self.temporal_window_size = temporal_window_size
        self.debug_morphology = debug_morphology

        # Components (lazy-initialized)
        self.db: Optional[ECGDatabase] = None
        self.pipeline: Optional[ECGPipeline] = None
        self.producer: Optional[ECGEventProducer] = None
        self.consumer: Optional[KafkaMessagePoller] = None
        self.processor: Optional[RawECGStreamProcessor] = None
        self.patient_cache: Optional[PatientMetadataCache] = None

        # Running state
        self._running = False
        self._epoch_offset: Optional[float] = None

    def start(self) -> None:
        """Initialize all components."""
        logger.info("Starting ECG Kafka Service...")

        # Database
        try:
            self.db = ECGDatabase(self.db_config)
            logger.info("Database connected.")
        except Exception as e:
            logger.error("Failed to connect to database: %s", e)
            raise

        # Pipeline
        try:
            self.pipeline = ECGPipeline(
                db_config=self.db_config,
                enable_agent=self.enable_agent,
                temporal_window_size=self.temporal_window_size,
                enable_morphology_debug=self.debug_morphology,
            )
            logger.info("ECG Pipeline initialized.")
        except Exception as e:
            logger.error("Failed to initialize pipeline: %s", e)
            raise

        # Kafka producer
        self.producer = ECGEventProducer(self.kafka_config)

        # Wire Kafka producer into pipeline for direct event publishing
        if self.pipeline:
            self.pipeline.set_kafka_producer(self.producer)

        # Kafka consumer for raw ECG input
        self.consumer = KafkaMessagePoller(self.kafka_config)

        # Raw ECG stream processor
        self.processor = RawECGStreamProcessor()

        # Patient metadata cache
        self.patient_cache = PatientMetadataCache()

        # Start Stream 4 consumer (patient registration) in background thread
        self._start_patient_registration_consumer()

        # Register signal handlers for graceful shutdown (main thread only)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            logger.debug("Signal handlers not registered (not in main thread)")

        # Set epoch offset for timestamp conversion
        self._epoch_offset = time.time()
        logger.info("Real-time baseline epoch: %.3f", self._epoch_offset)

        self._running = True
        logger.info("ECG Kafka Service started.")

    def _start_patient_registration_consumer(self) -> None:
        """Start a background thread to consume patient registration from Stream 4."""
        import threading

        def _consume_patient_registrations():
            """Consume from ecg.patient.registration and populate cache + database."""
            try:
                from kafka import KafkaConsumer
                consumer = KafkaConsumer(
                    self.kafka_config.get("patient_registration_topic", "ecg.patient.registration"),
                    bootstrap_servers=self.kafka_config["bootstrap_servers"],
                    group_id=f"{self.kafka_config['consumer_group']}-patient",
                    auto_offset_reset="latest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    consumer_timeout_ms=1000,
                )
                logger.info("Patient registration consumer started (topic=%s)",
                            self.kafka_config.get("patient_registration_topic"))
                while self._running:
                    msgs = consumer.poll(timeout_ms=1000)
                    for tp, records in msgs.items():
                        for record in records:
                            val = record.value
                            patient_id = val.get("patient_id")
                            if patient_id:
                                # 1. Cache in memory for fast lookup
                                self.patient_cache.set(patient_id, val)
                                
                                # 2. Persist to database so metadata survives restarts
                                if self.db is not None:
                                    mandatory = val.get("mandatory", {})
                                    optional = val.get("optional", {})
                                    try:
                                        self.db.register_patient(
                                            patient_id=patient_id,
                                            age=mandatory.get("age"),
                                            sex=mandatory.get("sex"),
                                            smoking_status=optional.get("smoking_status", ""),
                                            comorbidities=optional.get("known_diagnoses", []),
                                        )
                                        logger.info("Persisted patient %s to database", patient_id)
                                    except Exception as db_err:
                                        logger.warning("Failed to persist patient %s: %s", patient_id, db_err)
                consumer.close()
            except ImportError:
                logger.debug("kafka-python not available — patient registration consumer disabled")
            except Exception as e:
                logger.warning("Patient registration consumer error: %s", e)

        thread = threading.Thread(target=_consume_patient_registrations, daemon=True)
        thread.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %s — shutting down...", signum)
        self._running = False

    def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Stopping ECG Kafka Service...")
        self._running = False

        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        if self.pipeline:
            self.pipeline.close()
        if self.db:
            self.db.close()

        logger.info("ECG Kafka Service stopped.")

    def _get_predicted_class(self, result: Dict) -> str:
        """Extract predicted class from pipeline result."""
        pred = result.get("beat_prediction") or {}
        class_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
        pc = pred.get("predicted_class")
        if pc:
            return pc
        pl = pred.get("predicted_label")
        if pl is not None:
            return class_map.get(pl, "?")
        return "?"

    def run(self, max_beats: Optional[int] = None) -> None:
        """
        Main event loop: consume raw ECG → process beats → publish results.

        Args:
            max_beats: Optional limit for testing.
        """
        self.start()

        session_id = f"kafka-session-{uuid.uuid4().hex[:8]}"
        logger.info("Session ID: %s", session_id)

        LOG_EVERY_N_BEATS = 1

        try:
            for beat_dict in self.processor.iter_beats(
                poll_function=self.consumer.poll,
                session_id=session_id,
                max_beats=max_beats,
            ):
                if not self._running:
                    break

                beat_index = beat_dict.get("beat_index", 0)

                # ── Add epoch offset to timestamp (fixes 1970 dates) ──
                if self._epoch_offset is not None:
                    beat_dict["timestamp"] = self._epoch_offset + beat_dict["timestamp"]

                # ── Fetch patient metadata with registration check ──
                patient_id = beat_dict.get("patient_id", session_id)
                patient_metadata = self.patient_cache.get_or_fetch(patient_id, self.db)
                if not patient_metadata:
                    logger.warning("Patient %s not registered — using empty metadata", patient_id)

                # ── Process through pipeline ──
                try:
                    result = self.pipeline.process_beat(
                        signal=beat_dict["cnn_signal"],
                        session_id=session_id,
                        timestamp=beat_dict["timestamp"],
                        rr_interval=beat_dict["rr_interval"],
                        patient_metadata=patient_metadata,
                        original_samples=beat_dict.get("original_samples"),
                        context_samples=beat_dict.get("context_samples"),
                        context_rpeaks=beat_dict.get("context_rpeaks"),
                        context_target_r=beat_dict.get("context_target_r"),
                        beat_start=beat_dict.get("beat_start", 0),
                        context_info=beat_dict.get("context_info"),
                    )

                    # ── Beat trace log (matching realtime_stream.py format) ──
                    predicted_class = self._get_predicted_class(result)
                    pred_conf = result.get("beat_prediction", {}).get("prediction_confidence", 0.0)
                    events = result.get("events", [])
                    rr = beat_dict.get("rr_interval", 0)
                    ts = beat_dict.get("timestamp", 0)
                    
                    if beat_index % LOG_EVERY_N_BEATS == 0:
                        logger.info(
                            "Beat %5d | t=%8.2fs | pred=%s (%.2f) | RR=%6.1fms | events=%d",
                            beat_index, ts, predicted_class, pred_conf, rr, len(events),
                        )

                    # ── Publish events to Kafka with console logging ──
                    self._publish_events(result, session_id, patient_id, patient_metadata)

                except Exception as e:
                    logger.error("Pipeline error at beat %s: %s",
                                 beat_dict.get("beat_index", "?"), e)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal.")
        finally:
            self.stop()

    def _publish_events(
        self,
        result: Dict[str, Any],
        session_id: str,
        patient_id: str,
        patient_metadata: Dict[str, Any],
    ) -> None:
        """Publish events from pipeline result to appropriate Kafka topics."""
        if self.producer is None:
            return

        session_metadata = {
            "session_id": session_id,
            "patient_id": patient_id,
            **patient_metadata,
        }

        for event in result.get("events", []):
            # Add session context to event
            event["session_id"] = session_id
            event["patient_id"] = patient_id

            # Get event rule
            event_rule = event.get("event_manager", {})

            # ── Closed events: publish with full episode data (duration, start/end) ──
            if event.get("storage_status") == "closed":
                # Closed events carry the full episode summary (duration,
                # start/end beat, start/end timestamp, reason, findings).
                # They go to the temporal topic as status updates.
                published = self.producer.publish_closed_event(
                    event=event,
                    session_metadata=session_metadata,
                )
                if published:
                    ep = event.get("episode", {})
                    logger.info(
                        "  [CLOSED] %s (severity=%s, duration=%.1fs, beats=%d, episode_id=%s, ended_reason=%s)",
                        event["event_type"], event.get("severity"),
                        ep.get("duration_sec", 0),
                        ep.get("duration_beats", 0),
                        event.get("episode_id"),
                        ep.get("ended_reason", "condition_no_longer_true"),
                    )
                continue

            # Publish to appropriate stream with console log
            if event_rule.get("trigger_agent", False):
                # Clinical result (Stream 3)
                agent_responses = result.get("agent_responses", [])
                agent_response = None
                for ar in agent_responses:
                    if ar.get("event_type") == event.get("event_type"):
                        agent_response = ar.get("response")
                        break

                published = self.producer.publish_clinical_result(
                    event=event,
                    agent_response=agent_response,
                    features=result.get("features"),
                    session_metadata=session_metadata,
                )
                if published:
                    logger.info("  [CLINICAL] %s (severity=%s, confidence=%s, episode=%s)",
                                event["event_type"], event.get("severity"),
                                event.get("metadata_json", {}).get("confidence", "N/A"),
                                event.get("episode_action", "N/A"))
            else:
                # Temporal event (Stream 2)
                published = self.producer.publish_temporal_event(
                    event=event,
                    session_metadata=session_metadata,
                )
                if published:
                    logger.info("  [TEMPORAL] %s (severity=%s, episode=%s)",
                                event["event_type"], event.get("severity"),
                                event.get("episode_action", "N/A"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for the Kafka ECG service."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Kafka-based ECG Processing Service")
    parser.add_argument("--max-beats", type=int, default=None, help="Stop after N beats (testing)")
    parser.add_argument("--no-agent", action="store_true", help="Disable RAG agent")
    parser.add_argument("--window-size", type=int, default=300, help="Temporal window size")
    parser.add_argument("--debug-morphology", action="store_true",
                        help="Print detailed morphology diagnostics for every beat")
    args = parser.parse_args()

    service = ECGKafkaService(
        enable_agent=not args.no_agent,
        temporal_window_size=args.window_size,
        debug_morphology=args.debug_morphology,
    )

    try:
        service.run(max_beats=args.max_beats)
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error("Fatal error: %s", e)
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()