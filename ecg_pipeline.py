"""
ECG Clinical Intelligence Pipeline — Orchestrator

Connects every existing module into a single end-to-end execution path:

    Raw ECG Beat
     → CNN Prediction          (model_service.ECGModelService.predict_single)
     → Feature Extraction      (feature_engineering.process_beat)
     → Database Storage        (database.ECGDatabase.insert_beat)
     → Temporal Analysis       (temporal_analysis.analyze_temporal_window)
     → Event Detection         (temporal_analysis  internal detectors)
     → Event Storage           (temporal_analysis  process_event_with_cooldown)
     → Event Manager Rules     (Event_Manager.evaluate_event)
     → Agent Trigger           (agent.ECGAgent.analyze)
     → Clinical Output
"""

import logging
import traceback
from typing import Dict, Any, List, Optional

# ─────────────────────────────────────────────
# Resolve a cross‑module dependency:
#   feature_engineering.process_beat() calls
#   estimate_qt_interval() and calculate_qtc()
#   which are defined in temporal_analysis.py
#   but never imported in feature_engineering.py.
#
#   We inject them here so process_beat() works
#   without modifying feature_engineering.py.
# ─────────────────────────────────────────────
from temporal_analysis import estimate_qt_interval, calculate_qtc
import feature_engineering as _fe_module

_fe_module.estimate_qt_interval = estimate_qt_interval
_fe_module.calculate_qtc = calculate_qtc

# ─────────────────────────────────────────────
# Core module imports (existing codebase)
# ─────────────────────────────────────────────
from model_service import ECGModelService
from feature_engineering import process_beat as extract_beat_features
from database import ECGDatabase
from temporal_analysis import analyze_temporal_window
from Event_Manager import evaluate_event
from agent import ECGAgent

logger = logging.getLogger(__name__)


# =============================================
# ECGPipeline
# =============================================

class ECGPipeline:
    """
    Orchestrates the full clinical ECG processing pipeline.

    Lifecycle
    ---------
    1. ``__init__``  – load model, database, and (optionally) the LLM agent.
    2. ``process_beat`` – run the full pipeline for one incoming beat.
    3. ``close``       – release database connections.
    """

    def __init__(
        self,
        model_path: str = "ecg_cnn_model.keras",
        db_config: Optional[Dict[str, str]] = None,
        enable_agent: bool = True,
        temporal_window_size: int = 50,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the Keras CNN model file.
        db_config : dict, optional
            Passed through to ``ECGDatabase``.
        enable_agent : bool
            If ``False``, the LLM agent is never loaded or called.
            Useful for fast testing or batch feature‑extraction runs.
        temporal_window_size : int
            Number of recent beats fetched for temporal analysis.
        """
        logger.info("Initializing ECG Pipeline …")

        # ── Deep‑Learning Model ──
        try:
            self.model_service = ECGModelService(model_path)
            logger.info("CNN model loaded.")
        except Exception as exc:
            logger.error(f"Failed to load CNN model: {exc}")
            raise

        # ── PostgreSQL Database ──
        try:
            self.db = ECGDatabase(db_config)
            logger.info("Database connection pool ready.")
        except Exception as exc:
            logger.error(f"Failed to connect to database: {exc}")
            raise

        # ── Clinical Reasoning Agent (optional) ──
        self.agent: Optional[ECGAgent] = None
        if enable_agent:
            try:
                self.agent = ECGAgent()
                logger.info("ECG Agent loaded.")
            except Exception as exc:
                logger.warning(
                    f"Agent failed to load — pipeline will run without LLM reasoning: {exc}"
                )

        # ── Configuration ──
        self.temporal_window_size = temporal_window_size

        # Per‑session beat counter
        self._beat_counters: Dict[str, int] = {}

        logger.info("ECG Pipeline ready.")

    # ─────────────────────────────────────────
    # Main entry‑point
    # ─────────────────────────────────────────

    def process_beat(
        self,
        signal: List[float],
        session_id: str,
        timestamp: float,
        rr_interval: float = 0.0,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single incoming ECG beat through the full pipeline.

        Parameters
        ----------
        signal : list[float]
            187‑sample ECG window (Lead II, normalised).
        session_id : str
            Unique identifier for the monitoring session.
        timestamp : float
            UNIX epoch timestamp (seconds) of this beat.
        rr_interval : float
            RR interval preceding this beat (seconds or ms — the
            feature‑engineering layer auto‑detects the unit).
        patient_metadata : dict, optional
            Clinical metadata (age, hypertension, diabetes …)
            consumed by temporal risk‑augmentation functions.

        Returns
        -------
        dict
            {
                "beat_prediction": { ... } or None,
                "features":        { ... } or None,
                "events":          [ ... ],
                "agent_responses": [ ... ]
            }
        """
        result: Dict[str, Any] = {
            "beat_prediction": None,
            "features": None,
            "events": [],
            "agent_responses": [],
        }

        # ── Beat index bookkeeping ──
        if session_id not in self._beat_counters:
            self._beat_counters[session_id] = 0
        beat_index = self._beat_counters[session_id]
        self._beat_counters[session_id] += 1

        # ================================================
        # STEP 1 — CNN Prediction
        # ================================================
        try:
            prediction = self.model_service.predict_single(signal)
            result["beat_prediction"] = prediction
            logger.debug(
                f"Beat {beat_index}: label={prediction['predicted_class']} "
                f"conf={prediction['prediction_confidence']:.2f}"
            )
        except Exception as exc:
            logger.error(f"[Step 1] Model prediction failed: {exc}")
            return result  # Cannot proceed without a label

        # ================================================
        # STEP 2 — Feature Extraction
        # ================================================
        try:
            beat_data = extract_beat_features(
                ecg_samples=signal,
                rr_interval=rr_interval,
                predicted_label=prediction["predicted_label"],
                confidence=prediction["prediction_confidence"],
                session_id=session_id,
                timestamp=timestamp,
                beat_index=beat_index,
            )
            result["features"] = beat_data
        except Exception as exc:
            logger.error(f"[Step 2] Feature extraction failed: {exc}")
            return result  # Cannot proceed without features

        # ================================================
        # STEP 3 — Store Beat in PostgreSQL
        # ================================================
        try:
            self.db.insert_beat(beat_data)
        except Exception as exc:
            logger.error(f"[Step 3] Beat storage failed: {exc}")
            # Non‑fatal: temporal analysis can still use whatever is in DB

        # ================================================
        # STEP 4 — Temporal Analysis + Event Detection
        # ================================================
        detected_events: List[Dict[str, Any]] = []
        try:
            detected_events = analyze_temporal_window(
                session_id=session_id,
                db_connection=self.db,
                patient_metadata=patient_metadata,
                window_size=self.temporal_window_size,
            )
            result["events"] = detected_events
            if detected_events:
                logger.info(
                    f"Beat {beat_index}: {len(detected_events)} event(s) detected — "
                    + ", ".join(e["event_type"] for e in detected_events)
                )
        except Exception as exc:
            logger.error(f"[Step 4] Temporal analysis failed: {exc}")

        # ================================================
        # STEP 5 — Event Manager Evaluation + Agent Trigger
        # ================================================
        #   analyze_temporal_window already:
        #     • evaluates every event via Event_Manager.evaluate_event
        #       (attaches dict at event["event_manager"])
        #     • stores / updates events via process_event_with_cooldown
        #
        #   Here we only need to check the trigger flags and call the agent.
        # ================================================
        for event in detected_events:
            event_rule = event.get("event_manager", {})

            if not event_rule.get("known_event", False):
                continue

            if not event_rule.get("trigger_agent", False):
                continue

            if self.agent is None:
                logger.debug(
                    f"Agent trigger skipped (agent disabled) for {event['event_type']}"
                )
                continue

            try:
                logger.info(
                    f"Triggering agent for event: {event['event_type']} "
                    f"(severity={event.get('severity')})"
                )
                response = self.agent.analyze(
                    session_id,
                    event["event_type"],
                )
                result["agent_responses"].append({
                    "event_type": event["event_type"],
                    "severity": event.get("severity"),
                    "priority": event_rule.get("priority"),
                    "escalation_level": event_rule.get("escalation_level"),
                    "response": response,
                })
            except Exception as exc:
                logger.error(
                    f"[Step 5] Agent reasoning failed for "
                    f"{event['event_type']}: {exc}\n{traceback.format_exc()}"
                )

        return result

    # ─────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────

    def close(self):
        """Release database connections."""
        try:
            if self.db:
                self.db.close()
                logger.info("Database connections closed.")
        except Exception as exc:
            logger.error(f"Error closing database: {exc}")
