"""
ECG Clinical Intelligence Pipeline — Orchestrator (Dual-Branch)

Architecture
------------

    Original ECG (360 Hz beat)
         │
         ├──► CNN Branch
         │        signal (187-sample, 125 Hz, pre-resampled) → CNN → label
         │
         └──► Clinical Branch
                   360 Hz beat → feature_engineering.process_beat()
                                   (single NeuroKit2 delineation call,
                                    inside neurokit_feature_extractor.py)
                                 → Temporal Analysis
                                 → Event Engine
                                 → PDFSemanticECGAgent (pgvector RAG)

IMPORTANT — source of truth for inputs
---------------------------------------
This pipeline does NOT resample raw signal to the CNN's 187-sample format,
and it does NOT run its own NeuroKit2 delineation pass to find P/T peak
positions. Both of those are the responsibility of upstream data
preparation:

  * CNN-ready samples (`signal`, 187 @ 125 Hz) come pre-computed from
    convert_wfdb_to_csv_W.py for batch/offline data, or from whatever
    live acquisition layer performs the equivalent resample for a real
    streaming deployment. This pipeline only calls the already-trained
    CNN on whatever 187-sample array it's given (model_service.py does
    no further resampling itself).

  * Clinical-branch features are computed exactly once, inside
    feature_engineering.process_beat() -> neurokit_feature_extractor.py,
    from `original_samples` (the raw 360 Hz beat). This is the single
    source of truth for P/QRS/T delineation, intervals, amplitudes, and
    all derived morphology features.

  * t_peak_position_360 / p_peak_position_360 / rt_interval_ms /
    pr_interval_ms (below) are GROUND-TRUTH / EXPERT ANNOTATION labels,
    not inputs the clinical branch needs to compute features. They are
    stored alongside the pipeline's own NeuroKit2-derived values purely
    so accuracy can be validated against expert annotations (e.g. from
    MIT-BIH's P/T-wave annotation files). Do not recompute these live by
    re-running NeuroKit2 delineation in a caller — pull them from the
    pre-built CSV/dataset that already carries the expert annotation, or
    omit them entirely in a real deployment with no expert labels
    available (they default to -1 / "not provided").

If only `signal` (187-sample CNN input) is provided with no
`original_samples`, the clinical branch is skipped gracefully and only
the CNN prediction is returned.
"""

import config

import logging
import traceback
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

from model_service import ECGModelService
from feature_engineering import process_beat as extract_beat_features
from database import ECGDatabase
from temporal_analysis import analyze_temporal_window
from pdf_semantic_rag import PDFSemanticECGAgent

logger = logging.getLogger(__name__)


class ECGPipeline:
    """
    Dual-branch ECG processing pipeline.

    Lifecycle
    ---------
    1. __init__       — load model, database, optional agent.
    2. process_beat   — run the full pipeline for one beat.
    3. close          — release database connections.
    """

    def __init__(
        self,
        model_path: str = "ecg_cnn_model.keras",
        db_config: Optional[Dict[str, str]] = None,
        enable_agent: bool = True,
        temporal_window_size: int = 300,
        agent_llm_mode: str = config.LLM_MODE or "groq",
        agent_local_model_path: Optional[str] = None,
        pdf_document_name: Optional[str] = None,
        pdf_top_k: Optional[int] = None,
        debug_afib: bool = False,
        debug_afl: bool = False,
        enable_morphology_debug: bool = False,
    ):
        logger.info("Initializing ECG Pipeline (dual-branch) …")

        try:
            self.model_service = ECGModelService(model_path)
            logger.info("CNN model loaded.")
        except Exception as exc:
            logger.error(f"Failed to load CNN model: {exc}")
            raise

        try:
            self.db = ECGDatabase(db_config)
            logger.info("Database connection pool ready.")
        except Exception as exc:
            logger.error(f"Failed to connect to database: {exc}")
            raise

        self.agent: Optional[PDFSemanticECGAgent] = None
        if enable_agent:
            try:
                agent_kwargs: Dict[str, Any] = dict(
                    db=self.db,
                    llm_mode=agent_llm_mode,
                    local_model_path=agent_local_model_path,
                )
                if pdf_document_name is not None:
                    agent_kwargs["document_name"] = pdf_document_name
                if pdf_top_k is not None:
                    agent_kwargs["top_k"] = pdf_top_k
                self.agent = PDFSemanticECGAgent(**agent_kwargs)
                logger.info(
                    "PDF semantic RAG agent loaded with %s LLM backend.",
                    agent_llm_mode,
                )
            except Exception as exc:
                logger.warning(
                    f"Agent failed to load — pipeline running without LLM: {exc}"
                )

        self.temporal_window_size = temporal_window_size
        self._agent_cooldown_sec = 300
        self._last_agent_trigger_time: Dict[str, float] = {}
        self.debug_afib = debug_afib
        self.debug_afl = debug_afl
        self.enable_morphology_debug = enable_morphology_debug
        self._beat_counters: Dict[str, int] = {}
        self.kafka_producer = None  # Set externally via set_kafka_producer()
        # Track last beat_index per session+event_type to implement
        # pattern-based cooldown (skip duplicates within same pattern window)
        self._last_triggered: Dict[str, Dict[str, int]] = {}

        # ── Background thread pool for async agent execution ──────────
        # Agent LLM calls (embedding + retrieval + Groq API) can take
        # 10-60+ seconds. Running them synchronously inside process_beat()
        # blocks the entire streaming loop. Offload them to a single
        # background worker thread so the main thread stays free for the
        # next beat. Each async call gets its own DB connection (psycopg2
        # connections are not thread-safe) — see _run_agent_async().
        self._agent_executor: Optional[ThreadPoolExecutor] = None
        self._agent_thread_db: Optional[ECGDatabase] = None
        # Stash these for the background thread to recreate the agent.
        self._agent_llm_mode = agent_llm_mode
        self._agent_local_model_path = agent_local_model_path
        self._agent_pdf_document_name = pdf_document_name
        self._agent_pdf_top_k = pdf_top_k
        if self.agent is not None:
            self._agent_executor = ThreadPoolExecutor(max_workers=1)
            # Each background agent call needs its own DB connection.
            self._agent_thread_db = ECGDatabase(db_config)

        logger.info("ECG Pipeline ready.")

    def process_beat(
        self,
        signal: List[float],
        session_id: str,
        timestamp: float,
        rr_interval: float = 0.0,
        patient_metadata: Optional[Dict[str, Any]] = None,
        # ── Clinical branch: raw 360 Hz beat ──────────────────────
        original_samples: Optional[List[float]] = None,
        original_beat_json: Optional[str] = None,   # JSON string from CSV
        # ── Expert / ground-truth peak annotations (360 Hz coords) ─
        # Validation-only — NOT used to compute clinical features.
        # See module docstring.
        t_peak_position_360: int = -1,
        p_peak_position_360: int = -1,
        rt_interval_ms: float = -1.0,
        pr_interval_ms: float = -1.0,
        # ── NeuroKit2 context window (real delineation) ──
        context_samples: Optional[Any] = None,
        context_rpeaks: Optional[Any] = None,
        context_target_r: Optional[int] = None,
        beat_start: int = 0,
        context_info: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process one beat through the dual-branch pipeline.

        Parameters
        ----------
        signal : list[float]
            187-sample CNN beat (125 Hz, normalised). Must already be
            resampled/normalised upstream — this method performs no
            resampling.
        original_samples : list[float], optional
            Raw 360 Hz beat (prev_R → next_R). If omitted, the clinical
            branch is skipped and only the CNN prediction is returned.
        original_beat_json : str, optional
            JSON-encoded original beat, used when original_samples is
            None (e.g. reading directly from a CSV column).
        t_peak_position_360 / p_peak_position_360 / rt_interval_ms /
        pr_interval_ms : expert/ground-truth annotations for accuracy
            validation only. Leave at the -1 default if unavailable.
        """
        result: Dict[str, Any] = {
            "beat_prediction": None,
            "features": None,
            "events": [],
            "agent_responses": [],
            "ground_truth": {
                "t_peak_position_360": t_peak_position_360,
                "p_peak_position_360": p_peak_position_360,
                "rt_interval_ms": rt_interval_ms,
                "pr_interval_ms": pr_interval_ms,
            },
        }

        # ── Resolve original (360 Hz) beat ───────────────────────
        if original_samples is None and original_beat_json:
            try:
                original_samples = json.loads(original_beat_json)
            except Exception:
                original_samples = None

        clinical_branch_available = original_samples is not None

        # ── Patient metadata ──────────────────────────────────────
        if patient_metadata is None:
            try:
                patient_metadata = self.db.get_patient_metadata(session_id)
            except Exception as e:
                logger.warning(f"Could not fetch patient metadata: {e}")
                patient_metadata = {}

        # ── Beat counter ──────────────────────────────────────────
        if session_id not in self._beat_counters:
            self._beat_counters[session_id] = 0
        beat_index = self._beat_counters[session_id]
        self._beat_counters[session_id] += 1

        # ════════════════════════════════════════════════════════
        # STEP 1 — CNN Prediction  (187-sample beat, 125 Hz)
        # ════════════════════════════════════════════════════════
        try:
            prediction = self.model_service.predict_single(signal)
            result["beat_prediction"] = prediction
            logger.debug(
                f"Beat {beat_index}: label={prediction['predicted_class']} "
                f"conf={prediction['prediction_confidence']:.2f}"
            )
        except Exception as exc:
            logger.error(f"[Step 1] Model prediction failed: {exc}")
            return result

        if not clinical_branch_available:
            logger.debug(
                f"Beat {beat_index}: no original_samples provided — "
                f"clinical branch skipped, CNN-only result returned."
            )
            return result

        # ════════════════════════════════════════════════════════
        # STEP 2 — Feature Extraction  (360 Hz clinical branch,
        #          single NeuroKit2 pass inside feature_engineering.py)
        # ════════════════════════════════════════════════════════
        try:
            beat_data = extract_beat_features(
                cnn_samples=signal,
                original_samples=original_samples,
                rr_interval=rr_interval,
                predicted_label=prediction["predicted_label"],
                confidence=prediction["prediction_confidence"],
                session_id=session_id,
                timestamp=timestamp,
                beat_index=beat_index,
                t_peak_position=t_peak_position_360,
                p_peak_position=p_peak_position_360,
                rt_interval_ms=rt_interval_ms,
                pr_interval_ms_gt=pr_interval_ms,
                context_samples=context_samples,
                context_rpeaks=context_rpeaks,
                context_target_r=context_target_r,
                beat_start=beat_start,
                context_info=context_info,
                debug_morphology=self.enable_morphology_debug,
            )
            result["features"] = beat_data
            logger.debug(f"Beat {beat_index}: features extracted")
        except Exception as exc:
            logger.error(f"[Step 2] Feature extraction failed: {exc}\n{traceback.format_exc()}")
            return result

        # ════════════════════════════════════════════════════════
        # STEP 3 — Store Beat
        # ════════════════════════════════════════════════════════
        try:
            self.db.insert_beat(beat_data)
        except Exception as exc:
            logger.error(f"[Step 3] Beat storage failed: {exc}")

        # ════════════════════════════════════════════════════════
        # STEP 4 — Temporal Analysis + Event Detection
        #          (Event_Manager evaluation happens inside
        #          analyze_temporal_window — not called directly here)
        # ════════════════════════════════════════════════════════
        detected_events: List[Dict[str, Any]] = []
        try:
            detected_events = analyze_temporal_window(
                session_id=session_id,
                db_connection=self.db,
                patient_metadata=patient_metadata,
                window_size=self.temporal_window_size,
                debug_afib=getattr(self, 'debug_afib', False),
                debug_afl=getattr(self, 'debug_afl', False),
            )
            result["events"] = detected_events
            if detected_events:
                logger.info(
                    f"Beat {beat_index}: {len(detected_events)} event(s) — "
                    + ", ".join(e["event_type"] for e in detected_events)
                )
        except Exception as exc:
            logger.error(f"[Step 4] Temporal analysis failed: {exc}\n{traceback.format_exc()}")

        # ════════════════════════════════════════════════════════
        # STEP 5 — Event Manager + Agent Trigger  (async in
        #          background thread — does NOT block main loop)
        # ════════════════════════════════════════════════════════
        for event in detected_events:
            event_rule = event.get("event_manager", {})
            if event.get("storage_status") != "created":
                continue
            if not event_rule.get("known_event", False):
                continue
            if not event_rule.get("trigger_agent", False):
                continue
            if self.agent is None or self._agent_executor is None:
                logger.debug(
                    f"Agent trigger skipped (agent disabled): {event['event_type']}"
                )
                continue

            # ── Agent time-based cooldown ──────────────────────────
            # Prevents calling the LLM more than once per N seconds
            # for the same event type in the same session.
            trigger_key = f"{session_id}:{event['event_type']}"
            now = time.time()
            last_trigger = self._last_agent_trigger_time.get(trigger_key, 0)
            if now - last_trigger < self._agent_cooldown_sec:
                logger.info(
                    f"Agent cooldown active for {event['event_type']} "
                    f"({now - last_trigger:.0f}s < {self._agent_cooldown_sec}s) — skipped"
                )
                continue
            self._last_agent_trigger_time[trigger_key] = now

            logger.info(
                f"Triggering RAG agent for: {event['event_type']} "
                f"(severity={event.get('severity')}) — async"
            )

            # Submit the agent call to the background thread pool.
            # The callable creates its own agent + DB connection to
            # avoid thread-safety issues with psycopg2.
            self._agent_executor.submit(
                self._run_agent_async,
                session_id=session_id,
                event_type=event["event_type"],
                event=event,
                beat_data=beat_data,
                patient_metadata=patient_metadata,
            )

            # Record a placeholder in the result so the caller knows
            # the agent was dispatched. The full response is stored
            # in the DB by the background thread.
            result["agent_responses"].append({
                "event_type": event["event_type"],
                "severity": event.get("severity"),
                "priority": event_rule.get("priority"),
                "escalation_level": event_rule.get("escalation_level"),
                "response": {"status": "submitted", "mode": "async"},
            })

        return result

    # ── Background agent runner (runs in executor thread) ──────────

    def _run_agent_async(
        self,
        session_id: str,
        event_type: str,
        event: Dict[str, Any],
        beat_data: Dict[str, Any],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run a single agent.analyze() call in a background thread.

        Uses a **separate** ``ECGDatabase`` connection (``self._agent_thread_db``)
        because ``psycopg2`` connections are not thread-safe and the main
        pipeline's ``self.db`` is in active use by the streaming loop.

        The agent call stores its own result in the database via
        ``insert_agent_interaction()``, so no explicit persistence is needed
        here — this method primarily exists to fire the LLM request without
        blocking ``process_beat()``.
        """
        if self.agent is None:
            logger.debug("[async] Agent disabled — skipping.")
            return

        thread_name = threading.current_thread().name
        try:
            logger.info(
                "[async:%s] Starting agent analysis for %s …",
                thread_name, event_type,
            )
            # Build a temporary agent that uses the thread-local DB connection.
            # Use the same LLM mode/path/doc/top_k as the main pipeline agent.
            from pdf_semantic_rag import PDFSemanticECGAgent as _AgentCls

            thread_agent_kwargs: Dict[str, Any] = dict(
                db=self._agent_thread_db,
                llm_mode=self._agent_llm_mode or config.LLM_MODE or "groq",
                local_model_path=self._agent_local_model_path,
            )
            if self._agent_pdf_document_name is not None:
                thread_agent_kwargs["document_name"] = self._agent_pdf_document_name
            if self._agent_pdf_top_k is not None:
                thread_agent_kwargs["top_k"] = self._agent_pdf_top_k
            thread_agent = _AgentCls(**thread_agent_kwargs)

            response = thread_agent.analyze(
                session_id=session_id,
                event_type=event_type,
                event=event,
                beat_data=beat_data,
                patient_metadata=patient_metadata,
            )
            logger.info(
                "[async:%s] Completed agent analysis for %s.",
                thread_name, event_type,
            )
        except Exception as exc:
            logger.error(
                "[async:%s] Agent failed for %s: %s\n%s",
                thread_name, event_type, exc, traceback.format_exc(),
            )

    def set_kafka_producer(self, producer) -> None:
        """Inject a Kafka producer for publishing events to Stream 2 & 3."""
        self.kafka_producer = producer

    def close(self):
        """Release database connections and shut down background executor."""
        # Shut down the background thread pool first so no new agent
        # calls are accepted while we're closing connections.
        if self._agent_executor is not None:
            try:
                self._agent_executor.shutdown(wait=True)
                logger.debug("Background agent executor shut down (waited for current task).")
            except Exception as exc:
                logger.error(f"Error shutting down agent executor: {exc}")

        # Close the thread-local DB connection.
        if self._agent_thread_db is not None:
            try:
                self._agent_thread_db.close()
                logger.debug("Background thread DB connection closed.")
            except Exception as exc:
                logger.error(f"Error closing thread DB: {exc}")

        # Close the main pipeline DB connection.
        try:
            if self.db:
                self.db.close()
                logger.info("Database connections closed.")
        except Exception as exc:
            logger.error(f"Error closing database: {exc}")

# """
# ECG Clinical Intelligence Pipeline — Orchestrator (Dual-Branch)

# Architecture after refactor:

#     Original ECG (360 Hz beat)
#          │
#          ├──► CNN Branch
#          │        resample → 125 Hz → 187 samples → CNN → label
#          │
#          └──► Clinical Branch
#                   360 Hz beat → Feature Engineering
#                                 → Temporal Analysis
#                                 → Event Engine
#                                 → Agent

# The pipeline receives BOTH cnn_samples (187) and original_samples (360 Hz).
# If only cnn_samples is provided (legacy / synthetic tests), clinical branch
# is skipped gracefully.
# """

# import logging
# import traceback
# import json
# from typing import Dict, Any, List, Optional

# from model_service import ECGModelService
# from feature_engineering import process_beat as extract_beat_features
# from database import ECGDatabase
# from temporal_analysis import analyze_temporal_window
# from Event_Manager import evaluate_event
# from pdf_semantic_rag import PDFSemanticECGAgent

# logger = logging.getLogger(__name__)


# class ECGPipeline:
#     """
#     Dual-branch ECG processing pipeline.

#     Lifecycle
#     ---------
#     1. __init__       — load model, database, optional agent.
#     2. process_beat   — run the full pipeline for one beat.
#     3. close          — release database connections.
#     """

#     def __init__(
#         self,
#         model_path: str = "ecg_cnn_model.keras",
#         db_config: Optional[Dict[str, str]] = None,
#         enable_agent: bool = True,
#         temporal_window_size: int = 50,
#         agent_llm_mode: str = "groq",
#         agent_local_model_path: Optional[str] = None,
#     ):
#         logger.info("Initializing ECG Pipeline (dual-branch) …")

#         try:
#             self.model_service = ECGModelService(model_path)
#             logger.info("CNN model loaded.")
#         except Exception as exc:
#             logger.error(f"Failed to load CNN model: {exc}")
#             raise

#         try:
#             self.db = ECGDatabase(db_config)
#             logger.info("Database connection pool ready.")
#         except Exception as exc:
#             logger.error(f"Failed to connect to database: {exc}")
#             raise

#         self.agent: Optional[PDFSemanticECGAgent] = None
#         if enable_agent:
#             try:
#                 self.agent = PDFSemanticECGAgent(
#                     db=self.db,
#                     llm_mode=agent_llm_mode,
#                     local_model_path=agent_local_model_path,
#                 )
#                 logger.info(
#                     "PDF semantic ECG Agent loaded with %s LLM backend.",
#                     agent_llm_mode,
#                 )
#             except Exception as exc:
#                 logger.warning(
#                     f"Agent failed to load — pipeline running without LLM: {exc}"
#                 )

#         self.temporal_window_size = temporal_window_size
#         self._beat_counters: Dict[str, int] = {}
#         logger.info("ECG Pipeline ready.")


#     def process_beat(
#         self,
#         signal: List[float],
#         session_id: str,
#         timestamp: float,
#         rr_interval: float = 0.0,
#         patient_metadata: Optional[Dict[str, Any]] = None,
#         # ── Clinical branch: raw 360 Hz beat ──────────────────────
#         original_samples: Optional[List[float]] = None,
#         original_beat_json: Optional[str] = None,   # JSON string from CSV
#         # ── Expert peak annotations (360 Hz coordinates) ──────────
#         t_peak_position_360: int   = -1,
#         p_peak_position_360: int   = -1,
#         # ── Legacy / CNN-coordinate annotations (kept for compat.) ─
#         t_peak_position: int       = -1,
#         p_peak_position: int       = -1,
#         rt_interval_ms: float      = -1.0,
#         pr_interval_ms: float      = -1.0,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """
#         Process one beat through the dual-branch pipeline.

#         Parameters
#         ----------
#         signal : list[float]
#             187-sample CNN beat (125 Hz, normalised).
#         original_samples : list[float], optional
#             Raw 360 Hz beat (prev_R → next_R).  If omitted, clinical
#             features are computed on `signal` at 125 Hz (legacy mode).
#         original_beat_json : str, optional
#             JSON-encoded original beat from the CSV column
#             ``original_beat_json``.  Used when original_samples is None.
#         t_peak_position_360 : int
#             T-peak index in the 360 Hz ``original_samples`` array.
#         p_peak_position_360 : int
#             P-peak index in the 360 Hz ``original_samples`` array.
#         t_peak_position : int
#             T-peak index in the 187-sample CNN beat (legacy, kept for
#             backward compatibility — NOT used for clinical measurements).
#         """
#         result: Dict[str, Any] = {
#             "beat_prediction": None,
#             "features":        None,
#             "events":          [],
#             "agent_responses": [],
#         }

#         # ── Resolve original (360 Hz) beat ───────────────────────
#         if original_samples is None and original_beat_json:
#             try:
#                 original_samples = json.loads(original_beat_json)
#             except Exception:
#                 original_samples = None

#         # If still None, fall back to CNN beat (legacy / synthetic mode)
#         clinical_samples = original_samples if original_samples is not None else signal
#         # Resolve 360 Hz peak positions: prefer explicit _360 params,
#         # fall back to CNN-coordinate params (legacy CSV without _360 columns)
#         eff_t_pos = t_peak_position_360 if t_peak_position_360 != -1 else t_peak_position
#         eff_p_pos = p_peak_position_360 if p_peak_position_360 != -1 else p_peak_position

#         # ── Patient metadata ──────────────────────────────────────
#         if patient_metadata is None:
#             try:
#                 patient_metadata = self.db.get_patient_metadata(session_id)
#             except Exception as e:
#                 logger.warning(f"Could not fetch patient metadata: {e}")
#                 patient_metadata = {}

#         # ── Beat counter ──────────────────────────────────────────
#         if session_id not in self._beat_counters:
#             self._beat_counters[session_id] = 0
#         beat_index = self._beat_counters[session_id]
#         self._beat_counters[session_id] += 1

#         # ════════════════════════════════════════════════════════
#         # STEP 1 — CNN Prediction  (187-sample beat, 125 Hz)
#         # ════════════════════════════════════════════════════════
#         try:
#             prediction = self.model_service.predict_single(signal)
#             result["beat_prediction"] = prediction
#             logger.debug(
#                 f"Beat {beat_index}: label={prediction['predicted_class']} "
#                 f"conf={prediction['prediction_confidence']:.2f}"
#             )
#         except Exception as exc:
#             logger.error(f"[Step 1] Model prediction failed: {exc}")
#             return result

#         # ════════════════════════════════════════════════════════
#         # STEP 2 — Feature Extraction  (360 Hz clinical branch)
#         # ════════════════════════════════════════════════════════
#         try:
#             beat_data = extract_beat_features(
#                 cnn_samples=signal,
#                 original_samples=clinical_samples,
#                 rr_interval=rr_interval,
#                 predicted_label=prediction["predicted_label"],
#                 confidence=prediction["prediction_confidence"],
#                 session_id=session_id,
#                 timestamp=timestamp,
#                 beat_index=beat_index,
#                 t_peak_position=eff_t_pos,
#                 p_peak_position=eff_p_pos,
#                 rt_interval_ms=rt_interval_ms,
#                 pr_interval_ms_gt=pr_interval_ms,
#             )
#             result["features"] = beat_data
#             logger.debug(f"Beat {beat_index}: features extracted")
#         except Exception as exc:
#             logger.error(f"[Step 2] Feature extraction failed: {exc}\n{traceback.format_exc()}")
#             return result

#         # ════════════════════════════════════════════════════════
#         # STEP 3 — Store Beat
#         # ════════════════════════════════════════════════════════
#         try:
#             self.db.insert_beat(beat_data)
#         except Exception as exc:
#             logger.error(f"[Step 3] Beat storage failed: {exc}")

#         # ════════════════════════════════════════════════════════
#         # STEP 4 — Temporal Analysis + Event Detection
#         # ════════════════════════════════════════════════════════
#         detected_events: List[Dict[str, Any]] = []
#         try:
#             detected_events = analyze_temporal_window(
#                 session_id=session_id,
#                 db_connection=self.db,
#                 patient_metadata=patient_metadata,
#                 window_size=self.temporal_window_size,
#             )
#             result["events"] = detected_events
#             if detected_events:
#                 logger.info(
#                     f"Beat {beat_index}: {len(detected_events)} event(s) — "
#                     + ", ".join(e["event_type"] for e in detected_events)
#                 )
#         except Exception as exc:
#             logger.error(f"[Step 4] Temporal analysis failed: {exc}")

#         # ════════════════════════════════════════════════════════
#         # STEP 5 — Event Manager + Agent Trigger
#         # ════════════════════════════════════════════════════════
#         for event in detected_events:
#             event_rule = event.get("event_manager", {})
#             if event.get("storage_status") != "created":
#                 continue
#             if not event_rule.get("known_event", False):
#                 continue
#             if not event_rule.get("trigger_agent", False):
#                 continue
#             if self.agent is None:
#                 logger.debug(
#                     f"Agent trigger skipped (agent disabled): {event['event_type']}"
#                 )
#                 continue
#             try:
#                 logger.info(
#                     f"Triggering agent for: {event['event_type']} "
#                     f"(severity={event.get('severity')})"
#                 )
#                 response = self.agent.analyze(
#                     session_id=session_id,
#                     event_type=event["event_type"],
#                     event=event,
#                     beat_data=beat_data,
#                     patient_metadata=patient_metadata,
#                 )
#                 result["agent_responses"].append({
#                     "event_type":      event["event_type"],
#                     "severity":        event.get("severity"),
#                     "priority":        event_rule.get("priority"),
#                     "escalation_level": event_rule.get("escalation_level"),
#                     "response":        response,
#                 })
#             except Exception as exc:
#                 logger.error(
#                     f"[Step 5] Agent failed for {event['event_type']}: "
#                     f"{exc}\n{traceback.format_exc()}"
#                 )

#         return result

#     def close(self):
#         """Release database connections."""
#         try:
#             if self.db:
#                 self.db.close()
#                 logger.info("Database connections closed.")
#         except Exception as exc:
#             logger.error(f"Error closing database: {exc}")
