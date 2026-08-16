"""
realtime_stream.py — Real-time MIT-BIH simulation against ECGPipeline
========================================================================

Replaces both test_realtime_mitbih.py and stream_csv_to_pipeline.py.

Data flow (matches the project's real architecture)
-----------------------------------------------------

    Raw MIT-BIH record (360 Hz .dat/.hea/.atr + P/T .mat annotations)
         │
         ▼
    convert_wfdb_to_csv_W.iter_record_beats()      <- the ONE place beat
         │   (called live, beat by beat,               segmentation + CNN
         │    not pre-batched into a CSV)               resampling happens
         ▼
    ECGPipeline.process_beat()
         │   CNN prediction
         │   feature_engineering.py -> neurokit_feature_extractor.py
         │   (single NeuroKit2 delineation pass, inside the pipeline)
         │   temporal_analysis -> disease_detector -> Event_Manager
         │   PDFSemanticECGAgent (pgvector RAG) when an event triggers
         ▼
    Printed trace (+ optional accuracy/event statistics)

This script does not run NeuroKit2 itself, does not resample anything
itself, and does not read from a pre-built CSV. It only walks the raw
record in chronological R-peak order, asks convert_wfdb_to_csv_W for
each beat's converted form as it "arrives", and hands it straight to
the pipeline — simulating a real-time monitor streaming beats in.

Usage
-----
    # Fast simulation (no delay), default record:
    python realtime_stream.py --record 100 --fast

    # True real-time pacing (sleeps by each beat's RR interval):
    python realtime_stream.py --record 100

    # Limit beats, write a diagnostics report, keep full event tracing:
    python realtime_stream.py --record 208 --max-beats 2000 --report out.json

    # Trace only, no pipeline (sanity-check the data source):
    python realtime_stream.py --record 100 --dry-run --fast
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from convert_wfdb_to_csv_W import iter_record_beats, TARGET_LEN as CNN_N_SAMPLES


# Custom warning handler: print a short note for the known NeuroKit2 rate warning
def _handle_neurokit_warning(message, category, filename, lineno, file=None, line=None):
    if "Too few peaks detected to compute the rate" in str(message):
        print("NeuroKit warning")
    else:
        warnings.showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _handle_neurokit_warning

try:
    from ecg_pipeline import ECGPipeline
    HAS_PIPELINE = True
except ImportError as _e:
    HAS_PIPELINE = False
    print(f"[WARN] ECGPipeline not importable ({_e}). Use --dry-run.")

# Morphology diagnostic formatter (optional dependency)
try:
    from morphology_diagnostics import MorphologyDiagnosticFormatter, print_morphology_diagnostics
    _HAS_MORPH_DIAG = True
except ImportError:
    MorphologyDiagnosticFormatter = None  # type: ignore
    print_morphology_diagnostics = None  # type: ignore
    _HAS_MORPH_DIAG = False

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

SESSION_ID = "mitbih_realtime"
LOG_EVERY_N_BEATS = 1
STOP_ON_ERROR = False

# AAMI EC57 grouping — maps MIT-BIH annotation symbols to the model's
# 5-class label space (N / S / V / F / Q), matching model_service.CLASS_MAP.
AAMI_GROUPS: Dict[str, List[str]] = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q", "?"],
}
SYMBOL_TO_AAMI: Dict[str, str] = {
    sym: group for group, syms in AAMI_GROUPS.items() for sym in syms
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_stream")


# ────────────────────────────────────────────────────────────────────────────
# STATS
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    total_beats: int = 0
    pipeline_errors: int = 0
    skipped_malformed: int = 0
    events_fired: int = 0
    agent_responses: int = 0
    event_type_counts: Dict[str, int] = field(default_factory=dict)
    event_diagnostics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConfusionStats:
    """CNN predicted_label vs expert annotation, AAMI 5-class confusion matrix."""
    matrix: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {t: {p: 0 for p in AAMI_CLASSES} for t in AAMI_CLASSES}
    )
    unmapped_symbols: Dict[str, int] = field(default_factory=dict)
    total_scored: int = 0

    def update(self, expert_symbol: str, predicted_class: str) -> None:
        true_class = SYMBOL_TO_AAMI.get(expert_symbol)
        if true_class is None:
            self.unmapped_symbols[expert_symbol] = self.unmapped_symbols.get(expert_symbol, 0) + 1
            return
        if predicted_class not in AAMI_CLASSES:
            return
        self.matrix[true_class][predicted_class] += 1
        self.total_scored += 1

    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for cls in AAMI_CLASSES:
            tp = self.matrix[cls][cls]
            fn = sum(self.matrix[cls][p] for p in AAMI_CLASSES if p != cls)
            fp = sum(self.matrix[t][cls] for t in AAMI_CLASSES if t != cls)
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall))
                else float("nan")
            )
            metrics[cls] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        return metrics

    def overall_accuracy(self) -> float:
        if self.total_scored == 0:
            return float("nan")
        correct = sum(self.matrix[c][c] for c in AAMI_CLASSES)
        return correct / self.total_scored


# ────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ────────────────────────────────────────────────────────────────────────────

def validate_beat(beat: Dict[str, Any]) -> List[str]:
    """Sanity-check a beat dict from iter_record_beats() before sending it on."""
    warnings: List[str] = []

    cnn_signal = beat.get("cnn_signal") or []
    original_samples = beat.get("original_samples") or []

    if len(cnn_signal) != CNN_N_SAMPLES:
        warnings.append(f"cnn_signal length {len(cnn_signal)} != {CNN_N_SAMPLES}")
    if not np.all(np.isfinite(cnn_signal)):
        warnings.append("cnn_signal contains NaN or Inf")
    if not original_samples:
        warnings.append("original_samples is empty")
    elif not np.all(np.isfinite(original_samples)):
        warnings.append("original_samples contains NaN or Inf")

    rr = beat.get("rr_interval", 0.0) or 0.0
    if rr < 0:
        warnings.append(f"negative RR interval: {rr:.1f} ms")
    if rr > 0 and (rr < 200 or rr > 3000):
        warnings.append(f"unusual RR interval: {rr:.1f} ms")

    return warnings


# ────────────────────────────────────────────────────────────────────────────
# LOGGING HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def feature_snapshot(features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    features = features or {}
    keys = [
        "beat_index", "timestamp", "predicted_label", "prediction_confidence",
        "heart_rate", "rr_interval", "signal_quality_score",
        "qrs_width", "qrs_voltage", "r_amplitude", "q_amplitude",
        "pr_interval_ms", "p_wave_detected", "p_wave_inverted", "p_wave_width_ms",
        "qt_interval", "qtc", "qtc_bazett", "qtc_fridericia",
        "st_deviation", "t_wave_inverted",
        "delta_wave_detected", "epsilon_wave_detected",
        "electrical_alternans_detected", "flutter_baseline_detected",
    ]
    return {key: _json_safe(features.get(key)) for key in keys if key in features}


def collect_event_diagnostics(beat: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    features = result.get("features") or {}
    agents_by_type = {item.get("event_type"): item for item in result.get("agent_responses", [])}

    for event in result.get("events", []):
        metadata = event.get("metadata_json") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw_metadata": metadata}

        agent_response = agents_by_type.get(event.get("event_type"), {}).get("response")
        retrieved_chunks: List[Any] = []
        llm_preview = None
        if isinstance(agent_response, dict):
            retrieved_chunks = agent_response.get("retrieved_chunks", [])
            llm_text = agent_response.get("llm_response")
            llm_preview = llm_text[:500] if isinstance(llm_text, str) else None

        diagnostics.append({
            "beat_index": beat.get("beat_index"),
            "timestamp": beat.get("timestamp"),
            "expert_symbol": beat.get("label"),
            "event_type": event.get("event_type"),
            "severity": event.get("severity"),
            "storage_status": event.get("storage_status"),
            "event_manager": event.get("event_manager"),
            "metadata_json": _json_safe(metadata),
            "feature_snapshot": feature_snapshot(features),
            "agent_retrieved_chunks": _json_safe(retrieved_chunks),
            "agent_llm_preview": llm_preview,
        })
    return diagnostics


def log_event_diagnostics(diagnostics: List[Dict[str, Any]]) -> None:
    for item in diagnostics:
        meta = item.get("metadata_json") or {}
        confirmation = meta.get("confirmation") if isinstance(meta, dict) else None
        features = item.get("feature_snapshot") or {}
        logger.info(
            "EVENT_DIAG beat=%s event=%s status=%s HR=%s RR=%s QRS=%s QTcF=%s PR=%s label=%s",
            item["beat_index"], item.get("event_type"), item.get("storage_status"),
            features.get("heart_rate"), features.get("rr_interval"),
            features.get("qrs_width"), features.get("qtc_fridericia"),
            features.get("pr_interval_ms"), features.get("predicted_label"),
        )
        if confirmation:
            logger.info(
                "EVENT_CONFIRM event=%s policy=%s support=%s/%s required=%s ratio=%s",
                item.get("event_type"), confirmation.get("policy"),
                confirmation.get("supporting_beats"), confirmation.get("evaluated_beats"),
                confirmation.get("required_supporting_beats"), confirmation.get("support_ratio"),
            )
        if item.get("agent_llm_preview"):
            logger.info("AGENT_PREVIEW event=%s %s", item.get("event_type"),
                         item["agent_llm_preview"].replace("\n", " | "))


def log_stream_stats(stats: StreamStats, confusion: ConfusionStats) -> None:
    logger.info("=" * 78)
    logger.info("STREAM COMPLETE")
    logger.info(f"  Total beats processed   : {stats.total_beats}")
    logger.info(f"  Skipped (malformed)     : {stats.skipped_malformed}")
    logger.info(f"  Pipeline errors         : {stats.pipeline_errors}")
    logger.info(f"  Total events fired      : {stats.events_fired}")
    logger.info(f"  Total agent responses   : {stats.agent_responses}")
    if stats.event_type_counts:
        logger.info("  Event breakdown:")
        for k, v in sorted(stats.event_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {k:<40} {v:>4}x")
    if confusion.total_scored > 0:
        logger.info("-" * 78)
        logger.info("CLASSIFICATION ACCURACY (CNN predicted_label vs expert annotation, AAMI 5-class)")
        logger.info(f"  Beats scored            : {confusion.total_scored}")
        if confusion.unmapped_symbols:
            logger.info(f"  Unmapped expert symbols : {confusion.unmapped_symbols}")
        acc = confusion.overall_accuracy()
        logger.info(f"  Overall accuracy        : {acc:.4f}")
        metrics = confusion.per_class_metrics()
        logger.info(f"  {'Class':<6}{'Support':>9}{'Precision':>12}{'Recall':>10}{'F1':>10}")
        for cls in AAMI_CLASSES:
            m = metrics[cls]
            prec = f"{m['precision']:.3f}" if not np.isnan(m['precision']) else "n/a"
            rec = f"{m['recall']:.3f}" if not np.isnan(m['recall']) else "n/a"
            f1 = f"{m['f1']:.3f}" if not np.isnan(m['f1']) else "n/a"
            logger.info(f"  {cls:<6}{m['support']:>9}{prec:>12}{rec:>10}{f1:>10}")
        logger.info("  Confusion matrix (rows=true, cols=predicted):")
        header = "        " + "".join(f"{c:>6}" for c in AAMI_CLASSES)
        logger.info(header)
        for t in AAMI_CLASSES:
            row_str = "".join(f"{confusion.matrix[t][p]:>6}" for p in AAMI_CLASSES)
            logger.info(f"  {t:<5}{row_str}")
    logger.info("=" * 78)


def log_per_record_stats(record_name: str, stats: StreamStats, confusion: ConfusionStats) -> None:
    """Log statistics for a single record in a multi-record stream."""
    logger.info("=" * 78)
    logger.info(f"RECORD COMPLETE: {record_name}")
    logger.info(f"  Total beats processed   : {stats.total_beats}")
    logger.info(f"  Skipped (malformed)     : {stats.skipped_malformed}")
    logger.info(f"  Pipeline errors         : {stats.pipeline_errors}")
    logger.info(f"  Total events fired      : {stats.events_fired}")
    logger.info(f"  Total agent responses   : {stats.agent_responses}")
    if stats.event_type_counts:
        logger.info("  Event breakdown:")
        for k, v in sorted(stats.event_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {k:<40} {v:>4}x")
    if confusion.total_scored > 0:
        logger.info("-" * 78)
        logger.info("CLASSIFICATION ACCURACY (CNN predicted_label vs expert annotation, AAMI 5-class)")
        logger.info(f"  Beats scored            : {confusion.total_scored}")
        if confusion.unmapped_symbols:
            logger.info(f"  Unmapped expert symbols : {confusion.unmapped_symbols}")
        acc = confusion.overall_accuracy()
        logger.info(f"  Overall accuracy        : {acc:.4f}")
    logger.info("=" * 78)


def aggregate_stream_stats(stats_list: List[StreamStats]) -> StreamStats:
    """Aggregate multiple StreamStats objects into one."""
    aggregated = StreamStats()
    for stats in stats_list:
        aggregated.total_beats += stats.total_beats
        aggregated.pipeline_errors += stats.pipeline_errors
        aggregated.skipped_malformed += stats.skipped_malformed
        aggregated.events_fired += stats.events_fired
        aggregated.agent_responses += stats.agent_responses
        
        # Merge event_type_counts
        for event_type, count in stats.event_type_counts.items():
            aggregated.event_type_counts[event_type] = aggregated.event_type_counts.get(event_type, 0) + count
        
        # Concatenate event_diagnostics
        aggregated.event_diagnostics.extend(stats.event_diagnostics)
    
    return aggregated


def aggregate_confusion_stats(confusion_list: List[ConfusionStats]) -> ConfusionStats:
    """Aggregate multiple ConfusionStats objects into one."""
    aggregated = ConfusionStats()
    for confusion in confusion_list:
        aggregated.total_scored += confusion.total_scored
        
        # Merge unmapped_symbols
        for symbol, count in confusion.unmapped_symbols.items():
            aggregated.unmapped_symbols[symbol] = aggregated.unmapped_symbols.get(symbol, 0) + count
        
        # Sum confusion matrices
        for true_class in AAMI_CLASSES:
            for pred_class in AAMI_CLASSES:
                aggregated.matrix[true_class][pred_class] += confusion.matrix[true_class][pred_class]
    
    return aggregated


def write_diagnostics_report(
    path: Optional[Path],
    session_id: str,
    args: argparse.Namespace,
    stats: StreamStats,
    confusion: ConfusionStats,
    records: Optional[List[str]] = None,
    per_record_summaries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if path is None:
        return
    payload = {
        "session_id": session_id,
        "records": records or ["100"],  # backward compatibility
        "summary": {
            "total_beats": stats.total_beats,
            "skipped_malformed": stats.skipped_malformed,
            "pipeline_errors": stats.pipeline_errors,
            "events_fired": stats.events_fired,
            "agent_responses": stats.agent_responses,
            "event_type_counts": stats.event_type_counts,
        },
        "classification": {
            "total_scored": confusion.total_scored,
            "overall_accuracy": confusion.overall_accuracy(),
            "unmapped_symbols": confusion.unmapped_symbols,
            "per_class_metrics": confusion.per_class_metrics(),
            "confusion_matrix": confusion.matrix,
        },
        "event_diagnostics": stats.event_diagnostics,
    }
    # If per_record_summaries provided, add them
    if per_record_summaries:
        payload["per_record_summaries"] = per_record_summaries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str), encoding="utf-8")
    logger.info(f"Diagnostics report written: {path}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN STREAMING LOOP
# ────────────────────────────────────────────────────────────────────────────

def stream_record_to_pipeline(
    record_name: str,
    pipeline: Optional[Any],
    session_id: str = SESSION_ID,
    max_beats: Optional[int] = None,
    start_beat: Optional[int] = None,
    end_beat: Optional[int] = None,
    realtime: bool = False,
    dry_run: bool = False,
    diagnose_events: bool = True,
    debug_morphology: bool = False,
) -> tuple[StreamStats, ConfusionStats]:
    """
    Walk a raw MIT-BIH record beat by beat via iter_record_beats(),
    streaming each converted beat straight into the pipeline as it
    "arrives". When realtime=True, sleeps by each beat's own RR
    interval to simulate true real-time pacing.

    Parameters
    ----------
    start_beat : int, optional
        0-based beat index to start from (skips all earlier beats).
        Also accepted via ``--start-beat`` / ``--start-time`` on the CLI.
    end_beat : int, optional
        0-based beat index to stop after (inclusive). Also accepted
        via ``--end-beat`` / ``--end-time``.
    """
    stats = StreamStats()
    confusion = ConfusionStats()
    class_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

    logger.info(f"Streaming record {record_name} (realtime={realtime}, dry_run={dry_run}) …")
    logger.info(f"  Record-internal clock starts at t=0.00s (original record length ~30 min = 1800s)")
    if start_beat is not None:
        logger.info(f"  Start beat : {start_beat}")
    if end_beat is not None:
        logger.info(f"  End beat   : {end_beat}")

    # ── Compute a real-time baseline so rhythm_events timestamps
    #    are real epoch times, not recording-relative floats that
    #    datetime.fromtimestamp() would misinterpret as 1970-01-01.
    #    Works correctly in both --realtime and --fast modes.
    record_start_epoch = time.time()
    logger.info(f"  Real-time baseline epoch: {record_start_epoch:.3f}")

    logger.info("-" * 78)

    last_rel_ts = 0.0
    for idx, beat in enumerate(iter_record_beats(record_name)):
        # Convert recording-relative timestamp to real epoch time
        # before handing it to the pipeline. This makes all downstream
        # datetime.fromtimestamp() calls produce correct wall-clock times.
        rel_ts = beat["timestamp"]
        beat["timestamp"] = record_start_epoch + rel_ts
        # Skip beats before start_beat
        if start_beat is not None and idx < start_beat:
            continue
        # Stop after end_beat (inclusive — process it, then stop)
        if max_beats is not None and idx >= max_beats:
            break
        if end_beat is not None and idx > end_beat:
            break

        warns = validate_beat(beat)
        last_rel_ts = rel_ts
        if warns:
            for w in warns:
                logger.warning(f"Beat {beat['beat_index']} validation: {w}")
            if any("cnn_signal length" in w or "original_samples is empty" in w for w in warns):
                stats.skipped_malformed += 1
                continue

        if dry_run or pipeline is None:
            stats.total_beats += 1
            if idx % LOG_EVERY_N_BEATS == 0:
                m, s = divmod(int(rel_ts), 60)
                time_str = f"   {m:02d}:{s:02d}"
                logger.info(
                    f"[DRY-RUN] Beat {beat['beat_index']:>5} | t={rel_ts:>7.2f}s ({time_str}) | "
                    f"expert={beat['label']} | RR={beat['rr_interval']:>6.1f}ms | "
                    f"CNN_len={len(beat['cnn_signal'])} | orig_len={len(beat['original_samples'])}"
                )
            if realtime and beat["rr_interval"] > 0:
                time.sleep(beat["rr_interval"] / 1000.0)
            continue

        try:
            result = pipeline.process_beat(
                signal=beat["cnn_signal"],
                session_id=session_id,
                timestamp=beat["timestamp"],
                rr_interval=beat["rr_interval"],
                original_samples=beat["original_samples"],
                t_peak_position_360=beat["t_peak_position_360"],
                p_peak_position_360=beat["p_peak_position_360"],
                rt_interval_ms=beat["rt_interval_ms"],
                pr_interval_ms=beat["pr_interval_ms"],
                context_samples=beat.get("context_samples"),
                context_rpeaks=beat.get("context_rpeaks"),
                context_target_r=beat.get("context_target_r"),
                beat_start=beat.get("beat_start", 0),
                context_info=beat.get("context_info"),
            )

            stats.total_beats += 1

            pred = result.get("beat_prediction") or {}
            predicted_class = pred.get("predicted_class") or class_map.get(pred.get("predicted_label"))
            if predicted_class:
                confusion.update(beat["label"], predicted_class)

            events = result.get("events", [])
            for ev in events:
                et = ev.get("event_type", "unknown")
                stats.event_type_counts[et] = stats.event_type_counts.get(et, 0) + 1
            stats.events_fired += len(events)
            stats.agent_responses += len(result.get("agent_responses", []))

            if events:
                diagnostics = collect_event_diagnostics(beat, result)
                stats.event_diagnostics.extend(diagnostics)
                if diagnose_events:
                    log_event_diagnostics(diagnostics)

            # ── Morphology diagnostics (when --debug-morphology is enabled) ──
            if debug_morphology and _HAS_MORPH_DIAG and print_morphology_diagnostics is not None:
                morph_diag = result.get("features", {}).get("morphology_diagnostics")
                if morph_diag is not None:
                    # Faithfully reconstruct the collector from its own
                    # serialization using from_dict(), which preserves
                    # method-level state (context window availability,
                    # actual failure reasons, etc.) — unlike the old
                    # manual reconstruction that had ordering bugs.
                    from morphology_diagnostics import MorphologyDiagnosticCollector
                    collector = MorphologyDiagnosticCollector.from_dict(morph_diag)
                    print_morphology_diagnostics(collector)

            # Always print a one-line trace per beat at the configured
            # cadence — this is the "just print results to trace" mode.
            if idx % LOG_EVERY_N_BEATS == 0:
                m, s = divmod(int(rel_ts), 60)
                logger.info(
                    f"Beat {beat['beat_index']:>5} | t={rel_ts:>7.2f}s ({m:02d}:{s:02d} min) | "
                    f"expert={beat['label']} | pred={predicted_class} "
                    f"({pred.get('prediction_confidence', 0.0):.2f}) | "
                    f"RR={beat['rr_interval']:>6.1f}ms | events={len(events)}"
                )

        except Exception as exc:
            stats.pipeline_errors += 1
            logger.error(f"Pipeline error at beat {beat['beat_index']}: {exc}\n{traceback.format_exc()}")
            if STOP_ON_ERROR:
                logger.error("STOP_ON_ERROR is True — aborting stream.")
                break

        if realtime and beat["rr_interval"] > 0:
            time.sleep(beat["rr_interval"] / 1000.0)

    lm, ls = divmod(int(last_rel_ts), 60)
    logger.info(f"Record {record_name} finished — internal duration: {last_rel_ts:.2f}s ({lm:02d}:{ls:02d} min)")

    return stats, confusion


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time MIT-BIH simulation against ECGPipeline.")
    parser.add_argument("--record", action="append", default=None, help="MIT-BIH record name (e.g. 100, 208). Can be specified multiple times.")
    parser.add_argument("--max-beats", type=int, default=None, help="Stop after this many beats.")
    parser.add_argument("--start-beat", type=int, default=None, help="0-based beat index to start from (skip earlier beats).")
    parser.add_argument("--end-beat", type=int, default=None, help="0-based beat index to stop after (inclusive).")
    parser.add_argument("--start-time", type=float, default=None, help="Start from this timestamp in seconds (approximate beat index).")
    parser.add_argument("--end-time", type=float, default=None, help="Stop after this timestamp in seconds (approximate beat index).")
    parser.add_argument("--fast", action="store_true", help="No inter-beat delay (default behaviour).")
    parser.add_argument("--realtime", action="store_true", help="Sleep by each beat's RR interval.")
    parser.add_argument("--dry-run", action="store_true", help="Skip pipeline calls; just trace the data source.")
    parser.add_argument("--session-id", default=SESSION_ID)
    parser.add_argument("--model-path", default="ecg_cnn_model.keras")
    parser.add_argument("--agent-llm-mode", default="groq", choices=["groq", "inference", "test"])
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--no-agent", action="store_true", help="Disable the RAG agent entirely.")
    parser.add_argument("--report", type=str, default=None, help="Write a JSON diagnostics report to this path.")
    parser.add_argument("--quiet-events", action="store_true", help="Suppress per-event diagnostic logging.")
    parser.add_argument("--turbo", action="store_true", help="Fast testing mode: disables agent, event logging, and uses minimal thresholds.")
    parser.add_argument("--debug-afib", action="store_true", help="Print AFib diagnostic dashboard to terminal for every analyzed window.")
    parser.add_argument("--debug-afl", action="store_true",
                        help="Print Atrial Flutter diagnostic dashboard to terminal for every analyzed window.")
    parser.add_argument("--debug-morphology", action="store_true",
                        help="Print detailed morphology diagnostics for every beat, explaining how each feature was computed.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write all log output to this text file in addition to the terminal.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Add file logging if --log-file is specified ──────────────
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
        # Redirect print() to the logger so [AFL context] and NeuroKit
        # warnings also go to the file, not just the terminal.
        import builtins
        _original_print = builtins.print
        def _print_to_log(*pargs, **pkwargs):
            _original_print(*pargs, **pkwargs)  # still print to terminal
            msg = " ".join(str(a) for a in pargs)
            if msg.strip():
                logger.info(msg)
        builtins.print = _print_to_log
        logger.info(f"Log file: {args.log_file}")

    # Apply turbo mode optimizations
    if args.turbo:
        args.fast = True
        args.no_agent = True
        args.quiet_events = True
    
    realtime = args.realtime and not args.fast
    
    # Determine records to process (backward compatibility: default to ["100"])
    records = args.record if args.record else ["100"]
    logger.info(f"Streaming {len(records)} record(s): {', '.join(records)}")

    # Generate a unique session_id per run so each run gets its own
    # fresh set of events without cooldown conflicts from previous runs.
    session_id = f"mitbih_realtime-{uuid.uuid4().hex[:8]}"
    logger.info(f"Session ID: {session_id}")

    pipeline = None
    if not args.dry_run:
        if not HAS_PIPELINE:
            logger.error("ECGPipeline not importable — use --dry-run, or fix imports.")
            return 1
        pipeline = ECGPipeline(
            model_path=args.model_path,
            enable_agent=not args.no_agent,
            agent_llm_mode=args.agent_llm_mode,
            agent_local_model_path=args.local_model_path,
            debug_afib=args.debug_afib,
            debug_afl=args.debug_afl,
            enable_morphology_debug=args.debug_morphology,
        )

    all_stats: List[StreamStats] = []
    all_confusions: List[ConfusionStats] = []
    per_record_summaries: List[Dict[str, Any]] = []

    # Resolve start/end beat from time args (approximate — uses avg RR ~800ms)
    start_beat = args.start_beat
    end_beat = args.end_beat
    if args.start_time is not None and start_beat is None:
        start_beat = int(args.start_time / 0.8)  # approximate beat index from seconds
        logger.info(f"  --start-time {args.start_time}s → approx start_beat={start_beat}")
    if args.end_time is not None and end_beat is None:
        end_beat = int(args.end_time / 0.8)
        logger.info(f"  --end-time {args.end_time}s → approx end_beat={end_beat}")

    try:
        for record_name in records:
            stats, confusion = stream_record_to_pipeline(
                record_name=record_name,
                pipeline=pipeline,
                session_id=session_id,
                max_beats=args.max_beats,
                start_beat=start_beat,
                end_beat=end_beat,
                realtime=realtime,
                dry_run=args.dry_run,
                diagnose_events=not args.quiet_events,
                debug_morphology=args.debug_morphology,
            )
            all_stats.append(stats)
            all_confusions.append(confusion)
            
            # Log per-record summary
            log_per_record_stats(record_name, stats, confusion)
            
            # Build per-record summary for report
            per_record_summaries.append({
                "record": record_name,
                "summary": {
                    "total_beats": stats.total_beats,
                    "skipped_malformed": stats.skipped_malformed,
                    "pipeline_errors": stats.pipeline_errors,
                    "events_fired": stats.events_fired,
                    "agent_responses": stats.agent_responses,
                    "event_type_counts": stats.event_type_counts,
                },
                "classification": {
                    "total_scored": confusion.total_scored,
                    "overall_accuracy": confusion.overall_accuracy(),
                    "unmapped_symbols": confusion.unmapped_symbols,
                    "per_class_metrics": confusion.per_class_metrics(),
                    "confusion_matrix": confusion.matrix,
                },
            })
    finally:
        if pipeline is not None:
            pipeline.close()

    # Aggregate stats across all records
    agg_stats = aggregate_stream_stats(all_stats)
    agg_confusion = aggregate_confusion_stats(all_confusions)

    # Log aggregated summary
    log_stream_stats(agg_stats, agg_confusion)

    # Write report if requested
    if args.report:
        write_diagnostics_report(
            Path(args.report),
            args.session_id,
            args,
            agg_stats,
            agg_confusion,
            records=records,
            per_record_summaries=per_record_summaries,
        )

    return 0 if agg_stats.pipeline_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
