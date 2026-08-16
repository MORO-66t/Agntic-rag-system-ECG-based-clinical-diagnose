"""
episode_manager.py
====================
Generalized clinical-episode tracking, replacing flat duplicate-suppression
cooldown with real episode modeling.

Three primitives, chosen per event type by clinical persistence category
(NOT by raw detector window size — nearly every detector in this project
shares one 50-beat window, so window size is not a meaningful differentiator;
see the architecture review this module implements):

  RECURRENCE  — Category A (ectopy patterns) and, cautiously, Category C's
                per-window confirmations. The same underlying pattern keeps
                re-matching a sliding window as long as it persists. Opens on
                first trigger; each recurrence within `cooldown_beats` EXTENDS
                the same episode (resets the cooldown clock) instead of
                creating a new event. Closes after a full cooldown period
                with no recurrence. Duration = last_recurrence - start.

  STATE       — Category B (continuous rate state), Category C (sustained
                arrhythmia), Category E (conduction state). Represents an
                ongoing condition, not a recurring pattern. Opens when the
                detector reports the condition true; extends on every
                evaluation where it remains true; closes specifically when
                an evaluation reports the condition NO LONGER true (not
                merely "absent for N beats" — an explicit false reading ends
                it immediately). This is the key semantic difference from
                RECURRENCE.

  CLUSTER     — Category D (pauses, escape beats). Deliberately NOT modeled
                as a duration episode — collapsing multiple genuinely
                distinct pauses into one "long episode" would hide clinically
                important information (three separate 3s pauses is a
                different picture from one long pause). Occurrences within a
                tight beat-window are grouped into a cluster list; anything
                further apart is always a new, independent event.

Built on top of the ECGDatabase methods that already exist for this purpose
(get_active_event / update_active_event / close_event / insert_rhythm_event)
rather than a new in-memory tracker — this is what makes episode state
survive process restarts and work across multiple pipeline workers, which a
bigger version of the old `_last_triggered_tracker` dict never would.

Cooldown/recurrence decisions are made in BEAT-INDEX space (matching the
project's existing, already-correct HR-adaptive cooldown mechanism), while
episode duration is reported in both beats and wall-clock seconds (derived
from beat timestamps) since "AFib episode, 4 minutes" is more clinically
readable than "AFib episode, 312 beats" for anything downstream (RAG agent,
UI, chart review).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class EpisodePrimitive(str, Enum):
    RECURRENCE = "recurrence"
    STATE = "state"
    CLUSTER = "cluster"


@dataclass
class EpisodeConfig:
    primitive: EpisodePrimitive

    # RECURRENCE: how many beats of silence before the episode closes.
    cooldown_beats: int = 50

    # RECURRENCE / STATE: chaptering cap. None = never auto-chapter (used
    # only for the small set of always-critical conditions where
    # fragmenting a continuous record for reporting tidiness is the wrong
    # tradeoff — VF, third-degree block).
    max_episode_beats: Optional[int] = None

    # CLUSTER: max beat-gap between occurrences to still count as the same
    # cluster. Deliberately much shorter than a RECURRENCE cooldown, since
    # clustering is meant to catch only tight, obviously-related runs (e.g.
    # 3 pauses in 40 seconds from one underlying AV block episode) — not to
    # merge pauses that happen to occur in the same monitoring session.
    cluster_window_beats: int = 30


# Cooldown-beats and chaptering caps below are clinical-persistence-derived,
# not copied from any per-detector window size (see architecture review —
# nearly every detector shares the same 50-beat source window, so window
# size is not a meaningful basis for these numbers). Assume ~60-100bpm for
# the beats-to-minutes intuition in the comments; actual duration is always
# computed from real beat timestamps, never assumed from a fixed rate.
EPISODE_CONFIGS: Dict[str, EpisodeConfig] = {

    # ── Category A: recurring ectopy patterns ──────────────────────────
    # Cooldown = cycle_size * min_cycles + 1 — just enough to let the
    # pattern fully exit the detector's history window before closing,
    # preventing fragmentation while still closing promptly when the
    # pattern truly stops.
    # BIGEMINY: pattern "02" (2 beats), min_cycles=2 → 2*2+1=5
    "BIGEMINY":            EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=5, max_episode_beats=14400),
    # TRIGEMINY: pattern "002" (3 beats), min_cycles=2 → 3*2+1=7
    "TRIGEMINY":           EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=7, max_episode_beats=14400),
    # QUADRIGEMINY: pattern "0002" (4 beats), min_cycles=3 → 4*3+1=13
    "QUADRIGEMINY":        EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=13, max_episode_beats=14400),
    # ATRIAL_BIGEMINY: pattern "01" (2 beats), min_cycles=3 → 2*3+1=7
    "ATRIAL_BIGEMINY":     EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=7, max_episode_beats=14400),
    # ATRIAL_TRIGEMINY: pattern "001" (3 beats), min_cycles=2 → 3*2+1=7
    "ATRIAL_TRIGEMINY":    EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=7, max_episode_beats=14400),
    # ATRIAL_QUADRIGEMINY: pattern "0001" (4 beats), min_cycles=3 → 4*3+1=13
    "ATRIAL_QUADRIGEMINY": EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=13, max_episode_beats=14400),
    # Couplets/triplets are shorter, punchier findings than sustained
    # bigeminy/trigeminy. COUPLET: run of 2 V beats, min_count=1 → 2*1+1=3
    "COUPLET":             EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=3, max_episode_beats=7200),
    # ATRIAL_COUPLET: run of 2 S beats, min_count=1 → 2*1+1=3
    "ATRIAL_COUPLET":      EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=3, max_episode_beats=7200),
    # ATRIAL_TRIPLET: run of 3 S beats, min_count=1 → 3*1+1=4
    "ATRIAL_TRIPLET":      EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=4, max_episode_beats=7200),
    "HIGH_PVC_BURDEN":     EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=50, max_episode_beats=14400),

    # ── Category B: continuous rate state ───────────────────────────────
    # RECURRENCE with short cooldown because rate fluctuates around the
    # 60/100 bpm thresholds. STATE would fragment into many tiny episodes
    # when HR briefly crosses the boundary and back.
    "BRADYCARDIA":         EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=10, max_episode_beats=7200),
    "TACHYCARDIA":         EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=10, max_episode_beats=7200),
    # Emergency-tier rate states get a shorter chaptering cap — more
    # frequent re-summarization is appropriate when the underlying state is
    # already critical.
    "EXTREME_BRADYCARDIA": EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=1800),
    "EXTREME_TACHYCARDIA": EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=1800),

    # ── Category C: sustained arrhythmia ────────────────────────────────
    # AFIB and AFLUTTER use RECURRENCE with cooldown=16 so that brief score
    # drops (e.g. 3-second gaps) don't fragment a continuous episode into
    # many tiny records. 16 beats ≈ 14 seconds at 60 bpm.
    "AFIB_DETECTED":      EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=16, max_episode_beats=7200),
    "AFLUTTER_SUSPECTED": EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=16, max_episode_beats=7200),
    "VT_RUN":             EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=1800),
    # VF is always critical and continuously ongoing evidence should never
    # be split up merely for reporting neatness.
    "DISEASE_VENTRICULAR_FIBRILLATION": EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=None),

    # ── Category D: one-shot/point events — clustered, not durationed ──
    "PAUSE_DETECTED":          EpisodeConfig(EpisodePrimitive.CLUSTER, cluster_window_beats=30),
    "PROLONGED_ASYSTOLE":      EpisodeConfig(EpisodePrimitive.CLUSTER, cluster_window_beats=30),
    "JUNCTIONAL_ESCAPE_BEAT":  EpisodeConfig(EpisodePrimitive.CLUSTER, cluster_window_beats=20),
    "VENTRICULAR_ESCAPE_BEAT": EpisodeConfig(EpisodePrimitive.CLUSTER, cluster_window_beats=20),
    "ATRIAL_ESCAPE_BEAT":      EpisodeConfig(EpisodePrimitive.CLUSTER, cluster_window_beats=20),

    # ── Category E: conduction state ────────────────────────────────────
    "FIRST_DEGREE_AV_BLOCK":       EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=7200),
    "MOBITZ_I_AV_BLOCK":           EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=7200),
    "MOBITZ_II_AV_BLOCK":          EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=3600),
    "SECOND_DEGREE_AV_BLOCK_2TO1": EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=3600),
    "HIGH_GRADE_AV_BLOCK":         EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=1800),
    # Complete heart block is always critical — never auto-chapter.
    "THIRD_DEGREE_AV_BLOCK":       EpisodeConfig(EpisodePrimitive.STATE, max_episode_beats=None),
}

# Fallback for any event type not explicitly configured above (e.g. a new
# event added later without updating this table) — behaves like the old
# flat cooldown so nothing silently breaks, but is a RECURRENCE primitive
# so at minimum recurrences still merge into one episode rather than
# spamming duplicates.
_DEFAULT_CONFIG = EpisodeConfig(EpisodePrimitive.RECURRENCE, cooldown_beats=50, max_episode_beats=14400)


# Mirrors the cross-rhythm exclusion already implemented inside
# av_block_detector.detect_av_block()'s existing_events check, generalized
# so any STATE episode in this group closes automatically when a different
# member of the group opens — a competing rhythm diagnosis on the same
# beats should displace, not coexist with, the previous one.
STATE_MUTUAL_EXCLUSION_GROUP: List[str] = [
    "VT_RUN",
    "DISEASE_VENTRICULAR_FIBRILLATION", "SVT_SUSPECTED",
    "FIRST_DEGREE_AV_BLOCK", "MOBITZ_I_AV_BLOCK", "MOBITZ_II_AV_BLOCK",
    "SECOND_DEGREE_AV_BLOCK_2TO1", "HIGH_GRADE_AV_BLOCK", "THIRD_DEGREE_AV_BLOCK",
]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _update_confidence_evolution(
    existing: Optional[Dict[str, Any]],
    new_confidence: float,
    max_history: int = 20,
) -> Dict[str, Any]:
    """
    Tracks latest / peak / trend rather than collapsing recurrences into one
    number. Avoids both failure modes of a single aggregate: max()
    overstates certainty from a single strong recurrence; mean()
    understates a genuinely strengthening pattern. `history` is capped so
    metadata doesn't grow unboundedly across a very long episode.
    """
    history: List[float] = list(existing.get("history", [])) if existing else []
    history.append(round(float(new_confidence), 4))
    history = history[-max_history:]

    peak = max(history)
    latest = history[-1]

    trend = "stable"
    if len(history) >= 3:
        recent = history[-3:]
        if recent[-1] > recent[0] + 0.05:
            trend = "rising"
        elif recent[-1] < recent[0] - 0.05:
            trend = "falling"

    return {"history": history, "latest": latest, "peak": peak, "trend": trend}


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeManager:
    """
    Wraps ECGDatabase's existing active-event primitives
    (get_active_event / update_active_event / close_event /
    insert_rhythm_event) with clinical episode semantics.

    One instance can be shared across a session/pipeline; all state lives
    in the database, not in this object, so it is safe across process
    restarts and multiple workers.
    """

    def __init__(self, db):
        self.db = db

    # ── Public entry point ──────────────────────────────────────────────
    def process_event(
        self,
        session_id: str,
        event_type: str,
        beat_index: int,
        timestamp: float,
        confidence: float,
        severity: str,
        metadata_json: Optional[Dict[str, Any]] = None,
        condition_true: bool = True,
    ) -> Dict[str, Any]:
        """
        Route to the correct primitive for this event type and return a
        result dict describing what happened:
            {
                "action": "opened" | "extended" | "closed_and_reopened"
                           | "suppressed" | "closed" | "no_op",
                "episode_id": int | None,
                "episode": {...}            # current episode snapshot
            }
        """
        config = EPISODE_CONFIGS.get(event_type, _DEFAULT_CONFIG)
        metadata_json = dict(metadata_json or {})

        if config.primitive == EpisodePrimitive.RECURRENCE:
            return self._process_recurrence(
                session_id, event_type, beat_index, timestamp,
                confidence, severity, metadata_json, config,
            )
        elif config.primitive == EpisodePrimitive.STATE:
            return self._process_state(
                session_id, event_type, beat_index, timestamp,
                confidence, severity, metadata_json, config, condition_true,
            )
        else:  # CLUSTER
            return self._process_cluster(
                session_id, event_type, beat_index, timestamp,
                confidence, severity, metadata_json, config,
            )

    # ── RECURRENCE primitive (Category A) ───────────────────────────────
    def _process_recurrence(
        self, session_id, event_type, beat_index, timestamp,
        confidence, severity, metadata_json, config: EpisodeConfig,
    ) -> Dict[str, Any]:
        active = self.db.get_active_event(session_id, event_type)

        if active is not None:
            active_meta = _as_dict(active.get("metadata_json"))
            last_beat = active_meta.get("last_beat_index", active_meta.get("start_beat_index", beat_index))
            start_beat = active_meta.get("start_beat_index", last_beat)

            within_cooldown = (beat_index - last_beat) <= config.cooldown_beats
            within_cap = (
                config.max_episode_beats is None
                or (beat_index - start_beat) < config.max_episode_beats
            )

            if within_cooldown and within_cap:
                new_meta = dict(active_meta)
                new_meta["last_beat_index"] = beat_index
                new_meta["last_timestamp"] = timestamp
                new_meta["recurrence_count"] = active_meta.get("recurrence_count", 1) + 1
                new_meta["duration_beats"] = beat_index - start_beat
                new_meta["duration_sec"] = timestamp - active_meta.get("start_timestamp", timestamp)
                new_meta["confidence_evolution"] = _update_confidence_evolution(
                    active_meta.get("confidence_evolution"), confidence
                )
                new_meta["latest_detector_metadata"] = metadata_json
                self.db.update_active_event(
                    event_id=active["id"],
                    new_end_time=timestamp,
                    metadata_json=new_meta,
                    severity=severity,
                )
                return {"action": "extended", "episode_id": active["id"], "episode": new_meta}

            # Cooldown lapsed, or the chaptering cap was hit while still
            # recurring -> close the old episode and open a fresh one.
            closed_episode = dict(active_meta)
            closed_episode_id = active["id"]
            self.db.close_event(active["id"], close_time=active_meta.get("last_timestamp", timestamp))
            parent_id = (
                active_meta.get("parent_episode_id") or active["id"]
                if not within_cap else None
            )
            result = self._open_recurrence_episode(
                session_id, event_type, beat_index, timestamp,
                confidence, severity, metadata_json, parent_id,
            )
            result["closed_episode"] = closed_episode
            result["closed_episode_id"] = closed_episode_id
            return result

        return self._open_recurrence_episode(
            session_id, event_type, beat_index, timestamp,
            confidence, severity, metadata_json, parent_id=None,
        )

    def _open_recurrence_episode(
        self, session_id, event_type, beat_index, timestamp,
        confidence, severity, metadata_json, parent_id,
    ) -> Dict[str, Any]:
        meta = {
            "start_beat_index": beat_index,
            "start_timestamp": timestamp,
            "last_beat_index": beat_index,
            "last_timestamp": timestamp,
            "recurrence_count": 1,
            "duration_beats": 0,
            "duration_sec": 0.0,
            "confidence_evolution": _update_confidence_evolution(None, confidence),
            "latest_detector_metadata": metadata_json,
            "parent_episode_id": parent_id,
        }
        event_id = self.db.insert_rhythm_event({
            "session_id": session_id,
            "event_type": event_type,
            "event_start_time": timestamp,
            "event_end_time": timestamp,
            "severity": severity,
            "metadata_json": meta,
        })
        action = "closed_and_reopened" if parent_id else "opened"
        return {"action": action, "episode_id": event_id, "episode": meta}

    # ── STATE primitive (Categories B, C, E) ────────────────────────────
    def _process_state(
        self, session_id, event_type, beat_index, timestamp,
        confidence, severity, metadata_json, config: EpisodeConfig,
        condition_true: bool,
    ) -> Dict[str, Any]:
        active = self.db.get_active_event(session_id, event_type)

        if not condition_true:
            # State has ended. Close immediately -- this is the defining
            # difference from RECURRENCE: no cooldown grace period, because
            # an explicit "no longer true" reading is itself the evidence
            # the state ended, not merely evidence of silence.
            if active is not None:
                active_meta = _as_dict(active.get("metadata_json"))
                self.db.close_event(active["id"], close_time=timestamp)
                active_meta["duration_beats"] = beat_index - active_meta.get("start_beat_index", beat_index)
                active_meta["duration_sec"] = timestamp - active_meta.get("start_timestamp", timestamp)
                active_meta["ended_reason"] = "condition_no_longer_true"
                return {"action": "closed", "episode_id": active["id"], "episode": active_meta}
            return {"action": "no_op", "episode_id": None, "episode": None}

        # Condition is true.
        self._close_mutually_exclusive_states(session_id, event_type, timestamp)

        if active is not None:
            active_meta = _as_dict(active.get("metadata_json"))
            start_beat = active_meta.get("start_beat_index", beat_index)
            within_cap = (
                config.max_episode_beats is None
                or (beat_index - start_beat) < config.max_episode_beats
            )

            if within_cap:
                new_meta = dict(active_meta)
                new_meta["last_beat_index"] = beat_index
                new_meta["last_timestamp"] = timestamp
                new_meta["duration_beats"] = beat_index - start_beat
                new_meta["duration_sec"] = timestamp - active_meta.get("start_timestamp", timestamp)
                new_meta["evaluations_count"] = active_meta.get("evaluations_count", 1) + 1
                new_meta["confidence_evolution"] = _update_confidence_evolution(
                    active_meta.get("confidence_evolution"), confidence
                )
                new_meta["latest_detector_metadata"] = metadata_json
                self.db.update_active_event(
                    event_id=active["id"],
                    new_end_time=timestamp,
                    metadata_json=new_meta,
                    severity=severity,
                )
                return {"action": "extended", "episode_id": active["id"], "episode": new_meta}

            # Chaptering cap hit while state is still true -> close this
            # chapter, open a linked continuation chapter immediately.
            self.db.close_event(active["id"], close_time=timestamp)
            parent_id = active_meta.get("parent_episode_id") or active["id"]
            return self._open_state_episode(
                session_id, event_type, beat_index, timestamp,
                confidence, severity, metadata_json, parent_id,
            )

        return self._open_state_episode(
            session_id, event_type, beat_index, timestamp,
            confidence, severity, metadata_json, parent_id=None,
        )

    def _open_state_episode(
        self, session_id, event_type, beat_index, timestamp,
        confidence, severity, metadata_json, parent_id,
    ) -> Dict[str, Any]:
        meta = {
            "start_beat_index": beat_index,
            "start_timestamp": timestamp,
            "last_beat_index": beat_index,
            "last_timestamp": timestamp,
            "evaluations_count": 1,
            "duration_beats": 0,
            "duration_sec": 0.0,
            "confidence_evolution": _update_confidence_evolution(None, confidence),
            "latest_detector_metadata": metadata_json,
            "parent_episode_id": parent_id,
        }
        event_id = self.db.insert_rhythm_event({
            "session_id": session_id,
            "event_type": event_type,
            "event_start_time": timestamp,
            "event_end_time": timestamp,
            "severity": severity,
            "metadata_json": meta,
        })
        action = "closed_and_reopened" if parent_id else "opened"
        return {"action": action, "episode_id": event_id, "episode": meta}

    def _close_mutually_exclusive_states(self, session_id: str, event_type: str, timestamp: float) -> None:
        """Generalization of av_block_detector's existing_events exclusion check."""
        if event_type not in STATE_MUTUAL_EXCLUSION_GROUP:
            return
        for other_type in STATE_MUTUAL_EXCLUSION_GROUP:
            if other_type == event_type:
                continue
            other_active = self.db.get_active_event(session_id, other_type)
            if other_active is not None:
                other_meta = _as_dict(other_active.get("metadata_json"))
                other_meta["ended_reason"] = f"displaced_by_{event_type}"
                self.db.close_event(other_active["id"], close_time=timestamp)

    # ── CLUSTER primitive (Category D) ──────────────────────────────────
    def _process_cluster(
        self, session_id, event_type, beat_index, timestamp,
        confidence, severity, metadata_json, config: EpisodeConfig,
    ) -> Dict[str, Any]:
        active = self.db.get_active_event(session_id, event_type)

        if active is not None:
            active_meta = _as_dict(active.get("metadata_json"))
            last_beat = active_meta.get("last_beat_index", beat_index)

            if (beat_index - last_beat) <= config.cluster_window_beats:
                occurrences = list(active_meta.get("occurrences", []))
                occurrences.append({
                    "beat_index": beat_index,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "detector_metadata": metadata_json,
                })
                new_meta = dict(active_meta)
                new_meta["occurrences"] = occurrences
                new_meta["last_beat_index"] = beat_index
                new_meta["last_timestamp"] = timestamp
                new_meta["cluster_count"] = len(occurrences)
                self.db.update_active_event(
                    event_id=active["id"],
                    new_end_time=timestamp,
                    metadata_json=new_meta,
                    severity=severity,
                )
                return {"action": "extended", "episode_id": active["id"], "episode": new_meta}

            # Gap too large -> this is a new, independent occurrence, not
            # part of the same cluster. Close the old cluster as-is.
            closed_episode = dict(active_meta)
            closed_episode_id = active["id"]
            self.db.close_event(active["id"], close_time=active_meta.get("last_timestamp", timestamp))

        meta = {
            "start_beat_index": beat_index,
            "start_timestamp": timestamp,
            "last_beat_index": beat_index,
            "last_timestamp": timestamp,
            "cluster_count": 1,
            "occurrences": [{
                "beat_index": beat_index,
                "timestamp": timestamp,
                "confidence": confidence,
                "detector_metadata": metadata_json,
            }],
        }
        event_id = self.db.insert_rhythm_event({
            "session_id": session_id,
            "event_type": event_type,
            "event_start_time": timestamp,
            "event_end_time": timestamp,
            "severity": severity,
            "metadata_json": meta,
        })
        result = {"action": "opened", "episode_id": event_id, "episode": meta}
        if closed_episode is not None:
            result["closed_episode"] = closed_episode
            result["closed_episode_id"] = closed_episode_id
        return result


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}
