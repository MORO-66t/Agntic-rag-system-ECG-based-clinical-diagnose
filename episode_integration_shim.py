"""
episode_integration_shim.py
=============================
Shows the minimal change needed in temporal_analysis.py to route event
storage through EpisodeManager instead of the flat `_last_triggered_tracker`
+ `_PATTERN_COOLDOWN_BEATS` mechanism.

This is intentionally NOT a full rewrite of temporal_analysis.py (that file
is large and actively evolving elsewhere in the project) — it's the
call-site replacement to drop in where `process_event_with_cooldown` is
currently invoked, plus the one-line change needed for a STATE-primitive
event (e.g. TACHYCARDIA/AFIB) to correctly report `condition_true=False`
when a window's evaluation comes back negative, which the old flat cooldown
had no way to express at all (it only ever knew "fired" or "didn't fire",
never "this state has now ended").

Integration steps
------------------
1. In temporal_analysis.py, instantiate one EpisodeManager per pipeline
   (same lifetime as the ECGDatabase connection):

       from episode_manager import EpisodeManager
       _episode_manager = EpisodeManager(db_connection)   # or pass in via DI

2. Replace calls to `process_event_with_cooldown(...)` with
   `route_event_through_episode_manager(...)` below.

3. For RECURRENCE and CLUSTER events, `condition_true` can stay at its
   default (True) — those primitives only ever fire when the pattern IS
   present; there is no "explicitly absent" reading to report.

4. For STATE events (BRADYCARDIA/TACHYCARDIA/EXTREME_*/AFIB_DETECTED/
   AFLUTTER_SUSPECTED/VT_RUN/DISEASE_VENTRICULAR_FIBRILLATION/AV-block
   subtypes), the calling detector needs to report BOTH cases:
     - condition_true=True  when the window confirms the state
     - condition_true=False when the window explicitly does NOT confirm it
   This is what lets a StateEpisode close promptly instead of only via a
   cooldown timeout that was never designed for continuous conditions.
   Concretely: call this function once per detector evaluation, not only
   on the beats where `triggered=True` — a "not triggered" DetectionResult
   for one of the STATE event types is itself meaningful information
   (condition_true=False) that the old code discarded entirely.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from episode_manager import EpisodeManager


def route_event_through_episode_manager(
    episode_manager: EpisodeManager,
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
    Drop-in replacement for the old process_event_with_cooldown() call site.

    Returns the same kind of result dict analyze_temporal_window() already
    expects to attach to an event under "storage_status" -- mapped from
    EpisodeManager's action vocabulary so existing downstream code (the
    ECGPipeline Step 5 gate checking `storage_status == "created"`) keeps
    working with minimal change:

        opened               -> "created"       (new episode -- agent may fire)
        closed_and_reopened  -> "created"       (new chapter of a continuing
                                                   episode -- agent may fire
                                                   again for the new chapter)
        extended             -> "updated"       (same episode continuing --
                                                   agent should NOT re-fire)
        closed               -> "closed"        (state just ended -- useful
                                                   for a "resolved" notice)
        no_op                -> "skipped"
    """
    result = episode_manager.process_event(
        session_id=session_id,
        event_type=event_type,
        beat_index=beat_index,
        timestamp=timestamp,
        confidence=confidence,
        severity=severity,
        metadata_json=metadata_json,
        condition_true=condition_true,
    )

    storage_status_map = {
        "opened": "created",
        "closed_and_reopened": "created",
        "extended": "updated",
        "closed": "closed",
        "no_op": "skipped",
    }

    return {
        "storage_status": storage_status_map.get(result["action"], "skipped"),
        "episode_action": result["action"],
        "episode_id": result["episode_id"],
        "episode": result["episode"],
    }
