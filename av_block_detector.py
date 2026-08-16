"""
av_block_detector.py
=====================
AV Block detection using P-wave ratio analysis.

Uses the `all_p_waves` data stored in `raw_feature_json` by
feature_engineering.py (extracted via extract_all_p_waves() in
neurokit_feature_extractor.py). This gives us ALL detected P-waves
in the inter-beat window, not just the one nearest the current R-peak.

Key diagnostic metric: p_qrs_ratio = total P-waves / total QRS complexes
  - ~1.0  → 1:1 conduction (normal or 1st degree based on PR interval)
  - ~2.0  → 2:1 block
  - >2.5  → high-grade block
  - >1.0 with independent atrial/ventricular rates → 3rd degree

Clinical sources:
  AHA/ACC 2018 ECG Guidelines
  European Society of Cardiology 2021 Cardiac Pacing Guidelines
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PR_FIRST_DEGREE_MS: float = 200.0
MIN_BEATS_FOR_AV_ANALYSIS: int = 6
MIN_P_WAVES_FOR_ANALYSIS: int = 3
PR_VARIANCE_THRESHOLD_MS: float = 20.0


def _gather_p_events(beats_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect all P-wave events from the raw_feature_json of each beat.
    Returns a flat list of P-wave dicts with {timestamp, amplitude}.
    """
    p_events = []
    for beat in beats_history:
        raw = beat.get("raw_feature_json") or {}
        if isinstance(raw, str):
            try:
                import json
                raw = json.loads(raw)
            except Exception:
                continue
        if not isinstance(raw, dict):
            continue
        waves = raw.get("all_p_waves")
        if not waves or not isinstance(waves, list):
            continue
        for pw in waves:
            if isinstance(pw, dict) and pw.get("timestamp") is not None:
                p_events.append(pw)
    return p_events


def _associate_p_to_qrs(
    p_events: List[Dict[str, Any]],
    r_peak_times: List[float],
    min_pr_ms: float = 80.0,
    max_pr_ms: float = 400.0,
) -> tuple:
    """
    Greedy nearest-preceding-P matching.
    Returns (conducted[(p_t, r_t, pr_ms)], non_conducted_p[t], unassociated_r[t]).
    """
    p_sorted = sorted(p_events, key=lambda e: e["timestamp"])
    used_p = set()
    conducted, unassociated_r = [], []

    for r_time in r_peak_times:
        best_i, best_pr = None, None
        for i, p in enumerate(p_sorted):
            if i in used_p:
                continue
            pr_ms = (r_time - p["timestamp"]) * 1000.0
            if min_pr_ms <= pr_ms <= max_pr_ms:
                if best_pr is None or pr_ms < best_pr:
                    best_i, best_pr = i, pr_ms
        if best_i is not None:
            conducted.append((p_sorted[best_i]["timestamp"], r_time, best_pr))
            used_p.add(best_i)
        else:
            unassociated_r.append(r_time)

    non_conducted_p = [p["timestamp"] for i, p in enumerate(p_sorted) if i not in used_p]
    return conducted, non_conducted_p, unassociated_r


def detect_av_block(
    beats_history: List[Dict[str, Any]],
    existing_events: Optional[List[Dict[str, Any]]] = None,
    min_quality: float = 0.50,
    min_confidence: float = 0.70,
) -> Optional[Dict[str, Any]]:
    """
    AV block classifier using P-wave / QRS ratio analysis.

    Requires all_p_waves data in raw_feature_json (extracted by
    extract_all_p_waves() in feature_engineering.py).

    Returns the highest-severity AV block event detected, or None.
    """
    if not beats_history or len(beats_history) < MIN_BEATS_FOR_AV_ANALYSIS:
        return None

    # ── Skip if conflicting rhythms ──────────────────────────────
    if existing_events:
        conflicts = {"AFIB_DETECTED", "AFLUTTER_SUSPECTED", "VT_RUN",
                     "DISEASE_VENTRICULAR_FIBRILLATION", "SVT_SUSPECTED"}
        if any(e.get("event_type") in conflicts for e in existing_events):
            return None

    # ── Filter valid beats ──────────────────────────────────────
    valid = [b for b in beats_history
             if b.get("signal_quality_score", 0) >= min_quality
             and b.get("prediction_confidence", 0) >= min_confidence]
    if len(valid) < MIN_BEATS_FOR_AV_ANALYSIS:
        return None

    r_peak_times = sorted(b["timestamp"] for b in valid)
    p_events = _gather_p_events(valid)
    n_p = len(p_events)
    n_qrs = len(r_peak_times)

    if n_p < MIN_P_WAVES_FOR_ANALYSIS:
        return None  # Not enough P-wave data

    # ── Match P-waves to QRS ────────────────────────────────────
    conducted, non_conducted_p, unassociated_r = _associate_p_to_qrs(p_events, r_peak_times)
    p_qrs_ratio = n_p / n_qrs if n_qrs else float("inf")

    # ── PP and RR regularity ─────────────────────────────────────
    all_p_times = sorted(p["timestamp"] for p in p_events)
    pp_intervals = np.diff(all_p_times)
    rr_intervals = np.diff(r_peak_times)
    pp_regular = len(pp_intervals) > 2 and (np.std(pp_intervals) / max(np.mean(pp_intervals), 1e-6)) < 0.15
    rr_regular = len(rr_intervals) > 2 and (np.std(rr_intervals) / max(np.mean(rr_intervals), 1e-6)) < 0.15

    # ================================================================
    # TIER 1: 1:1 conduction — normal or 1st degree
    # ================================================================
    if len(non_conducted_p) == 0 and len(unassociated_r) == 0 and 0.9 <= p_qrs_ratio <= 1.1:
        pr_values = [pr for (_, _, pr) in conducted]
        if not pr_values:
            return None
        mean_pr = float(np.mean(pr_values))
        if mean_pr > PR_FIRST_DEGREE_MS:
            prolonged_ratio = sum(1 for pr in pr_values if pr > PR_FIRST_DEGREE_MS) / len(pr_values)
            if prolonged_ratio >= 0.80:
                return {
                    "event_type": "FIRST_DEGREE_AV_BLOCK",
                    "severity": "low",
                    "metadata_json": {
                        "mean_pr_ms": round(mean_pr, 1),
                        "prolonged_ratio": round(prolonged_ratio, 2),
                        "p_qrs_ratio": round(p_qrs_ratio, 2),
                        "analyzed_beats": len(pr_values),
                        "reason": f"First degree AV block: mean PR interval {mean_pr:.0f}ms exceeds threshold {PR_FIRST_DEGREE_MS:.0f}ms",
                        "ecg_findings": [f"PR interval {mean_pr:.0f}ms", f"PR prolonged in {prolonged_ratio*100:.0f}% of beats"],
                    },
                }
        return None

    # ================================================================
    # More P's than QRS: 2nd/3rd degree territory
    # ================================================================
    if p_qrs_ratio <= 1.15 or len(non_conducted_p) == 0:
        return None  # Not enough evidence of block

    pr_values = [pr for (_, _, pr) in conducted]

    # ── TIER 2: 3rd degree (complete AV dissociation) ─────────────
    if pp_regular and rr_regular and len(pr_values) > 2:
        atrial_rate = 60.0 / np.mean(pp_intervals) if np.mean(pp_intervals) > 0 else 0
        ventricular_rate = 60.0 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        pr_cv = float(np.std(pr_values)) / max(float(np.mean(pr_values)), 1e-6)

        if pr_cv > 0.30 and ventricular_rate < 60:
            qrs_widths = [
                float(b.get("qrs_width", 0))
                for b in valid
                if b.get("qrs_width") and b.get("qrs_width") > 0
            ]
            mean_qrs = float(np.mean(qrs_widths)) if qrs_widths else 0.0
            escape_type = "junctional" if mean_qrs < 120 else "ventricular"
            return {
                "event_type": "THIRD_DEGREE_AV_BLOCK",
                "severity": "critical",
                "metadata_json": {
                    "atrial_rate_bpm": round(atrial_rate, 1),
                    "ventricular_rate_bpm": round(ventricular_rate, 1),
                    "pr_coefficient_of_variation": round(pr_cv, 2),
                    "escape_qrs_width_ms": round(mean_qrs, 1),
                    "escape_rhythm_type": escape_type,
                    "p_qrs_ratio": round(p_qrs_ratio, 2),
                    "analyzed_beats": len(valid),
                },
            }

    # ── TIER 3: 2:1 block (indeterminate Mobitz I vs II) ────────
    if 1.8 <= p_qrs_ratio <= 2.2 and len(conducted) >= 2:
        qrs_widths = [
            float(b.get("qrs_width", 0))
            for b in valid
            if b.get("qrs_width") and b.get("qrs_width") > 0
        ]
        mean_qrs = float(np.mean(qrs_widths)) if qrs_widths else 0.0
        subtype = (
            "Mobitz_I_pattern (narrow QRS, AV-nodal-favored)"
            if (mean_qrs and mean_qrs < 120)
            else "Mobitz_II_pattern (wide QRS, infra-Hisian-favored)"
        )
        return {
            "event_type": "SECOND_DEGREE_AV_BLOCK_2TO1",
            "severity": "high",
            "metadata_json": {
                "p_qrs_ratio": round(p_qrs_ratio, 2),
                "qrs_width_ms": round(mean_qrs, 1),
                "heuristic_subtype": subtype,
                "note": "2:1 block cannot be definitively classified from rhythm alone.",
            },
        }

    # ── TIER 4: High-grade block (multiple consecutive dropped P's) ──
    if len(non_conducted_p) >= 2 and p_qrs_ratio >= 2.5:
        return {
            "event_type": "HIGH_GRADE_AV_BLOCK",
            "severity": "high",
            "metadata_json": {
                "p_qrs_ratio": round(p_qrs_ratio, 2),
                "non_conducted_p_count": len(non_conducted_p),
                "analyzed_beats": len(valid),
            },
        }

    # ── TIER 5: Mobitz I vs II (based on PR progression) ────────
    if len(pr_values) >= 3:
        pr_deltas = np.diff(pr_values)
        progressive_lengthening = np.sum(pr_deltas > 0) >= max(2, len(pr_deltas) - 1)
        # Check for pause following dropped P
        rr_vals = rr_intervals.copy() if len(rr_intervals) > 0 else []
        longest_rr = float(np.max(rr_vals)) if len(rr_vals) > 0 else 0
        shortest_rr = float(np.min(rr_vals)) if len(rr_vals) > 0 else 1
        has_pause = len(rr_vals) > 0 and longest_rr > shortest_rr * 1.5

        if progressive_lengthening and has_pause:
            return {
                "event_type": "MOBITZ_I_AV_BLOCK",
                "severity": "moderate",
                "metadata_json": {
                    "p_qrs_ratio": round(p_qrs_ratio, 2),
                    "pr_sequence_ms": [round(p, 1) for p in pr_values],
                    "pause_sec": round(longest_rr, 3),
                },
            }

        pr_std = float(np.std(pr_values))
        if pr_std < PR_VARIANCE_THRESHOLD_MS:
            return {
                "event_type": "MOBITZ_II_AV_BLOCK",
                "severity": "high",
                "metadata_json": {
                    "p_qrs_ratio": round(p_qrs_ratio, 2),
                    "pr_std_ms": round(pr_std, 1),
                    "non_conducted_p_count": len(non_conducted_p),
                },
            }

    return None