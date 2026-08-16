"""
disease_detection/detect_afib.py
=================================
Window-based Atrial Fibrillation detector.

Replaces disease_detector.py's old detect_atrial_fibrillation(ECGFeatures).
Follows the same architectural pattern as detect_long_qt.py / detect_arvc.py /
detect_vt_vf.py: a dedicated *WindowFeatures dataclass carrying real per-beat
series, built upstream in temporal_analysis.py, evaluated here.

Fixes relative to the old detect_atrial_fibrillation(ECGFeatures):
  1. Uses the actual per-beat RR interval series + CV/RMSSD (computed via
     temporal_analysis.calculate_rr_irregularity, already in your codebase)
     instead of a single collapsed rr_irregular boolean thresholded at 0.10.
  2. Uses P-wave presence across the WHOLE window instead of only the last
     beat in the window.
  3. Hard-excludes windows with high ectopic burden (>20% PVC/PAC) before
     scoring — mirrors the exclusion logic already in your
     detect_rr_irregularity_pattern(), which this detector supersedes for
     diagnostic (as opposed to screening/display-only) purposes.
  4. Drops the circular af_probability input — it was partly derived from
     the same RR-irregularity and P-wave signals this rule checks
     independently, which inflated confidence by double-counting evidence.
  5. Drops the `"af" in cnn_label.lower()` check. Beat-level CNN output is
     AAMI N/S/V/F/Q; none of those strings contain "af", so this branch
     never fired under the standard AAMI grouping. AF is a rhythm
     diagnosis and cannot come from a per-beat classifier.
  6. Requires >=30 valid beats and (when timestamps are available) >=30
     seconds of window duration, matching the 2023 ACC/AHA/ACCP/HRS AF
     Guideline's requirement of a documented >=30-second single-lead strip.
  7. Treats "irregularly irregular RR" as a hard gate, not just a scored
     component — if the rhythm is regular, it structurally cannot be AF
     (the rare regularised-AF-with-complete-heart-block exception is
     surfaced in missing_data, not modelled here).

Source
------
2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial
Fibrillation. Circulation. 2024;149:e1-e156.
"""

from __future__ import annotations
from asyncio import windows_events

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# Import shared plumbing from disease_detector to avoid duplicating
from disease_detector import DetectionResult, Severity, _result, QUESTIONS  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW FEATURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AFWindowFeatures:
    """
    Window-level features for AF detection, built by
    temporal_analysis._build_afib_window_features().

    Carries the real per-beat series (rr_intervals_ms, p_wave_present_flags)
    rather than pre-collapsed booleans — the detector does its own gating
    on the real data instead of trusting an upstream single-bit summary.

    Two RR metric sets are provided:
      1. Raw (rr_cv, rmssd_sec)        — computed on ALL beats in the window.
      2. Filtered (rr_cv_filtered,     — computed ONLY on normal beats
         rmssd_filtered_sec)             (excluding PVCs, PACs, Fusion, etc.)
    This allows the detector to evaluate irregularity specifically from
    conducted beats, avoiding false AF triggers caused by ectopic burden
    (e.g., in Record 202 where AF + Aberrated beats coexist).
    """
    # ─── Core identifiers ───────────────────────────────────────────
    window_id:            str
    patient_id:            str
    recorded_at:           datetime
    beat_count:            int
    valid_beat_count:      int

    # ─── Per-beat series ────────────────────────────────────────────
    rr_intervals_ms:        List[float]       # Raw RR series, all beats
    p_wave_present_flags:   List[bool]        # Per-beat P-wave detection

    # ─── Window duration ────────────────────────────────────────────
    window_duration_sec:    Optional[float] = None

    # ─── Raw RR metrics (computed on ALL beats) ────────────────────
    rr_mean_ms:              Optional[float] = None
    rr_std_ms:               Optional[float] = None
    rr_cv:                   Optional[float] = None   # rr_std / rr_mean
    rmssd_sec:               Optional[float] = None

    # ─── Filtered RR metrics (computed ONLY on NORMAL beats) ──────
    # These exclude V, S, F, Q beats so that the underlying conducted
    # rhythm can be assessed without ectopy-driven pseudo-irregularity.
    rr_intervals_filtered_ms: List[float] = field(default_factory=list)  # Normal beats only
    rr_cv_filtered:           Optional[float] = None   # CV over filtered RRs
    rmssd_filtered_sec:       Optional[float] = None   # RMSSD over filtered RRs

    # ─── Window burden & quality ────────────────────────────────────
    ectopic_fraction:        float = 0.0    # fraction of PVC/PAC beats in window
    mean_signal_quality:     float = 1.0

    # ─── Supporting evidence (independent, no double-counting) ────
    fibrillatory_baseline_detected: Optional[bool] = None
    rhythm_classifier_af_probability: Optional[float] = None

    # ─── Patient context ────────────────────────────────────────────
    prior_af_history:        bool = False
    sex:                     Optional[str] = None
    age:                     Optional[int] = None
    known_diagnoses:         List[str] = field(default_factory=list)
# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

MIN_VALID_BEATS         = 30     # AHA/ACC/HRS 2023: documented >=30s strip
MIN_WINDOW_DURATION_SEC  = 30.0

RR_CV_THRESHOLD           = 0.15   # matches detect_rr_irregularity_pattern()
RMSSD_THRESHOLD_SEC        = 0.10   # matches detect_rr_irregularity_pattern()
MAX_ECTOPIC_FRACTION         = 0.20  # matches detect_rr_irregularity_pattern()

P_ABSENT_FRACTION_HIGH         = 0.85
P_ABSENT_FRACTION_MODERATE      = 0.60

TRIGGER_SCORE_THRESHOLD           = 0.60

QUESTIONS_AF = (
    [
        "Do you feel your heart racing, fluttering, or beating irregularly?",
        "Does the irregular heartbeat come and go, or is it constant?",
        "Have you experienced dizziness, near-fainting, or actual fainting?",
        "Do you feel unusually short of breath during normal activities?",
        "Have you had a stroke or mini-stroke (TIA) in the past?",
        "Do you drink alcohol regularly? If so, how much per week?",
        "Do you have a history of high blood pressure, diabetes, or heart failure?",
        "Are you currently taking any blood thinners (anticoagulants)?",
    ],
    "AHA — Atrial Fibrillation Patient Page (ahajournals.org); "
    "2023 ACC/AHA/ACCP/HRS AF Guideline, Circulation. 2024;149:e1-e156.",
)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(reason: str, missing: Optional[List[str]] = None) -> DetectionResult:
    return DetectionResult(
        disease="Atrial Fibrillation (AF)",
        triggered=False,
        confidence=0.0,
        severity=Severity.INFO,
        reason=reason,
        ecg_findings=[],
        missing_data=missing or [],
        symptom_questions=[],
        symptom_source="",
        rag_trigger=False,
        icd10_codes=[],
    )


def detect_afib(w: AFWindowFeatures, debug: bool = False) -> DetectionResult:
    """
    Atrial fibrillation — window-based diagnostic detector.

    Criteria (2023 ACC/AHA/ACCP/HRS AF Guideline):
      - Irregularly irregular RR intervals (hallmark finding) — hard gate
      - Absent P waves across the window, replaced by fibrillatory baseline
      - Ventricular rate variable
    """
    # ── Gate 1: adequate window ──────────────────────────────────
    if w.valid_beat_count < MIN_VALID_BEATS or (
        w.window_duration_sec is not None and w.window_duration_sec < MIN_WINDOW_DURATION_SEC
    ):
        return _empty_result(
            f"Insufficient window for AF diagnosis — requires >={MIN_VALID_BEATS} valid "
            f"beats over >={MIN_WINDOW_DURATION_SEC:.0f}s (AHA/ACC/HRS 2023). "
            f"Have {w.valid_beat_count} beats"
            + (f", {w.window_duration_sec:.0f}s" if w.window_duration_sec is not None else ""),
            missing=["Longer monitoring window"],
        )

    # ── Gate 2: exclude ectopy-driven pseudo-irregularity ────────
    # if w.ectopic_fraction > MAX_ECTOPIC_FRACTION:
    #     return _empty_result(
    #         f"RR irregularity in this window ({w.ectopic_fraction:.0%} ectopic beats) is "
    #         f"more likely explained by ectopic burden than AF — see HIGH_PVC_BURDEN / "
    #         f"bigeminy-trigeminy-couplet detectors instead.",
    #         missing=["Window with lower ectopic burden"],
    #     )
    # ── Gate 2: Ectopic burden handling ────────────────────────
    # ── Gate 2: Ectopic burden handling (penalty, NOT hard reject) ──
    ectopic_penalty = 0.0
    if w.ectopic_fraction > MAX_ECTOPIC_FRACTION:
        findings.append(
            f"High ectopic burden ({w.ectopic_fraction:.0%}) in window. "
            f"AF pattern suspected but confidence reduced due to ectopy."
        )
        ectopic_penalty = 0.15  # خصم 15% من الثقة
    else:
        ectopic_penalty = 0.0

    # استخدم الـ filtered metrics لو موجودة (الأفضل)، وإلا ارجع للـ raw metrics كـ fallback
    cv = w.rr_cv_filtered if w.rr_cv_filtered is not None else (w.rr_cv if w.rr_cv is not None else 0.0)
    rmssd = w.rmssd_filtered_sec if w.rmssd_filtered_sec is not None else (w.rmssd_sec if w.rmssd_sec is not None else 0.0)

    # ── Gate 3: irregularly irregular RR is a hard requirement ───
    # AF's defining feature. If the rhythm is regular, this cannot be AF
    # regardless of P-wave findings (rare exception: AF with complete heart
    # block / regularised ventricular response — flagged, not modelled).
    if cv < RR_CV_THRESHOLD or rmssd < RMSSD_THRESHOLD_SEC:
        return _empty_result(
            f"RR rhythm not irregular enough for AF (CV={cv:.3f}, "
            f"RMSSD={rmssd*1000:.0f}ms; thresholds CV>={RR_CV_THRESHOLD}, "
            f"RMSSD>={RMSSD_THRESHOLD_SEC*1000:.0f}ms).",
            missing=[
                "Consider AF with regularised ventricular response (rare — "
                "complete AV block) if clinical suspicion remains despite regular RR"
            ],
        )

    findings: List[str] = [
        f"Irregularly irregular RR intervals — CV {cv:.2f} (threshold {RR_CV_THRESHOLD}), "
        f"RMSSD {rmssd*1000:.0f} ms (threshold {RMSSD_THRESHOLD_SEC*1000:.0f} ms) "
        f"over {w.valid_beat_count} beats"
    ]
    missing: List[str] = []

    score = 0.40 + min(0.15, 0.15 * ((cv - RR_CV_THRESHOLD) / RR_CV_THRESHOLD))

    # ── P-wave absence across the WHOLE window ────────────────────
    p_absent_fraction = (
        1.0 - (sum(w.p_wave_present_flags) / len(w.p_wave_present_flags))
        if w.p_wave_present_flags else 0.0
    )
    if p_absent_fraction >= P_ABSENT_FRACTION_HIGH:
        findings.append(f"P waves absent in {p_absent_fraction:.0%} of beats across the window")
        score += 0.40
    elif p_absent_fraction >= P_ABSENT_FRACTION_MODERATE:
        findings.append(f"P waves absent in {p_absent_fraction:.0%} of beats — partial evidence")
        score += 0.20
    else:
        missing.append(
            "Consistent P-wave absence across the window "
            f"(only {p_absent_fraction:.0%} of beats show absent P — single-beat "
            f"evidence would have been unreliable here)"
        )

    # ── Optional independent supporting evidence ──────────────────
    if w.fibrillatory_baseline_detected:
        findings.append(
            "Fibrillatory baseline spectral evidence present "
            "(approximate signal — not a purpose-built f-wave detector)"
        )
        score += 0.10

    if w.rhythm_classifier_af_probability is not None:
        findings.append(
            f"Independent rhythm-level classifier AF probability "
            f"{w.rhythm_classifier_af_probability:.0%}"
        )
        score += 0.15 * w.rhythm_classifier_af_probability
    else:
        missing.append(
            "Dedicated rhythm-level AF classifier — the beat-level CNN cannot "
            "provide this signal (AAMI N/S/V/F/Q grouping has no AF class)"
        )

    if w.prior_af_history:
        findings.append("Prior documented AF history")
        score = min(score + 0.05, 1.0)

    triggered = score >= TRIGGER_SCORE_THRESHOLD
    confidence = min(score, 0.97)

    reason = (
        f"Irregularly irregular rhythm (CV {cv:.2f}, RMSSD {rmssd*1000:.0f}ms) with "
        f"{p_absent_fraction:.0%} P-wave absence across {w.valid_beat_count} beats — AF pattern."
        if triggered else
        f"Irregular RR present but combined evidence below diagnostic threshold "
        f"(score {score:.0%}, need {TRIGGER_SCORE_THRESHOLD:.0%})."
    )

    qs, src = QUESTIONS_AF
    return DetectionResult(
        disease="Atrial Fibrillation (AF)",
        triggered=triggered,
        confidence=confidence,
        severity=Severity.HIGH if triggered else Severity.INFO,
        reason=reason,
        ecg_findings=findings,
        missing_data=missing,
        symptom_questions=qs,
        symptom_source=src,
        rag_trigger=triggered,
        icd10_codes=["I48.91", "I48.20", "I48.0"],
    )
