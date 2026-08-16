"""
detect_vt_vf.py
================
Window- and run-based redesign of `detect_vt()` and `detect_vf()`.

CLINICAL BASIS
--------------
Al-Khatib et al. 2017 AHA/ACC/HRS Guideline for Management of Patients
with Ventricular Arrhythmias and the Prevention of Sudden Cardiac Death.
Circulation. 2018;138:e272-e391.

Definitions used (Table 5 of the above guideline):
  VT (general):     ≥3 consecutive ventricular complexes at a rate
                     >100 bpm (cycle length <600 ms).
  Nonsustained VT:   ≥3 beats, terminating spontaneously, duration <30s,
                     without hemodynamic compromise.
  Sustained VT:      >30 seconds duration, OR requiring termination due
                     to hemodynamic compromise in <30 seconds.

This distinction is clinically load-bearing: sustained VT is always an
emergency regardless of the patient's underlying cardiac status; NSVT in
a structurally normal heart is frequently benign and does not always
warrant the same emergency pathway, though it always warrants workup.

WHY THIS REPLACES THE SINGLE-BEAT VERSION
-------------------------------------------
The previous `detect_vt()` scored a single feature snapshot (a boolean
vt_detected flag plus a single beat's HR/QRS) and treated every positive
as CRITICAL regardless of duration. This conflates a 3-beat, 2-second
run of NSVT (common, often benign, e.g. in athletes or during sleep)
with a sustained 45-second run causing hemodynamic collapse. Published
criteria explicitly separate these, and the response pathway differs:
  - Sustained VT -> IMMEDIATE (cardioversion / antiarrhythmic, per ACLS)
  - NSVT         -> workup indicated, but not an emergency call in
                    isolation, particularly with a structurally normal
                    heart and no symptoms

This version operates on a beat-run history (the sequence of consecutive
ventricular-classified beats within the current monitoring window),
mirroring the beat-history architecture already used for LQTS and ARVC.

CALLING CONTRACT
-----------------
    from disease_detection.detect_vt_vf import (
        detect_vt, detect_vf, VTWindowFeatures, VFWindowFeatures
    )
    vt_result = detect_vt(vt_window)
    vf_result = detect_vf(vf_window)

Depends on names defined in disease_detector.py:
    DetectionResult, Severity, _result, QUESTIONS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


def _import_detector_deps():
    """
    Lazy import from disease_detector to avoid circular imports.
    disease_detector imports from disease_detection.detect_vt_vf,
    and we need DetectionResult, Severity, _result, QUESTIONS from it.
    """
    from disease_detector import DetectionResult, Severity, _result, QUESTIONS as _Q
    return DetectionResult, Severity, _result, _Q


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

VT_MIN_CONSECUTIVE_BEATS: int = 3
# Source: Al-Khatib et al. 2017 AHA/ACC/HRS Guideline, Table 5.
# VT is defined as >=3 consecutive ventricular complexes. Fewer than 3
# consecutive ventricular beats is a couplet (2 beats) or an isolated PVC
# (1 beat), neither of which meets the VT definition.

VT_RATE_THRESHOLD_BPM: float = 100.0
# Source: Al-Khatib et al. 2017, Table 5 — rate >100 bpm (cycle length
# <600 ms) is part of the formal VT definition, distinguishing VT from
# slower idioventricular rhythms (20-40 bpm, a "backup" escape rhythm
# that is not tachycardia and has a different, usually benign, clinical
# significance and management).

SUSTAINED_VT_DURATION_SEC: float = 30.0
# Source: Al-Khatib et al. 2017, Table 5. Sustained VT = duration >30
#8seconds OR any duration requiring termination for hemodynamic
# compromise. This is a hard clinical threshold, not an engineering choice.

IDIOVENTRICULAR_RATE_MAX_BPM: float = 40.0
# Source: standard rhythm classification (accelerated idioventricular
# rhythm: 40-100 bpm sits between idioventricular rhythm and VT).
# A wide-complex rhythm below 40 bpm is idioventricular escape rhythm,
# not VT, and should not be scored as VT even if beats are ventricular
# in origin — it is often a protective backup pacemaker rhythm.

QRS_WIDE_THRESHOLD_MS: float = 120.0
# Source: Al-Khatib et al. 2017; standard wide-QRS definition used across
# ECG literature (StatPearls, Merck Manual). Used as a supporting feature,
# not a substitute for the ventricular-origin classification itself.

PVC_BURDEN_HIGH_THRESHOLD: int = 500
# Source: shared threshold with ARVC arrhythmia criterion (Marcus 2010 /
# Arbelo 2023 TFC) — >500 PVCs/24h is clinically significant burden
# warranting further workup (structural heart disease screening, possible
# PVC-induced cardiomyopathy) independent of any single VT episode.
# This is reported as context, not as a VT diagnostic criterion itself.


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VTWindowFeatures:
    """
    Features describing a run of consecutive ventricular-classified beats
    within the current monitoring window, assembled by the pipeline from
    the beat-classifier output and beat history.

    Fields
    ------
    window_id, patient_id, recorded_at : identity
    consecutive_vt_beats : int
        Length of the longest run of consecutive beats classified as
        ventricular-origin (by the CNN/beat classifier) within the
        current window. 0 if no such run exists.
    run_duration_sec : float
        Wall-clock duration of that run, computed from beat timestamps.
        This is the primary input for the sustained/nonsustained split.
    run_rate_bpm : Optional[float]
        Mean ventricular rate during the run. Used to distinguish VT
        (>100 bpm) from accelerated idioventricular rhythm (40-100 bpm)
        and idioventricular escape rhythm (<40 bpm).
    qrs_duration_ms : Optional[float]
        Median QRS duration during the run.
    terminated_for_compromise : bool
        True if the run was terminated (e.g., by cardioversion or the
        patient's condition prompted intervention) before reaching 30
        seconds, due to hemodynamic compromise. This independently
        satisfies the "sustained" definition regardless of duration.
        Populate from clinical/event data if available; default False
        for a purely automated monitoring pipeline with no intervention
        log — do not infer this from ECG alone.
    monomorphic : Optional[bool]
        True if beat morphology is consistent within the run (single
        QRS morphology), False if polymorphic (beat-to-beat morphology
        variation). None if not assessed. Monomorphic vs polymorphic VT
        have different differentials and immediate management
        implications (e.g., polymorphic VT with QT prolongation raises
        concern for torsades de pointes — cross-reference with the LQTS
        detector's tdp_detected flag if available).
    pvc_count_24h : int
        Total isolated-PVC-equivalent ventricular ectopic beat count
        over the past 24 hours from the beat history database, reported
        as context (PVC burden), not as a VT diagnostic criterion.
    known_diagnoses : List[str]
        Known patient conditions, checked for structural heart disease
        or channelopathy history that changes risk interpretation.
    """
    window_id: str
    patient_id: str
    recorded_at: datetime
    consecutive_vt_beats: int = 0
    run_duration_sec: float = 0.0
    run_rate_bpm: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    terminated_for_compromise: bool = False
    monomorphic: Optional[bool] = None
    pvc_count_24h: int = 0
    known_diagnoses: List[str] = field(default_factory=list)


@dataclass
class VFWindowFeatures:
    """
    Features describing the current window for VF screening.

    VF is a rhythm diagnosis (chaotic, disorganized electrical activity
    with no identifiable QRS/P/T), not a morphology diagnosis on a single
    beat, so the primary input is the rhythm classifier's VF flag plus a
    minimum sustained-beats requirement to reject transient artifact.

    Fields
    ------
    vf_flag_beats : int
        Number of consecutive beats/samples classified as VF by the
        rhythm classifier within the current window.
    signal_quality_ok : bool
        True if the recording passed lead-off / signal-quality checks
        during the flagged interval. VF-like chaotic baselines are a
        classic false positive from lead detachment or motion artifact;
        this flag exists specifically to catch that failure mode.
    """
    window_id: str
    patient_id: str
    recorded_at: datetime
    vf_flag_beats: int = 0
    signal_quality_ok: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# VT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_vt_rhythm(w: VTWindowFeatures) -> Tuple[str, List[str]]:
    """
    Classify the ventricular run into one of:
      "none", "too_short", "idioventricular", "accelerated_idioventricular",
      "nsvt", "sustained_vt"

    Returns (classification, findings).
    """
    findings: List[str] = []

    if w.consecutive_vt_beats < VT_MIN_CONSECUTIVE_BEATS:
        if w.consecutive_vt_beats == 2:
            findings.append(
                "Ventricular couplet (2 consecutive ventricular beats) — "
                "below the VT threshold of >=3 consecutive beats "
                "(Al-Khatib et al. 2017 AHA/ACC/HRS)."
            )
        elif w.consecutive_vt_beats == 1:
            findings.append(
                "Isolated ventricular ectopic beat (PVC) — not VT by definition."
            )
        return "too_short", findings

    rate = w.run_rate_bpm
    if rate is None:
        findings.append(
            f"{w.consecutive_vt_beats} consecutive ventricular beats detected, "
            f"but rate could not be determined — cannot classify as VT vs "
            f"idioventricular rhythm (rate is a required part of the VT definition)."
        )
        return "none", findings

    if rate < IDIOVENTRICULAR_RATE_MAX_BPM:
        findings.append(
            f"{w.consecutive_vt_beats} consecutive ventricular beats at "
            f"{rate:.0f} bpm — idioventricular rhythm (<{IDIOVENTRICULAR_RATE_MAX_BPM:.0f} "
            f"bpm), typically a protective escape rhythm, NOT ventricular "
            f"tachycardia. Do not treat as an arrhythmic emergency on this "
            f"basis alone."
        )
        return "idioventricular", findings

    if rate < VT_RATE_THRESHOLD_BPM:
        findings.append(
            f"{w.consecutive_vt_beats} consecutive ventricular beats at "
            f"{rate:.0f} bpm — accelerated idioventricular rhythm "
            f"({IDIOVENTRICULAR_RATE_MAX_BPM:.0f}-{VT_RATE_THRESHOLD_BPM:.0f} bpm). "
            f"Often seen with reperfusion post-MI; usually benign and "
            f"self-limiting, not scored as VT."
        )
        return "accelerated_idioventricular", findings

    # Rate >= 100 bpm and >=3 consecutive beats: meets formal VT definition
    morphology_str = ""
    if w.monomorphic is True:
        morphology_str = "monomorphic "
    elif w.monomorphic is False:
        morphology_str = (
            "POLYMORPHIC (beat-to-beat morphology varies — raises concern for "
            "torsades de pointes if QT prolongation is present; cross-check "
            "the LQTS detector's tdp_detected/QTc findings) "
        )

    if w.run_duration_sec > SUSTAINED_VT_DURATION_SEC or w.terminated_for_compromise:
        reason_suffix = (
            f"duration {w.run_duration_sec:.0f}s > {SUSTAINED_VT_DURATION_SEC:.0f}s"
            if w.run_duration_sec > SUSTAINED_VT_DURATION_SEC
            else "terminated due to hemodynamic compromise before 30s"
        )
        findings.append(
            f"SUSTAINED {morphology_str}VT: {w.consecutive_vt_beats} consecutive "
            f"ventricular beats at {rate:.0f} bpm, {reason_suffix} "
            f"(Al-Khatib et al. 2017 AHA/ACC/HRS: sustained VT definition)."
        )
        return "sustained_vt", findings

    findings.append(
        f"Nonsustained VT (NSVT): {w.consecutive_vt_beats} consecutive "
        f"{morphology_str}ventricular beats at {rate:.0f} bpm, duration "
        f"{w.run_duration_sec:.0f}s (<{SUSTAINED_VT_DURATION_SEC:.0f}s, "
        f"terminated spontaneously)."
    )
    return "nsvt", findings


def detect_vt(window: VTWindowFeatures) -> "DetectionResult":
    """
    Beat-run-based VT screen distinguishing sustained VT (emergency) from
    nonsustained VT (workup indicated, not necessarily an emergency call).

    Tiers
    -----
    SUSTAINED VT   -> triggered=True, CRITICAL, confidence 0.90
                      Fires regardless of monomorphic/polymorphic status;
                      polymorphic sustained VT is flagged as higher acuity
                      in the reason text (risk of degeneration to VF).

    NSVT           -> triggered=True, MODERATE, confidence 0.55
                      Always warrants cardiology workup (structural heart
                      disease screen, Holter correlation, echo) but is not
                      treated as an emergency call by itself, per the
                      guideline's distinct handling of NSVT vs sustained VT.

    ACCELERATED IDIOVENTRICULAR / IDIOVENTRICULAR
                   -> triggered=False, LOW/INFO
                      Explicitly NOT scored as VT; usually benign escape
                      rhythms. Findings note this to prevent downstream
                      over-alerting.

    COUPLET / ISOLATED PVC / NONE
                   -> triggered=False, INFO
                      Does not meet the >=3-beat VT definition. PVC 24h
                      burden is still reported as context if elevated.
    """
    DetectionResult, Severity, _result, QUESTIONS = _import_detector_deps()

    classification, findings = _classify_vt_rhythm(window)
    missing: List[str] = [
        "12-lead ECG during the episode — required for Brugada/Vereckei "
        "morphology algorithms (QRS >140ms, monophasic R in aVR, precordial "
        "concordance, absence of RS in all precordial leads) to confirm "
        "ventricular origin vs SVT with aberrancy.",
        "AV dissociation assessment (capture/fusion beats) — not detectable "
        "from a single lead; present in only ~20% of VT cases on surface ECG "
        "even when available.",
    ]

    if window.pvc_count_24h > PVC_BURDEN_HIGH_THRESHOLD:
        findings.append(
            f"PVC burden context: {window.pvc_count_24h:,} ventricular ectopic "
            f"beats in past 24h (>{PVC_BURDEN_HIGH_THRESHOLD} threshold) — "
            f"warrants evaluation for PVC-induced cardiomyopathy independent "
            f"of this VT episode."
        )

    present_structural = {"structural_heart_disease", "prior_mi", "cardiomyopathy"} & {
        d.lower() for d in window.known_diagnoses
    }
    if present_structural:
        findings.append(
            f"Known structural heart disease context: {', '.join(sorted(present_structural))} "
            f"— VT in this context carries substantially higher risk of degeneration "
            f"to VF than VT in a structurally normal heart."
        )

    qs, src = QUESTIONS["Ventricular Tachycardia"] if "Ventricular Tachycardia" in QUESTIONS else (
        [
            "Are you currently experiencing palpitations, dizziness, or chest pain?",
            "Have you ever fainted or nearly fainted during an episode of rapid heartbeat?",
            "Have you been diagnosed with any heart condition before?",
        ],
        "ACC/AHA/HRS 2017 VA Guideline — Patient Assessment",
    )

    if classification == "sustained_vt":
        polymorphic_note = ""
        if window.monomorphic is False:
            polymorphic_note = (
                " Polymorphic morphology raises additional concern for "
                "imminent degeneration to ventricular fibrillation — "
                "treat with the highest acuity."
            )
        return _result(
            disease="Ventricular Tachycardia (VT) — Sustained",
            triggered=True,
            confidence=0.90,
            severity=Severity.CRITICAL,
            reason=(
                f"Sustained VT: {window.consecutive_vt_beats} consecutive ventricular "
                f"beats, duration {window.run_duration_sec:.0f}s, rate "
                f"{window.run_rate_bpm:.0f} bpm. Meets the 2017 AHA/ACC/HRS "
                f"sustained VT definition (>30s or hemodynamic-compromise "
                f"termination).{polymorphic_note} IMMEDIATE evaluation required: "
                f"unstable -> synchronized cardioversion; stable -> antiarrhythmic "
                f"therapy per ACLS."
            ),
            findings=findings, missing=missing, questions=qs, source=src,
            icd10=["I47.2"], rag=True,
        )

    if classification == "nsvt":
        return _result(
            disease="Ventricular Tachycardia (VT) — Nonsustained",
            triggered=True,
            confidence=0.55,
            severity=Severity.MODERATE,
            reason=(
                f"Nonsustained VT: {window.consecutive_vt_beats} consecutive "
                f"ventricular beats, duration {window.run_duration_sec:.0f}s, "
                f"rate {window.run_rate_bpm:.0f} bpm. Terminated spontaneously "
                f"before 30s. Warrants cardiology workup (structural heart disease "
                f"screen, correlation with symptoms, Holter follow-up) but is not "
                f"by itself an emergency call — management depends heavily on "
                f"underlying structural heart disease, which this ECG-only system "
                f"cannot assess."
            ),
            findings=findings, missing=missing, questions=qs, source=src,
            icd10=["I47.2"], rag=True,
        )

    if classification in ("accelerated_idioventricular", "idioventricular"):
        return _result(
            disease="Ventricular Tachycardia (VT)",
            triggered=False,
            confidence=0.10,
            severity=Severity.LOW,
            reason=(
                "Wide-complex ventricular rhythm present but rate is below the "
                "VT threshold (100 bpm) — classified as idioventricular or "
                "accelerated idioventricular rhythm, typically a benign escape "
                "rhythm rather than a tachyarrhythmia. Not treated as VT."
            ),
            findings=findings, missing=missing, questions=[], source=src,
            icd10=[], rag=False,
        )

    return _result(
        disease="Ventricular Tachycardia (VT)",
        triggered=False,
        confidence=0.0,
        severity=Severity.INFO,
        reason=(
            "No run of >=3 consecutive ventricular beats at >=100 bpm in the "
            "current window — VT criteria not met."
        ),
        findings=findings, missing=missing, questions=[], source=src,
        icd10=[], rag=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VF DETECTION
# ─────────────────────────────────────────────────────────────────────────────

VF_MIN_CONSECUTIVE_BEATS_OR_SEC: float = 3.0
# ENGINEERING PARAMETER, not a specific published beat-count for VF.
# VF is recognized as a rhythm pattern (chaotic, no organized QRS), not
# counted in beats the way VT is. This minimum exists purely to reject
# single-sample/transient artifact spikes from being classified as VF;
# it is a signal-quality gate, not a clinical criterion. Document as such
# — unlike VT's 3-beat threshold, this number has no guideline citation.

def detect_vf(window: VFWindowFeatures) -> "DetectionResult":
    """
    VF screen: rhythm-pattern recognition with an explicit artifact/
    lead-quality gate, since chaotic-looking baseline from a detached or
    loose electrode is the classic single-lead false positive for VF.

    VF is immediately life-threatening and this function does not use a
    tiered response — any VF flag that survives the signal-quality gate
    fires CRITICAL immediately, consistent with ACLS practice of treating
    any VF-appearing rhythm as VF until proven otherwise.
    """
    DetectionResult, Severity, _result, QUESTIONS = _import_detector_deps()

    findings: List[str] = []
    missing: List[str] = [
        "Multi-lead confirmation (VF diagnosis from a single lead cannot "
        "exclude severe artifact from lead detachment or motion, though "
        "clinical correlation — pulselessness, loss of consciousness — "
        "is the definitive confirmation and should be sought immediately "
        "if this fires)."
    ]

    if window.vf_flag_beats <= 0:
        return _result(
            disease="Ventricular Fibrillation (VF)",
            triggered=False,
            confidence=0.0,
            severity=Severity.INFO,
            reason="No VF pattern detected.",
            findings=findings, missing=missing, questions=[],
            source="ACLS VF Algorithm (AHA)", icd10=["I49.01"], rag=False,
        )

    if not window.signal_quality_ok:
        findings.append(
            f"Chaotic/disorganized pattern detected over {window.vf_flag_beats} "
            f"beat-equivalents, BUT signal quality checks failed during this "
            f"interval — classic presentation of lead detachment or severe "
            f"motion artifact mimicking VF. NOT confirmed as VF; recommend "
            f"immediate signal check before treating as a cardiac emergency, "
            f"unless clinical correlation (pulselessness, unresponsiveness) "
            f"is independently available."
        )
        return _result(
            disease="Ventricular Fibrillation (VF)",
            triggered=False,
            confidence=0.15,
            severity=Severity.MODERATE,
            reason=(
                "VF-pattern flag present but signal quality failed during the "
                "flagged interval — likely artifact. Verify lead contact "
                "immediately; do not discard without checking, since a true "
                "VF event with simultaneous lead artifact cannot be ruled out "
                "by this check alone."
            ),
            findings=findings, missing=missing, questions=[],
            source="ACLS VF Algorithm (AHA)", icd10=["I49.01"], rag=True,
        )

    findings.append(
        f"Chaotic, disorganized ventricular electrical activity sustained over "
        f"{window.vf_flag_beats} beat-equivalents with signal quality confirmed "
        f"acceptable — VF pattern."
    )
    return _result(
        disease="Ventricular Fibrillation (VF)",
        triggered=True,
        confidence=0.95,
        severity=Severity.CRITICAL,
        reason=(
            "VENTRICULAR FIBRILLATION PATTERN — IMMEDIATE DEFIBRILLATION "
            "INDICATED. Confirm clinically (pulselessness, unresponsiveness) "
            "and begin ACLS protocol without delay; do not wait for further "
            "ECG confirmation."
        ),
        findings=findings, missing=missing, questions=[],
        source="ACLS VF Algorithm (AHA 2020)", icd10=["I49.01"], rag=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION NOTE
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add "Ventricular Tachycardia" to the QUESTIONS dict in disease_detector.py
#    (currently VT has no entry — the old detect_vt() passed [] directly).
#    Suggested entry, sourced from AHA/ACC VT patient assessment materials:
#
#    "Ventricular Tachycardia": ([
#        "Are you currently experiencing palpitations, dizziness, or chest pain?",
#        "Have you ever fainted or nearly fainted during an episode of rapid heartbeat?",
#        "Have you been diagnosed with any heart condition before?",
#        "Do you have a family history of sudden cardiac death or inherited heart conditions?",
#    ], "ACC/AHA/HRS 2017 VA Guideline — Patient Assessment"),
#
# 2. In disease_detector.py, replace the ECGFeatures-based detect_vt/detect_vf
#    imports with:
#        from disease_detection.detect_vt_vf import (
#            detect_vt, detect_vf, VTWindowFeatures, VFWindowFeatures
#        )
#
# 3. Update DiseaseDetector.evaluate() to dispatch by function name, following
#    the exact pattern already used for detect_long_qt and detect_arvc:
#        elif rule_fn.__name__ == "detect_vt":
#            result = rule_fn(vt_window) if vt_window is not None else <fallback>
#        elif rule_fn.__name__ == "detect_vf":
#            result = rule_fn(vf_window) if vf_window is not None else <fallback>
#
# 4. Pipeline must assemble VTWindowFeatures / VFWindowFeatures from the beat
#    classifier's run-length output (longest consecutive ventricular-beat run
#    in the window) rather than passing a single-beat vt_detected/vf_detected
#    boolean, which the old ECGFeatures schema used.
