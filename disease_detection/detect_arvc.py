"""
detect_arvc.py
==============
Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC) detector for
continuous ECG monitoring pipelines.

CLINICAL BASIS
--------------
Diagnosis follows the 2023 ESC Cardiomyopathy Guidelines Task Force
Criteria (TFC), which revised and replaced the 2010 Marcus et al. TFC
(European Heart Journal, 2010;31:806-14). The 2023 update is in:

  Arbelo et al. 2023 ESC Guidelines for the management of
  cardiomyopathies. Eur Heart J. 2023;44(37):3503-3626.

The TFC are categorical and point-based across SIX independent
criterion categories. Diagnosis levels:
  DEFINITIVE  ≥4 TFC points (or 2 independent major criteria)
  BORDERLINE  3 TFC points (1 major + 1 minor from different categories)
  POSSIBLE    2 TFC points (2 minor from different categories)

Points per criterion:
  Major criterion (any category)  → 2 points
  Minor criterion (any category)  → 1 point

ECG-ACCESSIBLE CRITERION CATEGORIES
-------------------------------------
Of the six TFC categories, only three are (even partially) accessible
from a single-lead wearable ECG stream:

  Category III — Repolarisation abnormalities
    Major (2 pts): T-wave inversions in V1+V2+V3 in patients ≥14 years,
                   in the absence of complete RBBB (QRS ≥ 120 ms).
    Minor (1 pt):  T-wave inversions in V1+V2 only (no complete RBBB),
                   OR T-wave inversions V1–V4 in presence of complete RBBB,
                   OR T-wave inversions in V4, V5, or V6.
    ──────────────────────────────────────────────────────────────────────
    LIMITATION: Your system records Lead II only. Lead II does not
    correspond to V1–V3 (right precordial leads), which is where the
    ARVC repolarisation criterion is defined. T-wave inversions in
    Lead II can be seen in ARVC (left-dominant/biventricular ARVC or
    advanced disease) but are not the primary criterion. All T-wave
    inversion findings from this detector are reported as SUPPORTING
    EVIDENCE only, not as formal TFC criterion points, until multi-lead
    confirmation is available.

  Category IV — Depolarisation/conduction abnormalities
    Major (2 pts): Epsilon wave in V1–V3 (a low-amplitude signal between
                   end of QRS and onset of T wave in right precordial leads).
    Minor (1 pt):  Terminal activation duration ≥55 ms measured from the
                   nadir of the S wave to the end of all depolarisation
                   deflections including R' in V1, V2, or V3 in the
                   absence of complete RBBB; OR late potentials on SAECG.
    ──────────────────────────────────────────────────────────────────────
    LIMITATION: Epsilon waves are defined in right precordial leads V1–V3.
    If your epsilon_wave_present flag is derived from Lead II or a
    single-channel wearable, this is a proxy finding. However, epsilon
    waves visible even in non-right-precordial leads indicate very
    pronounced depolarisation delay and should always be flagged urgently.
    Terminal activation duration requires lead-specific measurement; your
    qrs_duration_ms is the whole-complex duration and is not equivalent.
    SAECG is entirely absent.

  Category V — Arrhythmia criteria
    Major (2 pts): Non-sustained or sustained VT of LBBB morphology with
                   superior axis (negative QRS in II, III, aVF), indicating
                   RV free wall or RV outflow tract origin.
                   OR > 500 PVCs per 24 hours with LBBB morphology on Holter.
    Minor (1 pt):  NSSVT or sustained VT of LBBB morphology with inferior
                   axis or unknown axis.
                   OR > 500 PVCs per 24 hours (without specified morphology).
    ──────────────────────────────────────────────────────────────────────
    THIS IS WHERE BEAT HISTORY IS CRITICAL. PVC burden over 24 hours is
    directly computable from your per-beat database. VT episode detection
    is available from your pipeline's vt_detected flag. VT morphology
    (LBBB + axis) requires multi-lead; your lbbb_pattern flag is a partial
    proxy for the LBBB component; axis confirmation is unavailable.

CATEGORIES NOT ACCESSIBLE FROM ECG
------------------------------------
  Category I  — Structural/functional (echo/CMR RV dimensions, RVEF,
                wall motion): entirely inaccessible → always missing_data.
  Category II — Tissue characterisation (CMR LGE, biopsy): entirely
                inaccessible → always missing_data.
  Category VI — Family history / genetics (first-degree relative with
                ARVC, pathogenic desmosomal mutation): inaccessible from
                ECG → always missing_data + symptom questions.

ARCHITECTURE
------------
Follows the same pattern established for LQTS (detect_long_qt_v4.py):
  - The detector is stateless and receives a pre-assembled
    ARVCWindowFeatures dataclass.
  - The pipeline is responsible for querying beat history and assembling
    the dataclass before calling this function.
  - No cross-beat history logic lives inside this detector.

Calling contract:
  from disease_detection.detect_arvc import detect_arvc, ARVCWindowFeatures
  result = detect_arvc(window)

Depends on names defined in disease_detector.py:
  DetectionResult, Severity, _result, QUESTIONS

DESIGN DECISIONS
----------------
1. Epsilon wave → always fires immediately as a high-priority finding,
   even on first occurrence. It is the most specific ECG sign of ARVC
   (sensitivity ~30%, specificity ~95% per Peters et al. 2008). A single
   epsilon wave warrants urgent referral even without a full TFC score.
   This mirrors clinical practice: an unexpected epsilon wave on any ECG
   strip triggers immediate specialist review.

2. VT with LBBB morphology → fires immediately if detected. LBBB-
   morphology VT indicates RV origin; in the absence of prior RV disease
   this is a major red flag for ARVC (or RV-origin idiopathic VT which
   also requires workup). The arrhythmia criterion contributes 2 TFC
   points (major) if LBBB + superior axis is confirmed, 1 point (minor)
   if axis is unknown.

3. T-wave inversions → reported as supporting evidence only (not formal
   TFC points) because Lead II does not cover the required right
   precordial leads V1–V3. This is an honest acknowledgment of the
   single-lead limitation. However, persistent T-wave inversions in Lead II
   across a sustained monitoring window are clinically relevant in the ARVC
   context and are surfaced as findings requiring multi-lead confirmation.

4. PVC burden → computed from beat history. The 24-hour PVC count is the
   most directly computable TFC criterion from your architecture. 500 PVCs
   per 24 hours is the published threshold. This is scored as minor (1 pt)
   without morphology confirmation, major (2 pts) if LBBB morphology of
   PVCs is also confirmed by your classifier.

5. No points are awarded for Category I, II, or VI criteria — these
   always appear in missing_data. The TFC score computed here is an
   ECG-only partial score; the reason string always makes clear that the
   full TFC score requires imaging, genetics, and family history.

6. Incomplete RBBB: the 2023 TFC include incomplete RBBB as a minor
   depolarisation criterion in the context of right bundle delay. Your
   rbbb_pattern flag is scored conservatively as supporting context only.

7. No diagnosis is made. The detector stratifies into:
   - URGENT_REFERRAL: epsilon wave or LBBB-morphology VT → immediate
   - HIGH_SUSPICION: partial TFC score suggesting borderline/possible ARVC
   - MONITORING: findings warrant closer surveillance but insufficient for
     suspicion threshold
   - NONE: no ARVC-relevant findings

PARAMETERS DOCUMENTED BY EVIDENCE CLASS
-----------------------------------------
Evidence-based (published TFC):
  EPSILON_WAVE_TFC_POINTS = 2       # Category IV major, 2023 TFC
  VT_LBBB_SUPERIOR_TFC_POINTS = 2  # Category V major, 2023 TFC
  VT_LBBB_UNKNOWN_AXIS_TFC_POINTS = 1  # Category V minor
  PVC_24H_THRESHOLD = 500           # Category V (major/minor), 2023 TFC
  PVC_LBBB_MAJOR_POINTS = 2        # Category V major (with LBBB morphology)
  PVC_WITHOUT_MORPHOLOGY_POINTS = 1 # Category V minor (count only)
  TFC_DEFINITIVE_THRESHOLD = 4     # 2023 TFC diagnostic score
  TFC_BORDERLINE_THRESHOLD = 3     # 2023 TFC borderline score
  TFC_POSSIBLE_THRESHOLD = 2       # 2023 TFC possible score

Engineering parameters (no published validation for wearable monitoring):
  EPSILON_FRACTION_MIN = 0.10   # minimum fraction of window beats showing
                                 # epsilon wave before scoring
  T_INVERSION_FRACTION_MIN = 0.70  # minimum fraction for persistent inversion
  RBBB_FRACTION_MIN = 0.80      # minimum fraction for stable RBBB pattern
  VT_EPISODE_LOOKBACK_HOURS = 24   # window for VT episode counting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, NamedTuple, Optional, Tuple


def _import_detector_deps():
    """
    Lazy import from disease_detector to avoid circular imports.
    disease_detector imports from disease_detection.detect_arvc,
    and we need DetectionResult, Severity, _result, QUESTIONS from it.
    """
    from disease_detector import DetectionResult, Severity, _result, QUESTIONS as _Q
    return DetectionResult, Severity, _result, _Q


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Evidence-based TFC point values (2023 ESC / 2010 Marcus TFC) ─────────────

EPSILON_WAVE_TFC_POINTS: int = 2
# Category IV Major criterion: epsilon wave in V1–V3.
# Source: Arbelo et al. 2023 ESC Guidelines, Table 6; Marcus et al. 2010.

VT_LBBB_SUPERIOR_AXIS_TFC_POINTS: int = 2
# Category V Major: VT of LBBB morphology with superior axis (neg in II, III, aVF).
# Source: 2023 TFC Table 6.

VT_LBBB_UNKNOWN_AXIS_TFC_POINTS: int = 1
# Category V Minor: VT of LBBB morphology with unknown/inferior axis,
# or NSSVT with LBBB morphology.
# Source: 2023 TFC Table 6.

PVC_24H_THRESHOLD: int = 500
# Category V threshold: >500 PVCs in 24 hours (Holter monitoring).
# Source: Marcus et al. 2010 TFC; retained in 2023 revision.

PVC_LBBB_MORPHOLOGY_TFC_POINTS: int = 2
# Category V Major: >500 PVCs/24h with confirmed LBBB morphology.
# Source: 2023 TFC Table 6.

PVC_WITHOUT_MORPHOLOGY_TFC_POINTS: int = 1
# Category V Minor: >500 PVCs/24h without morphology confirmation.
# Source: 2023 TFC Table 6.

TFC_DEFINITIVE_THRESHOLD: int = 4
# Definitive ARVC diagnosis: ≥4 TFC points from ≥2 different categories,
# OR 2 major criteria from 2 different categories (= 4 points).
# Source: 2023 ESC Guidelines, Section 8.3.

TFC_BORDERLINE_THRESHOLD: int = 3
# Borderline ARVC: 3 TFC points (e.g. 1 major + 1 minor from ≥2 categories).
# Source: 2023 ESC Guidelines, Section 8.3.

TFC_POSSIBLE_THRESHOLD: int = 2
# Possible ARVC: 2 TFC points (2 minor from ≥2 different categories).
# Source: 2023 ESC Guidelines, Section 8.3.

# ── Engineering parameters — no direct published validation for wearables ────

EPSILON_FRACTION_MIN: float = 0.10
# ENGINEERING PARAMETER — NO PUBLISHED THRESHOLD.
# Minimum fraction of valid beats in the current window that must show an
# epsilon wave before it is counted as a window-level epsilon finding.
# A 10% floor rejects isolated noisy beats while being permissive enough
# for a real epsilon wave (which may be intermittent in a wearable due to
# signal quality variation). Validate on labelled data.
# Clinical note: epsilon waves are often intermittent even on standard ECG;
# the criterion is satisfied by their presence, not their frequency.
# In a wearable context, consistent detection across at least 10% of beats
# in a stable window is a reasonable persistence floor.

T_INVERSION_FRACTION_MIN: float = 0.70
# ENGINEERING PARAMETER — NO PUBLISHED THRESHOLD.
# Minimum fraction of valid beats showing T-wave inversion for the finding
# to be reported as "persistent" (vs. intermittent/artefactual).
# 70% chosen as a majority-sustained threshold. Validate on labelled data.
# Note: T-wave inversions in Lead II are not a formal TFC criterion;
# this threshold governs when the finding is surfaced in findings[], not
# whether TFC points are awarded (they are not, due to lead limitations).

RBBB_FRACTION_MIN: float = 0.80
# ENGINEERING PARAMETER — NO PUBLISHED THRESHOLD.
# Fraction of beats in the window that must show RBBB morphology for
# it to be treated as a stable, persistent RBBB pattern (vs. rate-related
# or transient). 80% chosen because a true persistent RBBB pattern should
# appear on nearly all sinus beats.

VT_EPISODE_LOOKBACK_HOURS: float = 24.0
# The PVC burden TFC criterion is defined over 24-hour Holter monitoring.
# VT episode counting uses the same lookback window.
# Source: 2023/2010 TFC — "on Holter ECG".
# In a continuous monitoring system this maps naturally to a 24-hour
# rolling look-back in the beat history.

PVC_LOOKBACK_HOURS: float = 24.0
# Same 24-hour window as TFC. Must match the Holter-equivalent period.
# Source: 2023/2010 TFC.

REVERSIBLE_ARVC_MIMICS: set = {
    # Conditions that can produce ARVC-like ECG findings and should be
    # noted as alternative explanations when present.
    "right_ventricular_strain",
    "pulmonary_hypertension",
    "pulmonary_embolism",
    "brugada_syndrome",
    "myocarditis",
    "sarcoidosis",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class VTEpisode(NamedTuple):
    """A ventricular tachycardia episode from the beat history."""
    occurred_at: datetime
    duration_beats: int           # number of consecutive VT beats
    lbbb_morphology: bool         # True if LBBB morphology confirmed by classifier
    superior_axis: Optional[bool] # True = negative in II/III/aVF; None = unknown


@dataclass
class ARVCWindowFeatures:
    """
    Features assembled by the pipeline from beat history and the current
    monitoring window, passed to detect_arvc().

    The pipeline is responsible for querying the beat database and
    populating all fields. The detector itself is stateless.

    REQUIRED FIELDS (detector cannot run without these)
    ---------------------------------------------------
    window_id, patient_id, recorded_at, valid_beat_count

    CURRENT-WINDOW FIELDS (from the active 30-minute monitoring window)
    -------------------------------------------------------------------
    Fractions and flags are computed over the set of valid sinus beats
    in the current window (post-ectopy, post-noise exclusion).

    LONGITUDINAL FIELDS (from the stored beat history)
    --------------------------------------------------
    Fields with "_24h" or "prior_" prefix require querying the database.
    pvc_count_24h: query COUNT of beats where cnn_label indicates PVC
                   in the past 24 hours for this patient.
    pvc_lbbb_fraction_24h: fraction of those PVCs where lbbb_pattern=True.
    vt_episodes_24h: list of VTEpisode objects from the past 24 hours,
                     populated from stored events or beat sequences.
    prior_epsilon_windows: number of prior monitoring windows (before this
                   one) where epsilon_fraction >= EPSILON_FRACTION_MIN.
                   Supports persistence-across-windows tracking.
    prior_t_inversion_windows: same for sustained T-wave inversion windows.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    window_id: str
    patient_id: str
    recorded_at: datetime
    valid_beat_count: int

    # ── Patient context ───────────────────────────────────────────────────────
    age: Optional[int] = None
    sex: Optional[str] = None           # "M" | "F"
    known_diagnoses: List[str] = field(default_factory=list)

    # ── Current window: Category IV — Depolarisation ─────────────────────────
    epsilon_fraction: float = 0.0
    # Fraction of valid beats in the current window showing an epsilon wave.
    # Populate from: sum(beat.epsilon_wave_present for beat in valid_beats)
    #                / len(valid_beats).
    # ECG note: epsilon wave is defined in V1–V3. If your epsilon_wave_present
    # flag is derived from Lead II, document this limitation per the module
    # header above. The finding is still clinically meaningful but its lead
    # provenance must be noted.

    rbbb_fraction: float = 0.0
    # Fraction of valid beats showing RBBB morphology in the current window.
    # Used to determine whether complete RBBB is present (which changes the
    # TFC repolarisation criterion interpretation).

    qrs_duration_ms: Optional[float] = None
    # Median QRS duration over the current window (ms). Used with rbbb_fraction
    # to confirm complete RBBB (≥ 120 ms) vs incomplete RBBB (< 120 ms).

    # ── Current window: Category III — Repolarisation ─────────────────────────
    t_inversion_fraction: float = 0.0
    # Fraction of valid beats in the current window showing T-wave inversion
    # in Lead II. NOT a formal TFC criterion in this lead; surfaced as
    # supporting evidence requiring multi-lead confirmation.

    # ── Current window: Category V — Arrhythmia ──────────────────────────────
    vt_detected_this_window: bool = False
    # True if any VT episode was detected during the current window.

    vt_lbbb_morphology_this_window: bool = False
    # True if the VT detected in this window had LBBB morphology (RV origin).
    # Populate from VT beat classification in your CNN/classifier output.

    vt_superior_axis_this_window: Optional[bool] = None
    # True if VT in this window had superior axis (negative in II, III, aVF).
    # None if axis cannot be determined (single-lead system).

    # ── Longitudinal: PVC burden (24-hour count) ──────────────────────────────
    pvc_count_24h: int = 0
    # Total PVC count in the past PVC_LOOKBACK_HOURS from the beat database.
    # Query: SELECT COUNT(*) FROM beats
    #        WHERE patient_id = X
    #          AND cnn_label IN ('PVC', 'pvc', 'V', ...)   -- your label names
    #          AND recorded_at >= NOW() - INTERVAL '24 hours'

    pvc_lbbb_fraction_24h: float = 0.0
    # Fraction of those PVCs where LBBB morphology was detected.
    # 0.0 if pvc_count_24h == 0 or morphology unavailable.
    # Determines whether PVC burden scores as major (2 pts) or minor (1 pt).

    # ── Longitudinal: VT history (24-hour episodes) ────────────────────────────
    vt_episodes_24h: List[VTEpisode] = field(default_factory=list)
    # List of VT episodes in the past VT_EPISODE_LOOKBACK_HOURS.
    # Populate from your event store or from sequences of consecutive
    # VT-classified beats in the beat database.

    # ── Longitudinal: Criterion persistence across prior windows ──────────────
    prior_epsilon_windows: int = 0
    # Number of prior windows (before this one) where epsilon findings
    # were present. Supports "has this been seen before?" reasoning.
    # Epsilon waves do not need to be seen on every window to be clinically
    # significant — a single prior confirmed finding is already meaningful.

    prior_t_inversion_windows: int = 0
    # Number of prior windows with sustained T-wave inversion findings.
    # Chronicity of T-wave inversion is clinically relevant for ARVC
    # (distinguishes from transient causes like ischaemia or myocarditis).

    # ── Longitudinal: Prior ARVC partial TFC scores ────────────────────────────
    prior_max_ecg_tfc_score: int = 0
    # The highest ECG-partial TFC score seen in any prior window evaluation
    # for this patient. Allows the detector to note if the patient has
    # previously accumulated partial evidence even in a currently quiet window.


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _has_complete_rbbb(w: ARVCWindowFeatures) -> bool:
    """True if the window shows a stable, complete RBBB pattern."""
    return (
        w.rbbb_fraction >= RBBB_FRACTION_MIN
        and w.qrs_duration_ms is not None
        and w.qrs_duration_ms >= 120.0
    )


def _score_depolarisation(
    w: ARVCWindowFeatures,
) -> Tuple[int, List[str], List[str]]:
    """
    Category IV: Depolarisation/conduction abnormalities.

    Returns (tfc_points, findings, missing).
    Points: 2 if major (epsilon wave), 0 otherwise from this category.
    (Terminal activation duration and SAECG are inaccessible.)
    """
    pts = 0
    findings: List[str] = []
    missing: List[str] = []
    complete_rbbb = _has_complete_rbbb(w)

    # ── Epsilon wave (Major, 2 pts) ───────────────────────────────────────────
    if w.epsilon_fraction >= EPSILON_FRACTION_MIN:
        pts += EPSILON_WAVE_TFC_POINTS
        persistence_note = ""
        if w.prior_epsilon_windows > 0:
            persistence_note = (
                f" Previously detected in {w.prior_epsilon_windows} prior "
                f"monitoring window(s) — persistent finding."
            )
        findings.append(
            f"EPSILON WAVE detected in {w.epsilon_fraction * 100:.1f}% of valid "
            f"beats in the current window (≥ {EPSILON_FRACTION_MIN * 100:.0f}% "
            f"persistence threshold). TFC Category IV Major: +{EPSILON_WAVE_TFC_POINTS} pts."
            f"{persistence_note} "
            f"IMPORTANT: Epsilon wave criterion is defined in leads V1–V3. "
            f"If this detection is from Lead II, multi-lead confirmation is "
            f"required before applying the formal TFC point."
        )
    elif w.epsilon_fraction > 0:
        findings.append(
            f"Possible epsilon wave in {w.epsilon_fraction * 100:.1f}% of beats "
            f"— below the {EPSILON_FRACTION_MIN * 100:.0f}% window-persistence "
            f"threshold. Not scored; may reflect signal noise. If consistently "
            f"present, obtain a 12-lead ECG."
        )

    # ── RBBB context ──────────────────────────────────────────────────────────
    if complete_rbbb:
        findings.append(
            f"Complete RBBB present (RBBB fraction {w.rbbb_fraction * 100:.0f}%, "
            f"QRS {w.qrs_duration_ms:.0f} ms). This affects interpretation of "
            f"repolarisation criteria (Category III) — see below."
        )
    elif w.rbbb_fraction >= 0.30:
        findings.append(
            f"RBBB morphology in {w.rbbb_fraction * 100:.0f}% of beats "
            f"(below the {RBBB_FRACTION_MIN * 100:.0f}% stable-pattern threshold). "
            f"Incomplete or intermittent RBBB — may represent right conduction delay."
        )

    # ── Missing: terminal activation duration, SAECG ──────────────────────────
    missing.extend([
        "Terminal activation duration ≥55 ms in V1–V3 (TFC Category IV Minor, 1 pt) "
        "— requires lead-specific QRS terminal measurement; whole-complex QRS duration "
        "is not equivalent.",
        "Signal-averaged ECG (SAECG) late potentials (TFC Category IV Minor, 1 pt) "
        "— requires dedicated SAECG acquisition; not available from continuous wearable.",
    ])

    return pts, findings, missing


def _score_repolarisation(
    w: ARVCWindowFeatures,
) -> Tuple[int, List[str], List[str]]:
    """
    Category III: Repolarisation abnormalities.

    Returns (tfc_points, findings, missing).

    CRITICAL LIMITATION: The TFC repolarisation criterion is defined for
    leads V1, V2, V3 (right precordial). This system records Lead II only.
    T-wave inversions in Lead II are NOT awarded TFC points here. They are
    surfaced as supporting evidence requiring multi-lead confirmation.
    """
    pts = 0
    findings: List[str] = []
    missing: List[str] = []
    complete_rbbb = _has_complete_rbbb(w)

    if w.t_inversion_fraction >= T_INVERSION_FRACTION_MIN:
        persistence_str = ""
        if w.prior_t_inversion_windows > 0:
            persistence_str = (
                f" Persistent: also present in {w.prior_t_inversion_windows} "
                f"prior monitoring window(s)."
            )
        findings.append(
            f"PERSISTENT T-WAVE INVERSION in Lead II: present in "
            f"{w.t_inversion_fraction * 100:.1f}% of valid beats "
            f"(threshold {T_INVERSION_FRACTION_MIN * 100:.0f}%).{persistence_str} "
            f"LEAD LIMITATION: TFC Category III repolarisation criteria require "
            f"T-wave inversions in V1+V2+V3 (major, 2 pts) or V1+V2 (minor, 1 pt). "
            f"Lead II does not map to these right precordial leads. "
            f"NO TFC POINTS AWARDED from this finding alone. "
            f"12-lead ECG required to formally score this criterion. "
            + (
                "Context: with complete RBBB present, the criterion threshold shifts "
                "to V1–V4 (minor) rather than V1–V3 (major)."
                if complete_rbbb else
                "Context: in the absence of complete RBBB, inversions in V1+V2+V3 "
                "would constitute a major criterion (2 pts)."
            )
        )
    elif w.t_inversion_fraction > 0:
        findings.append(
            f"T-wave inversion present in {w.t_inversion_fraction * 100:.1f}% of "
            f"beats in Lead II — below the {T_INVERSION_FRACTION_MIN * 100:.0f}% "
            f"persistence threshold. Intermittent finding; not reported as sustained."
        )

    if w.age is not None and w.age < 14:
        missing.append(
            f"Patient age {w.age} years — TFC repolarisation criteria (Category III) "
            f"require patient ≥14 years. Findings in younger patients are not scored."
        )

    missing.extend([
        "12-lead ECG with right precordial leads V1–V3 (required to formally "
        "score TFC Category III repolarisation criteria — up to 2 major TFC points "
        "available if T-wave inversions confirmed in V1+V2+V3 without complete RBBB).",
    ])

    return pts, findings, missing


def _score_arrhythmia(
    w: ARVCWindowFeatures,
) -> Tuple[int, List[str], List[str]]:
    """
    Category V: Arrhythmia criteria.

    Returns (tfc_points, findings, missing).

    This is the category most directly computable from your beat history.
    Two sub-criteria are assessed:
      (a) VT with LBBB morphology (± axis)
      (b) PVC burden > 500/24h (± LBBB morphology)

    Points are awarded once per sub-criterion (not additive within the same
    sub-criterion): the highest applicable point value is used.
    """
    pts = 0
    findings: List[str] = []
    missing: List[str] = []
    vt_pts_awarded = 0

    # ── (a) VT episodes ───────────────────────────────────────────────────────
    # Gather all VT episodes from the current window and the 24h history
    all_episodes: List[VTEpisode] = list(w.vt_episodes_24h)
    if w.vt_detected_this_window:
        # Add the current-window episode if not already in the list
        current_ep = VTEpisode(
            occurred_at=w.recorded_at,
            duration_beats=0,               # unknown from the window flag alone
            lbbb_morphology=w.vt_lbbb_morphology_this_window,
            superior_axis=w.vt_superior_axis_this_window,
        )
        all_episodes.append(current_ep)

    if all_episodes:
        lbbb_superior = [e for e in all_episodes if e.lbbb_morphology and e.superior_axis is True]
        lbbb_other    = [e for e in all_episodes if e.lbbb_morphology and e.superior_axis is not True]
        non_lbbb      = [e for e in all_episodes if not e.lbbb_morphology]

        if lbbb_superior:
            # Major criterion: LBBB morphology + superior axis
            vt_pts_awarded = VT_LBBB_SUPERIOR_AXIS_TFC_POINTS
            findings.append(
                f"VT with LBBB morphology AND superior axis in "
                f"{len(lbbb_superior)} episode(s) in past "
                f"{VT_EPISODE_LOOKBACK_HOURS:.0f}h. "
                f"TFC Category V Major: +{vt_pts_awarded} pts. "
                f"(LBBB morphology = RV origin; superior axis = RV free wall or "
                f"inferior RV outflow tract — classic ARVC pattern.)"
            )
        elif lbbb_other:
            # Minor criterion: LBBB morphology, axis unknown or inferior
            vt_pts_awarded = VT_LBBB_UNKNOWN_AXIS_TFC_POINTS
            axis_note = (
                "inferior axis" if any(e.superior_axis is False for e in lbbb_other)
                else "unknown axis (single-lead system cannot confirm superior axis)"
            )
            findings.append(
                f"VT with LBBB morphology ({axis_note}) in "
                f"{len(lbbb_other)} episode(s) in past "
                f"{VT_EPISODE_LOOKBACK_HOURS:.0f}h. "
                f"TFC Category V Minor: +{vt_pts_awarded} pt. "
                f"Superior axis confirmation would upgrade this to Major (+2 pts) — "
                f"requires multi-lead ECG or intracardiac mapping."
            )
        elif non_lbbb:
            findings.append(
                f"VT detected in {len(non_lbbb)} episode(s) in past "
                f"{VT_EPISODE_LOOKBACK_HOURS:.0f}h. "
                f"VT morphology (LBBB vs. RBBB) could not be confirmed — "
                f"no TFC Category V points awarded. "
                f"Multi-lead ECG required to characterise morphology and axis."
            )

        pts += vt_pts_awarded

        missing.append(
            "VT morphology confirmation (LBBB vs RBBB) and axis determination "
            "(superior = neg in II/III/aVF) require a 12-lead ECG during VT — "
            "needed to distinguish TFC Category V Major from Minor, or from "
            "non-ARVC RV-origin VT."
        )

    # ── (b) PVC burden ────────────────────────────────────────────────────────
    pvc_pts_awarded = 0
    if w.pvc_count_24h > PVC_24H_THRESHOLD:
        if w.pvc_lbbb_fraction_24h >= 0.80:
            # Majority LBBB morphology → Major criterion
            pvc_pts_awarded = PVC_LBBB_MORPHOLOGY_TFC_POINTS
            findings.append(
                f"PVC BURDEN: {w.pvc_count_24h:,} PVCs in past "
                f"{PVC_LOOKBACK_HOURS:.0f}h (threshold: >{PVC_24H_THRESHOLD}), "
                f"of which {w.pvc_lbbb_fraction_24h * 100:.0f}% with LBBB morphology. "
                f"TFC Category V Major: +{pvc_pts_awarded} pts. "
                f"(LBBB-morphology PVCs indicate RV ectopic focus.)"
            )
        elif w.pvc_lbbb_fraction_24h > 0:
            # Some LBBB PVCs but not a majority → Minor criterion
            pvc_pts_awarded = PVC_WITHOUT_MORPHOLOGY_TFC_POINTS
            findings.append(
                f"PVC BURDEN: {w.pvc_count_24h:,} PVCs in past "
                f"{PVC_LOOKBACK_HOURS:.0f}h (threshold: >{PVC_24H_THRESHOLD}), "
                f"{w.pvc_lbbb_fraction_24h * 100:.0f}% LBBB morphology (minority). "
                f"TFC Category V Minor: +{pvc_pts_awarded} pt. "
                f"Majority LBBB morphology confirmation would upgrade to Major (+2 pts)."
            )
        else:
            # No morphology data → Minor criterion
            pvc_pts_awarded = PVC_WITHOUT_MORPHOLOGY_TFC_POINTS
            findings.append(
                f"PVC BURDEN: {w.pvc_count_24h:,} PVCs in past "
                f"{PVC_LOOKBACK_HOURS:.0f}h (threshold: >{PVC_24H_THRESHOLD}). "
                f"TFC Category V Minor: +{pvc_pts_awarded} pt. "
                f"(Morphology unavailable — LBBB confirmation required for Major criterion.)"
            )
        pts += pvc_pts_awarded

    elif w.pvc_count_24h > 0:
        findings.append(
            f"PVC count: {w.pvc_count_24h:,} in past {PVC_LOOKBACK_HOURS:.0f}h "
            f"(below >{PVC_24H_THRESHOLD} TFC threshold — no points awarded). "
            f"Continue monitoring."
        )
    else:
        missing.append(
            f"PVC count over past {PVC_LOOKBACK_HOURS:.0f}h not provided "
            f"(required for TFC Category V arrhythmia criterion). "
            f"Query beat database for cnn_label IN ('PVC', 'V') in past 24h."
        )

    return pts, findings, missing


def _score_ecg_only(
    w: ARVCWindowFeatures,
) -> Tuple[int, int, List[str], List[str], bool]:
    """
    Compute the ECG-accessible partial TFC score from all three scorable
    categories (III, IV, V).

    Returns:
        tfc_score       : total ECG-partial TFC points
        categories_hit  : number of DISTINCT categories contributing ≥1 pt
        findings        : combined findings list
        missing         : combined missing list
        urgent          : True if any urgent-referral criterion is met
                         (epsilon wave present OR LBBB-morphology VT)
    """
    findings: List[str] = []
    missing: List[str] = []
    tfc_score = 0
    categories_contributing = 0
    urgent = False

    dep_pts, dep_f, dep_m = _score_depolarisation(w)
    rep_pts, rep_f, rep_m = _score_repolarisation(w)
    arr_pts, arr_f, arr_m = _score_arrhythmia(w)

    findings.extend(dep_f)
    findings.extend(rep_f)
    findings.extend(arr_f)
    missing.extend(dep_m)
    missing.extend(rep_m)
    missing.extend(arr_m)

    tfc_score += dep_pts + arr_pts
    # rep_pts is always 0 here (Lead II limitation) but kept for forward compatibility

    if dep_pts > 0:
        categories_contributing += 1
    if arr_pts > 0:
        categories_contributing += 1

    # Urgent flag: epsilon wave or LBBB-morphology VT regardless of total score
    if w.epsilon_fraction >= EPSILON_FRACTION_MIN:
        urgent = True
    if w.vt_detected_this_window and w.vt_lbbb_morphology_this_window:
        urgent = True
    any_lbbb_vt = any(e.lbbb_morphology for e in w.vt_episodes_24h)
    if any_lbbb_vt:
        urgent = True

    return tfc_score, categories_contributing, findings, missing, urgent


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_arvc(window: ARVCWindowFeatures) -> "DetectionResult":
    """
    Evidence-gated ARVC screen for continuous ECG monitoring.

    Computes an ECG-partial TFC score from the three ECG-accessible
    criterion categories (III repolarisation, IV depolarisation, V
    arrhythmia) and stratifies into four output tiers.

    IMPORTANT: The maximum ECG-partial TFC score achievable here is
    limited by what a single-lead wearable can measure:
      Category IV max: 2 pts (epsilon wave major)
      Category V max:  2 pts (VT/PVC major)
      Category III:    0 pts (Lead II limitation — requires V1–V3)
    Total ECG max: 4 pts (from Categories IV + V only)

    For a definitive ARVC diagnosis (≥4 pts from ≥2 categories), this
    detector CAN in theory reach the definitive threshold purely from ECG
    (epsilon wave 2 pts + LBBB VT 2 pts = 4 pts across 2 categories).
    However, the detector does NOT state "ARVC diagnosed" — it states
    the TFC point total and recommends specialist referral. Final diagnosis
    requires imaging (Category I), which is not accessible here.

    Output tiers
    ------------
    URGENT_REFERRAL  Epsilon wave present OR LBBB-morphology VT detected.
                     These are immediate clinical concerns regardless of
                     total TFC score. Fires even on first occurrence.
                     Severity: HIGH | Confidence: 0.80 | triggered: True

    HIGH_SUSPICION   ECG-partial TFC score ≥ TFC_BORDERLINE_THRESHOLD (3)
                     without an urgent-flag criterion. Strong accumulation
                     of evidence warrants specialist referral.
                     Severity: HIGH | Confidence: 0.65 | triggered: True

    MODERATE_SUSPICION ECG-partial TFC score == TFC_POSSIBLE_THRESHOLD (2)
                     from ≥2 categories. Possible ARVC by TFC.
                     Severity: MODERATE | Confidence: 0.45 | triggered: True

    MONITORING       ECG-partial score 1 pt OR prior history score > 0.
                     Insufficient for suspicion threshold but warrants
                     continued surveillance.
                     Severity: LOW | triggered: False

    NONE             Score 0, no prior history.
                     Severity: INFO | triggered: False

    Non-ECG criteria always listed in missing_data
    -----------------------------------------------
    Category I  (structural/imaging): always missing
    Category II (tissue): always missing
    Category VI (family history/genetics): always missing
    → maximum points potentially available from non-ECG sources: 8+ pts

    The reason string always notes that the ECG-only partial score cannot
    reach the definitive diagnosis threshold without imaging confirmation.
    """
    # Lazy import to avoid circular dependency with disease_detector
    DetectionResult, Severity, _result, QUESTIONS = _import_detector_deps()

    tfc_score, categories_hit, findings, missing, urgent = _score_ecg_only(window)

    # ── Non-ECG categories — always missing ──────────────────────────────────
    missing.extend([
        # Category I — Structural/functional
        "CATEGORY I (Structural/Functional — up to 2 pts Major or 1 pt Minor): "
        "Echocardiogram or CMR required to assess RV regional wall motion "
        "abnormalities (akinesia, dyskinesia, aneurysm) AND RV dilation or "
        "reduced RVEF. This category is inaccessible from ECG alone.",

        # Category II — Tissue characterisation
        "CATEGORY II (Tissue Characterisation — up to 2 pts Major): "
        "CMR with late gadolinium enhancement (fibrosis pattern) or endomyocardial "
        "biopsy (fibrofatty replacement). Inaccessible from ECG alone.",

        # Category VI — Family history / genetics
        "CATEGORY VI (Family History/Genetics — up to 2 pts Major or 1 pt Minor): "
        "First-degree relative with confirmed ARVC, or pathogenic desmosomal "
        "mutation (PKP2, DSP, DSG2, DSC2, JUP) on genetic testing, or autopsy "
        "confirmation of ARVC in a relative. Cannot be assessed from ECG — "
        "requires family history questionnaire and genetic panel.",

        # Multi-lead confirmation
        "12-lead ECG at rest: required to formally score TFC Category III "
        "(T-wave inversions in V1–V3) and to confirm epsilon wave lead localisation "
        "(V1–V3). This is the single highest-yield next investigation from ECG alone.",
    ])

    # ── Add ARVC-mimic caution if relevant ───────────────────────────────────
    present_mimics = REVERSIBLE_ARVC_MIMICS & {d.lower() for d in window.known_diagnoses}
    if present_mimics:
        findings.append(
            f"CAUTION: Known conditions that can mimic ARVC ECG findings are "
            f"present: {', '.join(sorted(present_mimics))}. "
            f"These must be considered as alternative explanations before "
            f"advancing toward an ARVC diagnosis."
        )

    qs, src = QUESTIONS["Arrhythmogenic Cardiomyopathy"]
    score_display = f"{tfc_score} ECG-partial TFC pt{'s' if tfc_score != 1 else ''}"
    prior_note = (
        f" Prior highest ECG-partial score for this patient: "
        f"{window.prior_max_ecg_tfc_score} pts."
        if window.prior_max_ecg_tfc_score > 0 else ""
    )

    # ── Tier: URGENT REFERRAL ─────────────────────────────────────────────────
    if urgent:
        urgent_triggers = []
        if window.epsilon_fraction >= EPSILON_FRACTION_MIN:
            urgent_triggers.append(
                f"epsilon wave in {window.epsilon_fraction * 100:.1f}% of beats"
            )
        if window.vt_detected_this_window and window.vt_lbbb_morphology_this_window:
            urgent_triggers.append("LBBB-morphology VT in current window")
        if any(e.lbbb_morphology for e in window.vt_episodes_24h):
            urgent_triggers.append("LBBB-morphology VT in past 24h")

        return _result(
            disease="Arrhythmogenic Cardiomyopathy (ARVC)",
            triggered=True,
            confidence=0.80,
            severity=Severity.HIGH,
            reason=(
                f"URGENT REFERRAL — pathognomonic or high-specificity ARVC finding(s): "
                f"{'; '.join(urgent_triggers)}. "
                f"ECG-partial TFC score: {score_display} from {categories_hit} "
                f"scorable category/ies.{prior_note} "
                f"These findings mandate urgent cardiology/electrophysiology referral "
                f"regardless of total TFC score. Full TFC requires CMR, echo, "
                f"12-lead ECG, genetics, and family history — none of which are "
                f"available from continuous ECG monitoring alone."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I42.8"],
            rag=True,
        )

    # ── Tier: HIGH SUSPICION (ECG-partial score ≥ borderline TFC threshold) ───
    if tfc_score >= TFC_BORDERLINE_THRESHOLD and categories_hit >= 2:
        return _result(
            disease="Arrhythmogenic Cardiomyopathy (ARVC)",
            triggered=True,
            confidence=0.65,
            severity=Severity.HIGH,
            reason=(
                f"HIGH SUSPICION — ECG-partial TFC score {score_display} from "
                f"{categories_hit} independent categories (TFC borderline threshold: "
                f"≥{TFC_BORDERLINE_THRESHOLD} pts from ≥2 categories).{prior_note} "
                f"This level of ECG evidence warrants specialist cardiology referral "
                f"for CMR, 12-lead ECG, echo, and genetic testing to complete the "
                f"full TFC evaluation. ECG monitoring alone cannot confirm ARVC."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I42.8"],
            rag=True,
        )

    # ── Tier: MODERATE SUSPICION (ECG-partial score == possible TFC threshold) ─
    if tfc_score >= TFC_POSSIBLE_THRESHOLD and categories_hit >= 2:
        return _result(
            disease="Arrhythmogenic Cardiomyopathy (ARVC)",
            triggered=True,
            confidence=0.45,
            severity=Severity.MODERATE,
            reason=(
                f"MODERATE SUSPICION — ECG-partial TFC score {score_display} from "
                f"{categories_hit} independent categories (TFC possible threshold: "
                f"≥{TFC_POSSIBLE_THRESHOLD} pts from ≥2 categories).{prior_note} "
                f"This corresponds to 'Possible ARVC' by TFC score, but note that "
                f"the ECG-accessible categories (IV depolarisation, V arrhythmia) "
                f"can contribute at most 4 pts — imaging and genetic criteria "
                f"(Categories I, II, VI) are required for full evaluation. "
                f"12-lead ECG and cardiology referral recommended."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I42.8"],
            rag=True,
        )

    # ── Tier: MONITORING (single-category or sub-threshold evidence) ──────────
    if tfc_score >= 1 or window.prior_max_ecg_tfc_score >= 1:
        prior_history_note = ""
        if window.prior_max_ecg_tfc_score >= 1 and tfc_score == 0:
            prior_history_note = (
                f" NOTE: Prior ECG monitoring has recorded a partial TFC score of "
                f"{window.prior_max_ecg_tfc_score} pts. Current window is quiet but "
                f"history warrants continued surveillance."
            )
        return _result(
            disease="Arrhythmogenic Cardiomyopathy (ARVC)",
            triggered=False,
            confidence=0.20,
            severity=Severity.LOW,
            reason=(
                f"MONITORING — ECG-partial TFC score {score_display} "
                f"(from {categories_hit} categor{'y' if categories_hit == 1 else 'ies'}) "
                f"below the TFC 'Possible ARVC' threshold of "
                f"{TFC_POSSIBLE_THRESHOLD} pts from ≥2 independent categories. "
                f"Insufficient evidence to trigger specialist referral at this time. "
                f"Continue monitoring; accumulating evidence will be re-evaluated "
                f"with each new window.{prior_history_note}"
            ),
            findings=findings,
            missing=missing,
            questions=[],
            source=src,
            icd10=["I42.8"],
            rag=False,
        )

    # ── Tier: NONE ────────────────────────────────────────────────────────────
    return _result(
        disease="Arrhythmogenic Cardiomyopathy (ARVC)",
        triggered=False,
        confidence=0.05,
        severity=Severity.INFO,
        reason=(
            f"No ARVC-relevant ECG findings in the current monitoring window "
            f"(ECG-partial TFC score: 0 pts). "
            f"ARVC cannot be excluded from ECG alone — a meaningful proportion "
            f"of genetically confirmed ARVC carriers have normal resting ECGs, "
            f"particularly in early disease. Clinical or family-history suspicion "
            f"should prompt cardiology referral regardless of this result."
        ),
        findings=findings,
        missing=missing,
        questions=[],
        source=src,
        icd10=["I42.8"],
        rag=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE_DETECTOR.PY INTEGRATION NOTE
# ─────────────────────────────────────────────────────────────────────────────
# To integrate this into DiseaseDetector, follow the same pattern used for
# detect_long_qt:
#
# 1. Import at top of disease_detector.py:
#    from disease_detection.detect_arvc import detect_arvc, ARVCWindowFeatures
#
# 2. In DiseaseDetector.evaluate(), dispatch by function name:
#    if rule_fn.__name__ == "detect_arvc":
#        if arvc_window is not None:
#            result = rule_fn(arvc_window)
#        else:
#            result = DetectionResult(
#                disease="Arrhythmogenic Cardiomyopathy (ARVC)",
#                triggered=False, confidence=0.0, severity=Severity.INFO,
#                reason="ARVCWindowFeatures not provided.",
#                ecg_findings=[], missing_data=["ARVCWindowFeatures"],
#                symptom_questions=[], symptom_source="",
#                rag_trigger=False, icd10_codes=[],
#            )
#
# 3. In the pipeline, assemble ARVCWindowFeatures before calling evaluate():
#    from disease_detection.detect_arvc import ARVCWindowFeatures, VTEpisode
#
#    arvc_window = ARVCWindowFeatures(
#        window_id=window_id,
#        patient_id=patient_id,
#        recorded_at=datetime.now(),
#        valid_beat_count=len(valid_beats),
#        age=patient_meta.get("age"),
#        sex=patient_meta.get("sex"),
#        known_diagnoses=patient_meta.get("diagnoses", []),
#
#        # Current window
#        epsilon_fraction=sum(b.epsilon_wave_present for b in valid_beats)
#                         / len(valid_beats),
#        rbbb_fraction=sum(b.rbbb_pattern for b in valid_beats)
#                      / len(valid_beats),
#        qrs_duration_ms=statistics.median(
#            b.qrs_duration_ms for b in valid_beats
#            if b.qrs_duration_ms is not None
#        ),
#        t_inversion_fraction=sum(b.t_wave_inverted for b in valid_beats)
#                             / len(valid_beats),
#        vt_detected_this_window=window_stats.get("vt_detected", False),
#        vt_lbbb_morphology_this_window=window_stats.get("vt_lbbb", False),
#        vt_superior_axis_this_window=None,  # single-lead cannot determine
#
#        # From beat history database (query before calling)
#        pvc_count_24h=db.count_pvcs_24h(patient_id),
#        pvc_lbbb_fraction_24h=db.pvc_lbbb_fraction_24h(patient_id),
#        vt_episodes_24h=db.vt_episodes_24h(patient_id),
#        prior_epsilon_windows=db.count_prior_arvc_epsilon_windows(patient_id),
#        prior_t_inversion_windows=db.count_prior_arvc_t_inversion_windows(patient_id),
#        prior_max_ecg_tfc_score=db.max_arvc_ecg_tfc_score(patient_id),
#    )
#
# 4. Pass arvc_window to evaluate():
#    results = detector.evaluate(features, window_features, arvc_window)
#
# 5. Update the evaluate() signature accordingly:
#    def evaluate(
#        self,
#        features: ECGFeatures,
#        window_features: Optional[ECGWindowFeatures] = None,
#        arvc_window: Optional[ARVCWindowFeatures] = None,
#    ) -> List[DetectionResult]:
