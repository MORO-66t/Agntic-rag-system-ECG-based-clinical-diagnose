"""
detect_long_qt_v4.py
=====================
Continuous-monitoring LQTS detector with temporal characterisation.

CHANGES FROM v3
---------------
v3 summarised each analysis window with a single median QTc and SDQT.
That is sufficient for characterising the *typical* state of the window
but misses three clinically important phenomena:

  1. Temporal evolution — QTc that rises or falls across the window.
     A rising trend is a recognised pre-arrhythmic signal even when the
     current median has not yet crossed a threshold (Hohnloser et al.
     1997, Circulation 96:2271-7).

  2. Sustained prolongation fraction — the proportion of beats above
     a threshold. A window where QTc exceeds 480 ms on 80% of beats is
     clinically different from one where only 20% do, even if their
     medians are similar. Halamek et al. (2012, J Electrocardiol
     45:610-4) showed that sustained-prolongation fraction is more
     reproducible and more predictive than mean/median QTc in ambulatory
     monitoring.

  3. Isolated extreme events — rare but verified beats with extreme QTc
     (> 550 ms) that are swamped by the median when a window contains
     thousands of normal beats. ICH E14 implementation guidance
     consistently treats extreme outlier QTc values (after signal-quality
     verification) as requiring flagging and investigation, not
     discarding. Pause-dependent QT prolongation (Viskin et al.) is a
     well-described mechanism, which is why post-ectopic beats must
     already be excluded upstream before a value appears in
     qtc_values_ms — any extreme value in the valid-beat list is
     therefore not a post-pause artefact by construction.

These three additions are implemented as:
  (a) New computed properties on ECGWindowFeatures (all derived from
      the existing qtc_values_ms list — no new upstream features needed).
  (b) A new _check_temporal() helper called from _score_window().
  (c) A new _check_extreme_events() helper that can escalate to CRITICAL
      independently of the window median and Schwartz sub-score.

Window duration is now an explicit documented engineering parameter
(WINDOW_DURATION_MINUTES) rather than an implicit assumption in prose.

Calling contract: unchanged from v3 — single ECGWindowFeatures object.

Depends on names already defined in disease_detector.py:
    DetectionResult, Severity, _result, QUESTIONS
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple


def _import_detector_deps():
    """
    Lazy import from disease_detector to avoid circular imports.
    disease_detector imports from disease_detection.detect_long_qt,
    and we need DetectionResult, Severity, _result, QUESTIONS from it.
    """
    from disease_detector import DetectionResult, Severity, _result, QUESTIONS as _Q
    return DetectionResult, Severity, _result, _Q


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_DURATION_MINUTES: int = 30
# ROLE: Expected duration of one analysis window, used only for prose in
# findings messages. The detector logic itself is beat-count-based and does
# not depend on wall-clock duration.
#
# ENGINEERING PARAMETER — NO PUBLISHED GUIDELINE SPECIFIES THIS VALUE.
# There is no clinical guideline or validation study that mandates a
# 30-minute window for continuous LQTS screening. The FDA 2012 guidance on
# Holter-based QT assessment recommends averaging over stable 10-beat epochs
# without specifying an overall window duration. The published ambulatory
# monitoring literature (Couderc et al., Rochester group) uses beat-level
# QT series analysis rather than fixed windows. 30 minutes was chosen as an
# engineering default: long enough to contain hundreds of valid beats at
# resting HR, short enough to give timely alerts. Validate empirically on
# labelled data and adjust if sensitivity/specificity analysis on your own
# patient population supports a different value.

MIN_VALID_BEATS: int = 25
# ROLE: Quality-control gate only — not a diagnostic threshold.
# Ensures the median QTc is computed from enough beats that the estimate
# is stable before it is acted on. Has no effect when the window contains
# far more beats than this (normal for a 30-minute monitoring window at
# resting HR; this gate only bites in heavily ectopy-interrupted or very
# short windows).
#
# STATISTICAL BASIS: Beat-to-beat QTc SD in stable sinus rhythm is
# approximately 10–15 ms (Molnar et al. 1996, JACC 28:1509-14). Standard
# error of the median ≈ SD / sqrt(n). To keep SE < 3 ms (which matters
# when the nearest diagnostic threshold is 480 ms):
#   n > (15 / 3)^2 = 25 beats.

QTC_SD_MAX_MS: float = 15.0
# ROLE: Within-window QTc stability filter. Windows where SDQT exceeds
# this value are flagged as having unstable QTc before the median is
# scored.
#
# CLINICAL BASIS: Malik et al. 2008 (Eur Heart J 29:1029-37) showed
# SDQT > 15 ms is associated with arrhythmic events and represents the
# upper bound of normal within-recording QTc variability. Molnar et al.
# 1996 established ~10–15 ms as the typical value in healthy subjects.
#
# NOTE: Elevated SDQT is not merely noise — it may itself be a risk
# marker. Always report it in findings; never suppress silently.

T_WAVE_BIPHASIC_FRACTION_MIN: float = 0.50
# ROLE: Persistence filter for the single-lead T-wave morphology proxy.
#
# ENGINEERING PARAMETER — NO DIRECT CLINICAL VALIDATION.
# The Schwartz criterion requires notched/biphasic T waves in >= 3 leads
# on a 12-lead ECG. This system records Lead II only; the literal
# criterion cannot be applied. This threshold (50% of valid beats) acts
# as a noise filter for automated single-lead morphology detection.
# The value has no published evidential basis and must be validated on
# labelled single-lead data before being treated as clinically equivalent
# to the published Schwartz criterion. All findings using this proxy
# include an explicit single-lead caveat.

TWA_MIN_CONSECUTIVE_PAIRS: int = 3
# ROLE: Noise filter for macroscopic T-wave alternans (TWA) detection.
#
# ENGINEERING PARAMETER — NO DIRECT CLINICAL VALIDATION.
# The Schwartz criterion for macroscopic TWA is qualitative and does not
# specify a minimum run length. 3 consecutive pairs is used to reject
# isolated beat-pair artefacts. See the newkit integration note at the
# bottom of this file for the critical distinction between macroscopic
# TWA (this criterion) and microvolt spectral/MMA TWA (a different,
# non-Schwartz marker).

QTC_EXTREME_THRESHOLD_MS: float = 550.0
# ROLE: Threshold above which a single verified beat triggers the extreme-
# event CRITICAL path, independently of the window median and Schwartz
# sub-score.
#
# CLINICAL BASIS: 550 ms is well above the existing CRITICAL threshold of
# 500 ms in the window median path. Values > 550 ms on a non-post-ectopic
# sinus beat are vanishingly rare in the absence of severe pathology
# (population 99th percentile is ~470–490 ms; published LQTS cohort data
# show events predominantly in the 480–550 ms range). ICH E14
# implementation documents (FDA 2017 E14/E15 Q&A) consistently recommend
# flagging rather than discarding extreme QTc outliers that survive signal-
# quality filtering. Post-ectopic beats must already be excluded upstream
# before values appear in qtc_values_ms, so any value above this threshold
# in the valid-beat list is not a pause-dependent artefact by construction.
#
# Note: This path reports an extreme event finding and escalates to
# CRITICAL. It does not replace the median-based Schwartz scoring; both
# are reported concurrently.

SPF_THRESHOLD_MS: float = 480.0
# ROLE: QTc threshold used to compute the Sustained Prolongation Fraction
# (SPF) — the proportion of valid beats exceeding this value within the
# window.
#
# CLINICAL BASIS: 480 ms is the Schwartz 'high probability' band threshold
# (Schwartz & Crotti 2011, JACC 57:802-12) and the threshold cited in the
# 2013 HRS/EHRA/APHRS consensus and 2015/2022 ESC guidelines for "QTc on
# repeated ECGs." Using the diagnostic threshold as the SPF reference point
# directly connects the SPF to the same clinical cutoff as the median-based
# scoring. Halamek et al. (2012, J Electrocardiol 45:610-4) used the
# relevant diagnostic threshold as their fraction reference in ambulatory
# monitoring.

SPF_CLINICALLY_SIGNIFICANT: float = 0.25
# ROLE: SPF level above which the finding is reported as clinically
# meaningful, even when the median QTc is below the threshold.
#
# ENGINEERING PARAMETER — NO DIRECT CLINICAL VALIDATION.
# There is no published SPF cutoff for continuous wearable monitoring.
# 25% (i.e., >= 25% of valid beats exceed 480 ms) is used as a reporting
# threshold based on clinical reasoning: a window where 1 in 4 sinus beats
# exceeds the diagnostic threshold likely represents a real pattern rather
# than tail-end variability around a normal mean. Validate on labelled data.

QTC_TREND_SLOPE_ALERT_MS_PER_BEAT: float = 0.05
# ROLE: Linear trend slope (ms per beat, from OLS regression of QTc vs.
# beat index) above which a rising QTc trend is flagged as a pre-arrhythmic
# signal in findings.
#
# ENGINEERING PARAMETER — PARTIALLY INFORMED BY LITERATURE.
# Hohnloser et al. (1997, Circulation 96:2271-7) documented progressive
# QTc prolongation in the minutes preceding TdP events. At a resting HR of
# ~60 bpm, 0.05 ms/beat = 3 ms/minute = ~90 ms/30 min — a clinically
# meaningful drift that would move a patient from the 440 ms range to near-
# critical over one analysis window. The slope threshold is not directly
# validated for wearable monitoring and should be tuned empirically.
# This finding does NOT add Schwartz points — it is a pre-arrhythmic
# warning signal, not a diagnostic criterion.

REVERSIBLE_QT_CAUSES: set = {
    "hypokalemia", "hypomagnesemia", "hypocalcemia",
    "qt_prolonging_medication", "hypothyroidism",
}
# Per Schwartz et al. 1993 (Circulation 88:782-4): the score is only
# interpretable in the absence of secondary causes of QT prolongation.


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTING DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class ExtremeQTEvent(NamedTuple):
    """
    A single verified beat whose QTc exceeds QTC_EXTREME_THRESHOLD_MS.

    beat_index  : Index of the beat within qtc_values_ms (0-based).
    qtc_ms      : The verified QTc value for this beat (ms).
    rr_ms       : The RR interval of this beat (ms), if available. Used
                  by reviewers to assess rate-dependency. None if the
                  per-beat RR is not exposed in the window features.
    """
    beat_index: int
    qtc_ms: float
    rr_ms: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ECGWindowFeatures:
    """
    Aggregated features over a multi-beat continuous-monitoring window.

    Build this from your stored per-beat features. For LQTS screening,
    the recommended window is the longest stretch of resting, rhythmically
    stable sinus beats available in the current monitoring session
    (see WINDOW_DURATION_MINUTES for duration guidance and its caveats).

    `valid_beat_count` and `qtc_values_ms` must already EXCLUDE, upstream:
      - ectopic beats (PACs, PVCs, etc.)
      - the beat immediately following an ectopic beat (post-ectopic QT
        is unreliable due to the altered preceding RR interval)
      - beats flagged as noisy or low signal quality
      - beats not in sinus rhythm
    This exclusion must happen before this dataclass is populated.

    Fields
    ------
    window_id         : Unique ID for this analysis window.
    patient_id        : Patient identifier.
    recorded_at       : Timestamp of the start of this window.
    beat_count        : Total beats observed (including excluded beats).
    valid_beat_count  : Number of valid sinus beats used for QTc stats.
    qtc_values_ms     : Per-beat QTc values (ms) for valid beats only,
                        IN TEMPORAL ORDER (beat 0 is earliest). Order is
                        required for trend slope computation.
                        Should be Fridericia-corrected (QTcF); see
                        Bazett vs. Fridericia note in _score_window().
    rr_values_ms      : Per-beat RR intervals (ms) for valid beats, in
                        the same temporal order as qtc_values_ms.
                        Optional; used only to enrich ExtremeQTEvent
                        records for reviewer inspection. May be None or
                        an empty list if not available.
    sex               : "M" or "F"; None if unknown.
    age               : Age in years; None if unknown.
    resting           : True if HR is in resting range with no recent
                        exertion. QTc thresholds assume a resting state.
    rhythm_stable     : True if no significant ectopy or irregularity
                        occurred. Ectopy distorts the QTc distribution.
    vt_detected       : True if VT was detected during the window.
    tdp_detected      : True if polymorphic VT consistent with TdP was
                        detected. Leave False if your rhythm classifier
                        does not distinguish TdP specifically.
    t_wave_biphasic_fraction : Fraction of valid beats (0.0–1.0) in
                        Lead II showing biphasic/notched T-wave morphology.
                        Single-lead proxy only — see T_WAVE_BIPHASIC_
                        FRACTION_MIN documentation.
    macroscopic_twa_present : True if macroscopic beat-to-beat T-wave
                        alternans was detected. See newkit note at the
                        bottom of this file before wiring this in.
    macroscopic_twa_consecutive_pairs : Consecutive alternating beat-
                        pairs supporting the TWA finding.
    bradycardia_for_age : True if sustained sinus bradycardia for age
                        was present throughout the window (Schwartz 0.5pt).
    known_diagnoses   : Active diagnoses / conditions; checked against
                        REVERSIBLE_QT_CAUSES.
    """

    window_id: str
    patient_id: str
    recorded_at: datetime
    beat_count: int
    valid_beat_count: int
    qtc_values_ms: List[float] = field(default_factory=list)
    rr_values_ms: Optional[List[float]] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    resting: bool = True
    rhythm_stable: bool = True
    vt_detected: bool = False
    tdp_detected: bool = False
    t_wave_biphasic_fraction: float = 0.0
    macroscopic_twa_present: bool = False
    macroscopic_twa_consecutive_pairs: int = 0
    bradycardia_for_age: bool = False
    known_diagnoses: List[str] = field(default_factory=list)

    # ── Derived statistics ────────────────────────────────────────────────────

    @property
    def qtc_median_ms(self) -> Optional[float]:
        """Median QTc across all valid beats (ms)."""
        if not self.qtc_values_ms:
            return None
        return statistics.median(self.qtc_values_ms)

    @property
    def qtc_sd_ms(self) -> Optional[float]:
        """
        Beat-to-beat QTc standard deviation within the window (ms).
        Returns None if fewer than 2 valid beats are present.
        Basis: Malik et al. 2008 (Eur Heart J); Molnar et al. 1996 (JACC).
        """
        if len(self.qtc_values_ms) < 2:
            return None
        return statistics.stdev(self.qtc_values_ms)

    @property
    def qtc_p95_ms(self) -> Optional[float]:
        """
        95th-percentile QTc across valid beats (ms).
        Captures the upper tail of the within-window QTc distribution,
        providing a less outlier-sensitive complement to the window
        maximum while still reflecting the worst sustained cluster.
        """
        n = len(self.qtc_values_ms)
        if n == 0:
            return None
        s = sorted(self.qtc_values_ms)
        # Nearest-rank method
        rank = max(1, math.ceil(0.95 * n))
        return s[rank - 1]

    @property
    def qtc_sustained_prolongation_fraction(self) -> Optional[float]:
        """
        Fraction of valid beats with QTc > SPF_THRESHOLD_MS (default 480 ms).

        Sustained Prolongation Fraction (SPF): a proportion-of-time-above-
        threshold metric. In ambulatory monitoring, SPF is more reproducible
        and more predictive of arrhythmic events than mean/median QTc
        (Halamek et al. 2012, J Electrocardiol 45:610-4). A window where
        the majority of beats exceed the diagnostic threshold represents
        a different physiological state than one where only a few do, even
        if their medians are similar.

        Returns None if the beat list is empty.
        """
        if not self.qtc_values_ms:
            return None
        n_above = sum(1 for q in self.qtc_values_ms if q > SPF_THRESHOLD_MS)
        return n_above / len(self.qtc_values_ms)

    @property
    def qtc_trend_slope_ms_per_beat(self) -> Optional[float]:
        """
        Linear trend slope of QTc vs. beat index within the window
        (ms per beat), from ordinary least squares regression.

        A positive slope means QTc is rising across the window; negative
        means falling. At resting HR (~60 bpm), 0.05 ms/beat corresponds
        to ~3 ms/minute — a clinically meaningful drift rate based on
        pre-TdP progression patterns (Hohnloser et al. 1997, Circulation
        96:2271-7).

        qtc_values_ms must be in temporal order (index 0 = earliest beat).
        Returns None if fewer than 2 valid beats are present.

        This slope does NOT contribute to the Schwartz sub-score.
        It is a pre-arrhythmic warning signal reported separately in
        findings.
        """
        n = len(self.qtc_values_ms)
        if n < 2:
            return None
        xs = list(range(n))
        xm = sum(xs) / n
        ym = sum(self.qtc_values_ms) / n
        ss_xx = sum((x - xm) ** 2 for x in xs)
        if ss_xx == 0:
            return None
        ss_xy = sum((xs[i] - xm) * (self.qtc_values_ms[i] - ym) for i in range(n))
        return ss_xy / ss_xx

    @property
    def extreme_qt_events(self) -> List[ExtremeQTEvent]:
        """
        List of verified beats whose QTc exceeds QTC_EXTREME_THRESHOLD_MS,
        drawn from the already-filtered valid-beat list.

        Because post-ectopic beats are excluded upstream before populating
        qtc_values_ms, any value above the threshold here is not a
        pause-dependent artefact by construction — it represents a true
        physiological QTc on a normal-conducted sinus beat.

        Each event records the beat index, QTc value, and RR interval (if
        rr_values_ms is populated) to allow downstream rate-dependency
        assessment. Basis: ICH E14/E15 Q&A (FDA 2017) — extreme QTc
        outliers surviving signal-quality filtering should be flagged and
        investigated, not discarded.
        """
        events: List[ExtremeQTEvent] = []
        rr = self.rr_values_ms or []
        for i, qtc in enumerate(self.qtc_values_ms):
            if qtc > QTC_EXTREME_THRESHOLD_MS:
                rr_val = rr[i] if i < len(rr) else None
                events.append(ExtremeQTEvent(beat_index=i, qtc_ms=qtc, rr_ms=rr_val))
        return events

    @property
    def has_sufficient_beats(self) -> bool:
        """True if valid_beat_count meets the MIN_VALID_BEATS gate."""
        return self.valid_beat_count >= MIN_VALID_BEATS


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _qtc_band(qtc: float, sex: Optional[str]) -> str:
    """Map a QTc value to a named Schwartz-score band."""
    sex = (sex or "").upper()
    if qtc >= 500:
        return "critical"
    if qtc >= 480:
        return "high"
    if qtc >= 460:
        return "intermediate"
    if qtc >= 450 and sex == "M":
        return "borderline_male"
    if qtc >= 440:
        return "borderline"
    return "normal"


def _check_extreme_events(
    w: ECGWindowFeatures,
) -> Tuple[bool, List[str]]:
    """
    Inspect the valid-beat QTc list for extreme events independently of
    the window median.

    Returns (extreme_critical: bool, extreme_findings: List[str]).

    extreme_critical=True escalates the result to CRITICAL in the main
    entry point, regardless of the Schwartz sub-score and quality gates.
    This mirrors the principle that a single verified extreme event cannot
    be ignored just because thousands of surrounding normal beats drag the
    median below the threshold.

    All events in w.extreme_qt_events have already passed upstream signal-
    quality filtering; they are not artefacts by assumption.
    """
    events = w.extreme_qt_events
    if not events:
        return False, []

    extreme_findings: List[str] = []
    max_event = max(events, key=lambda e: e.qtc_ms)

    extreme_findings.append(
        f"EXTREME QT EVENT(S) DETECTED: {len(events)} verified beat(s) with "
        f"QTc > {QTC_EXTREME_THRESHOLD_MS:.0f} ms within the window "
        f"(threshold: ICH E14 extreme-outlier flag). "
        f"Peak: QTc {max_event.qtc_ms:.0f} ms at beat index {max_event.beat_index}"
        + (f", RR {max_event.rr_ms:.0f} ms" if max_event.rr_ms is not None else "")
        + f". These beats survived upstream signal-quality filtering and are "
        f"not post-ectopic artefacts by construction. "
        f"The window median QTc may underrepresent risk because these events "
        f"are diluted by surrounding normal beats."
    )

    if len(events) > 1:
        extreme_findings.append(
            f"Additional extreme events: beat indices "
            f"{[e.beat_index for e in events if e != max_event]} "
            f"with QTc values "
            f"{[round(e.qtc_ms, 0) for e in events if e != max_event]} ms."
        )

    return True, extreme_findings


def _check_temporal(
    w: ECGWindowFeatures,
) -> Tuple[List[str], List[str]]:
    """
    Evaluate temporal characteristics of the QTc series within the window:
      (a) Sustained Prolongation Fraction (SPF)
      (b) Linear trend slope (rising/falling QTc)
      (c) 95th-percentile QTc

    Returns (temporal_findings: List[str], temporal_missing: List[str]).

    None of these contribute to the Schwartz sub-score — they are
    supplementary findings reported alongside it.
    """
    t_findings: List[str] = []
    t_missing: List[str] = []

    if not w.qtc_values_ms:
        return t_findings, t_missing

    # ── (a) Sustained Prolongation Fraction ──────────────────────────────────
    spf = w.qtc_sustained_prolongation_fraction
    if spf is not None:
        spf_pct = spf * 100
        t_findings.append(
            f"Sustained Prolongation Fraction (SPF): {spf_pct:.1f}% of valid beats "
            f"exceed {SPF_THRESHOLD_MS:.0f} ms within the window "
            f"(basis: Halamek et al. 2012, J Electrocardiol — SPF is more "
            f"reproducible and predictive than median QTc in ambulatory monitoring)."
        )
        if spf >= SPF_CLINICALLY_SIGNIFICANT:
            t_findings.append(
                f"SPF >= {SPF_CLINICALLY_SIGNIFICANT * 100:.0f}% — a clinically "
                f"meaningful proportion of beats exceed the diagnostic threshold "
                f"even if the window median is near or below it. "
                f"(SPF_CLINICALLY_SIGNIFICANT is an engineering threshold requiring "
                f"validation on labelled data.)"
            )
        else:
            t_findings.append(
                f"SPF < {SPF_CLINICALLY_SIGNIFICANT * 100:.0f}% — prolonged QTc "
                f"is not a sustained pattern across this window."
            )

    # ── (b) 95th-percentile QTc ───────────────────────────────────────────────
    p95 = w.qtc_p95_ms
    if p95 is not None:
        t_findings.append(
            f"95th-percentile QTc: {p95:.0f} ms "
            f"(upper-tail summary; less outlier-sensitive than the window maximum)."
        )
        if p95 >= 500:
            t_findings.append(
                f"P95 QTc {p95:.0f} ms >= 500 ms — the upper 5% of beats in this "
                f"window reach the TdP-risk threshold even if the median does not."
            )
        elif p95 >= 480:
            t_findings.append(
                f"P95 QTc {p95:.0f} ms >= 480 ms — the upper tail of the window "
                f"QTc distribution reaches the Schwartz 'high probability' band."
            )

    # ── (c) Linear trend slope ────────────────────────────────────────────────
    slope = w.qtc_trend_slope_ms_per_beat
    if slope is None:
        t_missing.append(
            "QTc trend slope unavailable — fewer than 2 valid beats in window."
        )
    else:
        # Convert slope to approximate ms/minute at resting HR ~60 bpm
        slope_per_min = slope * 60
        t_findings.append(
            f"QTc linear trend: {slope:+.4f} ms/beat ({slope_per_min:+.1f} ms/min "
            f"at resting HR ~60 bpm). "
            f"(qtc_values_ms must be in temporal order for this to be valid.)"
        )
        if slope >= QTC_TREND_SLOPE_ALERT_MS_PER_BEAT:
            t_findings.append(
                f"RISING QTc TREND: slope {slope:+.4f} ms/beat "
                f"({slope_per_min:+.1f} ms/min) meets the alert threshold "
                f"({QTC_TREND_SLOPE_ALERT_MS_PER_BEAT} ms/beat = ~"
                f"{QTC_TREND_SLOPE_ALERT_MS_PER_BEAT * 60:.1f} ms/min). "
                f"Progressive QTc prolongation within a monitoring window is a "
                f"recognised pre-arrhythmic signal preceding TdP "
                f"(Hohnloser et al. 1997, Circulation 96:2271-7). "
                f"This finding does NOT add Schwartz points but warrants "
                f"heightened monitoring. "
                f"(QTC_TREND_SLOPE_ALERT_MS_PER_BEAT is an engineering threshold "
                f"requiring validation on labelled data.)"
            )
        elif slope <= -QTC_TREND_SLOPE_ALERT_MS_PER_BEAT:
            t_findings.append(
                f"FALLING QTc TREND: slope {slope:+.4f} ms/beat "
                f"({slope_per_min:+.1f} ms/min) — QTc is improving across the window. "
                f"May reflect resolution of a transient cause; monitor for recurrence."
            )

    return t_findings, t_missing


def _score_window(
    w: ECGWindowFeatures,
) -> Tuple[float, List[str], List[str], bool, bool]:
    """
    Compute the ECG-only Schwartz sub-score for a single monitoring window.

    Returns: (subscore, findings, missing, critical, quality_ok)

    `quality_ok` is False when quality gates fail hard enough that the
    score should not be acted on even if numerically >= a tier threshold.
    CRITICAL findings (QTc >= 500 ms median, TdP/VT, or extreme events)
    override quality_ok — acute arrhythmic risk cannot wait.

    Bazett vs. Fridericia note
    --------------------------
    Published Schwartz score thresholds (480 / 460 / 450 ms) were derived
    using Bazett-corrected QTc (QTcB). Fridericia correction (QTcF) is
    preferred in monitoring pipelines because QTcB over-corrects at
    elevated HR and under-corrects at low HR. Feeding QTcF into Bazett-
    calibrated thresholds is a reasonable modern adaptation but means this
    is not strictly the validated Schwartz instrument. Document in clinical
    reporting output.
    """
    findings: List[str] = []
    missing: List[str] = []
    subscore = 0.0
    critical = False
    quality_ok = True

    # ── Quality gates ────────────────────────────────────────────────────────
    if not w.has_sufficient_beats:
        quality_ok = False
        missing.append(
            f"Only {w.valid_beat_count} valid sinus beats in window "
            f"(minimum {MIN_VALID_BEATS} required for a stable median QTc estimate). "
            f"Window may be too short or too ectopy-interrupted."
        )
    if not w.resting:
        quality_ok = False
        missing.append(
            "Window not confirmed resting — Schwartz QTc thresholds assume a resting "
            "recording. QTc during or after exertion is not directly comparable."
        )
    if not w.rhythm_stable:
        quality_ok = False
        missing.append(
            "Rhythm not stable during window — significant ectopy or irregularity "
            "reduces QTc reliability even after ectopic beats are excluded."
        )

    # ── QTc statistics ───────────────────────────────────────────────────────
    qtc = w.qtc_median_ms
    if qtc is None:
        missing.append("No valid QTc measurements in window — cannot score.")
        return 0.0, findings, missing, False, False

    sd = w.qtc_sd_ms
    if sd is not None:
        sd_flag = sd > QTC_SD_MAX_MS
        findings.append(
            f"Median QTc {qtc:.0f} ms | SDQT {sd:.1f} ms "
            f"across {w.valid_beat_count} valid beats"
            + (" [SDQT ELEVATED — QTc unstable within window]" if sd_flag else "")
        )
        if sd_flag:
            quality_ok = False
            missing.append(
                f"SDQT {sd:.1f} ms exceeds the {QTC_SD_MAX_MS:.0f} ms stability "
                f"threshold (Malik et al. 2008, Eur Heart J). The window QTc is "
                f"unstable; the median may not represent a sustained true QTc. "
                f"Note: elevated SDQT may itself be an arrhythmic risk marker — "
                f"review independently rather than treating as simple noise."
            )
    else:
        findings.append(
            f"Median QTc {qtc:.0f} ms (SD unavailable — fewer than 2 valid beats)"
        )

    # ── Temporal characterisation (SPF, trend, P95) ──────────────────────────
    t_findings, t_missing = _check_temporal(w)
    findings.extend(t_findings)
    missing.extend(t_missing)

    # ── Schwartz ECG sub-score ───────────────────────────────────────────────
    sex = (w.sex or "").upper()

    if qtc >= 500:
        subscore += 3.0
        critical = True
        findings.append(
            f"Median QTc {qtc:.0f} ms >= 500 ms — CRITICAL: marked torsades de "
            f"pointes risk (Schwartz 3.0 pts; fires immediately)"
        )
    elif qtc >= 480:
        subscore += 3.0
        findings.append(
            f"Median QTc {qtc:.0f} ms >= 480 ms (Schwartz 3.0 pts — 'high "
            f"probability' band)"
        )
    elif qtc >= 460:
        subscore += 2.0
        findings.append(
            f"Median QTc {qtc:.0f} ms in 460–479 ms band (Schwartz 2.0 pts)"
        )
    elif qtc >= 450 and sex == "M":
        subscore += 1.0
        findings.append(
            f"Median QTc {qtc:.0f} ms in 450–459 ms band, male "
            f"(Schwartz 1.0 pt — male-specific threshold)"
        )
    elif qtc >= 440:
        findings.append(
            f"Median QTc {qtc:.0f} ms — borderline, below Schwartz scoring "
            f"threshold (no points awarded)"
        )
    else:
        findings.append(
            f"Median QTc {qtc:.0f} ms — within normal range (no points awarded)"
        )

    if sex not in ("M", "F") and 450 <= qtc < 460:
        missing.append(
            "Patient sex unknown — needed to apply the 450–459 ms male-specific "
            "Schwartz point (1.0 pt)."
        )

    # Polymorphic VT / TdP with QT prolongation (Schwartz 2.0 pts + CRITICAL)
    if (w.vt_detected or w.tdp_detected) and qtc >= 460:
        subscore += 2.0
        critical = True
        label = "TdP" if w.tdp_detected else "polymorphic VT"
        findings.append(
            f"Documented {label} with QT prolongation in window "
            f"(Schwartz 2.0 pts; CRITICAL override)"
        )

    # T-wave morphology — single-lead Lead II proxy (Schwartz 1.0 pt)
    if w.t_wave_biphasic_fraction >= T_WAVE_BIPHASIC_FRACTION_MIN:
        subscore += 1.0
        findings.append(
            f"Biphasic/notched T wave in {w.t_wave_biphasic_fraction * 100:.0f}% "
            f"of valid beats in Lead II (>= {T_WAVE_BIPHASIC_FRACTION_MIN * 100:.0f}% "
            f"threshold met). PROXY CRITERION: single-lead approximation of Schwartz "
            f"'notched T in >= 3 leads' (1.0 pt). Multi-lead validation is not "
            f"possible with Lead II only. Treat with caution."
        )
    elif w.t_wave_biphasic_fraction > 0:
        findings.append(
            f"Biphasic/notched T wave in {w.t_wave_biphasic_fraction * 100:.0f}% "
            f"of valid beats in Lead II — below the "
            f"{T_WAVE_BIPHASIC_FRACTION_MIN * 100:.0f}% persistence threshold; "
            f"not scored."
        )

    # Macroscopic T-wave alternans (Schwartz 1.0 pt)
    if w.macroscopic_twa_present:
        if w.macroscopic_twa_consecutive_pairs >= TWA_MIN_CONSECUTIVE_PAIRS:
            subscore += 1.0
            findings.append(
                f"Macroscopic T-wave alternans over "
                f"{w.macroscopic_twa_consecutive_pairs} consecutive beat-pairs "
                f"(Schwartz 1.0 pt). Verify this is gross resting-rate T-wave "
                f"alternation (Schwartz criterion), not microvolt spectral/MMA TWA "
                f"— see newkit note."
            )
        else:
            findings.append(
                f"Possible macroscopic TWA over "
                f"{w.macroscopic_twa_consecutive_pairs} beat-pair(s) — below the "
                f"{TWA_MIN_CONSECUTIVE_PAIRS}-pair persistence threshold; not scored."
            )

    # Bradycardia for age (Schwartz 0.5 pt)
    if w.bradycardia_for_age:
        subscore += 0.5
        findings.append(
            "Sinus bradycardia for age sustained throughout window "
            "(Schwartz 0.5 pts)"
        )

    # Reversible / acquired QT-prolonging causes
    present_reversible = REVERSIBLE_QT_CAUSES & {d.lower() for d in w.known_diagnoses}
    if present_reversible:
        findings.append(
            f"CAUTION — known reversible QT-prolonging factor(s): "
            f"{', '.join(sorted(present_reversible))}. "
            f"Schwartz score is only interpretable in the absence of secondary "
            f"causes (Schwartz et al. 1993, Circulation 88:782-4). Score may "
            f"reflect acquired, not congenital, LQTS."
        )

    return subscore, findings, missing, critical, quality_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_long_qt(window: ECGWindowFeatures) -> "DetectionResult":
    """
    Evidence-gated LQTS screen for continuous cardiac monitoring.

    Operates on a single ECGWindowFeatures object representing a
    continuous monitoring window (default ~30 min; see
    WINDOW_DURATION_MINUTES). Evaluates three complementary layers:

      Layer 1 — Schwartz ECG sub-score (median QTc, morphology,
                bradycardia, VT/TdP). Drives the main tier logic.

      Layer 2 — Temporal characterisation (SPF, trend slope, P95 QTc).
                Reported as supplementary findings; does NOT add Schwartz
                points. A rising trend or high SPF may be reported even
                when the sub-score is below the trigger threshold.

      Layer 3 — Extreme event detection (any valid beat > 550 ms).
                Escalates independently to CRITICAL regardless of the
                window median and quality gates.

    Tier logic
    ----------
    CRITICAL   Any of:
               • Median QTc >= 500 ms
               • Documented TdP / polymorphic VT with QT prolongation
               • Any verified beat with QTc > QTC_EXTREME_THRESHOLD_MS
                 (550 ms) after upstream quality filtering
               → Fires immediately.
               Confidence 0.95 | Severity CRITICAL

    HIGH       ECG sub-score >= 3.5 pts AND all quality gates pass.
               → Fires.
               Confidence 0.85 | Severity HIGH

    MODERATE   ECG sub-score >= 1.5 pts AND all quality gates pass.
               → Fires; proceeds to symptom/family-history questioning.
               Confidence 0.55 | Severity MODERATE

    PENDING    Sub-score >= 1.5 pts BUT quality gates failed.
               → Does NOT trigger; recommend higher-quality window.
               HIGH-pending: Confidence 0.40 | Severity MODERATE
               MID-pending:  Confidence 0.25 | Severity LOW

    NONE       Sub-score < 1.5 pts.
               → Does NOT trigger.
               Confidence 0.05–0.10 | Severity INFO

    Note on temporal findings
    -------------------------
    Temporal findings (SPF, trend, P95) are reported in `findings`
    regardless of which tier fires. A rising trend or high SPF at the
    NONE tier is included so the pipeline can surface a pre-arrhythmic
    warning even when the current Schwartz score is below the trigger
    threshold.

    Important limitations
    ---------------------
    - ECG sub-score only. Missing Schwartz points (syncope history,
      family history, genetic testing) are always listed in `missing`.
    - NONE does NOT exclude LQTS. Concealed LQTS (normal QTc in
      genetically confirmed carriers) is well-documented.
    - T-wave morphology uses a single-lead proxy. See documentation.
    - QTcF values fed into Bazett-calibrated thresholds: an adapted
      instrument, not the strictly validated Schwartz score.
    """
    # Lazy import to avoid circular dependency with disease_detector
    DetectionResult, Severity, _result, QUESTIONS = _import_detector_deps()

    # ── Layer 1 + 2: Schwartz score + temporal (via _score_window) ───────────
    subscore, findings, missing, critical, quality_ok = _score_window(window)

    # ── Layer 3: Extreme event detection (independent of median) ─────────────
    extreme_critical, extreme_findings = _check_extreme_events(window)
    if extreme_findings:
        findings.extend(extreme_findings)
    if extreme_critical:
        critical = True

    # ── Non-ECG Schwartz points — always listed as missing ────────────────────
    missing.extend([
        "Syncope history — number of episodes, triggers, and stress-relatedness "
        "(stress-related: 2.0 pts; non-stress-related: 1.0 pt) "
        "[requires symptom questionnaire]",
        "Congenital deafness / Jervell-Lange-Nielsen syndrome — 0.5 pt "
        "[requires clinical history]",
        "Family history: confirmed LQTS in a first-degree relative — 1.0 pt "
        "[requires family history questionnaire]",
        "Family history: unexplained sudden cardiac death < 30 years in an "
        "immediate family member — 0.5 pt [requires family history questionnaire]",
        "Genetic testing result (pathogenic LQTS mutation = 3.5 pts in the "
        "modified Schwartz / 2013 HRS criteria) — not available from ECG",
    ])

    qs, src = QUESTIONS["Long QT Syndrome"]
    qtc_display = (
        f"{window.qtc_median_ms:.0f} ms"
        if window.qtc_median_ms is not None
        else "n/a"
    )

    # ── Tier: CRITICAL ───────────────────────────────────────────────────────
    if critical:
        # Build a concise reason that distinguishes the critical cause
        critical_causes = []
        if window.qtc_median_ms is not None and window.qtc_median_ms >= 500:
            critical_causes.append(f"median QTc {qtc_display} >= 500 ms")
        if window.vt_detected or window.tdp_detected:
            critical_causes.append(
                "documented " + ("TdP" if window.tdp_detected else "polymorphic VT")
                + " with QT prolongation"
            )
        if extreme_critical:
            peak = max(window.extreme_qt_events, key=lambda e: e.qtc_ms)
            critical_causes.append(
                f"verified extreme QT event (beat {peak.beat_index}: "
                f"QTc {peak.qtc_ms:.0f} ms > {QTC_EXTREME_THRESHOLD_MS:.0f} ms threshold)"
            )
        cause_str = "; ".join(critical_causes) if critical_causes else "see findings"

        return _result(
            disease="Long QT Syndrome (LQTS)",
            triggered=True,
            confidence=0.95,
            severity=Severity.CRITICAL,
            reason=(
                f"CRITICAL — {cause_str}. "
                f"ECG sub-score {subscore:.1f} pts, {window.valid_beat_count} valid beats. "
                f"Fires immediately; critical findings override quality gates."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I45.81"],
            rag=True,
        )

    # ── Tier: HIGH ───────────────────────────────────────────────────────────
    if subscore >= 3.5 and quality_ok:
        return _result(
            disease="Long QT Syndrome (LQTS)",
            triggered=True,
            confidence=0.85,
            severity=Severity.HIGH,
            reason=(
                f"ECG sub-score {subscore:.1f} pts (median QTc {qtc_display}) from "
                f"{window.valid_beat_count} valid beats — reaches the Schwartz 'high "
                f"probability' band (>= 3.5 pts) with all quality gates passed. "
                f"Proceeding to symptom/family-history questioning."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I45.81"],
            rag=True,
        )

    # ── Tier: PENDING (high sub-score, quality failed) ───────────────────────
    if subscore >= 3.5 and not quality_ok:
        return _result(
            disease="Long QT Syndrome (LQTS)",
            triggered=False,
            confidence=0.40,
            severity=Severity.MODERATE,
            reason=(
                f"ECG sub-score {subscore:.1f} pts (median QTc {qtc_display}) would "
                f"reach the HIGH tier but one or more quality gates failed "
                f"(see missing items). Holding pending a higher-quality window."
            ),
            findings=findings,
            missing=missing,
            questions=[],
            source=src,
            icd10=["I45.81"],
            rag=False,
        )

    # ── Tier: MODERATE (intermediate, quality ok) ────────────────────────────
    if subscore >= 1.5 and quality_ok:
        return _result(
            disease="Long QT Syndrome (LQTS)",
            triggered=True,
            confidence=0.55,
            severity=Severity.MODERATE,
            reason=(
                f"ECG sub-score {subscore:.1f} pts (median QTc {qtc_display}) from "
                f"{window.valid_beat_count} valid beats — intermediate Schwartz band, "
                f"sustained over a ~{WINDOW_DURATION_MINUTES}-minute monitoring window "
                f"with all quality gates passed. Proceeding to symptom/family-history "
                f"questioning."
            ),
            findings=findings,
            missing=missing,
            questions=qs,
            source=src,
            icd10=["I45.81"],
            rag=True,
        )

    # ── Tier: PENDING (intermediate sub-score, quality failed) ───────────────
    if subscore >= 1.5 and not quality_ok:
        return _result(
            disease="Long QT Syndrome (LQTS)",
            triggered=False,
            confidence=0.25,
            severity=Severity.LOW,
            reason=(
                f"ECG sub-score {subscore:.1f} pts (median QTc {qtc_display}), but "
                f"one or more quality gates failed for this window (insufficient valid "
                f"beats, elevated SDQT, non-resting, or rhythm instability — see "
                f"missing items). Holding pending a higher-quality window."
            ),
            findings=findings,
            missing=missing,
            questions=[],
            source=src,
            icd10=["I45.81"],
            rag=False,
        )

    # ── Tier: NONE ───────────────────────────────────────────────────────────
    # Temporal findings (trend, SPF) are still included in findings here,
    # so a rising-trend pre-arrhythmic warning surfaces even at NONE tier.
    return _result(
        disease="Long QT Syndrome (LQTS)",
        triggered=False,
        confidence=0.10 if subscore > 0 else 0.05,
        severity=Severity.INFO,
        reason=(
            f"ECG sub-score {subscore:.1f} pts (median QTc {qtc_display}) — below "
            f"the intermediate Schwartz band (1.5 pts). Insufficient ECG evidence to "
            f"warrant symptom questioning in this window. "
            f"IMPORTANT: This result does NOT exclude LQTS — a clinically meaningful "
            f"proportion of genetically confirmed LQTS carriers have a normal-range "
            f"QTc (concealed LQTS). Clinical or family-history suspicion should "
            f"prompt genetic testing regardless of this ECG result."
        ),
        findings=findings,
        missing=missing,
        questions=[],
        source=src,
        icd10=["I45.81"],
        rag=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON newkit T-WAVE ALTERNANS INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
# The Schwartz criterion scored here is MACROSCOPIC T-wave alternans:
# a visible, gross beat-to-beat alternation in T-wave amplitude and/or
# morphology/polarity, occurring in sinus rhythm at the patient's own
# resting rate over a consecutive run of alternating beats.
#
# This is DIFFERENT from microvolt T-wave alternans (MTWA) / Modified
# Moving Average (MMA) TWA testing, which:
#   - detects sub-visible (microvolt-level) fluctuations
#   - requires HR held at ~105-110 bpm via exercise or pacing
#   - needs ~100+ beats at that sustained rate
#   - is a general arrhythmic risk test, NOT the Schwartz LQTS criterion
#
# Before wiring macroscopic_twa_present / macroscopic_twa_consecutive_pairs
# in from newkit, confirm which variant it detects:
#
#   (a) Gross beat-to-beat T-wave amplitude/morphology alternation at the
#       patient's own resting sinus rate
#       → maps directly onto these two fields; use as-is
#
#   (b) Microvolt spectral/MMA TWA requiring an elevated-HR protocol
#       → do NOT feed into this score. Expose as a separate feature/finding
#         (still clinically useful as a general arrhythmic risk marker,
#         just not this Schwartz criterion).
#
# Required fields if (a):
#   macroscopic_twa_present: bool
#   macroscopic_twa_consecutive_pairs: int  (used by TWA_MIN_CONSECUTIVE_PAIRS)
