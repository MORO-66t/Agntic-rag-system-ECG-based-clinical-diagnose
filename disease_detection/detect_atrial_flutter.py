"""
detect_atrial_flutter.py
=========================
Window-level Atrial Flutter (AFL) detection.

Clinical basis:
  - Atrial rate 220-350/min (occasionally 200-400) — Waldo AL, JACC 2000;
    Chugh SS et al., Circulation 2001.
  - Classic 2:1 block -> ventricular rate ~150 bpm with extreme RR regularity
    (more regular than other SVTs at similar rates) — standard ECG teaching.
  - Fixed whole-number AV ratios (2:1, 3:1, 4:1); variable block gives
    "regularly irregular" (group-beating) or irregularly-irregular rhythm —
    Bun SS et al., Eur Heart J 2015 (Afl review).
  - Spectral organization (narrow dominant peak vs. broad AF spectrum) is a
    validated AF/AFL discriminator — Holm M et al., JACC 2005 (spectral peak
    height ratio); Corino VD et al. (spectral entropy ~92% accuracy).
  - Atrial flutter is defined by replacement of discrete P waves by continuous
    F (flutter) waves — mandatory morphological criterion. Discrete P waves
    exclude typical atrial flutter (Atwood JE et al., Circulation 1983;
    Cosío FG, Europace 2017 — flutter vs. sinus vs. atrial tachycardia
    distinction rests on absence of an isoelectric baseline / discrete P).

Honest limitation: single-lead input cannot confirm true sawtooth polarity
(negative in II/III/aVF, positive in V1) required to call typical
CTI-dependent flutter, and cannot exclude atypical flutter. This module
flags AFL as a *rhythm pattern*, not a confirmed electrophysiological
diagnosis — that distinction is preserved in missing_data/symptom_questions
exactly like the rest of this file's rules already do for other conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AFLWindowFeatures:
    window_id: str
    patient_id: str
    recorded_at: datetime
    valid_beat_count: int

    # Ventricular / RR
    ventricular_rate_bpm: Optional[float] = None
    rr_cv: Optional[float] = None                    # std/mean of RR over window
    group_beating_detected: bool = False              # alternating short/long RR

    # Atrial / baseline
    p_wave_present_fraction: float = 1.0
    flutter_baseline_detected_fraction: float = 0.0
    atrial_rate_bpm: Optional[float] = None            # dominant_hz * 60, averaged
    atrial_rate_std_bpm: Optional[float] = None        # stability of the estimate
    organization_index: Optional[float] = None         # 0-1, spectral peak sharpness

    # Window-level spectral features (computed from averaged periodogram)
    window_dominant_hz: Optional[float] = None          # dominant 4-9 Hz from avg spectrum
    window_flutter_ratio: Optional[float] = None         # band_power / total_power from avg spectrum
    window_flutter_baseline_detected: bool = False       # ratio > 0.35 on avg spectrum

    # Derived
    av_block_ratio: Optional[float] = None             # atrial_rate / ventricular_rate
    av_block_ratio_is_integer_like: bool = False

    age: Optional[int] = None
    sex: Optional[str] = None
    known_diagnoses: List[str] = field(default_factory=list)


from disease_detector import DetectionResult, Severity, _result, QUESTIONS  # noqa: E402


_PLAUSIBLE_RATE_RANGE = (60.0, 170.0)
_MIN_VALID_BEATS = 12

# ── Mandatory P-wave-absence gate ─────────────────────────────────────
# Atrial flutter requires replacement of discrete P waves by continuous F
# waves. If P waves are present in more than this fraction of the window,
# flutter is excluded as a diagnosis regardless of any other feature.
# 0.20 allows for a small fraction of mis-detected P-on-T or noisy segments
# without ever firing flutter on a rhythm that genuinely shows P waves.
_P_WAVE_ABSENT_THRESHOLD = 0.20


def detect_atrial_flutter(w: AFLWindowFeatures) -> DetectionResult:
    findings: List[str] = []
    missing: List[str] = []
    score = 0.0

    qs, src = QUESTIONS.get("Atrial Flutter", ([], ""))

    if w.valid_beat_count < _MIN_VALID_BEATS:
        return _result(
            "Atrial Flutter (AFL)", False, 0.0, Severity.INFO,
            f"Insufficient valid beats ({w.valid_beat_count}) for a window-level "
            f"rhythm judgement — need at least {_MIN_VALID_BEATS}.",
            [], ["Longer stable window"], qs, src, ["I48.3", "I48.4"], rag=False,
        )

    # ── MANDATORY GATE: absent P waves required ─────────────────────
    # Atrial flutter is defined by absence of discrete P waves (replaced by
    # continuous F waves). If discrete P waves are present beyond a small
    # tolerance, this detector MUST NOT fire — no other feature can
    # compensate for the presence of P waves.
    if w.p_wave_present_fraction > _P_WAVE_ABSENT_THRESHOLD:
        return _result(
            "Atrial Flutter (AFL)", False, 0.0, Severity.INFO,
            f"Discrete P waves present in {w.p_wave_present_fraction*100:.0f}% of "
            f"the window — atrial flutter requires absent P waves (replaced by "
            f"continuous F waves). Mandatory criterion not met; not firing.",
            [],
            ["Sawtooth flutter-wave morphology in inferior leads (needs multi-lead)"],
            qs, src, ["I48.3", "I48.4"], rag=False,
        )

    # Past this point, P waves are confirmed absent — record the mandatory
    # finding and proceed to the rest of the criteria.
    findings.append(
        f"Mandatory criterion met: discrete P waves absent "
        f"(present in only {w.p_wave_present_fraction*100:.0f}% of window) "
        f"— consistent with replacement by flutter (F) waves"
    )
    score += 0.15   # contributes, but absence alone is not sufficient to fire

    # ── MANDATORY GATE: sustained RR regularity required ──────────
    # Atrial flutter with fixed-ratio AV conduction (2:1, 3:1, 4:1)
    # produces extremely regular RR intervals (CV < 0.08). Without this
    # regularity the rhythm cannot be flutter with consistent conduction;
    # group-beating alone (variable block) is not sufficient to fire the
    # detector because sinus rhythm with blocked PACs, 2nd-degree AV
    # block, or atrial tachycardia with variable block can all produce a
    # similar regularly-irregular pattern without being flutter.
    rr_regular = w.rr_cv is not None and w.rr_cv < 0.10
    if not rr_regular:
        missing.append("Sustained RR regularity to confirm fixed-ratio conduction")
        return _result(
            "Atrial Flutter (AFL)", False, 0.0, Severity.INFO,
            f"RR interval variability (CV={w.rr_cv:.3f}) exceeds the "
            f"fixed-ratio AV conduction threshold (< 0.08). Sustained RR "
            f"regularity required for atrial flutter with consistent "
            f"conduction — mandatory criterion not met; not firing.",
            ["Mandatory P-wave absence: OK"],
            ["Sustained RR regularity to confirm fixed-ratio conduction"],
            qs, src, ["I48.3", "I48.4"], rag=False,
        )

    # ── 1. Ventricular rate in plausible flutter range ──
    rate_in_plausible_range = (
        w.ventricular_rate_bpm is not None
        and _PLAUSIBLE_RATE_RANGE[0] <= w.ventricular_rate_bpm <= _PLAUSIBLE_RATE_RANGE[1]
    )

    # Ventricular rate in [60, 170] is a plausible range gate. To earn full score (+0.35),
    # it must be accompanied by active flutter evidence (flutter baseline, integer AV ratio, or rate >= 130 bpm).
    has_afl_evidence = (
        w.flutter_baseline_detected_fraction >= 0.20 or
        w.av_block_ratio_is_integer_like or
        w.group_beating_detected or
        (w.ventricular_rate_bpm is not None and w.ventricular_rate_bpm >= 130.0)
    )

    if rate_in_plausible_range and has_afl_evidence:
        findings.append(
            f"Ventricular rate {w.ventricular_rate_bpm:.0f} bpm with extreme RR "
            f"regularity (CV {w.rr_cv:.3f}) and flutter evidence — consistent with fixed-ratio conduction"
        )
        score += 0.35
    elif rate_in_plausible_range:
        findings.append(
            f"Ventricular rate {w.ventricular_rate_bpm:.0f} bpm in plausible range, "
            f"but lacks specific flutter baseline/conduction evidence"
        )
        score += 0.10
    else:
        findings.append(f"Ventricular rate {w.ventricular_rate_bpm:.0f} bpm — outside "
                        f"typical flutter range ({_PLAUSIBLE_RATE_RANGE[0]}-"
                        f"{_PLAUSIBLE_RATE_RANGE[1]}) despite RR regularity")
        score += 0.0
        missing.append("Ventricular rate outside typical flutter range")

    # ── 3. Atrial rate in the flutter range ─────────────────────────
    if w.atrial_rate_bpm is not None:
        if 220 <= w.atrial_rate_bpm <= 350:
            findings.append(f"Estimated atrial rate {w.atrial_rate_bpm:.0f} bpm (typical flutter range)")
            score += 0.20
        elif 200 <= w.atrial_rate_bpm < 220 or 350 < w.atrial_rate_bpm <= 400:
            findings.append(f"Estimated atrial rate {w.atrial_rate_bpm:.0f} bpm (atypical flutter range)")
            score += 0.10
        if w.atrial_rate_std_bpm is not None and w.atrial_rate_bpm > 0:
            if (w.atrial_rate_std_bpm / w.atrial_rate_bpm) < 0.08:
                findings.append("Stable atrial-rate estimate across the window")
                score += 0.05
    else:
        missing.append("Atrial (F-wave) rate estimate from baseline spectral analysis")

    # ── 4. Spectral organization — DISABLED ──────────────────────────
    missing.append("Baseline spectral organization index (disabled — insufficient per-beat frequency resolution)")

    # ── 5. Integer AV-block ratio ───────
    if w.av_block_ratio_is_integer_like:
        findings.append(f"Atrial:ventricular ratio ≈ {w.av_block_ratio:.1f}:1 (whole-number conduction ratio)")
        score += 0.15

    # ── 6. Flutter baseline present (supporting) ────────────────────
    if w.flutter_baseline_detected_fraction >= 0.50:
        findings.append("Flutter baseline present across most of the window")
        score += 0.10

    missing.append("Sawtooth F-wave polarity in inferior leads / V1 (requires multi-lead ECG)")
    missing.append("Confirmation of typical (CTI-dependent) vs. atypical flutter circuit")

    score = max(0.0, min(score, 1.0))
    triggered = score >= 0.45
    conf = min(score, 0.90)
    sev = (
        Severity.CRITICAL if triggered and w.ventricular_rate_bpm is not None and w.ventricular_rate_bpm >= 200 else
        Severity.HIGH if triggered else
        Severity.INFO
    )
    reason = (
        f"Regular/organized rhythm at a flutter-typical rate with "
        f"{'an atrial:ventricular ratio near ' + str(round(w.av_block_ratio, 1)) + ':1' if w.av_block_ratio_is_integer_like and w.av_block_ratio is not None else 'flutter-baseline evidence'} "
        f"and absent discrete P waves — atrial flutter pattern suspected. "
        f"Confirm with multi-lead review."
        if triggered else
        "Atrial flutter pattern criteria not met on this window "
        "(P waves confirmed absent, but other supportive criteria insufficient)."
    )

    return _result(
        "Atrial Flutter (AFL)", triggered, conf, sev, reason,
        findings, missing, qs, src, ["I48.3", "I48.4"], rag=triggered,
    )