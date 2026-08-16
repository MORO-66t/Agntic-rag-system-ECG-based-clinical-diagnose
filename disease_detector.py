"""
disease_detector.py
===================
ECG-based cardiac disease detection module.

Path : D:\\DEPI Project\\Graduation project\\disease_detector.py

Architecture
------------
Each disease is a standalone DetectionRule that:
  1. Receives aggregated ECG features from the clinical branch
  2. Returns a DetectionResult (triggered / not triggered, confidence,
     missing data flags, symptom questions to ask)

The module plugs into ECGPipeline at Step 4 (after temporal analysis)
or can be called stand-alone on any feature dict — no pipeline required.

Usage – inside pipeline (agent trigger)
----------------------------------------
    from disease_detector import DiseaseDetector
    detector = DiseaseDetector()
    results  = detector.evaluate(beat_features, window_features)

Usage – stand-alone (ECG only, no pipeline)
--------------------------------------------
    from disease_detector import DiseaseDetector, ECGFeatures
    features = ECGFeatures(
        rr_mean_ms        = 750.0,
        rr_std_ms         = 180.0,
        rr_irregular      = True,
        p_wave_present    = False,
        pr_interval_ms    = None,
        qrs_duration_ms   = 88.0,
        qt_interval_ms    = 380.0,
        qtc_ms            = 420.0,
        st_deviation_mv   = 0.05,
        t_wave_inverted   = False,
        heart_rate_bpm    = 110.0,
    )
    detector  = DiseaseDetector()
    results   = detector.evaluate_ecg_only(features)
    for r in results:
        if r.triggered:
            print(r.disease, r.confidence, r.reason)

Data sources for each rule
--------------------------
Every threshold in this file is traceable to a published source:
  • AHA/ACC 2022 ECG Interpretation Guidelines
  • ESC 2021 Arrhythmia Guidelines
  • Brugada Consensus Report (Heart Rhythm 2022)
  • ACLS ECG Recognition Criteria (AHA)
  • UpToDate ECG criteria per condition (referenced inline)
  • PhysioNet MIT-BIH annotation guide for rhythm labels

Symptom question banks
-----------------------
Each rule carries a list of clinical questions.  These are drawn from:
  • AHA "Could It Be My Heart?" patient screening tools
  • ACC ClinicalConnect condition-specific intake questions
  • ESC Patient Education ECG condition checklists
  • NICE CKS (UK) condition-specific question sets
  • UpToDate "Approach to the patient" sections per condition
Source URLs are embedded as comments per question group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class Confidence(float, Enum):
    """Qualitative confidence levels mapped to float scores."""
    LOW    = 0.35
    MEDIUM = 0.60
    HIGH   = 0.85
    DEFINITIVE = 0.97


class Severity(str, Enum):
    INFO     = "info"
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class ECGFeatures:
    """
    Flat feature bag consumed by every detection rule.

    All interval fields are in milliseconds; amplitude fields in millivolts.
    Fields that could not be measured should be set to None (not 0 / NaN).

    Mapping from pipeline feature_engineering output:
        rr_interval         → rr_mean_ms, rr_std_ms, rr_irregular
        pr_interval_ms_gt   → pr_interval_ms
        qrs_duration_ms     → qrs_duration_ms
        qt_interval_ms      → qt_interval_ms  (compute qtc_ms separately)
        st_segment_mv       → st_deviation_mv (positive = elevation)
        t_wave_amplitude    → t_wave_amplitude_mv
        p_wave_amplitude    → p_wave_amplitude_mv (None if p absent)
        heart_rate_bpm      → heart_rate_bpm
        predicted_label     → cnn_label
    """
    # ── Rhythm ──────────────────────────────────────────────────
    rr_mean_ms:          Optional[float] = None   # mean RR over window
    rr_std_ms:           Optional[float] = None   # beat-to-beat variability
    rr_irregular:        bool            = False  # True if CoV > 0.10
    heart_rate_bpm:      Optional[float] = None

    # ── P wave ───────────────────────────────────────────────────
    p_wave_present:      bool            = True
    p_wave_amplitude_mv: Optional[float] = None   # None = absent
    p_duration_ms:       Optional[float] = None
    p_wave_peaked:       bool            = False   # > 2.5 mm in lead II
    p_wave_broad:        bool            = False   # > 120 ms

    # ── PR interval ──────────────────────────────────────────────
    pr_interval_ms:      Optional[float] = None   # None = unmeasurable
    pr_progressively_lengthening: bool   = False  # Wenckebach pattern

    # ── QRS ──────────────────────────────────────────────────────
    qrs_duration_ms:     Optional[float] = None   # normal < 120
    qrs_axis_degrees:    Optional[float] = None   # normal -30 to +90
    lbbb_pattern:        bool            = False   # LBBB morphology
    rbbb_pattern:        bool            = False   # RBBB morphology
    delta_wave_present:  bool            = False   # WPW pre-excitation
    epsilon_wave_present:bool            = False   # ARVC
    r_wave_amplitude_mv: Optional[float] = None
    q_wave_deep:         bool            = False   # pathological Q
    low_voltage:         bool            = False   # < 5mm limb / < 10mm precordial
    electrical_alternans:bool            = False   # beat-to-beat QRS amplitude alternation

    # ── ST segment ───────────────────────────────────────────────
    st_deviation_mv:     Optional[float] = None   # +elevation / -depression
    st_slope:            Optional[str]   = None   # "flat" | "downsloping" | "upsloping"

    # ── T wave ───────────────────────────────────────────────────
    t_wave_amplitude_mv: Optional[float] = None
    t_wave_inverted:     bool            = False
    t_wave_peaked:       bool            = False   # hyperacute T
    t_wave_biphasic:     bool            = False

    # ── QT / repolarisation ──────────────────────────────────────
    qt_interval_ms:      Optional[float] = None
    qtc_ms:              Optional[float] = None   # Fridericia preferred
    u_wave_prominent:    bool            = False   # > 1 mm V2-V3
    flutter_baseline_detected: bool      = False   # sawtooth/f-wave baseline evidence

    # ── Window-level derived features ───────────────────────────
    af_probability:      float           = 0.0    # 0-1 from rhythm classifier
    vt_detected:         bool            = False
    vf_detected:         bool            = False
    paced_rhythm:        bool            = False
    cnn_label:           Optional[str]   = None   # CNN predicted beat class
    window_beat_count:   int             = 0

    # ── Patient context (from metadata, optional) ────────────────
    age:                 Optional[int]   = None
    sex:                 Optional[str]   = None   # "M" | "F"
    known_diagnoses:     List[str]       = field(default_factory=list)


@dataclass
class DetectionResult:
    """Output of a single disease detection rule."""
    disease:      str
    triggered:    bool
    confidence:   float          # 0.0 – 1.0
    severity:     Severity
    reason:       str            # human-readable explanation
    ecg_findings: List[str]      # specific ECG criteria met
    missing_data: List[str]      # features needed but not available
    symptom_questions: List[str] # questions to ask the patient
    symptom_source:    str       # reference for the question set
    rag_trigger:  bool           # should this fire the RAG agent?
    icd10_codes:  List[str]      # relevant ICD-10 codes


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Optional[float], default: float = 0.0) -> float:
    return value if value is not None else default


def _qtc_fridericia(qt_ms: float, rr_ms: float) -> float:
    """Fridericia QTc = QT / RR^(1/3).  RR in ms."""
    if rr_ms <= 0:
        return qt_ms
    rr_s = rr_ms / 1000.0
    return qt_ms / (rr_s ** (1.0 / 3.0))


def _result(
    disease:   str,
    triggered: bool,
    confidence: float,
    severity:  Severity,
    reason:    str,
    findings:  List[str],
    missing:   List[str],
    questions: List[str],
    source:    str,
    icd10:     List[str],
    rag:       bool = True,
) -> DetectionResult:
    return DetectionResult(
        disease            = disease,
        triggered          = triggered,
        confidence         = confidence,
        severity           = severity,
        reason             = reason,
        ecg_findings       = findings,
        missing_data       = missing,
        symptom_questions  = questions,
        symptom_source     = source,
        rag_trigger        = triggered and rag,
        icd10_codes        = icd10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM QUESTION BANKS
# Source: AHA/ACC/ESC patient-facing intake tools and clinical intake guides
# ─────────────────────────────────────────────────────────────────────────────

_Q_SOURCE_AHA_AF    = "AHA — Atrial Fibrillation Patient Page (ahajournals.org)"
_Q_SOURCE_AHA_ACS   = "AHA — Heart Attack Warning Signs (heart.org)"
_Q_SOURCE_AHA_HF    = "AHA — Heart Failure Patient Page (heart.org)"
_Q_SOURCE_ESC_BRUGADA = "ESC 2022 Brugada Syndrome Patient Q&A (escardio.org)"
_Q_SOURCE_ESC_LQTS  = "ESC/ACC — Long QT Syndrome Patient Guide"
_Q_SOURCE_ESC_HCM   = "ESC 2023 Cardiomyopathy Guidelines — History Taking"
_Q_SOURCE_AHA_PE    = "AHA — Pulmonary Embolism Patient Page (heart.org)"
_Q_SOURCE_ESC_PERI  = "ESC 2015 Pericarditis Guidelines — Clinical Intake"
_Q_SOURCE_NICE_HBP  = "NICE CKS — Hypertension History (cks.nice.org.uk)"
_Q_SOURCE_ESC_PH    = "ESC 2022 Pulmonary Hypertension Guidelines"
_Q_SOURCE_ACC_WPW   = "ACC/AHA WPW — Electrophysiology Study Indications"
_Q_SOURCE_ESC_ARVC  = "ESC 2023 Cardiomyopathy Guidelines — ARVC History"
_Q_SOURCE_ESC_TAMPO = "ESC Cardiac Tamponade Clinical Assessment"
_Q_SOURCE_ESC_AMY   = "ESC 2021 Amyloidosis Working Group"


QUESTIONS: Dict[str, tuple[List[str], str]] = {

    "Atrial Fibrillation": ([
        "Do you feel your heart racing, fluttering, or beating irregularly?",
        "Does the irregular heartbeat come and go, or is it constant?",
        "Have you experienced dizziness, near-fainting, or actual fainting?",
        "Do you feel unusually short of breath during normal activities?",
        "Have you had a stroke or mini-stroke (TIA) in the past?",
        "Do you drink alcohol regularly? If so, how much per week?",
        "Do you have a history of high blood pressure, diabetes, or heart failure?",
        "Are you currently taking any blood thinners (anticoagulants)?",
    ], _Q_SOURCE_AHA_AF),

    "Atrial Flutter": ([
        "Do you feel a rapid, regular fluttering in your chest?",
        "Does the sensation come on suddenly and stop suddenly?",
        "Have you had any lightheadedness or near-fainting spells?",
        "Do you have a history of atrial fibrillation or prior ablation?",
        "Do you have thyroid disease, lung disease, or prior heart surgery?",
    ], _Q_SOURCE_AHA_AF),

    "Acute Coronary Syndrome / STEMI": ([
        "Do you have chest pain, pressure, tightness, or squeezing right now?",
        "Does the pain radiate to your arm, jaw, neck, or back?",
        "Are you sweating, nauseated, or short of breath along with the pain?",
        "When exactly did the chest discomfort begin?",
        "Have you had a heart attack or stent/bypass surgery before?",
        "Do you have diabetes, high blood pressure, or high cholesterol?",
        "Do you smoke or have you smoked in the past?",
    ], _Q_SOURCE_AHA_ACS),

    "Heart Failure": ([
        "Are you short of breath when walking, climbing stairs, or lying flat?",
        "Do you wake at night gasping for air (paroxysmal nocturnal dyspnoea)?",
        "Have your ankles, feet, or legs been swelling recently?",
        "Have you gained more than 2 kg in the past week?",
        "Do you feel extremely tired even with minimal exertion?",
        "How many pillows do you sleep on to breathe comfortably?",
        "Have you had any recent change in medications or diet?",
    ], _Q_SOURCE_AHA_HF),

    "Brugada Syndrome": ([
        "Have you ever fainted or had a seizure during sleep or rest?",
        "Has a family member died suddenly and unexpectedly, especially before age 45?",
        "Have you been told you have a 'right bundle branch block' on a previous ECG?",
        "Do your symptoms get worse with fever?",
        "Are you taking any sodium channel-blocking medications?",
        "Do you have a history of unexplained palpitations at rest?",
    ], _Q_SOURCE_ESC_BRUGADA),

    "Long QT Syndrome": ([
        "Have you ever fainted during exercise, excitement, or a sudden noise?",
        "Have you had a seizure that was not explained by epilepsy?",
        "Has anyone in your family died suddenly under age 40?",
        "Are you taking any medications (list all, including over-the-counter)?",
        "Have your potassium, magnesium, or calcium levels been checked recently?",
        "Do you have a history of an eating disorder?",
    ], _Q_SOURCE_ESC_LQTS),

    "Hypertrophic Cardiomyopathy": ([
        "Do you get short of breath, chest pain, or dizzy during exercise?",
        "Have you ever fainted or nearly fainted during physical activity?",
        "Has anyone in your family died suddenly from a heart condition?",
        "Have you been told your heart muscle is thickened on an echocardiogram?",
        "Do you participate in competitive sports or intense exercise?",
    ], _Q_SOURCE_ESC_HCM),

    "Arrhythmogenic Cardiomyopathy": ([
        "Do you have palpitations, dizziness, or fainting during exercise?",
        "Has a family member had sudden cardiac death or a heart transplant?",
        "Have you been told you have frequent 'ectopic beats' from the right side?",
        "Do you participate in endurance sports (marathon, cycling, football)?",
        "Has any family member been diagnosed with ARVC or cardiomyopathy?",
    ], _Q_SOURCE_ESC_ARVC),

    "Pulmonary Embolism": ([
        "Did you develop sudden shortness of breath within the past few hours or days?",
        "Do you have calf pain, redness, or swelling suggesting a blood clot in the leg?",
        "Have you had a recent long flight, car journey, or period of bed rest?",
        "Have you had surgery or a hospital admission in the past 4 weeks?",
        "Are you taking the contraceptive pill, HRT, or other oestrogen-containing drugs?",
        "Do you have cancer, or are you receiving chemotherapy?",
        "Have you coughed up any blood?",
    ], _Q_SOURCE_AHA_PE),

    "Pericarditis": ([
        "Do you have sharp chest pain that is worse when you lie flat?",
        "Does sitting forward or leaning forward reduce the chest pain?",
        "Have you recently had a viral illness, fever, or flu-like symptoms?",
        "Do you have an autoimmune condition such as lupus or rheumatoid arthritis?",
        "Have you had pericarditis before?",
        "Have you had recent heart surgery or a heart attack?",
    ], _Q_SOURCE_ESC_PERI),

    "Hypertension / Left Ventricular Hypertrophy": ([
        "What is your usual blood pressure reading?",
        "Have you been diagnosed with high blood pressure?",
        "Are you taking any blood pressure medications?",
        "Do you experience headaches, visual changes, or nosebleeds?",
        "Do you have a family history of high blood pressure or stroke?",
        "How much salt do you consume daily?",
    ], _Q_SOURCE_NICE_HBP),

    "Pulmonary Hypertension": ([
        "Are you increasingly short of breath during activities that used to be easy?",
        "Have you fainted or felt near-fainting during exertion?",
        "Do your legs or ankles swell, especially later in the day?",
        "Do you have an autoimmune disease, HIV, or liver disease?",
        "Have you been told your pulmonary (lung) artery pressure is elevated?",
        "Are you taking or have you taken appetite suppressants?",
    ], _Q_SOURCE_ESC_PH),

    "WPW / Pre-excitation": ([
        "Do you have sudden-onset, rapid palpitations that stop abruptly?",
        "Have you ever fainted during palpitations?",
        "Have you been told you have a 'short PR interval' or 'delta wave' before?",
        "Does straining or bearing down (Valsalva) slow the palpitations?",
        "Do you engage in competitive sports?",
    ], _Q_SOURCE_ACC_WPW),

    "Ventricular Tachycardia": ([
        "Are you currently experiencing palpitations, dizziness, or chest pain?",
        "Have you ever fainted or nearly fainted during an episode of rapid heartbeat?",
        "Have you been diagnosed with any heart condition before?",
        "Do you have a family history of sudden cardiac death or inherited heart conditions?",
    ], "ACC/AHA/HRS 2017 VA Guideline — Patient Assessment"),

    "Cardiac Tamponade": ([
        "Do you feel short of breath even at rest?",
        "Has your heart rate been faster than usual lately?",
        "Have you been told you have fluid around your heart?",
        "Do you have cancer, kidney failure, or a recent heart procedure?",
        "Do you feel dizzy or faint when you stand up?",
    ], _Q_SOURCE_ESC_TAMPO),

    "Amyloidosis": ([
        "Do you have carpal tunnel syndrome (wrist pain/numbness)?",
        "Have you experienced peripheral neuropathy (tingling hands/feet)?",
        "Is your heart failure getting worse despite standard medications?",
        "Have you had unexplained weight loss?",
        "Is there a family history of amyloidosis or heart failure at an older age?",
    ], _Q_SOURCE_ESC_AMY),
}


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION RULES — one function per disease
# Each function returns a DetectionResult.
# Threshold references are in inline comments.
# ─────────────────────────────────────────────────────────────────────────────







# def detect_stemi(f: ECGFeatures) -> DetectionResult:
#     """
#     STEMI (AHA/ACC 4th Universal Definition of MI):
#       - ST elevation ≥ 1 mm in ≥2 contiguous limb leads, OR
#       - ST elevation ≥ 2 mm in ≥2 contiguous precordial leads, OR
#       - New LBBB
#     Uses single-beat st_deviation_mv; full multi-lead requires 12-lead pass.
#     """
#     findings, missing = [], []
#     score = 0.0
#     st = _safe(f.st_deviation_mv, 0.0)

#     if st >= 0.2:          # ≥2mm — strong criterion
#         findings.append(f"ST elevation {st*10:.1f} mm (≥2 mm criterion)")
#         score += 0.70
#     elif st >= 0.1:        # ≥1mm — moderate criterion
#         findings.append(f"ST elevation {st*10:.1f} mm (≥1 mm criterion)")
#         score += 0.45

#     if f.t_wave_peaked:
#         findings.append("Hyperacute T waves (earliest STEMI sign)")
#         score += 0.20

#     if f.lbbb_pattern and st >= 0.05:
#         findings.append("New LBBB pattern — STEMI equivalent")
#         score += 0.30

#     if f.q_wave_deep:
#         findings.append("Pathological Q waves (evolving infarction)")
#         score += 0.15

#     if not f.st_deviation_mv:
#         missing.append("ST deviation measurement")
#     missing.append("Multi-lead localisation (inferior/anterior/lateral/posterior)")

#     triggered = score >= 0.55
#     conf = min(score, 0.97)
#     sev = Severity.CRITICAL if score >= 0.60 else (Severity.HIGH if triggered else Severity.INFO)
#     qs, src = QUESTIONS["Acute Coronary Syndrome / STEMI"]
#     reason = (
#         f"ST elevation {st*10:.1f} mm with hyperacute T waves — STEMI until proven otherwise. CALL IMMEDIATELY."
#         if triggered else "STEMI criteria not met on current beat."
#     )
#     return _result("Acute Coronary Syndrome / STEMI", triggered, conf,
#                    sev, reason, findings, missing, qs, src,
#                    ["I21.9", "I21.0", "I21.1", "I21.2"])


# def detect_nstemi_ua(f: ECGFeatures) -> DetectionResult:
#     """
#     NSTEMI / Unstable Angina (AHA/ACC):
#       - ST depression ≥ 0.5 mm, OR
#       - T-wave inversions in ischaemic pattern, OR
#       - Downsloping ST depression
#     """
#     findings, missing = [], []
#     score = 0.0
#     st = _safe(f.st_deviation_mv, 0.0)

#     if st <= -0.05:        # ≥0.5 mm depression
#         findings.append(f"ST depression {abs(st)*10:.1f} mm")
#         score += 0.50
#     if f.t_wave_inverted:
#         findings.append("T-wave inversions in ischaemic distribution")
#         score += 0.30
#     if f.st_slope == "downsloping":
#         findings.append("Downsloping ST depression (high ischaemic specificity)")
#         score += 0.20

#     missing.append("Troponin result (essential for NSTEMI vs UA distinction)")
#     missing.append("Multi-lead ST distribution")

#     triggered = score >= 0.45
#     conf = min(score, 0.85)
#     qs, src = QUESTIONS["Acute Coronary Syndrome / STEMI"]
#     reason = (
#         f"ST depression {abs(st)*10:.1f} mm {'with T-wave inversions ' if f.t_wave_inverted else ''}"
#         f"— NSTEMI/UA suspected. Troponin required."
#         if triggered else "NSTEMI/UA criteria not met."
#     )
#     return _result("NSTEMI / Unstable Angina", triggered, conf,
#                    Severity.CRITICAL if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I21.4", "I20.0"])


# def detect_heart_failure(f: ECGFeatures) -> DetectionResult:
#     """
#     Heart Failure ECG markers (AHA/ESC HF Guidelines 2021):
#       - LBBB (CRT candidate criterion: QRS ≥ 130 ms + LBBB)
#       - LVH voltage criteria
#       - AF in context of HF
#       - Low voltage (dilated non-ischaemic)
#     """
#     findings, missing = [], []
#     score = 0.0
#     qrs = _safe(f.qrs_duration_ms, 0.0)

#     if f.lbbb_pattern and qrs >= 130:
#         findings.append(f"LBBB with QRS {qrs:.0f} ms — CRT candidacy criterion")
#         score += 0.50
#     elif f.lbbb_pattern:
#         findings.append(f"LBBB pattern (QRS {qrs:.0f} ms)")
#         score += 0.30

#     if f.low_voltage:
#         findings.append("Low voltage — dilated or infiltrative cardiomyopathy")
#         score += 0.25

#     if f.rr_irregular and not f.p_wave_present:
#         findings.append("AF — common precipitant/consequence of HF")
#         score += 0.20

#     missing.append("Echo LVEF (essential for HFrEF vs HFpEF distinction)")
#     missing.append("BNP/NT-proBNP")

#     triggered = score >= 0.45
#     conf = min(score, 0.80)
#     qs, src = QUESTIONS["Heart Failure"]
#     reason = (
#         f"LBBB {'+ QRS >130ms ' if qrs >= 130 else ''}pattern with {'low voltage and ' if f.low_voltage else ''}"
#         f"possible HF ECG features."
#         if triggered else "Heart failure ECG criteria not met."
#     )
#     return _result("Heart Failure", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I50.9", "I50.2", "I50.3"])


# def detect_brugada(f: ECGFeatures) -> DetectionResult:
#     """
#     Brugada Syndrome (Brugada Consensus Report, Heart Rhythm 2022):
#       - Type 1: Coved ST elevation ≥ 2mm with J-point elevation in V1-V2,
#         descending into negative T wave — spontaneous or provoked
#       - Type 2 / 3: Saddle-back pattern (not diagnostic alone)
#     Note: definitive diagnosis requires V1-V2 lead-specific analysis.
#     """
#     findings, missing = [], []
#     score = 0.0
#     st = _safe(f.st_deviation_mv, 0.0)

#     if st >= 0.2 and f.rbbb_pattern:
#         findings.append("ST elevation ≥2mm with RBBB morphology in V1-V2 (Type 1 pattern suspected)")
#         score += 0.70
#     elif f.rbbb_pattern and st >= 0.05:
#         findings.append("RBBB morphology with mild ST elevation — possible Type 2 Brugada")
#         score += 0.30

#     missing.extend([
#         "V1-V2 specific lead morphology (coved vs saddle-back)",
#         "T-wave polarity in V1-V2 (must be negative for Type 1)",
#         "Sodium channel blocker provocation test result",
#         "Family history of SCD / genetic testing (SCN5A)",
#     ])

#     triggered = score >= 0.55
#     conf = min(score, 0.85)
#     qs, src = QUESTIONS["Brugada Syndrome"]
#     reason = (
#         f"Coved ST elevation with RBBB morphology — Brugada Type 1 pattern. "
#         f"Provocation test and genetics recommended."
#         if triggered else "Brugada criteria not met (V1-V2 morphology detail required)."
#     )
#     return _result("Brugada Syndrome", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I49.8"])


# # Old detect_long_qt removed - now using the new version from disease_detection.detect_long_qt
# # which takes ECGWindowFeatures instead of ECGFeatures for temporal analysis


# def detect_wpw(f: ECGFeatures) -> DetectionResult:
#     """
#     WPW / Pre-excitation (ACC/AHA WPW Guidelines):
#       - Short PR < 120 ms + delta wave + wide QRS
#       - QRS 110-140 ms with slurred upstroke
#     """
#     findings, missing = [], []
#     score = 0.0
#     pr = _safe(f.pr_interval_ms, 999.0)
#     qrs = _safe(f.qrs_duration_ms, 0.0)

#     if f.delta_wave_present:
#         findings.append("Delta wave detected (slurred QRS onset)")
#         score += 0.50

#     if pr < 120 and f.pr_interval_ms is not None:
#         findings.append(f"Short PR interval {pr:.0f} ms (< 120 ms)")
#         score += 0.30

#     if 110 <= qrs <= 140:
#         findings.append(f"Widened QRS {qrs:.0f} ms (typical WPW range)")
#         score += 0.20

#     missing.append("Pathway localisation (12-lead delta wave polarity analysis)")

#     triggered = score >= 0.55
#     conf = min(score, 0.90)
#     qs, src = QUESTIONS["WPW / Pre-excitation"]
#     reason = (
#         f"Delta wave + short PR {pr:.0f} ms + wide QRS {qrs:.0f} ms — WPW triad."
#         if triggered else "WPW criteria not met."
#     )
#     return _result("WPW / Pre-excitation Syndrome", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I45.6"])


# def detect_hcm(f: ECGFeatures) -> DetectionResult:
#     """
#     Hypertrophic Cardiomyopathy (ESC 2023 HCM Guidelines):
#       - Extreme LVH voltage (high R or S amplitude)
#       - Deep narrow Q waves in lateral leads (septal hypertrophy)
#       - T-wave inversions (apical HCM — Yamaguchi pattern in V3-V6)
#       - LA enlargement
#     """
#     findings, missing = [], []
#     score = 0.0
#     r_amp = _safe(f.r_wave_amplitude_mv, 0.0)

#     if r_amp >= 3.5:   # > 35mm Sokolow equivalent
#         findings.append(f"High R-wave amplitude {r_amp*10:.1f} mm — LVH criteria")
#         score += 0.35

#     if f.q_wave_deep:
#         findings.append("Deep narrow Q waves (septal hypertrophy pattern)")
#         score += 0.35

#     if f.t_wave_inverted:
#         findings.append("T-wave inversions (apical HCM pattern)")
#         score += 0.25

#     if f.p_wave_broad:
#         findings.append("Broad P wave — left atrial enlargement")
#         score += 0.10

#     missing.extend([
#         "Echocardiogram — maximum wall thickness measurement",
#         "Genetic testing (sarcomere mutations)",
#         "Family history of HCM or SCD",
#     ])

#     triggered = score >= 0.50
#     conf = min(score, 0.80)
#     qs, src = QUESTIONS["Hypertrophic Cardiomyopathy"]
#     reason = (
#         f"LVH voltage + septal Q waves + T inversions — HCM pattern suspected."
#         if triggered else "HCM ECG criteria not met."
#     )
#     return _result("Hypertrophic Cardiomyopathy (HCM)", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I42.1", "I42.2"])


# def detect_arvc(f: ECGFeatures) -> DetectionResult:
#     """
#     Arrhythmogenic Cardiomyopathy / ARVC (ESC 2023 Task Force Criteria):
#       - Epsilon wave in V1-V3
#       - T-wave inversions V1-V4 (without RBBB)
#       - RBBB morphology VT (LBBB-morphology from RV)
#       - Terminal QRS activation > 55 ms in V1-V3
#     """
#     findings, missing = [], []
#     score = 0.0

#     if f.epsilon_wave_present:
#         findings.append("Epsilon wave detected (pathognomonic for ARVC)")
#         score += 0.65

#     if f.t_wave_inverted and f.rbbb_pattern:
#         findings.append("T-wave inversions with RBBB pattern — ARVC criterion")
#         score += 0.35
#     elif f.t_wave_inverted:
#         findings.append("T-wave inversions (V1-V4 without RBBB — major ARVC criterion)")
#         score += 0.45

#     missing.extend([
#         "Signal-averaged ECG (late potentials — ARVC criterion)",
#         "Cardiac MRI (fatty infiltration / fibrosis)",
#         "Family history / genetic testing (DSP, PKP2, DSG2)",
#         "VT morphology (LBBB morphology = RV origin)",
#     ])

#     triggered = score >= 0.45
#     conf = min(score, 0.85)
#     qs, src = QUESTIONS["Arrhythmogenic Cardiomyopathy"]
#     reason = (
#         f"Epsilon wave {'and T-wave inversions ' if f.t_wave_inverted else ''}"
#         f"— ARVC major criteria met. Cardiac MRI and genetics required."
#         if triggered else "ARVC criteria not met on current ECG."
#     )
#     print(f"ARVC: {triggered}, {conf}, {reason}, {findings}, {missing}, {qs}, {src}")
#     return _result("Arrhythmogenic Cardiomyopathy (ARVC)", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I42.8"])


# def detect_pulmonary_embolism(f: ECGFeatures) -> DetectionResult:
#     """
#     Pulmonary Embolism ECG (AHA PE Guidelines 2011; ESC 2019):
#       - Sinus tachycardia (most sensitive sign — present ~44%)
#       - S1Q3T3 pattern (S in I, Q in III, T inversion in III)
#       - T-wave inversions V1-V4 (RV strain)
#       - New RBBB
#       - Right axis deviation
#     S1Q3T3 sensitivity only ~20% — sinus tachycardia alone is more sensitive.
#     """
#     findings, missing = [], []
#     score = 0.0
#     hr = _safe(f.heart_rate_bpm, 0.0)
#     axis = _safe(f.qrs_axis_degrees, 0.0)

#     if hr > 100:
#         findings.append(f"Sinus tachycardia {hr:.0f} bpm (most sensitive PE sign)")
#         score += 0.25

#     if f.rbbb_pattern:
#         findings.append("New RBBB — right heart pressure overload")
#         score += 0.30

#     if f.t_wave_inverted:
#         findings.append("T-wave inversions (V1-V4 RV strain pattern)")
#         score += 0.25

#     if axis > 90:
#         findings.append(f"Right axis deviation {axis:.0f}° — RV overload")
#         score += 0.20

#     missing.extend([
#         "S1Q3T3 pattern (requires leads I, III individually)",
#         "D-dimer and clinical Wells score",
#         "CT pulmonary angiography",
#     ])

#     triggered = score >= 0.50
#     conf = min(score, 0.75)   # PE ECG alone has low specificity
#     qs, src = QUESTIONS["Pulmonary Embolism"]
#     reason = (
#         f"Sinus tachycardia + RBBB + RV strain T inversions — PE pattern. "
#         f"Urgent CT-PA and D-dimer required."
#         if triggered else "PE ECG criteria not met (ECG alone cannot exclude PE)."
#     )
#     return _result("Pulmonary Embolism (PE)", triggered, conf,
#                    Severity.CRITICAL if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I26.99", "I26.09"])


# def detect_pericarditis(f: ECGFeatures) -> DetectionResult:
#     """
#     Pericarditis (ESC 2015 Pericarditis Guidelines):
#       - Diffuse concave (saddle-shaped) ST elevation
#       - PR depression
#       - Absence of reciprocal ST changes (unlike STEMI)
#       - Spodick sign (downsloping TP segment)
#     Single-lead limitation: diffuse nature requires multi-lead.
#     """
#     findings, missing = [], []
#     score = 0.0
#     st = _safe(f.st_deviation_mv, 0.0)

#     if 0.05 <= st <= 0.50 and not f.q_wave_deep:
#         findings.append(f"ST elevation {st*10:.1f} mm without Q waves (pericarditis pattern)")
#         score += 0.45

#     if f.t_wave_amplitude_mv and f.t_wave_amplitude_mv > 0.15:
#         findings.append("Upright T waves with ST elevation (Stage 1 pericarditis)")
#         score += 0.20

#     missing.extend([
#         "PR depression (key differentiating sign — requires full 12-lead)",
#         "Absence of reciprocal ST depression (distinguishes from STEMI)",
#         "CRP / ESR (inflammatory markers)",
#         "Echocardiogram (pericardial effusion assessment)",
#     ])

#     triggered = score >= 0.40
#     conf = min(score, 0.70)   # multi-lead required for high confidence
#     qs, src = QUESTIONS["Pericarditis"]
#     reason = (
#         f"ST elevation {st*10:.1f} mm without Q waves — pericarditis possible. "
#         f"Full 12-lead and inflammatory markers required."
#         if triggered else "Pericarditis criteria not met on single-lead ECG."
#     )
#     return _result("Pericarditis", triggered, conf,
#                    Severity.MODERATE if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I30.9", "I30.0"])


# def detect_cardiac_tamponade(f: ECGFeatures) -> DetectionResult:
#     """
#     Cardiac Tamponade ECG (ESC / ACC):
#       - Electrical alternans (beat-to-beat QRS amplitude variation)
#       - Low voltage (< 5mm all limb leads)
#       - Sinus tachycardia
#     Electrical alternans + low voltage = highly specific.
#     """
#     findings, missing = [], []
#     score = 0.0
#     hr = _safe(f.heart_rate_bpm, 0.0)

#     if f.electrical_alternans:
#         findings.append("Electrical alternans (pathognomonic for large effusion + tamponade)")
#         score += 0.65

#     if f.low_voltage:
#         findings.append("Low voltage (fluid insulation of electrical signal)")
#         score += 0.30

#     if hr > 100:
#         findings.append(f"Sinus tachycardia {hr:.0f} bpm (compensatory)")
#         score += 0.15

#     missing.extend([
#         "Echocardiogram (urgent — gold standard)",
#         "Clinical triad: hypotension, JVD, muffled heart sounds (Beck's triad)",
#         "Pulsus paradoxus > 10 mmHg",
#     ])

#     triggered = score >= 0.55
#     conf = min(score, 0.92)
#     sev = Severity.CRITICAL if f.electrical_alternans else (
#         Severity.HIGH if triggered else Severity.INFO
#     )
#     qs, src = QUESTIONS["Cardiac Tamponade"]
#     reason = (
#         f"Electrical alternans + low voltage — cardiac tamponade. EMERGENCY ECHO REQUIRED."
#         if triggered else "Tamponade criteria not met."
#     )
#     return _result("Cardiac Tamponade", triggered, conf, sev,
#                    reason, findings, missing, qs, src, ["I31.9", "I31.2"])


# def detect_lvh_hypertension(f: ECGFeatures) -> DetectionResult:
#     """
#     LVH / Hypertension (Sokolow-Lyon / Cornell criteria — AHA):
#       - Sokolow-Lyon: SV1 + RV5/V6 > 35mm (3.5 mV)
#       - Cornell: R_aVL + S_V3 > 28mm men / 20mm women
#       - Strain pattern: ST depression + T-wave inversions V4-V6
#     Using R-wave amplitude as proxy (single lead).
#     """
#     findings, missing = [], []
#     score = 0.0
#     r_amp = _safe(f.r_wave_amplitude_mv, 0.0)

#     if r_amp >= 3.5:
#         findings.append(f"R-wave amplitude {r_amp*10:.1f} mm — Sokolow-Lyon LVH range")
#         score += 0.45

#     if f.t_wave_inverted and r_amp > 2.0:
#         findings.append("Strain pattern — ST depression with T-wave inversions")
#         score += 0.30

#     if f.p_wave_broad:
#         findings.append("Broad P wave — left atrial enlargement from hypertension")
#         score += 0.20

#     missing.extend([
#         "Multi-lead voltage (SV1 + RV5/V6 for Sokolow-Lyon)",
#         "Cornell voltage (R aVL + S V3)",
#         "Blood pressure measurement",
#     ])

#     triggered = score >= 0.40
#     conf = min(score, 0.80)
#     qs, src = QUESTIONS["Hypertension / Left Ventricular Hypertrophy"]
#     reason = (
#         f"LVH voltage {r_amp*10:.1f} mm with {'strain pattern ' if f.t_wave_inverted else ''}"
#         f"— hypertensive heart disease suspected."
#         if triggered else "LVH criteria not met."
#     )
#     return _result("Hypertension / LVH", triggered, conf,
#                    Severity.MODERATE if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I10", "I11.9"])


# def detect_pulmonary_hypertension(f: ECGFeatures) -> DetectionResult:
#     """
#     Pulmonary Hypertension ECG (ESC 2022 PH Guidelines):
#       - Right axis deviation > 110°
#       - RVH pattern (R > S in V1)
#       - P pulmonale (peaked P > 2.5 mm in II)
#       - Strain: T inversions V1-V4
#     """
#     findings, missing = [], []
#     score = 0.0
#     axis = _safe(f.qrs_axis_degrees, 0.0)

#     if axis > 110:
#         findings.append(f"Right axis deviation {axis:.0f}° — RV pressure overload")
#         score += 0.35

#     if f.p_wave_peaked:
#         findings.append("P pulmonale — right atrial hypertrophy")
#         score += 0.30

#     if f.rbbb_pattern:
#         findings.append("RBBB — right ventricular pressure overload")
#         score += 0.25

#     if f.t_wave_inverted:
#         findings.append("T-wave inversions V1-V4 — RV strain")
#         score += 0.20

#     missing.extend([
#         "R > S ratio in V1 (RVH voltage criterion)",
#         "Echo — RVSP / tricuspid regurgitation velocity",
#         "Right heart catheterisation (definitive diagnosis)",
#     ])

#     triggered = score >= 0.55
#     conf = min(score, 0.80)
#     qs, src = QUESTIONS["Pulmonary Hypertension"]
#     reason = (
#         f"Right axis deviation + P pulmonale + RBBB — pulmonary hypertension pattern."
#         if triggered else "Pulmonary hypertension ECG criteria not met."
#     )
#     return _result("Pulmonary Hypertension", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["I27.0", "I27.2"])


# def detect_amyloidosis(f: ECGFeatures) -> DetectionResult:
#     """
#     Cardiac Amyloidosis ECG (ESC 2021 Amyloidosis Working Group):
#       - Low voltage + LVH on echo = pathognomonic voltage-wall mismatch
#       - Pseudo-infarction Q waves (multiple territories without true MI)
#       - Conduction disease (AV block, BBB)
#       - AF common
#     """
#     findings, missing = [], []
#     score = 0.0

#     if f.low_voltage:
#         findings.append("Low QRS voltage (< 5mm limb leads / < 10mm precordial)")
#         score += 0.45

#     if f.q_wave_deep and f.low_voltage:
#         findings.append("Pseudo-infarction Q waves + low voltage (amyloid hallmark)")
#         score += 0.30

#     if f.pr_interval_ms and f.pr_interval_ms > 200:
#         findings.append(f"Prolonged PR {f.pr_interval_ms:.0f} ms — conduction system infiltration")
#         score += 0.20

#     missing.extend([
#         "Echo — LV wall thickness (voltage-wall mismatch is pathognomonic)",
#         "CMR with LGE (subendocardial enhancement pattern)",
#         "99mTc-DPD/PYP scintigraphy (ATTR-specific)",
#         "Serum/urine protein electrophoresis (AL amyloidosis)",
#     ])

#     triggered = score >= 0.50
#     conf = min(score, 0.85)
#     qs, src = QUESTIONS["Amyloidosis"]
#     reason = (
#         f"Low voltage + pseudo-Q waves — cardiac amyloidosis ECG pattern. "
#         f"Echo voltage-wall mismatch assessment urgent."
#         if triggered else "Amyloidosis ECG criteria not met."
#     )
#     return _result("Cardiac Amyloidosis", triggered, conf,
#                    Severity.HIGH if triggered else Severity.INFO,
#                    reason, findings, missing, qs, src, ["E85.4"])
# ─────────────────────────────────────────────────────────────────────────────
# DISEASE DETECTOR — main entry point
# ─────────────────────────────────────────────────────────────────────────────

from disease_detection.detect_long_qt import detect_long_qt, ECGWindowFeatures
from disease_detection.detect_arvc import detect_arvc, ARVCWindowFeatures
from disease_detection.detect_vt_vf import (
    detect_vt, detect_vf,
    VTWindowFeatures, VFWindowFeatures,
)
from disease_detection.detect_atrial_flutter import detect_atrial_flutter, AFLWindowFeatures
from disease_detection.detect_afib import detect_afib, AFWindowFeatures

#: Full registry of detection functions, ordered by clinical priority
_RULE_REGISTRY = [
    detect_vf,
    detect_vt,
    # detect_stemi,
    # detect_cardiac_tamponade,
    # detect_nstemi_ua,
    # detect_long_qt,
    # detect_brugada,
    detect_afib,
    detect_atrial_flutter,
    # detect_pulmonary_embolism,
    # detect_hcm,
    detect_arvc,
    # detect_wpw,
    # detect_heart_failure,
    # detect_pericarditis,
    # detect_pulmonary_hypertension,
    # detect_lvh_hypertension,
    # detect_amyloidosis,
]


class DiseaseDetector:
    """
    Runs all detection rules against ECGFeatures and ECGWindowFeatures.

    Typical integration inside ECGPipeline (Step 4 / Event Manager):
    -----------------------------------------------------------------
        from disease_detector import DiseaseDetector, ECGFeatures
        from disease_detection.detect_long_qt import ECGWindowFeatures

        detector = DiseaseDetector()

        # Build ECGFeatures from beat_data (output of feature_engineering)
        features = ECGFeatures(
            rr_mean_ms        = beat_data.get("rr_interval"),
            rr_std_ms         = window_stats.get("rr_std"),
            rr_irregular      = window_stats.get("rr_irregular", False),
            p_wave_present    = beat_data.get("p_wave_amplitude", 0) > 0.05,
            pr_interval_ms    = beat_data.get("pr_interval_ms"),
            qrs_duration_ms   = beat_data.get("qrs_duration_ms"),
            qt_interval_ms    = beat_data.get("qt_interval_ms"),
            st_deviation_mv   = beat_data.get("st_deviation_mv"),
            t_wave_inverted   = beat_data.get("t_wave_amplitude", 1) < -0.05,
            heart_rate_bpm    = 60000 / beat_data.get("rr_interval", 800),
            cnn_label         = beat_data.get("predicted_label"),
            af_probability    = window_stats.get("af_probability", 0.0),
            vt_detected       = window_stats.get("vt_detected", False),
            vf_detected       = window_stats.get("vf_detected", False),
            low_voltage       = window_stats.get("low_voltage", False),
            electrical_alternans = window_stats.get("electrical_alternans", False),
        )

        # Build ECGWindowFeatures for LQTS temporal analysis
        window_features = ECGWindowFeatures(
            window_id="window_001",
            patient_id="patient_123",
            recorded_at=datetime.now(),
            beat_count=len(beats),
            valid_beat_count=len(valid_beats),
            qtc_values_ms=[beat['qtc_ms'] for beat in valid_beats],
            rr_values_ms=[beat['rr_interval'] for beat in valid_beats],
            sex="M",
            age=45,
            resting=True,
            rhythm_stable=True,
            vt_detected=window_stats.get("vt_detected", False),
            tdp_detected=window_stats.get("tdp_detected", False),
            t_wave_biphasic_fraction=window_stats.get("t_wave_biphasic_fraction", 0.0),
            macroscopic_twa_present=window_stats.get("macroscopic_twa_present", False),
            bradycardia_for_age=window_stats.get("bradycardia_for_age", False),
            known_diagnoses=[],
        )

        results = detector.evaluate(features, window_features)
        triggered = [r for r in results if r.triggered]
    """

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self._rules = _RULE_REGISTRY

    def evaluate(
    self,
    features: ECGFeatures,
    window_features: Optional[ECGWindowFeatures] = None,
    arvc_window: Optional[ARVCWindowFeatures] = None,
    vt_window: Optional[VTWindowFeatures] = None,
    vf_window: Optional[VFWindowFeatures] = None,
    afl_window: Optional[AFLWindowFeatures] = None,
    afib_window: Optional[AFWindowFeatures] = None,
) -> List[DetectionResult]:
        """
        Run all rules. Returns ALL results (triggered and not triggered).
        Filter by .triggered for actionable findings.
        
        Parameters
        ----------
        features : ECGFeatures
            Single-beat or aggregated ECG features for most detection rules.
        window_features : Optional[ECGWindowFeatures]
            Window-level features for LQTS detection (temporal analysis).
            Required for detect_long_qt to work correctly.
        arvc_window : Optional[ARVCWindowFeatures]
            Window-level features for ARVC detection.
            Required for detect_arvc to work correctly.
        vt_window : Optional[VTWindowFeatures]
            Run-level features for VT detection (consecutive ventricular beats).
            Required for detect_vt to work correctly.
        vf_window : Optional[VFWindowFeatures]
            Window-level features for VF detection.
            Required for detect_vf to work correctly.
        """
        results = []
        for rule_fn in self._rules:
            try:
                # detect_long_qt requires ECGWindowFeatures, others use ECGFeatures
                if rule_fn.__name__ == "detect_long_qt":
                    if window_features is not None:
                        result = rule_fn(window_features)
                    else:
                        result = DetectionResult(
                            disease="Long QT Syndrome (LQTS)",
                            triggered=False,
                            confidence=0.0,
                            severity=Severity.INFO,
                            reason="ECGWindowFeatures not provided — temporal LQTS analysis requires window-level QTc series.",
                            ecg_findings=[],
                            missing_data=["ECGWindowFeatures with qtc_values_ms list"],
                            symptom_questions=[],
                            symptom_source="",
                            rag_trigger=False,
                            icd10_codes=[],
                        )
                elif rule_fn.__name__ == "detect_arvc":
                    if arvc_window is not None:
                        result = rule_fn(arvc_window)
                    else:
                        result = DetectionResult(
                            disease="Arrhythmogenic Cardiomyopathy (ARVC)",
                            triggered=False, confidence=0.0, severity=Severity.INFO,
                            reason="ARVCWindowFeatures not provided.",
                            ecg_findings=[], missing_data=["ARVCWindowFeatures"],
                            symptom_questions=[], symptom_source="",
                            rag_trigger=False, icd10_codes=[],
                        )
                elif rule_fn.__name__ == "detect_vt":
                    if vt_window is not None:
                        result = rule_fn(vt_window)
                    else:
                        result = DetectionResult(
                            disease="Ventricular Tachycardia (VT)",
                            triggered=False, confidence=0.0, severity=Severity.INFO,
                            reason="VTWindowFeatures not provided — VT detection requires run-level features (consecutive ventricular beats).",
                            ecg_findings=[], missing_data=["VTWindowFeatures"],
                            symptom_questions=[], symptom_source="",
                            rag_trigger=False, icd10_codes=[],
                        )
                elif rule_fn.__name__ == "detect_vf":
                    if vf_window is not None:
                        result = rule_fn(vf_window)
                    else:
                        result = DetectionResult(
                            disease="Ventricular Fibrillation (VF)",
                            triggered=False, confidence=0.0, severity=Severity.INFO,
                            reason="VFWindowFeatures not provided — VF detection requires window-level VF flag features.",
                            ecg_findings=[], missing_data=["VFWindowFeatures"],
                            symptom_questions=[], symptom_source="",
                            rag_trigger=False, icd10_codes=[],
                        )
                elif rule_fn.__name__ == "detect_atrial_flutter":
                    if afl_window is not None:
                        result = rule_fn(afl_window)
                    else:
                        result = DetectionResult(
                            disease="Atrial Flutter (AFL)", triggered=False, confidence=0.0,
                            severity=Severity.INFO,
                            reason="AFLWindowFeatures not provided.",
                            ecg_findings=[], missing_data=["AFLWindowFeatures"],
                            symptom_questions=[], symptom_source="", rag_trigger=False, icd10_codes=[],
                        )
                elif rule_fn.__name__ == "detect_afib":
                    if afib_window is not None:
                        result = rule_fn(afib_window)
                    else:
                        result = DetectionResult(
                            disease="Atrial Fibrillation (AF)", triggered=False, confidence=0.0,
                            severity=Severity.INFO,
                            reason="AFWindowFeatures not provided.",
                            ecg_findings=[], missing_data=["AFWindowFeatures"],
                            symptom_questions=[], symptom_source="", rag_trigger=False, icd10_codes=[],
                        )
                else:
                    result = rule_fn(features)
                results.append(result)
            except Exception as exc:
                results.append(DetectionResult(
                    disease=rule_fn.__name__.replace("detect_", ""),
                    triggered=False,
                    confidence=0.0,
                    severity=Severity.INFO,
                    reason=f"Rule error: {exc}",
                    ecg_findings=[],
                    missing_data=[],
                    symptom_questions=[],
                    symptom_source="",
                    rag_trigger=False,
                    icd10_codes=[],
                ))
        return results

    def evaluate_ecg_only(self, features: ECGFeatures) -> List[DetectionResult]:
        """
        Convenience wrapper: same as evaluate() but returns only triggered
        results above min_confidence — intended for stand-alone ECG-only use.
        """
        return [
            r for r in self.evaluate(features)
            if r.triggered and r.confidence >= self.min_confidence
        ]

    def get_symptom_questions(self, disease_name: str) -> tuple:
        """Return (questions list, source string) for a disease by name."""
        for key, (qs, src) in QUESTIONS.items():
            if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
                return qs, src
        return [], ""

    def rag_triggers(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """Return only results that should trigger the RAG agent."""
        return [r for r in results if r.rag_trigger]

    def critical_alerts(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """Return CRITICAL severity results — require immediate action."""
        return [r for r in results if r.severity == Severity.CRITICAL and r.triggered]

    def summary(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Compact summary dict suitable for logging and database storage."""
        triggered = [r for r in results if r.triggered]
        return {
            "triggered_count": len(triggered),
            "critical_count":  sum(1 for r in triggered if r.severity == Severity.CRITICAL),
            "rag_trigger_count": sum(1 for r in triggered if r.rag_trigger),
            "diseases": [
                {
                    "name":       r.disease,
                    "confidence": round(r.confidence, 3),
                    "severity":   r.severity.value,
                    "icd10":      r.icd10_codes,
                    "rag":        r.rag_trigger,
                }
                for r in triggered
            ],
        }
