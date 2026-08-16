from Event_Manager import evaluate_event
from episode_manager import EpisodeManager
from episode_integration_shim import route_event_through_episode_manager
from typing import List, Dict, Any, Optional
from datetime import datetime
from disease_detector import DiseaseDetector, ECGFeatures
from disease_detection.detect_long_qt import ECGWindowFeatures
from disease_detection.detect_arvc import ARVCWindowFeatures, VTEpisode
from disease_detection.detect_vt_vf import VTWindowFeatures, VFWindowFeatures
from av_block_detector import detect_av_block
# from Graduation project.ecg-project1 import history
# Constants based on typical MIT-BIH AAMI mapping
LABEL_N = 0  # Normal
LABEL_S = 1  # Supraventricular ectopic (APC)
LABEL_V = 2  # Ventricular ectopic (PVC)
LABEL_F = 3  # Fusion
LABEL_Q = 4  # Unknown

_DISEASE_DETECTOR = DiseaseDetector(min_confidence=0.35)

_DISEASE_EVENT_MAP = {
    "atrial fibrillation": "AFIB_DETECTED",
    "atrial flutter": "AFLUTTER_SUSPECTED",
    "ventricular tachycardia": "VT_RUN",
    "long qt": "POSSIBLE_LONG_QT",
    "stemi": "POSSIBLE_ISCHEMIC_PATTERN",
    "nstemi": "POSSIBLE_ISCHEMIC_PATTERN",
    "unstable angina": "POSSIBLE_ISCHEMIC_PATTERN",
    "heart failure": "POSSIBLE_HEART_FAILURE_PATTERN",
    "wpw": "DISEASE_WPW_PREEXCITATION",
    "pre-excitation": "DISEASE_WPW_PREEXCITATION",
    "brugada": "DISEASE_BRUGADA_SYNDROME",
    "hypertrophic cardiomyopathy": "DISEASE_HCM",
    "arrhythmogenic cardiomyopathy": "DISEASE_ARVC",
    "pulmonary embolism": "DISEASE_PULMONARY_EMBOLISM",
    "pericarditis": "DISEASE_PERICARDITIS",
    "tamponade": "DISEASE_CARDIAC_TAMPONADE",
    "pulmonary hypertension": "DISEASE_PULMONARY_HYPERTENSION",
    "hypertension": "DISEASE_LVH_HYPERTENSION",
    "amyloidosis": "DISEASE_CARDIAC_AMYLOIDOSIS",
    "ventricular fibrillation": "DISEASE_VENTRICULAR_FIBRILLATION",
}

_SEVERITY_RANK = {
    "info": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

_DISEASE_MANAGED_EVENT_TYPES = {
    "AFIB_DETECTED",
    "AFLUTTER_SUSPECTED",
    "VT_RUN",
    "POSSIBLE_LONG_QT",
    "POSSIBLE_ISCHEMIC_PATTERN",
    "POSSIBLE_HEART_FAILURE_PATTERN",
    "DISEASE_WPW_PREEXCITATION",
    "DISEASE_BRUGADA_SYNDROME",
    "DISEASE_HCM",
    "DISEASE_ARVC",
    "DISEASE_PULMONARY_EMBOLISM",
    "DISEASE_PERICARDITIS",
    "DISEASE_CARDIAC_TAMPONADE",
    "DISEASE_LVH_HYPERTENSION",
    "DISEASE_PULMONARY_HYPERTENSION",
    "DISEASE_CARDIAC_AMYLOIDOSIS",
    "DISEASE_VENTRICULAR_FIBRILLATION",
    "DISEASE_DETECTED",
}

def calculate_rr_window_features(beats_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates statistics over the RR intervals in the window.
    """
    rr_intervals = [b['rr_interval'] for b in beats_history if b['rr_interval'] is not None and b['rr_interval'] > 0]
    
    if not rr_intervals:
        return {"rr_mean": 0.0, "rr_std": 0.0, "rr_min": 0.0, "rr_max": 0.0}
        
    return {
        "rr_mean": sum(rr_intervals) / len(rr_intervals),
        "rr_std": float(pow(sum((x - (sum(rr_intervals) / len(rr_intervals)))**2 for x in rr_intervals) / len(rr_intervals), 0.5)),
        "rr_min": min(rr_intervals),
        "rr_max": max(rr_intervals)
    }
def calculate_abnormal_burden(beats_history: List[Dict[str, Any]],
                              min_quality: float = 0.50,
                              min_valid_beats: int = 10) -> Dict[str, float]:
    """
    Calculates ectopic burden (%) of PVCs and PACs in the given beat window.

    Returns:
        dict with keys: abnormal_burden (%), pvc_burden (%), apc_burden (%).
        All values are 0.0 if the window has fewer than min_valid_beats.
    """
    if not beats_history:
        return {"abnormal_burden": 0.0, "pvc_burden": 0.0, "apc_burden": 0.0}

    valid_beats = [b for b in beats_history if b.get('signal_quality_score', 0) >= min_quality]
    total_valid = len(valid_beats)

    if total_valid < min_valid_beats:
        return {"abnormal_burden": 0.0, "pvc_burden": 0.0, "apc_burden": 0.0}

    pvcs = sum(1 for b in valid_beats if b.get('predicted_label') == 2)   # LABEL_V
    apcs = sum(1 for b in valid_beats if b.get('predicted_label') == 1)   # LABEL_S
    abnormal = sum(1 for b in valid_beats if b.get('is_abnormal', False))

    pvc_burden = (pvcs / total_valid) * 100.0
    apc_burden = (apcs / total_valid) * 100.0
    abnormal_burden = (abnormal / total_valid) * 100.0

    return {
        "abnormal_burden": round(abnormal_burden, 1),
        "pvc_burden": round(pvc_burden, 1),
        "apc_burden": round(apc_burden, 1)
    }
def _get_labels_str(beats_history: List[Dict[str, Any]]) -> str:
    # Convert labels to string for easy regex/pattern matching. e.g. "020202"
    return "".join(str(b['predicted_label']) for b in beats_history)


def detect_rhythm_patterns(
    beats_history,
    # Ventricular thresholds
    min_bigeminy_cycles=2,
    min_trigeminy_cycles=2,
    min_quadrigeminy_cycles=3,
    min_couplet_count=1,
    min_triplet_count=1,
    # Atrial thresholds (can be set separately if desired, here defaulted to same)
    min_atrial_bigeminy_cycles=3,
    min_atrial_trigeminy_cycles=2,
    min_atrial_quadrigeminy_cycles=3,
    min_atrial_couplet_count=1,
    min_atrial_triplet_count=1,
    min_atrial_run_beats=3,     # runs of ≥3 S beats to count as atrial salvo
    # Confidence & quality filters
    min_confidence=0.50,
    min_quality=0.50,
    # Feature flags
    include_atrial=True,
    include_counts=True
):
    """
    Detect ventricular AND supraventricular ectopy patterns from beat labels,
    plus quality counters.

    Labels: 0=N, 1=S (supraventricular), 2=V (ventricular), 3=F (fusion), 4=Q (unknown)

    Returns a dictionary with pattern flags, counts, and confidence.
    """
    # --- 1. Filter beats, build label string ---
    labels = []
    for beat in beats_history:
        if not isinstance(beat, dict):
            continue
        if beat.get('prediction_confidence', 0) < min_confidence:
            continue
        if beat.get('signal_quality_score', 0) < min_quality:
            continue
        labels.append(str(beat['predicted_label']))

    labels_str = "".join(labels)
    total_beats = len(labels)

    # --- 2. Result skeleton ---
    result = {
        # Ventricular patterns
        "bigeminy": False,
        "trigeminy": False,
        "quadrigeminy": False,
        "couplet": False,
        "couplet_count": 0,
        "triplet": False,
        "triplet_count": 0,
        "ventricular_run_count": 0,          # runs of ≥3 V beats (raw VT input)
        "max_ventricular_run_length": 0,
        # Atrial patterns
        "atrial_bigeminy": False,
        "atrial_trigeminy": False,
        "atrial_quadrigeminy": False,
        "atrial_couplet": False,
        "atrial_couplet_count": 0,
        "atrial_triplet": False,
        "atrial_triplet_count": 0,
        "atrial_run_count": 0,               # runs of ≥3 S beats
        "max_atrial_run_length": 0,
        # Counters
        "total_ventricular_beats": 0,
        "total_supraventricular_beats": 0,
        "total_fusion_beats": 0,
        "total_unknown_beats": 0,
        "total_evaluated_beats": total_beats,
        # Confidence
        "pattern_confidence": 0.0,
        "data_quality_warning": False
    }

    # --- 3. Basic counts ---
    if include_counts:
        for lbl in labels:
            if lbl == '2':
                result["total_ventricular_beats"] += 1
            elif lbl == '1':
                result["total_supraventricular_beats"] += 1
            elif lbl == '3':
                result["total_fusion_beats"] += 1
            elif lbl == '4':
                result["total_unknown_beats"] += 1
        if total_beats > 0 and result["total_unknown_beats"] / total_beats > 0.2:
            result["data_quality_warning"] = True

    # Helper: generic pattern searcher
    def _find_pattern(ectopic_label, patterns_list, cycles):
        """Return True if any pattern repeated `cycles` times exists in labels_str."""
        for pat in patterns_list:
            if (pat * cycles) in labels_str:
                return True
        return False

    # Helper: count runs of exact length & return counts and max run length
    def _count_runs(ectopic_label, exact_length=None):
        """
        Returns (count_of_exact_length, count_of_runs_ge_3, max_run_length).
        If exact_length is given, count only runs of that length.
        """
        exact_count = 0
        run_count_ge3 = 0
        max_run = 0
        i = 0
        L = len(labels_str)
        while i < L:
            if labels_str[i] == ectopic_label:
                run_len = 1
                j = i + 1
                while j < L and labels_str[j] == ectopic_label:
                    run_len += 1
                    j += 1
                if run_len > max_run:
                    max_run = run_len
                if exact_length is not None and run_len == exact_length:
                    exact_count += 1
                if run_len >= 3:
                    run_count_ge3 += 1
                i = j
            else:
                i += 1
        return exact_count, run_count_ge3, max_run

    # --- 4. Ventricular patterns ---
    # Bigeminy
    if _find_pattern('2', ["02", "20"], min_bigeminy_cycles):
        result["bigeminy"] = True
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.85)

    # Trigeminy
    if _find_pattern('2', ["002", "200", "020"], min_trigeminy_cycles):
        result["trigeminy"] = True
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.80)

    # Quadrigeminy
    if _find_pattern('2', ["0002", "2000", "0200", "0020"], min_quadrigeminy_cycles):
        result["quadrigeminy"] = True
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.75)

    # Couplets (exactly 2)
    couplet_cnt, _, _ = _count_runs('2', exact_length=2)
    if couplet_cnt >= min_couplet_count:
        result["couplet"] = True
        result["couplet_count"] = couplet_cnt
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.75)

    # Triplets (exactly 3)
    triplet_cnt, run3plus, max_v_run = _count_runs('2', exact_length=3)
    # note: we also need runs of ≥3 for ventricular_run_count and max length
    _, ventricular_run_count, max_run_len = _count_runs('2')  # full scan again, but okay for clarity
    # But we already have run3plus and max_v_run from the triplet call? Actually _count_runs when called with exact_length still returns run_count_ge3 and max_run. So we can use that.
    # Safer: call once without exact_length to get all run stats, then check couplet/triplet separately.
    # Let's refactor for efficiency.

    # Efficient run counting for ventricular beats
    def _ventricular_runs():
        """
        Return (couplet_count, triplet_count, run_ge3_count, max_run) for V beats.
        
        Couplets and triplets are counted WITHIN longer runs too:
        - A run of 7 V beats = 3 couplets + 1 VT run (run_len >= 3)
        - A run of 3 V beats = 1 triplet + 1 VT run
        """
        couplets = 0
        triplets = 0
        runs_ge3 = 0
        max_run = 0
        i = 0
        L = len(labels_str)
        while i < L:
            if labels_str[i] == '2':
                run_len = 1
                j = i + 1
                while j < L and labels_str[j] == '2':
                    run_len += 1
                    j += 1
                # Count couplets within ANY run (including long runs)
                if run_len >= 2:
                    couplets += run_len // 2
                # Count triplets within ANY run
                if run_len >= 3:
                    triplets += run_len // 3
                # Runs of 3+ are VT/salvos
                if run_len >= 3:
                    runs_ge3 += 1
                if run_len > max_run:
                    max_run = run_len
                i = j
            else:
                i += 1
        return couplets, triplets, runs_ge3, max_run

    v_couplets, v_triplets, v_runs_ge3, v_max_run = _ventricular_runs()
    result["couplet_count"] = v_couplets
    result["couplet"] = v_couplets >= min_couplet_count
    if result["couplet"]:
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.75)

    result["triplet_count"] = v_triplets
    result["triplet"] = v_triplets >= min_triplet_count
    if result["triplet"]:
        result["pattern_confidence"] = max(result["pattern_confidence"], 0.70)

    result["ventricular_run_count"] = v_runs_ge3
    result["max_ventricular_run_length"] = v_max_run

    # --- 5. Atrial patterns (if enabled) ---
    if include_atrial:
        # Bigeminy
        if _find_pattern('1', ["01", "10"], min_atrial_bigeminy_cycles):
            result["atrial_bigeminy"] = True
            result["pattern_confidence"] = max(result["pattern_confidence"], 0.80)

        # Trigeminy
        if _find_pattern('1', ["001", "100", "010"], min_atrial_trigeminy_cycles):
            result["atrial_trigeminy"] = True
            result["pattern_confidence"] = max(result["pattern_confidence"], 0.75)

        # Quadrigeminy
        if _find_pattern('1', ["0001", "1000", "0100", "0010"], min_atrial_quadrigeminy_cycles):
            result["atrial_quadrigeminy"] = True
            result["pattern_confidence"] = max(result["pattern_confidence"], 0.70)

        # Atrial runs (couplets, triplets, ≥3 runs)
        def _atrial_runs():
            couplets = 0
            triplets = 0
            runs_ge3 = 0
            max_run = 0
            i = 0
            L = len(labels_str)
            while i < L:
                if labels_str[i] == '1':
                    run_len = 1
                    j = i + 1
                    while j < L and labels_str[j] == '1':
                        run_len += 1
                        j += 1
                    if run_len == 2:
                        couplets += 1
                    elif run_len == 3:
                        triplets += 1
                    if run_len >= min_atrial_run_beats:   # default 3
                        runs_ge3 += 1
                    if run_len > max_run:
                        max_run = run_len
                    i = j
                else:
                    i += 1
            return couplets, triplets, runs_ge3, max_run

        a_couplets, a_triplets, a_runs_ge3, a_max_run = _atrial_runs()
        result["atrial_couplet_count"] = a_couplets
        result["atrial_couplet"] = a_couplets >= min_atrial_couplet_count
        if result["atrial_couplet"]:
            result["pattern_confidence"] = max(result["pattern_confidence"], 0.70)

        result["atrial_triplet_count"] = a_triplets
        result["atrial_triplet"] = a_triplets >= min_atrial_triplet_count
        if result["atrial_triplet"]:
            result["pattern_confidence"] = max(result["pattern_confidence"], 0.65)

        result["atrial_run_count"] = a_runs_ge3
        result["max_atrial_run_length"] = a_max_run

    # Build reason and ecg_findings from the detected patterns
    pattern_names = []
    if result["bigeminy"]: pattern_names.append("Bigeminy")
    if result["trigeminy"]: pattern_names.append("Trigeminy")
    if result["quadrigeminy"]: pattern_names.append("Quadrigeminy")
    if result["couplet"]: pattern_names.append("Couplet")
    if result.get("atrial_bigeminy"): pattern_names.append("Atrial bigeminy")
    if result.get("atrial_trigeminy"): pattern_names.append("Atrial trigeminy")
    if result.get("atrial_quadrigeminy"): pattern_names.append("Atrial quadrigeminy")
    if result.get("atrial_couplet"): pattern_names.append("Atrial couplet")
    if result.get("atrial_triplet"): pattern_names.append("Atrial triplet")
    if pattern_names:
        result["reason"] = f"Pattern detected: {', '.join(pattern_names)} (confidence={result['pattern_confidence']:.2f})"
        result["ecg_findings"] = [f"Pattern: {', '.join(pattern_names)}"]
    if result["ventricular_run_count"] > 0:
        result["ecg_findings"].append(f"Ventricular runs: {result['ventricular_run_count']} (max {result['max_ventricular_run_length']} beats)")
    result["ecg_findings"] = result.get("ecg_findings", [])
    return result

import numpy as np

def detect_pauses(
    beats_history,
    pause_threshold_ms: float = 2000.0,
    min_quality: float = 0.50,
    min_confidence: float = 0.70
):
    """
    Detect clinically significant pauses with tiered severity.

    Clinical pause tiers (AHA/ACC):
      - ≥2000ms (2s):  Sinus pause / arrest — moderate
      - ≥3000ms (3s):  Pathological pause — high
      - ≥4000ms (4s):  Prolonged asystole — critical
      - ≥6000ms (6s):  Emergency asystole — critical (separate event type)

    Uses the entire window instead of only the latest beat.
    Returns the HIGHEST severity event only (one per call).
    """

    if not beats_history:
        return None

    valid_rrs = []

    for beat in beats_history:

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        rr = beat.get("rr_interval")

        if rr is None:
            continue

        if rr <= 0:
            continue

        # Normalize to milliseconds
        rr_ms = (
            rr
            if rr > 10
            else rr * 1000.0
        )

        valid_rrs.append(rr_ms)

    if not valid_rrs:
        return None

    pause_rrs = [
        rr
        for rr in valid_rrs
        if rr >= pause_threshold_ms
    ]

    if not pause_rrs:
        return None

    longest_pause_ms = max(pause_rrs)

    pause_count = len(pause_rrs)

    # ── Tiered severity ──────────────────────────────────────
    if longest_pause_ms >= 6000.0:
        return {
            "event_type": "PROLONGED_ASYSTOLE",
            "severity": "critical",
            "metadata_json": {
                "pause_threshold_ms": pause_threshold_ms,
                "pause_count": pause_count,
                "longest_pause_ms": longest_pause_ms,
                "mean_pause_ms": (
                    sum(pause_rrs) / len(pause_rrs)
                ),
                "total_valid_rr_intervals": len(valid_rrs),
                "asystole_tier": ">=6s_emergency",
                "reason": f"Prolonged asystole: longest pause {longest_pause_ms:.0f}ms (>=6000ms emergency threshold)",
                "ecg_findings": [f"Longest pause {longest_pause_ms:.0f}ms", f"Pause count: {pause_count}", f"Asystole tier: >=6s emergency"],
            }
        }
    elif longest_pause_ms >= 4000.0:
        return {
            "event_type": "PROLONGED_ASYSTOLE",
            "severity": "critical",
            "metadata_json": {
                "pause_threshold_ms": pause_threshold_ms,
                "pause_count": pause_count,
                "longest_pause_ms": longest_pause_ms,
                "mean_pause_ms": (
                    sum(pause_rrs) / len(pause_rrs)
                ),
                "total_valid_rr_intervals": len(valid_rrs),
                "asystole_tier": ">=4s_prolonged",
                "reason": f"Prolonged asystole: longest pause {longest_pause_ms:.0f}ms (>=4000ms critical threshold)",
                "ecg_findings": [f"Longest pause {longest_pause_ms:.0f}ms", f"Pause count: {pause_count}", f"Asystole tier: >=4s prolonged"],
            }
        }
    elif longest_pause_ms >= 3000.0:
        severity = "high"
    else:
        severity = "moderate"

    return {
        "event_type": "PAUSE_DETECTED",
        "severity": severity,
        "metadata_json": {
            "pause_threshold_ms": pause_threshold_ms,
            "pause_count": pause_count,
            "longest_pause_ms": longest_pause_ms,
            "mean_pause_ms": (
                sum(pause_rrs) / len(pause_rrs)
            ),
            "total_valid_rr_intervals": len(valid_rrs),
            "reason": f"Pause detected: longest pause {longest_pause_ms:.0f}ms (threshold {pause_threshold_ms:.0f}ms)",
            "ecg_findings": [f"Longest pause {longest_pause_ms:.0f}ms", f"Pause count: {pause_count}", f"Threshold: {pause_threshold_ms:.0f}ms"],
        }
    }

def detect_rate_abnormalities(
    beats_history,
    tachy_threshold: float = 100.0,
    brady_threshold: float = 60.0,
    severe_tachy_threshold: float = 140.0,
    severe_brady_threshold: float = 40.0,
    extreme_tachy_beats: int = 17,          # من PhysioNet/CinC 2015
    extreme_brady_beats: int = 5,           # من PhysioNet/CinC 2015
    min_duration_sec: float = 15.0,         # للنافذة الزمنية
    required_ratio: float = 0.80,
    min_confidence: float = 0.70,
    min_quality: float = 0.50
):
    """
    كشف تسارع/تباطؤ القلب بطريقتين:
    1. شديد (Extreme): عدد ضربات متتالية فوق/تحت عتبة شديدة.
    2. مستدام (Sustained): نافذة زمنية بنسبة 80%.
    """
    if not beats_history:
        return []

    # فلترة أولية للضربات الصالحة (بدون استبعاد الوقت)
    valid_beats = []
    for beat in beats_history:
        if not isinstance(beat, dict):
            continue
        if beat.get('prediction_confidence', 0) < min_confidence:
            continue
        if beat.get('signal_quality_score', 0) < min_quality:
            continue
        # استبعاد الضربات البطينية (لأن VT لها كاشف خاص)
        if beat.get('predicted_label') == 2:
            continue
        hr = beat.get('heart_rate')
        if hr is None or hr <= 0:
            continue
        valid_beats.append(beat)

    events = []

    # ============================================================
    # المرحلة الأولى: الكشف عن الحالات الشديدة (Extreme)
    # ============================================================
    # نبحث في كل الضربات الصالحة (وليس فقط نافذة زمنية)
    extreme_tachy_found = False
    extreme_brady_found = False

    tachy_consecutive = 0
    brady_consecutive = 0

    for beat in valid_beats:
        hr = beat['heart_rate']
        if hr >= severe_tachy_threshold:
            tachy_consecutive += 1
            brady_consecutive = 0
        elif hr <= severe_brady_threshold:
            brady_consecutive += 1
            tachy_consecutive = 0
        else:
            tachy_consecutive = 0
            brady_consecutive = 0

        if tachy_consecutive >= extreme_tachy_beats and not extreme_tachy_found:
            extreme_tachy_found = True
            # حساب متوسط الـ HR خلال هذه النوبة للشدة
            avg_hr = sum(b['heart_rate'] for b in valid_beats[-tachy_consecutive:]) / tachy_consecutive
            events.append({
                "event_type": "EXTREME_TACHYCARDIA",
                "triggered": True,
                "severity": "critical",
                "metadata_json": {
                    "average_hr": avg_hr,
                    "consecutive_beats": tachy_consecutive,
                    "threshold": severe_tachy_threshold,
                    "detection_method": "PhysioNet/CinC 2015 extreme rule",
                    "reason": f"Extreme tachycardia: {tachy_consecutive} consecutive beats at HR {avg_hr:.1f} bpm (threshold {severe_tachy_threshold} bpm)",
                    "ecg_findings": [f"HR {avg_hr:.1f} bpm", f"{tachy_consecutive} consecutive beats above {severe_tachy_threshold} bpm"],
                }
            })
            # لا نكسر الحلقة، نستمر لجمع أحداث أخرى (مثلاً تباطؤ شديد لاحقًا)
            # لكن نمنع التكرار
            break  # نكتفي بإنذار واحد من هذا النوع لكل استدعاء

        if brady_consecutive >= extreme_brady_beats and not extreme_brady_found:
            extreme_brady_found = True
            avg_hr = sum(b['heart_rate'] for b in valid_beats[-brady_consecutive:]) / brady_consecutive
            events.append({
                "event_type": "EXTREME_BRADYCARDIA",
                "triggered": True,
                "severity": "critical",
                "metadata_json": {
                    "average_hr": avg_hr,
                    "consecutive_beats": brady_consecutive,
                    "threshold": severe_brady_threshold,
                    "detection_method": "PhysioNet/CinC 2015 extreme rule",
                    "reason": f"Extreme bradycardia: {brady_consecutive} consecutive beats at HR {avg_hr:.1f} bpm (threshold {severe_brady_threshold} bpm)",
                    "ecg_findings": [f"HR {avg_hr:.1f} bpm", f"{brady_consecutive} consecutive beats below {severe_brady_threshold} bpm"],
                }
            })
            break

    # إذا انطلق إنذار شديد، لا نكمل تحليل النافذة الزمنية (اختياري)
    # لكن سريريًا قد نريد إنذار "استمرار" بالإضافة للإنذار الشديد
    # سنستمر هنا لتوليد كليهما إذا تحققت الشروط

    # ============================================================
    # المرحلة الثانية: الكشف المستدام (نافذة زمنية)
    # ============================================================
    if not valid_beats:
        return events

    latest_time = valid_beats[-1]['timestamp']
    window_start = latest_time - min_duration_sec
    window_beats = [b for b in valid_beats if b['timestamp'] >= window_start]

    if len(window_beats) < 3:
        return events

    # حساب معدل القلب الحقيقي في النافذة
    count_beats = len(window_beats)
    duration = window_beats[-1]['timestamp'] - window_beats[0]['timestamp']
    window_start_beat = window_beats[0].get('beat_index', 0) if window_beats else 0
    if duration <= 0:
        return events
    avg_hr = (count_beats / duration) * 60.0

    heart_rates = [b['heart_rate'] for b in window_beats]
    tachy_count = sum(1 for hr in heart_rates if hr >= tachy_threshold)
    brady_count = sum(1 for hr in heart_rates if hr <= brady_threshold)

    tachy_ratio = tachy_count / count_beats
    brady_ratio = brady_count / count_beats

    if tachy_ratio >= required_ratio and avg_hr >= tachy_threshold:
        severity = "high" if avg_hr >= severe_tachy_threshold else "moderate"
        events.append({
            "event_type": "TACHYCARDIA",
            "triggered": True,
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "tachy_threshold": tachy_threshold,
                "tachy_ratio": tachy_ratio,
                "observed_duration_sec": duration,
                "detection_method": "sustained time window",
                "window_start_time": window_start,
                "window_start_beat": window_start_beat,
                "reason": f"Average HR {avg_hr:.1f} bpm exceeds threshold {tachy_threshold} bpm ({tachy_ratio*100:.0f}% of beats above threshold)",
                "ecg_findings": [f"Average HR {avg_hr:.1f} bpm", f"HR threshold {tachy_threshold} bpm", f"{tachy_ratio*100:.0f}% of beats above threshold"],
            }
        })

    elif brady_ratio >= required_ratio and avg_hr <= brady_threshold:
        severity = "high" if avg_hr <= severe_brady_threshold else "moderate"
        events.append({
            "event_type": "BRADYCARDIA",
            "triggered": True,
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "brady_threshold": brady_threshold,
                "brady_ratio": brady_ratio,
                "observed_duration_sec": duration,
                "detection_method": "sustained time window",
                "window_start_time": window_start,
                "window_start_beat": window_start_beat,
                "reason": f"Average HR {avg_hr:.1f} bpm below threshold {brady_threshold} bpm ({brady_ratio*100:.0f}% of beats below threshold)",
                "ecg_findings": [f"Average HR {avg_hr:.1f} bpm", f"HR threshold {brady_threshold} bpm", f"{brady_ratio*100:.0f}% of beats below threshold"],
            }
        })

    return events

# def detect_rate_abnormalities(
#     beats_history,
#     tachy_threshold: float = 100.0,
#     brady_threshold: float = 60.0,
#     severe_tachy_threshold: float = 140.0,
#     severe_brady_threshold: float = 40.0,
#     min_duration_beats: int = 5,
#     required_ratio: float = 0.80,
#     min_confidence: float = 0.70,
#     min_quality: float = 0.50
# ):
#     """
#     Detect sustained sinus tachycardia and bradycardia.

#     Excludes ventricular beats because VT/PVC runs
#     are handled by dedicated detectors.
#     """

#     if not beats_history:
#         return []

#     if len(beats_history) < min_duration_beats:
#         return []

#     recent_beats = beats_history[-min_duration_beats:]

#     valid_beats = []

#     for beat in recent_beats:

#         if not isinstance(beat, dict):
#             continue

#         if (
#             beat.get(
#                 "prediction_confidence",
#                 0.0
#             )
#             < min_confidence
#         ):
#             continue

#         if (
#             beat.get(
#                 "signal_quality_score",
#                 0.0
#             )
#             < min_quality
#         ):
#             continue

#         # Exclude ventricular beats
#         if (
#             beat.get(
#                 "predicted_label"
#             ) == 2
#         ):
#             continue

#         hr = beat.get(
#             "heart_rate"
#         )

#         if hr is None:
#             continue

#         if hr <= 0:
#             continue

#         valid_beats.append(
#             beat
#         )

#     if len(valid_beats) < min_duration_beats:
#         return []

#     heart_rates = [
#         float(
#             beat["heart_rate"]
#         )
#         for beat in valid_beats
#     ]

#     avg_hr = float(
#         sum(heart_rates)
#         /
#         len(heart_rates)
#     )

#     start_time = valid_beats[0].get(
#         "timestamp"
#     )

#     end_time = valid_beats[-1].get(
#         "timestamp"
#     )

#     observed_duration_sec = None

#     if (
#         start_time is not None
#         and end_time is not None
#     ):
#         observed_duration_sec = max(
#             0.0,
#             float(end_time)
#             -
#             float(start_time)
#         )

#     tachy_beats = sum(
#         1
#         for hr in heart_rates
#         if hr >= tachy_threshold
#     )

#     brady_beats = sum(
#         1
#         for hr in heart_rates
#         if hr <= brady_threshold
#     )

#     tachy_ratio = (
#         tachy_beats
#         /
#         len(heart_rates)
#     )

#     brady_ratio = (
#         brady_beats
#         /
#         len(heart_rates)
#     )

#     tachy_triggered = (
#         tachy_ratio >= required_ratio
#         and avg_hr >= tachy_threshold
#     )

#     brady_triggered = (
#         brady_ratio >= required_ratio
#         and avg_hr <= brady_threshold
#     )

#     events = []

#     # ---------------------------------------
#     # Tachycardia
#     # ---------------------------------------

#     if tachy_triggered:

#         severity = (
#             "high"
#             if avg_hr >= severe_tachy_threshold
#             else "moderate"
#         )

#         events.append({
#             "event_type": "TACHYCARDIA",
#             "triggered": True,
#             "severity": severity,
#             "metadata_json": {

#                 "average_hr":
#                     avg_hr,

#                 "tachy_threshold":
#                     tachy_threshold,

#                 "tachy_ratio":
#                     tachy_ratio,

#                 "tachy_beats":
#                     tachy_beats,

#                 "evaluated_beats":
#                     len(valid_beats),

#                 "observed_duration_sec":
#                     observed_duration_sec
#             }
#         })

#     # ---------------------------------------
#     # Bradycardia
#     # ---------------------------------------

#     elif brady_triggered:

#         severity = (
#             "high"
#             if avg_hr <= severe_brady_threshold
#             else "moderate"
#         )

#         events.append({
#             "event_type": "BRADYCARDIA",
#             "triggered": True,
#             "severity": severity,
#             "metadata_json": {

#                 "average_hr":
#                     avg_hr,

#                 "brady_threshold":
#                     brady_threshold,

#                 "brady_ratio":
#                     brady_ratio,

#                 "brady_beats":
#                     brady_beats,

#                 "evaluated_beats":
#                     len(valid_beats),

#                 "observed_duration_sec":
#                     observed_duration_sec
#             }
#         })

#     return events



# Pattern-based cooldown windows (in beats) per event type.
# An event will only fire again when at least this many new beats
# have arrived since the last trigger, ensuring the sliding window
# has moved past the original pattern that caused the event.
_PATTERN_COOLDOWN_BEATS: Dict[str, int] = {
    "COUPLET": 50,
    "BIGEMINY": 50,
    "TRIGEMINY": 50,
    "QUADRIGEMINY": 50,
    "ATRIAL_BIGEMINY": 50,
    "ATRIAL_TRIGEMINY": 50,
    "ATRIAL_QUADRIGEMINY": 50,
    "ATRIAL_COUPLET": 50,
    "ATRIAL_TRIPLET": 50,
    "HIGH_PVC_BURDEN": 50,
    "BRADYCARDIA": 50,
    "TACHYCARDIA": 50,
    "EXTREME_BRADYCARDIA": 50,
    "EXTREME_TACHYCARDIA": 50,
    "VT_RUN": 50,
    "AFIB_DETECTED": 50,
    "AFLUTTER_SUSPECTED": 50,
    "POSSIBLE_LONG_QT": 50,
    "POSSIBLE_ISCHEMIC_PATTERN": 50,
    "POSSIBLE_HEART_FAILURE_PATTERN": 50,
    "DISEASE_WPW_PREEXCITATION": 50,
    "DISEASE_BRUGADA_SYNDROME": 50,
    "DISEASE_HCM": 50,
    "DISEASE_ARVC": 50,
    "DISEASE_PULMONARY_EMBOLISM": 50,
    "DISEASE_PERICARDITIS": 50,
    "DISEASE_CARDIAC_TAMPONADE": 50,
    "DISEASE_LVH_HYPERTENSION": 50,
    "DISEASE_PULMONARY_HYPERTENSION": 50,
    "DISEASE_CARDIAC_AMYLOIDOSIS": 50,
    "DISEASE_VENTRICULAR_FIBRILLATION": 50,
    "DISEASE_DETECTED": 50,
    "PAUSE_DETECTED": 50,
    "PROLONGED_ASYSTOLE": 50,
    "RR_IRREGULARITY_SUGGESTIVE": 50,
    "LOW_SIGNAL_QUALITY": 50,
    "FIRST_DEGREE_AV_BLOCK": 50,
    "MOBITZ_I_AV_BLOCK": 50,
    "MOBITZ_II_AV_BLOCK": 50,
    "SECOND_DEGREE_AV_BLOCK_2TO1": 50,
    "HIGH_GRADE_AV_BLOCK": 50,
    "THIRD_DEGREE_AV_BLOCK": 50,
}

# Global in-memory tracker: {session_id: {event_type: last_beat_index}}
_last_triggered_tracker: Dict[str, Dict[str, int]] = {}

def process_event_with_cooldown(
    db_connection,
    session_id,
    event,
    beat_index: int = 0,
    window_size: int = 50,
):
    """
    Store a detected event in the database with pattern-based cooldown.
    
    An event is only stored (and returned as "created") if enough beats
    have passed since the last time the same event type was triggered.
    This prevents the same pattern from re-firing on every beat while
    the sliding window still contains the original pattern.
    
    The cooldown is in beats, not seconds, so it naturally adapts to
    the heart rate. For example, with a window_size of 50 beats,
    an event can fire again once 50 new beats have arrived.
    """
    # Use window_size as cooldown to ensure pattern moves out of analysis window
    # Dynamically fall back or resolve using _PATTERN_COOLDOWN_BEATS, ensuring it's at least the window_size
    event_type = event.get("event_type", "")
    cooldown = max(_PATTERN_COOLDOWN_BEATS.get(event_type, window_size), window_size)
    
    # Check if this event was recently triggered
    session_tracker = _last_triggered_tracker.setdefault(session_id, {})
    last_beat = session_tracker.get(event.get("event_type", ""), -cooldown)
    
    beats_since_last = beat_index - last_beat
    if beats_since_last < cooldown:
        return "cooldown"
    
    # Store the event and update tracker
    db_connection.insert_rhythm_event(event)
    session_tracker[event.get("event_type", "")] = beat_index
    return "created"

import math


def calculate_rr_irregularity(beats_history):
    """
    Calculate RR interval variability metrics.

    Returns:
    {
        "rr_mean": float,
        "rr_median": float,
        "rr_std": float,
        "rr_cv": float,
        "rmssd": float,
        "rr_min": float,
        "rr_max": float
    }

    Notes:
    - RR intervals are normalized to seconds.
    - No heuristic irregularity score is produced.
    - Intended for AF, Flutter, SVT and AV-block support logic.
    """

    rr_intervals = []

    for beat in beats_history:

        rr = beat.get("rr_interval")

        if rr is None:
            continue

        if rr <= 0:
            continue

        # ---------------------------------------
        # Normalize RR to seconds
        # ---------------------------------------
        if rr > 10:
            rr = rr / 1000.0

        rr_intervals.append(float(rr))

    if len(rr_intervals) < 5:
        return None

    rr_mean = (
        sum(rr_intervals)
        / len(rr_intervals)
    )

    if rr_mean <= 0:
        return None

    rr_sorted = sorted(rr_intervals)

    if len(rr_sorted) % 2 == 0:

        mid = len(rr_sorted) // 2

        rr_median = (
            rr_sorted[mid - 1]
            + rr_sorted[mid]
        ) / 2.0

    else:

        rr_median = rr_sorted[
            len(rr_sorted) // 2
        ]

    # ---------------------------------------
    # Standard deviation
    # Using sample SD
    # ---------------------------------------

    variance = (
        sum(
            (rr - rr_mean) ** 2
            for rr in rr_intervals
        )
        / (len(rr_intervals) - 1)
    )

    rr_std = math.sqrt(variance)

    # ---------------------------------------
    # Coefficient of variation
    # ---------------------------------------

    rr_cv = rr_std / rr_mean

    # ---------------------------------------
    # RMSSD
    # Root Mean Square of Successive Differences
    # ---------------------------------------

    successive_diffs = []

    for i in range(1, len(rr_intervals)):

        successive_diffs.append(
            rr_intervals[i]
            - rr_intervals[i - 1]
        )

    rmssd = math.sqrt(
        sum(
            diff ** 2
            for diff in successive_diffs
        )
        / len(successive_diffs)
    )

    return {
        "rr_mean": float(rr_mean),
        "rr_median": float(rr_median),
        "rr_std": float(rr_std),
        "rr_cv": float(rr_cv),
        "rmssd": float(rmssd),
        "rr_min": float(min(rr_intervals)),
        "rr_max": float(max(rr_intervals))
    }

def detect_rr_irregularity_pattern(
    beats_history,
    min_window_beats: int = 30,
    cv_threshold: float = 0.15,
    rmssd_threshold_sec: float = 0.10,
    max_pvc_apc_fraction: float = 0.20,
):
    """
    Lightweight, non-diagnostic flag for RR-interval irregularity that
    is *suggestive of* possible AF, based purely on RR statistics over
    a short rolling window (30-60 beats is the range supported by
    published RR-based AF-screening work — see project research notes).

    This is deliberately NOT the same thing as AFIB_DETECTED. It does
    not analyze P-wave presence/absence, does not require sustained
    duration, and is explicitly a backend/display-only alert
    (Event_Manager: trigger_agent=False). Its only purpose is to
    surface "this rhythm looks irregular, might be worth a closer
    look" without waking up the agent — persistent/strengthening
    irregularity is expected to eventually cross into the real
    AFIB_DETECTED disease-detector path, which does the fuller
    analysis.

    Uses the coefficient of variation (CV = SD/mean) and RMSSD of RR
    intervals as the two irregularity signals, both drawn from
    calculate_rr_irregularity() (already computed elsewhere in this
    module, from the rr_interval field NeuroKit2 provides per beat —
    no new features required). A high proportion of ventricular/
    supraventricular ectopic beats in the window can itself inflate RR
    irregularity without reflecting true AF-pattern chaos, so windows
    with high ectopy burden are excluded here to avoid double-counting
    with the bigeminy/trigeminy/couplet detectors above.
    """

    if not beats_history or len(beats_history) < min_window_beats:
        return None

    window = beats_history[-min_window_beats:]

    ectopic_count = sum(
        1 for beat in window
        if beat.get("predicted_label") in (LABEL_S, LABEL_V)
    )
    ectopic_fraction = ectopic_count / len(window)
    if ectopic_fraction > max_pvc_apc_fraction:
        # Irregularity here is more likely explained by ectopy burden
        # than by an AF-like pattern — defer to HIGH_PVC_BURDEN /
        # bigeminy-trigeminy-couplet detectors instead.
        return None

    stats = calculate_rr_irregularity(window)
    if stats is None:
        return None

    cv_flag = stats["rr_cv"] >= cv_threshold
    rmssd_flag = stats["rmssd"] >= rmssd_threshold_sec

    if not (cv_flag and rmssd_flag):
        return None

    severity = (
        "moderate"
        if stats["rr_cv"] >= (cv_threshold * 1.5)
        else "low"
    )

    return {
        "event_type": "RR_IRREGULARITY_SUGGESTIVE",
        "severity": severity,
        "metadata_json": {
            "rr_cv": stats["rr_cv"],
            "rmssd_sec": stats["rmssd"],
            "rr_mean_sec": stats["rr_mean"],
            "rr_std_sec": stats["rr_std"],
            "cv_threshold": cv_threshold,
            "rmssd_threshold_sec": rmssd_threshold_sec,
            "ectopic_fraction": ectopic_fraction,
            "evaluated_beats": len(window),
            "note": (
                "Backend/display alert only — RR-pattern irregularity, "
                "not a confirmed AFIB diagnosis. See AFIB_DETECTED for "
                "the full disease-detector evaluation."
            ),
        },
    }


def detect_signal_quality_event(
    beats_history: List[Dict[str, Any]],
    min_quality_threshold: float = 0.30,
    min_window_beats: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Standalone signal quality event. Emitted when the average signal quality
    in the window falls below a threshold, indicating noise/artifact.

    This is a backend/display alert only — it explains why other detectors
    may not fire, rather than being a clinical finding itself.
    """
    if not beats_history or len(beats_history) < min_window_beats:
        return None

    quality_scores = [
        float(b.get("signal_quality_score", 1.0))
        for b in beats_history[-min_window_beats:]
        if b.get("signal_quality_score") is not None
    ]
    if not quality_scores:
        return None

    avg_quality = float(np.mean(quality_scores))
    if avg_quality >= min_quality_threshold:
        return None

    poor_ratio = sum(1 for q in quality_scores if q < min_quality_threshold) / len(quality_scores)

    return {
        "event_type": "LOW_SIGNAL_QUALITY",
        "severity": "moderate" if avg_quality < 0.20 else "low",
        "metadata_json": {
            "avg_signal_quality": round(avg_quality, 3),
            "poor_quality_ratio": round(poor_ratio, 2),
            "threshold": min_quality_threshold,
            "evaluated_beats": len(quality_scores),
            "note": "Backend/display alert — signal quality is low, other detectors may be unreliable.",
        },
    }


def _has_condition(patient_metadata, condition_name):
    if not patient_metadata:
        return False
    # Check flat keys
    if patient_metadata.get(condition_name):
        return True
    
    # Check comorbidities
    comorbidities = patient_metadata.get("comorbidities", [])
    if isinstance(comorbidities, str):
        comorbidities = [comorbidities]
    if any(condition_name.lower() in str(c).lower() for c in comorbidities):
        return True
        
    # Check dynamic_symptoms
    dynamic = patient_metadata.get("dynamic_symptoms", {})
    if str(dynamic.get(condition_name, "")).strip().lower() in ["yes", "true", "1", "y"]:
        return True
        
    return False

def estimate_qt_interval(
    samples,
    sampling_rate=360,
    t_peak_position=-1,
    p_peak_position=-1
):
    """Estimate QT interval in ms from QRS onset to T-wave end."""
    signal = np.asarray(samples, dtype=np.float64)
    if len(signal) < int(0.30 * sampling_rate):
        return None

    search_start = len(signal) // 8
    search_end = 7 * len(signal) // 8
    if search_end <= search_start:
        return None

    r_peak = search_start + int(np.argmax(signal[search_start:search_end]))

    back_limit = int(0.08 * sampling_rate)
    bl_start = max(0, r_peak - int(0.14 * sampling_rate))
    bl_end = max(1, r_peak - int(0.08 * sampling_rate))
    baseline_q = (
        float(np.mean(signal[bl_start:bl_end]))
        if bl_end > bl_start
        else float(signal[0])
    )

    qrs_onset = r_peak
    for index in range(r_peak, max(0, r_peak - back_limit), -1):
        if abs(signal[index] - baseline_q) < 0.05:
            qrs_onset = index
            break

    t_end = None
    if t_peak_position != -1 and t_peak_position > r_peak:
        t_peak_position = int(t_peak_position)
        tangent_window = int(0.32 * sampling_rate)
        search_end_t = min(len(signal) - 1, t_peak_position + tangent_window)
        bl_t_start = max(0, search_end_t - int(0.08 * sampling_rate))
        baseline_t = float(np.mean(signal[bl_t_start:search_end_t]))

        is_inverted = signal[t_peak_position] < baseline_t
        max_slope = 0.0
        max_slope_idx = t_peak_position

        for index in range(t_peak_position, search_end_t):
            slope = (
                signal[index + 1] - signal[index]
                if is_inverted
                else signal[index] - signal[index + 1]
            )
            if slope > max_slope:
                max_slope = float(slope)
                max_slope_idx = index

        if max_slope > 1e-4:
            slope = max_slope if is_inverted else -max_slope
            y1 = float(signal[max_slope_idx])
            t_end_calc = int((baseline_t - y1) / slope + max_slope_idx)
            t_end = max(
                t_peak_position,
                min(t_end_calc, t_peak_position + int(0.25 * sampling_rate))
            )
            t_end = min(t_end, len(signal) - 1)
        else:
            t_end = min(
                t_peak_position + int(0.12 * sampling_rate),
                len(signal) - 1
            )
    else:
        peak_value = float(signal[r_peak])
        baseline_len = max(1, int(0.15 * sampling_rate))
        baseline = float(np.mean(signal[:baseline_len]))
        threshold = baseline + (peak_value - baseline) * 0.15
        scan_start = r_peak + int(0.12 * sampling_rate)
        for index in range(scan_start, len(signal)):
            if signal[index] <= threshold:
                t_end = index
                break

    if t_end is None:
        return None

    qt_samples = t_end - qrs_onset
    if qt_samples <= 0:
        return None

    return float(qt_samples / sampling_rate * 1000.0)


def calculate_qtc(qt_ms, rr_interval):
    """Bazett QTc in ms from QT ms and RR seconds or ms."""
    if qt_ms is None or rr_interval is None or rr_interval <= 0:
        return None

    rr_sec = rr_interval / 1000 if rr_interval > 10 else rr_interval
    if rr_sec <= 0:
        return None

    try:
        return float(qt_ms / math.sqrt(rr_sec))
    except Exception:
        return None

def detect_escape_beats(beats_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Detect escape beats with rate range validation.

    Clinical rate ranges (AHA/ACC):
      - Junctional escape: 40-60 bpm
      - Ventricular escape: 30-40 bpm
      - Atrial escape: varies (typically 50-90 bpm, no strict range)
    """
    if len(beats_history) < 5:
        return None
    
    last_beat = beats_history[-1]
    last_rr = last_beat.get("rr_interval", 0)
    last_hr = last_beat.get("heart_rate", 0)
    
    prev_rrs = [b.get("rr_interval", 0) for b in beats_history[-5:-1]]
    avg_prev_rr = sum(prev_rrs) / len(prev_rrs) if prev_rrs else 0
    
    if avg_prev_rr > 0 and last_rr > 1.2 * avg_prev_rr and last_rr > 1000:
        p_detected = last_beat.get("p_wave_detected", False)
        qrs_width = last_beat.get("qrs_width") or 0
        
        # Classify by morphology
        if qrs_width > 120:
            escape_type = "VENTRICULAR"
            # Ventricular escape rate: 30-40 bpm
            if last_hr and (last_hr < 20 or last_hr > 45):
                return None  # Rate outside ventricular escape range
        elif not p_detected:
            escape_type = "JUNCTIONAL"
            # Junctional escape rate: 40-60 bpm
            if last_hr and (last_hr < 35 or last_hr > 65):
                return None  # Rate outside junctional escape range
        else:
            escape_type = "ATRIAL"
            # Atrial escape: typically 50-90 bpm, wider range
            if last_hr and (last_hr < 40 or last_hr > 100):
                return None
        
        return {
            "event_type": f"{escape_type}_ESCAPE_BEAT",
            "severity": "low",
            "metadata_json": {
                "escape_rr": last_rr,
                "avg_prev_rr": avg_prev_rr,
                "heart_rate": last_hr,
                "escape_type": escape_type,
                "reason": f"{escape_type} escape beat: RR {last_rr:.0f}ms (avg prev RR {avg_prev_rr:.0f}ms, HR {last_hr:.0f} bpm)",
                "ecg_findings": [f"Escape type: {escape_type}", f"RR {last_rr:.0f}ms", f"HR {last_hr:.0f} bpm"],
            }
        }
    return None


def _as_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "detected", "positive"}
    return bool(value)


def _metadata_value(patient_metadata: Optional[Dict[str, Any]], *keys, default=None):
    if not patient_metadata:
        return default
    for key in keys:
        if key in patient_metadata and patient_metadata[key] is not None:
            return patient_metadata[key]
    return default


def _build_disease_detector_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None
) -> ECGFeatures:
    valid = [beat for beat in beats_history if _as_float(beat.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history
    last = source[-1] if source else {}

    rr_values = [
        _as_float(beat.get("rr_interval"))
        for beat in source
        if _as_float(beat.get("rr_interval")) and _as_float(beat.get("rr_interval")) > 0
    ]
    rr_mean = float(np.mean(rr_values)) if rr_values else None
    rr_std = float(np.std(rr_values)) if len(rr_values) > 1 else 0.0
    rr_irregular = bool(rr_mean and rr_std and (rr_std / rr_mean) > 0.10)

    heart_rates = [
        _as_float(beat.get("heart_rate"))
        for beat in source
        if _as_float(beat.get("heart_rate")) and _as_float(beat.get("heart_rate")) > 0
    ]
    heart_rate = float(np.mean(heart_rates)) if heart_rates else (60000.0 / rr_mean if rr_mean else None)

    p_amp = _as_float(last.get("p_wave_amplitude"))
    p_duration = _as_float(last.get("p_wave_width_ms"))
    qrs_duration = _as_float(last.get("qrs_width"))
    qrs_axis = _as_float(last.get("qrs_axis_deg"))
    qtc_ms = _as_float(last.get("qtc_fridericia"), _as_float(last.get("qtc"), _as_float(last.get("qtc_bazett"))))
    t_amp = _as_float(last.get("t_wave_amplitude"))

    qrs_values = [
        _as_float(beat.get("qrs_width"))
        for beat in source
        if _as_float(beat.get("qrs_width")) is not None
    ]
    r_values = [
        abs(_as_float(beat.get("r_amplitude"), 0.0))
        for beat in source
        if _as_float(beat.get("r_amplitude")) is not None
    ]
    st_values = [
        _as_float(beat.get("st_deviation"))
        for beat in source
        if _as_float(beat.get("st_deviation")) is not None
    ]

    event_types = {event.get("event_type") for event in existing_events or []}
    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", "medical_history", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]

    sex = _metadata_value(patient_metadata, "sex", "gender")
    age = _metadata_value(patient_metadata, "age")
    try:
        age = int(age) if age is not None else None
    except (TypeError, ValueError):
        age = None

    af_probability = 0.0
    if "AFIB_DETECTED" in event_types:
        af_probability = 0.85
    elif rr_irregular and source:
        missing_p_ratio = sum(1 for beat in source if not _as_bool(beat.get("p_wave_detected"), True)) / len(source)
        if missing_p_ratio >= 0.50:
            af_probability = 0.70

    return ECGFeatures(
        rr_mean_ms=rr_mean,
        rr_std_ms=rr_std,
        rr_irregular=rr_irregular,
        heart_rate_bpm=heart_rate,
        p_wave_present=_as_bool(last.get("p_wave_detected"), p_amp is not None),
        p_wave_amplitude_mv=p_amp,
        p_duration_ms=p_duration,
        p_wave_peaked=bool(p_amp is not None and abs(p_amp) >= 0.25),
        p_wave_broad=bool(p_duration is not None and p_duration > 120.0),
        pr_interval_ms=_as_float(last.get("pr_interval_ms")),
        qrs_duration_ms=qrs_duration,
        qrs_axis_degrees=qrs_axis,
        lbbb_pattern=bool(qrs_duration is not None and qrs_duration >= 120.0 and (qrs_axis is None or qrs_axis <= -30.0)),
        rbbb_pattern=bool(qrs_duration is not None and qrs_duration >= 120.0 and (qrs_axis is None or qrs_axis >= 90.0)),
        delta_wave_present=_as_bool(last.get("delta_wave_detected")),
        epsilon_wave_present=_as_bool(last.get("epsilon_wave_detected")),
        r_wave_amplitude_mv=_as_float(last.get("r_amplitude"), _as_float(last.get("qrs_voltage"))),
        q_wave_deep=bool(_as_float(last.get("q_amplitude"), 0.0) <= -0.30),
        low_voltage=bool(r_values and float(np.mean(r_values)) < 0.50),
        electrical_alternans=_as_bool(last.get("electrical_alternans_detected")),
        st_deviation_mv=float(np.mean(st_values)) if st_values else _as_float(last.get("st_deviation")),
        st_slope=last.get("st_slope"),
        t_wave_amplitude_mv=t_amp,
        t_wave_inverted=_as_bool(last.get("t_wave_inverted")) or str(last.get("t_wave_polarity", "")).lower() == "negative",
        t_wave_peaked=bool(t_amp is not None and abs(t_amp) >= 1.0),
        qt_interval_ms=_as_float(last.get("qt_interval")),
        qtc_ms=qtc_ms,
        u_wave_prominent=_as_bool(last.get("u_wave_detected")) and abs(_as_float(last.get("u_wave_amplitude"), 0.0)) >= 0.10,
        flutter_baseline_detected=_as_bool(last.get("flutter_baseline_detected")),
        af_probability=af_probability,
        vt_detected="VT_RUN" in event_types,
        vf_detected=False,
        paced_rhythm=last.get("rhythm_classification") == "paced",
        cnn_label=last.get("predicted_label"),
        window_beat_count=len(source),
        age=age,
        sex=sex,
        known_diagnoses=list(diagnoses) if diagnoses else [],
    )


def _build_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None
) -> Optional[ECGWindowFeatures]:
    """
    Build ECGWindowFeatures for temporal LQTS analysis.
    Returns None if insufficient data.
    """
    valid = [beat for beat in beats_history if _as_float(beat.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history
    
    if len(source) < 25:  # MIN_VALID_BEATS requirement from ECGWindowFeatures
        return None
    
    # Extract QTc values in temporal order
    qtc_values = []
    rr_values = []
    for beat in source:
        qtc = _as_float(
            beat.get("qtc_fridericia"),
            _as_float(beat.get("qtc"), _as_float(beat.get("qtc_bazett")))
        )
        rr = _as_float(beat.get("rr_interval"))
        if qtc is not None:
            qtc_values.append(qtc)
        if rr is not None and rr > 0:
            rr_values.append(rr)
    
    if len(qtc_values) < 25:
        return None
    
    event_types = {event.get("event_type") for event in existing_events or []}
    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", "medical_history", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]
    
    sex = _metadata_value(patient_metadata, "sex", "gender")
    age = _metadata_value(patient_metadata, "age")
    try:
        age = int(age) if age is not None else None
    except (TypeError, ValueError):
        age = None
    
    # Check for bradycardia for age
    heart_rates = [
        _as_float(beat.get("heart_rate"))
        for beat in source
        if _as_float(beat.get("heart_rate")) and _as_float(beat.get("heart_rate")) > 0
    ]
    avg_hr = float(np.mean(heart_rates)) if heart_rates else None
    bradycardia_for_age = False
    if age and avg_hr:
        if age < 1 and avg_hr < 100:
            bradycardia_for_age = True
        elif age < 3 and avg_hr < 90:
            bradycardia_for_age = True
        elif age < 10 and avg_hr < 70:
            bradycardia_for_age = True
        elif avg_hr < 60:
            bradycardia_for_age = True
    
    # T-wave biphasic fraction
    biphasic_count = sum(1 for beat in source if str(beat.get("t_wave_polarity", "")).lower() == "biphasic")
    t_wave_biphasic_fraction = biphasic_count / len(source) if source else 0.0
    
    return ECGWindowFeatures(
        window_id=f"temporal_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        beat_count=len(beats_history),
        valid_beat_count=len(source),
        qtc_values_ms=qtc_values,
        rr_values_ms=rr_values if rr_values else None,
        sex=sex,
        age=age,
        resting=True,  # Assume resting for continuous monitoring
        rhythm_stable=not bool("AFIB_DETECTED" in event_types or "VT_RUN" in event_types),
        vt_detected="VT_RUN" in event_types,
        tdp_detected=False,  # Would need specific TdP detection
        t_wave_biphasic_fraction=t_wave_biphasic_fraction,
        macroscopic_twa_present=_as_bool(source[-1].get("macroscopic_twa_detected")) if source else False,
        macroscopic_twa_consecutive_pairs=0,  # Would need TWA detection
        bradycardia_for_age=bradycardia_for_age,
        known_diagnoses=list(diagnoses) if diagnoses else [],
    )


def _disease_event_type(disease_name: str) -> str:
    normalized = disease_name.lower()
    for key, event_type in _DISEASE_EVENT_MAP.items():
        if key in normalized:
            return event_type
    return "DISEASE_DETECTED"


def _disease_result_to_event(result) -> Dict[str, Any]:
    severity = getattr(result.severity, "value", result.severity)
    metadata = {
        "source": "disease_detector",
        "disease": result.disease,
        "confidence": float(result.confidence),
        "severity": severity,
        "reason": result.reason,
        "ecg_findings": result.ecg_findings,
        "missing_data": result.missing_data,
        "symptom_questions": result.symptom_questions,
        "symptom_source": result.symptom_source,
        "rag_trigger": bool(result.rag_trigger),
        "icd10_codes": result.icd10_codes,
    }
    return {
        "event_type": _disease_event_type(result.disease),
        "severity": severity,
        "metadata_json": metadata
    }


def _merge_disease_events(events: List[Dict[str, Any]], disease_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = list(events)
    by_type = {event.get("event_type"): event for event in merged}

    for disease_event in disease_events:
        event_type = disease_event.get("event_type")
        existing = by_type.get(event_type)
        if not existing:
            merged.append(disease_event)
            by_type[event_type] = disease_event
            continue

        existing_meta = existing.setdefault("metadata_json", {})
        disease_meta = disease_event.get("metadata_json", {})
        detector_results = existing_meta.setdefault("disease_detector_results", [])
        detector_results.append(disease_meta)

        old_rank = _SEVERITY_RANK.get(str(existing.get("severity", "info")).lower(), 0)
        new_rank = _SEVERITY_RANK.get(str(disease_event.get("severity", "info")).lower(), 0)
        if new_rank > old_rank:
            existing["severity"] = disease_event["severity"]

    return merged


def _build_arvc_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None,
    db_connection: Any = None,
    session_id: Optional[str] = None
) -> Optional[ARVCWindowFeatures]:
    """
    Build ARVCWindowFeatures for ARVC detection.
    Returns None if insufficient data.
    
    Requires db_connection and session_id for longitudinal queries
    (PVC count, VT episodes, etc.). If not provided, longitudinal
    fields default to 0/empty and the detector will still run but
    with reduced sensitivity.
    """
    valid = [beat for beat in beats_history if _as_float(beat.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history
    
    if not source:
        return None
    
    last = source[-1] if source else {}
    event_types = {event.get("event_type") for event in existing_events or []}
    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", "medical_history", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]
    
    sex = _metadata_value(patient_metadata, "sex", "gender")
    age = _metadata_value(patient_metadata, "age")
    try:
        age = int(age) if age is not None else None
    except (TypeError, ValueError):
        age = None
    
    # ── Current window: epsilon wave fraction ──────────────────────
    epsilon_count = sum(1 for beat in source if _as_bool(beat.get("epsilon_wave_detected")))
    epsilon_fraction = epsilon_count / len(source) if source else 0.0
    
    # ── Current window: RBBB fraction ──────────────────────────────
    rbbb_count = sum(
        1 for beat in source
        if _as_float(beat.get("qrs_width"), 0) >= 120.0
        and (_as_float(beat.get("qrs_axis_deg"), 0) >= 90.0)
    )
    rbbb_fraction = rbbb_count / len(source) if source else 0.0
    
    # ── Current window: median QRS duration ────────────────────────
    qrs_values = [
        _as_float(beat.get("qrs_width"))
        for beat in source
        if _as_float(beat.get("qrs_width")) is not None
    ]
    qrs_duration_ms = float(np.median(qrs_values)) if qrs_values else None
    
    # ── Current window: T-wave inversion fraction ──────────────────
    t_inv_count = sum(
        1 for beat in source
        if _as_bool(beat.get("t_wave_inverted"))
        or str(beat.get("t_wave_polarity", "")).lower() == "negative"
    )
    t_inversion_fraction = t_inv_count / len(source) if source else 0.0
    
    # ── Current window: VT detection ───────────────────────────────
    vt_detected_this_window = "VT_RUN" in event_types
    vt_lbbb_morphology_this_window = (
        vt_detected_this_window
        and _as_float(last.get("qrs_width"), 0) >= 120.0
        and (_as_float(last.get("qrs_axis_deg"), 0) <= -30.0)
    )
    
    # ── Longitudinal: PVC burden (24h) ─────────────────────────────
    pvc_count_24h = 0
    pvc_lbbb_fraction_24h = 0.0
    vt_episodes_24h: List[VTEpisode] = []
    prior_epsilon_windows = 0
    prior_t_inversion_windows = 0
    prior_max_ecg_tfc_score = 0
    
    if db_connection is not None and session_id is not None:
        try:
            pvc_count_24h = db_connection.count_pvcs_last_24h(session_id)
            pvc_lbbb_count = db_connection.count_pvcs_lbbb_last_24h(session_id)
            pvc_lbbb_fraction_24h = (
                pvc_lbbb_count / pvc_count_24h
                if pvc_count_24h > 0
                else 0.0
            )
            vt_episode_count = db_connection.count_vt_episodes_last_24h(session_id)
            # Fetch actual VT events with timestamps for episode tracking
            vt_events = db_connection.get_session_events(session_id)
            vt_events_24h = [
                ev for ev in vt_events
                if ev.get("event_type") == "VT_RUN"
                and ev.get("event_start_time", 0) >= (
                    datetime.now().timestamp() - 86400
                )
            ]
            for ev in vt_events_24h:
                vt_episodes_24h.append(VTEpisode(
                    occurred_at=datetime.fromtimestamp(
                        float(ev.get("event_start_time", 0))
                    ),
                    duration_beats=0,  # would need consecutive beat count
                    lbbb_morphology=bool(
                        _as_float(last.get("qrs_width"), 0) >= 120.0
                        and _as_float(last.get("qrs_axis_deg"), 0) <= -30.0
                    ),
                    superior_axis=None,  # single-lead cannot determine
                ))
            prior_epsilon_windows = db_connection.count_prior_arvc_epsilon_windows(session_id)
            prior_t_inversion_windows = db_connection.count_prior_arvc_t_inversion_windows(session_id)
            prior_max_ecg_tfc_score = db_connection.get_max_arvc_ecg_tfc_score(session_id)
        except Exception:
            pass
    
    return ARVCWindowFeatures(
        window_id=f"arvc_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        valid_beat_count=len(source),
        age=age,
        sex=sex,
        known_diagnoses=list(diagnoses) if diagnoses else [],
        epsilon_fraction=epsilon_fraction,
        rbbb_fraction=rbbb_fraction,
        qrs_duration_ms=qrs_duration_ms,
        t_inversion_fraction=t_inversion_fraction,
        vt_detected_this_window=vt_detected_this_window,
        vt_lbbb_morphology_this_window=vt_lbbb_morphology_this_window,
        vt_superior_axis_this_window=None,  # single-lead cannot determine
        pvc_count_24h=pvc_count_24h,
        pvc_lbbb_fraction_24h=pvc_lbbb_fraction_24h,
        vt_episodes_24h=vt_episodes_24h,
        prior_epsilon_windows=prior_epsilon_windows,
        prior_t_inversion_windows=prior_t_inversion_windows,
        prior_max_ecg_tfc_score=prior_max_ecg_tfc_score,
    )


def _build_vt_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None,
    db_connection: Any = None,
    session_id: Optional[str] = None
) -> Optional[VTWindowFeatures]:
    """
    Build VTWindowFeatures from beat history for VT detection.

    Scans beats_history for the longest run of consecutive ventricular
    beats (predicted_label == 2) and computes run-level metrics.

    Returns None if no ventricular run is found.
    """
    valid = [beat for beat in beats_history if _as_float(beat.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history

    if not source:
        return None

    # ── Find longest consecutive ventricular run ────────────────
    max_run_length = 0
    current_run = 0
    run_start_idx = -1
    best_run_start_idx = -1

    for i, beat in enumerate(source):
        if beat.get("predicted_label") == 2:  # LABEL_V
            if current_run == 0:
                run_start_idx = i
            current_run += 1
            if current_run > max_run_length:
                max_run_length = current_run
                best_run_start_idx = run_start_idx
        else:
            current_run = 0

    if max_run_length < 1:
        return None

    # ── Extract beats in the best run ───────────────────────────
    run_beats = source[best_run_start_idx:best_run_start_idx + max_run_length]

    # ── Run duration (wall-clock) ───────────────────────────────
    start_ts = run_beats[0].get("timestamp")
    end_ts = run_beats[-1].get("timestamp")
    run_duration_sec = 0.0
    if start_ts is not None and end_ts is not None:
        run_duration_sec = max(0.0, float(end_ts) - float(start_ts))

    # ── Run rate (mean heart rate during run) ───────────────────
    hr_values = []
    for beat in run_beats:
        hr = _as_float(beat.get("heart_rate"))
        if hr is not None and hr > 0:
            hr_values.append(hr)
        else:
            rr = _as_float(beat.get("rr_interval"))
            if rr is not None and rr > 0:
                hr_values.append(60000.0 / rr)
    run_rate_bpm = float(np.mean(hr_values)) if hr_values else None

    # ── Median QRS duration during run ──────────────────────────
    qrs_values = [
        _as_float(beat.get("qrs_width"))
        for beat in run_beats
        if _as_float(beat.get("qrs_width")) is not None
    ]
    qrs_duration_ms = float(np.median(qrs_values)) if qrs_values else None

    # ── Morphology consistency (monomorphic vs polymorphic) ─────
    # Simple heuristic: check if QRS morphology varies significantly
    # within the run. Uses QRS width + axis as a proxy.
    qrs_widths_run = [
        _as_float(beat.get("qrs_width"))
        for beat in run_beats
        if _as_float(beat.get("qrs_width")) is not None
    ]
    qrs_axes_run = [
        _as_float(beat.get("qrs_axis_deg"))
        for beat in run_beats
        if _as_float(beat.get("qrs_axis_deg")) is not None
    ]
    monomorphic = None
    if len(qrs_widths_run) >= 3:
        width_range = max(qrs_widths_run) - min(qrs_widths_run)
        # If QRS width varies by <15ms throughout, likely monomorphic
        monomorphic = width_range < 15.0

    # ── PVC count 24h ───────────────────────────────────────────
    pvc_count_24h = 0
    if db_connection is not None and session_id is not None:
        try:
            pvc_count_24h = db_connection.count_pvcs_last_24h(session_id)
        except Exception:
            pass

    event_types = {event.get("event_type") for event in existing_events or []}
    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", "medical_history", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]
    known_diags = list(diagnoses) if diagnoses else []

    # ── Terminated for compromise (from event context) ──────────
    # In a pure ECG-monitoring pipeline without intervention logs,
    # this defaults to False.
    terminated_for_compromise = False

    return VTWindowFeatures(
        window_id=f"vt_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        consecutive_vt_beats=max_run_length,
        run_duration_sec=run_duration_sec,
        run_rate_bpm=run_rate_bpm,
        qrs_duration_ms=qrs_duration_ms,
        terminated_for_compromise=terminated_for_compromise,
        monomorphic=monomorphic,
        pvc_count_24h=pvc_count_24h,
        known_diagnoses=known_diags,
    )


def _build_vf_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None
) -> Optional[VFWindowFeatures]:
    """
    Build VFWindowFeatures from beat history for VF detection.

    VF is a global rhythm diagnosis — checked from beat-classifier flags
    rather than individual beat morphology. This builder checks for a
    VF-classified run in the window's beat history.

    Returns None if no VF evidence is found.
    """
    valid = [beat for beat in beats_history if _as_float(beat.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history

    if not source:
        return None

    # ── Count consecutive VF-flagged beats ──────────────────────
    vf_count = 0
    for beat in source:
        if _as_bool(beat.get("vf_detected"), False):
            vf_count += 1
        else:
            vf_count = 0  # reset — want consecutive

    # ── Signal quality during the window ────────────────────────
    quality_scores = [
        _as_float(beat.get("signal_quality_score"), 1.0)
        for beat in source
    ]
    avg_quality = float(np.mean(quality_scores)) if quality_scores else 1.0
    signal_quality_ok = avg_quality >= 0.50

    if vf_count <= 0:
        return None

    return VFWindowFeatures(
        window_id=f"vf_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        vf_flag_beats=vf_count,
        signal_quality_ok=signal_quality_ok,
    )


def _detect_group_beating(rr_ms: List[float], min_beats: int = 10, sub_window: int = 20) -> bool:
    """
    Heuristic for variable-block flutter: looks for natural RR-value clusters
    in the most recent sub-window of beats, rather than a whole-window even/odd
    split. Real variable-block flutter produces RR clusters that drift over
    time (e.g. 2:1 sliding toward 3:1), which the old even/odd split couldn't
    detect because the drift washed out into noise over 100+ beats.

    Uses a simple gap-detection heuristic on sorted RR values: if the largest
    gap between adjacent sorted values is much larger than the typical spacing,
    the RR values form natural clusters (group beating). Not a validated
    clinical algorithm — a screening heuristic only.
    """
    if len(rr_ms) < min_beats:
        return False
    # Look at the most recent sub_window beats only, so a slow drift
    # doesn't wash out into noise.
    recent = rr_ms[-sub_window:] if len(rr_ms) >= sub_window else rr_ms
    if len(recent) < min_beats:
        return False
    sorted_rr = sorted(recent)
    diffs = [sorted_rr[i+1] - sorted_rr[i] for i in range(len(sorted_rr)-1)]
    if not diffs:
        return False
    max_gap = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    # A real cluster gap should be much larger than the typical
    # adjacent-value spacing within a cluster.
    has_natural_split = max_gap > 3.0 * mean_diff
    return bool(has_natural_split)


def _print_afl_diagnostic(
    beat_index: int,
    w: Optional["AFLWindowFeatures"],
    result: "DetectionResult",
    beats_history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Print a comprehensive Atrial Flutter diagnostic dashboard to the terminal.
    Only prints when AFLWindowFeatures was provided (non-None).

    Displays every intermediate feature, gate, score contribution, and decision
    in the exact order the detector thinks, making every decision fully explainable.
    """
    from disease_detection.detect_atrial_flutter import _PLAUSIBLE_RATE_RANGE, _MIN_VALID_BEATS

    if w is None:
        return

    sep = "=" * 72
    dash = "-" * 72
    sub = "·" * 40

    # ── Replicate detector logic for score breakdown ──────────────
    rate_in_plausible_range = (
        w.ventricular_rate_bpm is not None
        and _PLAUSIBLE_RATE_RANGE[0] <= w.ventricular_rate_bpm <= _PLAUSIBLE_RATE_RANGE[1]
    )
    extremely_regular = w.rr_cv is not None and w.rr_cv < 0.08
    score = 0.0
    score_components: List[tuple[str, float, str]] = []

    # Component 1: Ventricular rate + regularity
    if rate_in_plausible_range and extremely_regular:
        score_components.append((
            "Ventricular Rate + Regularity",
            0.35,
            f"rate={w.ventricular_rate_bpm:.0f} bpm in [{_PLAUSIBLE_RATE_RANGE[0]}-{_PLAUSIBLE_RATE_RANGE[1]}], CV={w.rr_cv:.3f} < 0.08"
        ))
        score += 0.35
    elif rate_in_plausible_range:
        score_components.append((
            "Ventricular Rate (partial)",
            0.15,
            f"rate={w.ventricular_rate_bpm:.0f} bpm in plausible range, but CV={w.rr_cv:.3f} >= 0.08"
        ))
        score += 0.15

    # Component 2: Group beating
    if w.group_beating_detected and not extremely_regular:
        score_components.append((
            "Group Beating (variable block)",
            0.20,
            "Alternating short/long RR pattern detected"
        ))
        score += 0.20

    # Component 3: Atrial rate
    if w.atrial_rate_bpm is not None:
        if 220 <= w.atrial_rate_bpm <= 350:
            score_components.append((
                "Atrial Rate (typical range)",
                0.20,
                f"atrial_rate={w.atrial_rate_bpm:.0f} bpm in [220-350]"
            ))
            score += 0.20
        elif 200 <= w.atrial_rate_bpm < 220 or 350 < w.atrial_rate_bpm <= 400:
            score_components.append((
                "Atrial Rate (atypical range)",
                0.10,
                f"atrial_rate={w.atrial_rate_bpm:.0f} bpm in [200-400]"
            ))
            score += 0.10
        if w.atrial_rate_std_bpm is not None and w.atrial_rate_bpm > 0:
            rate_stability = w.atrial_rate_std_bpm / w.atrial_rate_bpm
            if rate_stability < 0.08:
                score_components.append((
                    "Atrial Rate Stability",
                    0.05,
                    f"std/mean={rate_stability:.4f} < 0.08"
                ))
                score += 0.05

    # Component 4: AV ratio integer-like
    if w.av_block_ratio_is_integer_like:
        score_components.append((
            "Integer AV Ratio",
            0.15,
            f"ratio={w.av_block_ratio:.1f}:1 ≈ {round(w.av_block_ratio)}:1"
        ))
        score += 0.15

    # Component 5: P-wave absence / flutter baseline
    if w.p_wave_present_fraction < 0.30 or w.flutter_baseline_detected_fraction >= 0.50:
        score_components.append((
            "P-wave Absence / Flutter Baseline",
            0.10,
            f"p_frac={w.p_wave_present_fraction:.2f}, base_frac={w.flutter_baseline_detected_fraction:.2f}"
        ))
        score += 0.10

    score = max(0.0, min(score, 1.0))
    triggered = score >= 0.45

    # ── Determine which gate failed (if any) ─────────────────────
    gate_failed = None
    if w.valid_beat_count < _MIN_VALID_BEATS:
        gate_failed = f"valid_beat_count={w.valid_beat_count} < {_MIN_VALID_BEATS}"
    elif w.p_wave_present_fraction >= 0.70 and w.flutter_baseline_detected_fraction < 0.20:
        gate_failed = (
            f"Discrete P waves in {w.p_wave_present_fraction*100:.0f}% of window "
            f"with no flutter baseline (fraction={w.flutter_baseline_detected_fraction:.2f} < 0.20)"
        )

    # ── Build missing evidence list ──────────────────────────────
    missing_evidence: List[str] = []
    if w.ventricular_rate_bpm is not None:
        if not rate_in_plausible_range:
            missing_evidence.append(
                f"Ventricular rate {w.ventricular_rate_bpm:.0f} bpm outside plausible range [{_PLAUSIBLE_RATE_RANGE[0]:.0f}-{_PLAUSIBLE_RATE_RANGE[1]:.0f}]"
            )
    if w.rr_cv is not None and w.rr_cv >= 0.10:
        missing_evidence.append(f"RR CV {w.rr_cv:.3f} >= 0.10 (not regular enough)")
    if w.atrial_rate_bpm is None:
        missing_evidence.append("Atrial rate estimate unavailable (no flutter baseline detected)")
    elif not (220 <= w.atrial_rate_bpm <= 350):
        missing_evidence.append(f"Atrial rate {w.atrial_rate_bpm:.0f} bpm outside typical flutter range [220-350]")
    if not w.av_block_ratio_is_integer_like and w.av_block_ratio is not None:
        missing_evidence.append(f"AV ratio {w.av_block_ratio:.2f}:1 not integer-like")
    if w.flutter_baseline_detected_fraction < 0.20:
        missing_evidence.append(f"Flutter baseline fraction {w.flutter_baseline_detected_fraction:.2f} < 0.20")
    if w.p_wave_present_fraction >= 0.30 and w.flutter_baseline_detected_fraction < 0.50:
        missing_evidence.append(f"Discrete P waves still present in {w.p_wave_present_fraction*100:.0f}% of beats")

    # ── Build feature evolution (last 10 beats) ──────────────────
    evolution_lines: List[str] = []
    if beats_history:
        recent = beats_history[-10:]
        for b in recent:
            bi = b.get("beat_index", "?")
            brr = _as_float(b.get("rr_interval"))
            bcv = None
            bscore = 0.0
            if brr and w.rr_cv is not None:
                # Approximate per-beat contribution
                bscore = score
            evolution_lines.append(
                f"  Beat {str(bi):>5}  RR={brr or 0:.0f}ms"
            )

    # ══════════════════════════════════════════════════════════════
    # PRINT DASHBOARD
    # ══════════════════════════════════════════════════════════════
    print()
    print(sep)
    print(f"  [AFL DIAGNOSTIC DASHBOARD]  |  beat_index={beat_index}")
    print(sep)

    # ── Section 1: Window Summary ────────────────────────────────
    print(f"  [1] WINDOW SUMMARY")
    print(dash)
    print(f"    Current Beat Index .: {beat_index}")
    print(f"    Valid Beats ........: {w.valid_beat_count}  {'>= ' + str(_MIN_VALID_BEATS) if w.valid_beat_count >= _MIN_VALID_BEATS else '< ' + str(_MIN_VALID_BEATS)}  |  {'PASS' if w.valid_beat_count >= _MIN_VALID_BEATS else 'FAIL'}")
    if w.ventricular_rate_bpm is not None:
        print(f"    Heart Rate .........: {w.ventricular_rate_bpm:.1f} bpm")
    if w.rr_cv is not None:
        print(f"    RR CV ..............: {w.rr_cv:.4f}  {'< 0.08' if w.rr_cv < 0.08 else '>= 0.08'}  |  {'REGULAR' if w.rr_cv < 0.08 else 'IRREGULAR'}")
    print(f"    Group Beating ......: {'YES' if w.group_beating_detected else 'NO'}")
    print(f"    P-wave Present .....: {w.p_wave_present_fraction:.2f} fraction")
    print(f"    Flutter Baseline ...: {w.flutter_baseline_detected_fraction:.2f} fraction")
    if w.window_dominant_hz is not None:
        print(f"    Window Dominant Hz .: {w.window_dominant_hz:.2f} Hz")
    if w.window_flutter_ratio is not None:
        print(f"    Window Flutter Ratio: {w.window_flutter_ratio:.4f}  {'> 0.35' if w.window_flutter_ratio > 0.35 else '<= 0.35'}  |  {'PASS' if w.window_flutter_ratio > 0.35 else 'FAIL'}")
    if w.atrial_rate_bpm is not None:
        print(f"    Atrial Rate ........: {w.atrial_rate_bpm:.0f} bpm  {'[220-350]' if 220 <= w.atrial_rate_bpm <= 350 else 'outside range'}")
    if w.av_block_ratio is not None:
        print(f"    AV Ratio ...........: {w.av_block_ratio:.2f}:1  {'INTEGER-LIKE' if w.av_block_ratio_is_integer_like else 'not integer'}")
    if w.organization_index is not None:
        print(f"    Organization Index .: {w.organization_index:.4f}")
    print(f"    Window Accepted? ...: {'YES' if gate_failed is None else 'NO'}")
    if gate_failed:
        print(f"    Rejected because ...: {gate_failed}")

    # ── Section 2: Morphology Summary ────────────────────────────
    print()
    print(f"  [2] MORPHOLOGY SUMMARY")
    print(dash)
    print(f"    P-wave present fraction ........: {w.p_wave_present_fraction:.3f}")
    print(f"    Flutter baseline fraction ......: {w.flutter_baseline_detected_fraction:.3f}")
    if w.atrial_rate_bpm is not None:
        print(f"    Atrial rate (bpm) .............: {w.atrial_rate_bpm:.1f}")
        if w.atrial_rate_std_bpm is not None:
            print(f"    Atrial rate std (bpm) .........: {w.atrial_rate_std_bpm:.1f}")
    if w.ventricular_rate_bpm is not None:
        print(f"    Ventricular rate (bpm) ........: {w.ventricular_rate_bpm:.1f}")
    if w.rr_cv is not None:
        print(f"    RR CV .........................: {w.rr_cv:.4f}")
    if w.av_block_ratio is not None:
        print(f"    AV block ratio ................: {w.av_block_ratio:.2f}:1")
    print(f"    AV ratio integer-like ..........: {'YES' if w.av_block_ratio_is_integer_like else 'NO'}")
    print(f"    Group beating detected .........: {'YES' if w.group_beating_detected else 'NO'}")
    if w.organization_index is not None:
        print(f"    Organization index ............: {w.organization_index:.4f}")

    # ── Section 3: Flutter Feature Values (gate table) ───────────
    print()
    print(f"  [3] FLUTTER FEATURE VALUES")
    print(dash)
    print(f"    {'Feature':<40} {'Value':<15} {'Threshold':<20} {'Result'}")
    print(f"    {'─'*40} {'─'*15} {'─'*20} {'─'*6}")

    def _gate(name: str, value: Any, threshold: str, passed: bool) -> None:
        v = str(value)[:14]
        print(f"    {name:<40} {v:<15} {threshold:<20} {'PASS' if passed else 'FAIL'}")

    _gate("valid_beat_count", w.valid_beat_count, f">= {_MIN_VALID_BEATS}", w.valid_beat_count >= _MIN_VALID_BEATS)
    _gate("p_wave_present_fraction", f"{w.p_wave_present_fraction:.2f}", "< 0.70", w.p_wave_present_fraction < 0.70)
    _gate("flutter_baseline_fraction", f"{w.flutter_baseline_detected_fraction:.2f}", ">= 0.20", w.flutter_baseline_detected_fraction >= 0.20)
    if w.ventricular_rate_bpm is not None:
        _gate("ventricular_rate_bpm", f"{w.ventricular_rate_bpm:.0f}", f"[{_PLAUSIBLE_RATE_RANGE[0]:.0f}-{_PLAUSIBLE_RATE_RANGE[1]:.0f}] (plausible gate)", rate_in_plausible_range)
    _gate("rr_cv", f"{w.rr_cv:.4f}" if w.rr_cv is not None else "None", "< 0.08", w.rr_cv is not None and w.rr_cv < 0.08)
    if w.atrial_rate_bpm is not None:
        _gate("atrial_rate_bpm", f"{w.atrial_rate_bpm:.0f}", "[220-350]", 220 <= w.atrial_rate_bpm <= 350)
    _gate("av_ratio_integer_like", "YES" if w.av_block_ratio_is_integer_like else "NO", "== True", w.av_block_ratio_is_integer_like)
    _gate("group_beating", "YES" if w.group_beating_detected else "NO", "== True", w.group_beating_detected)

    # ── Section 4: Confidence Score Breakdown ────────────────────
    print()
    print(f"  [4] CONFIDENCE SCORE BREAKDOWN")
    print(dash)
    for name, contrib, reason in score_components:
        sign = "+" if contrib >= 0 else ""
        print(f"    {name:<45} {sign}{contrib:.2f}")
        print(f"    {'Reason:':<12} {reason}")
        print(f"    {sub}")
    print(f"    {'─' * 55}")
    print(f"    {'FINAL SCORE':<45} {score:.4f}  {'>= 0.45' if score >= 0.45 else '< 0.45'}  |  {'FIRED' if triggered else 'REJECTED'}")

    # ── Section 5: Decision Tree ─────────────────────────────────
    print()
    print(f"  [5] DECISION TREE")
    print(dash)
    print(f"    Window accepted")
    print(f"    {'↓':>8}")
    print(f"    Enough valid beats ({w.valid_beat_count} >= {_MIN_VALID_BEATS})?")
    if w.valid_beat_count >= _MIN_VALID_BEATS:
        print(f"    {'→ YES':>10}")
        print(f"    {'↓':>8}")
        print(f"    Discrete P waves absent OR flutter baseline present?")
        p_gate = w.p_wave_present_fraction < 0.70 or w.flutter_baseline_detected_fraction >= 0.20
        if p_gate:
            print(f"    {'→ YES':>10}  (p_frac={w.p_wave_present_fraction:.2f}, base_frac={w.flutter_baseline_detected_fraction:.2f})")
            print(f"    {'↓':>8}")
            print(f"    Ventricular rate in plausible range?")
            if rate_in_plausible_range:
                print(f"    {'→ YES':>10}  ({w.ventricular_rate_bpm:.0f} bpm in range)")
                print(f"    {'↓':>8}")
                print(f"    Rhythm regular enough (CV < 0.08)?")
                if w.rr_cv is not None and w.rr_cv < 0.08:
                    print(f"    {'→ YES':>10}  (CV={w.rr_cv:.4f})")
                    print(f"    {'↓':>8}")
                    print(f"    Atrial rate in flutter range?")
                    if w.atrial_rate_bpm is not None and 220 <= w.atrial_rate_bpm <= 350:
                        print(f"    {'→ YES':>10}  ({w.atrial_rate_bpm:.0f} bpm)")
                        print(f"    {'↓':>8}")
                        print(f"    AV ratio integer-like?")
                        if w.av_block_ratio_is_integer_like:
                            print(f"    {'→ YES':>10}  ({w.av_block_ratio:.1f}:1)")
                            print(f"    {'↓':>8}")
                            print(f"    Score >= 0.45?")
                            if score >= 0.45:
                                print(f"    {'→ YES':>10}  ({score:.2f})")
                                print(f"    {'↓':>8}")
                                print(f"    🔥 AFLUTTER DETECTED")
                            else:
                                print(f"    {'→ NO':>10}  (score={score:.2f} < 0.45)")
                        else:
                            print(f"    {'→ NO':>10}  (ratio={w.av_block_ratio:.2f}:1 not integer)")
                    else:
                        reason = f"atrial_rate={w.atrial_rate_bpm or 'None'} bpm"
                        print(f"    {'→ NO':>10}  ({reason})")
                else:
                    print(f"    {'→ NO':>10}  (CV={w.rr_cv:.4f} >= 0.08)")
            else:
                print(f"    {'→ NO':>10}  ({w.ventricular_rate_bpm:.0f} bpm outside [{_PLAUSIBLE_RATE_RANGE[0]:.0f}-{_PLAUSIBLE_RATE_RANGE[1]:.0f}])")
        else:
            print(f"    {'→ NO':>10}  (P waves in {w.p_wave_present_fraction*100:.0f}%, no flutter baseline)")
    else:
        print(f"    {'→ NO':>10}  (only {w.valid_beat_count} valid beats)")

    # ── Section 6: Missing Evidence (only when rejected) ─────────
    if not triggered:
        print()
        print(f"  [6] MISSING EVIDENCE")
        print(dash)
        if missing_evidence:
            for ev in missing_evidence:
                print(f"    ✗ {ev}")
        needed = max(0.0, 0.45 - score)
        if needed > 0:
            print(f"    Missing {needed:.2f} confidence points to reach threshold (0.45)")
            # Suggest what would have pushed it over
            if w.flutter_baseline_detected_fraction < 0.20:
                print(f"    → Would fire if flutter_baseline_detected_fraction >= 0.50 (+0.10)")
            if w.rr_cv is not None and w.rr_cv >= 0.08:
                print(f"    → Would fire if RR CV < 0.08 (+0.35)")
            if w.atrial_rate_bpm is not None and not (220 <= w.atrial_rate_bpm <= 350):
                print(f"    → Would fire if atrial rate in [220-350] (+0.20)")
        else:
            print(f"    No missing evidence — score {score:.2f} >= 0.45 but detector rejected (check confidence cap)")
    else:
        print()
        print(f"  [6] MISSING EVIDENCE — NONE (detector fired)")

    # ── Section 7: Detector Inputs (raw values) ──────────────────
    print()
    print(f"  [7] DETECTOR INPUTS (raw window values)")
    print(dash)
    if beats_history:
        valid = [b for b in beats_history if _as_float(b.get("signal_quality_score"), 1.0) >= 0.50]
        source = valid if valid else beats_history
        rr_vals = [_as_float(b.get("rr_interval")) for b in source if _as_float(b.get("rr_interval"))]
        p_vals = [_as_bool(b.get("p_wave_detected"), False) for b in source]
        base_vals = [_as_bool(b.get("flutter_baseline_detected")) for b in source]
        dom_vals = [_as_float(b.get("flutter_baseline_dominant_hz")) for b in source if _as_float(b.get("flutter_baseline_dominant_hz")) is not None]
        org_vals = [_as_float(b.get("flutter_organization_index")) for b in source if _as_float(b.get("flutter_organization_index")) is not None]
        print(f"    RR intervals (ms) ........: {len(rr_vals)} values, mean={np.mean(rr_vals):.0f}ms" if rr_vals else "    RR intervals: None")
        print(f"    P-wave flags .............: {sum(p_vals)}/{len(p_vals)} present" if p_vals else "    P-wave flags: None")
        print(f"    Flutter baseline flags ...: {sum(base_vals)}/{len(base_vals)} detected" if base_vals else "    Flutter baseline flags: None")
        if dom_vals:
            print(f"    Dominant frequencies (Hz) : {len(dom_vals)} values, mean={np.mean(dom_vals):.1f} Hz")
            print(f"      First few: {[f'{v:.1f}' for v in dom_vals[:6]]}")
        if org_vals:
            print(f"    Organization indices .....: {len(org_vals)} values, mean={np.mean(org_vals):.3f}")
        print(f"    Signal quality scores ....: {len(source)} beats, mean={np.mean([_as_float(b.get('signal_quality_score'), 1.0) for b in source]):.3f}")

    # ── Section 8: Feature Evolution ─────────────────────────────
    print()
    print(f"  [8] FEATURE EVOLUTION (last 10 beats)")
    print(dash)
    if beats_history:
        recent = beats_history[-10:]
        print(f"    {'Beat':>6}  {'RR(ms)':>8}  {'CV':>8}  {'Score':>8}")
        print(f"    {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
        # Compute approximate per-beat score evolution
        for i, b in enumerate(recent):
            bi = b.get("beat_index", "?")
            brr = _as_float(b.get("rr_interval"))
            # Approximate: show final score for all (we don't have per-beat scores)
            print(f"    {str(bi):>6}  {brr or 0:>8.0f}  {w.rr_cv or 0:>8.4f}  {score:>8.2f}")
    else:
        print(f"    (no beat history available)")

    # ── Section 9: Final Diagnostic ──────────────────────────────
    print()
    print(f"  [9] FINAL DIAGNOSTIC")
    print(dash)
    if triggered:
        reasons = []
        if rate_in_plausible_range:
            reasons.append(f"ventricular rate {w.ventricular_rate_bpm:.0f} bpm in plausible flutter range")
        if w.atrial_rate_bpm is not None and 220 <= w.atrial_rate_bpm <= 350:
            reasons.append(f"atrial rate {w.atrial_rate_bpm:.0f} bpm")
        if w.av_block_ratio_is_integer_like and w.av_block_ratio is not None:
            reasons.append(f"integer AV ratio {w.av_block_ratio:.1f}:1")
        if w.flutter_baseline_detected_fraction >= 0.50:
            reasons.append(f"flutter baseline in {w.flutter_baseline_detected_fraction*100:.0f}% of beats")
        if w.group_beating_detected:
            reasons.append("group beating pattern")
        print(f"    Flutter detected: {', '.join(reasons)}.")
        print(f"    Score {score:.2f} >= 0.45 threshold.")
    elif gate_failed:
        print(f"    Detection rejected because: {gate_failed}")
    elif missing_evidence:
        print(f"    Detection rejected because:")
        for ev in missing_evidence[:3]:
            print(f"      • {ev}")
        if score > 0:
            print(f"    Score {score:.2f} < 0.45 threshold (needs {0.45 - score:.2f} more).")
    else:
        print(f"    Detection rejected: criteria not met (score={score:.2f}).")
    print(sep)
    print()


def _build_afl_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
) -> Optional["AFLWindowFeatures"]:
    from disease_detection.detect_atrial_flutter import AFLWindowFeatures
    import numpy as np

    valid = [b for b in beats_history if _as_float(b.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history
    if len(source) < 12:
        return None

    rr_stats = calculate_rr_irregularity(source)
    ventricular_rate_bpm = 60.0 / rr_stats["rr_mean"] if rr_stats and rr_stats["rr_mean"] else None
    rr_cv = rr_stats["rr_cv"] if rr_stats else None

    p_flags = [_as_bool(b.get("p_wave_detected"), False) for b in source]
    p_wave_present_fraction = sum(p_flags) / len(p_flags) if p_flags else 1.0

    # ══════════════════════════════════════════════════════════════════
    # Window-level spectral features from 10-second context window
    # ══════════════════════════════════════════════════════════════════
    # The per-beat TP segment (p_offset to qrs_onset) is 0-1 samples at
    # ventricular rates >100 bpm, making per-beat spectral analysis
    # impossible. Inter-beat segments (qrs_offset to next qrs_onset) also
    # fail because qrs_offset/qrs_onset are often None in database-retrieved
    # beats (they require context-window DWT which itself fails at high HR).
    #
    # Instead, we use the 10-second context window (3600 samples at 360 Hz)
    # that is stored alongside each beat. This gives 0.1 Hz frequency
    # resolution — more than enough to resolve the 4-9 Hz flutter band.
    # The context window contains multiple R-peaks and the full flutter-
    # wave signal, so we can detect organized baseline oscillations without
    # depending on per-beat landmark quality.
    _FS = 360.0
    _CONTEXT_LEN_SAMPLES = int(10.0 * _FS)  # 3600
    _FFT_LEN = 4096  # zero-padded FFT for ~0.088 Hz resolution
    window_dominant_hz = None
    window_flutter_ratio = None
    window_flutter_baseline_detected = False
    window_atrial_rate_bpm = None
    window_organization_index = None

    # Use the most recent beat's context_samples if available.
    # The context window is the full 10-second window from which this beat
    # was extracted — it contains the baseline oscillation signal we need.
    ctx = source[-1].get("context_samples") if source else None
    if ctx is not None and isinstance(ctx, (list, np.ndarray)) and len(ctx) > 0:
        ctx_arr = np.asarray(ctx, dtype=np.float64)
        if len(ctx_arr) >= int(2.0 * _FS):
            # Subtract mean and apply Hanning window to reduce spectral leakage
            centered = ctx_arr - np.mean(ctx_arr)
            windowed = centered * np.hanning(len(centered))
            # Zero-pad to FFT_LEN for fine frequency resolution
            padded = np.zeros(_FFT_LEN)
            n = min(len(windowed), _FFT_LEN)
            padded[:n] = windowed[:n]
            # Compute power spectral density
            spectrum = np.abs(np.fft.rfft(padded)) ** 2
            freqs = np.fft.rfftfreq(_FFT_LEN, d=1.0 / _FS)
            # 4-9 Hz flutter band
            band_bins = (freqs >= 4.0) & (freqs <= 9.0)
            if np.any(band_bins):
                band_power = spectrum[band_bins]
                total_power = float(np.sum(spectrum))
                band_power_sum = float(np.sum(band_power))
                window_flutter_ratio = band_power_sum / total_power if total_power > 0 else 0.0
                window_flutter_baseline_detected = window_flutter_ratio > 0.35
                # Dominant frequency in the band
                dom_bin_idx = int(np.argmax(band_power))
                band_freqs = freqs[band_bins]
                window_dominant_hz = float(band_freqs[dom_bin_idx])
                # Only report atrial rate if a flutter baseline pattern is actually detected
                if window_flutter_baseline_detected:
                    window_atrial_rate_bpm = window_dominant_hz * 60.0
                # Organization index: peak concentration in dominant bin
                mean_band_power = float(np.mean(band_power)) if len(band_power) > 0 else 0.0
                peak_power = float(band_power[dom_bin_idx])
                window_organization_index = (
                    min(1.0, peak_power / (mean_band_power * len(band_power)))
                    if mean_band_power > 0 and len(band_power) > 0
                    else 0.0
                )
        # if window_dominant_hz is not None:
        #     print(f"[AFL context] {len(ctx_arr)} samples, "
        #           f"dominant_hz={window_dominant_hz:.3f}, "
        #           f"flutter_ratio={window_flutter_ratio:.4f}, "
        #           f"detected={window_flutter_baseline_detected}")
    else:
        # Only print when debug_afl would have been useful but context is missing
        pass

    # Use window-level features for atrial rate and flutter baseline
    atrial_rate_bpm = window_atrial_rate_bpm
    atrial_rate_std_bpm = None
    flutter_baseline_detected_fraction = 1.0 if window_flutter_baseline_detected else 0.0
    org_values = [
        _as_float(b.get("flutter_organization_index"))
        for b in source
        if _as_float(b.get("flutter_organization_index")) is not None
    ]
    organization_index = float(np.mean(org_values)) if org_values else None
    # Override with window-level org index if available (more stable)
    if window_organization_index is not None:
        organization_index = window_organization_index

    av_block_ratio = None
    av_ratio_is_integer_like = False
    if atrial_rate_bpm and ventricular_rate_bpm and ventricular_rate_bpm > 0:
        av_block_ratio = atrial_rate_bpm / ventricular_rate_bpm
        nearest = round(av_block_ratio)
        av_ratio_is_integer_like = bool(1 <= nearest <= 8 and abs(av_block_ratio - nearest) <= 0.20)

    group_beating_detected = _detect_group_beating([
        _as_float(b.get("rr_interval")) for b in source if _as_float(b.get("rr_interval"))
    ])

    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]

    return AFLWindowFeatures(
        window_id=f"afl_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        valid_beat_count=len(source),
        ventricular_rate_bpm=ventricular_rate_bpm,
        rr_cv=rr_cv,
        group_beating_detected=group_beating_detected,
        p_wave_present_fraction=p_wave_present_fraction,
        flutter_baseline_detected_fraction=flutter_baseline_detected_fraction,
        atrial_rate_bpm=atrial_rate_bpm,
        atrial_rate_std_bpm=atrial_rate_std_bpm,
        organization_index=organization_index,
        av_block_ratio=av_block_ratio,
        av_block_ratio_is_integer_like=av_ratio_is_integer_like,
        age=_metadata_value(patient_metadata, "age"),
        sex=_metadata_value(patient_metadata, "sex", "gender"),
        known_diagnoses=list(diagnoses) if diagnoses else [],
        window_dominant_hz=window_dominant_hz,
        window_flutter_ratio=window_flutter_ratio,
        window_flutter_baseline_detected=window_flutter_baseline_detected,
    )

def _build_afib_window_features(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
) -> Optional["AFWindowFeatures"]:
    from disease_detection.detect_afib import AFWindowFeatures

    valid = [b for b in beats_history if _as_float(b.get("signal_quality_score"), 1.0) >= 0.50]
    source = valid if valid else beats_history
    if len(source) < 30:  # Matches MIN_VALID_BEATS from detect_afib.py
        return None

    rr_stats = calculate_rr_irregularity(source)
    if not rr_stats:
        return None

    # ── تحديد النبضات خارج الرحم ──────────────────────────────────
    # LABEL_N = 0 (طبيعي), LABEL_S = 1 (APC), LABEL_V = 2 (PVC),
    # LABEL_F = 3 (Fusion), LABEL_Q = 4 (Unknown)
    ectopic_flags = [
        b.get("predicted_label") in (LABEL_S, LABEL_V, LABEL_F, LABEL_Q)
        for b in source
    ]
    ectopic_fraction = sum(ectopic_flags) / len(ectopic_flags) if ectopic_flags else 0.0

    # ── استخراج RR intervals للنبضات الطبيعية بس (N و J و أي نبضات تعتبرها طبيعية) ──
    normal_rr_intervals = []
    for beat in source:
        label = beat.get("predicted_label")
        # استبعد الـ Ectopic Beats (V, S, F, Q)
        # خلّي الـ Normal (0) و Junctional Escape (J) لو موجودة عندك
        if label in (0,):  # 0 = N (طبيعي)
            rr = _as_float(beat.get("rr_interval"))
            if rr is not None and rr > 0:
                normal_rr_intervals.append(rr)

    # ── حساب الـ CV و RMSSD على النبضات الطبيعية ──────────────────
    rr_cv_filtered = None
    rmssd_filtered_sec = None
    
    if len(normal_rr_intervals) >= 5:  # نفس العتبة بتاعة calculate_rr_irregularity
        import numpy as np
        rr_mean = float(np.mean(normal_rr_intervals))
        rr_std = float(np.std(normal_rr_intervals, ddof=1)) if len(normal_rr_intervals) > 1 else 0.0
        rr_cv_filtered = rr_std / rr_mean if rr_mean > 0 else 0.0
        
        # حساب RMSSD
        diffs = []
        for i in range(1, len(normal_rr_intervals)):
            diffs.append(normal_rr_intervals[i] - normal_rr_intervals[i - 1])
        if diffs:
            import math
            rmssd_filtered_sec = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs)) / 1000.0  # تحويل لثواني

    p_flags = [_as_bool(b.get("p_wave_detected"), False) for b in source]
    rr_intervals_ms = [_as_float(b.get("rr_interval")) for b in source if _as_float(b.get("rr_interval")) is not None]

    baseline_flags = [_as_bool(b.get("flutter_baseline_detected")) for b in source]
    fibrillatory_baseline_detected = (sum(baseline_flags) / len(baseline_flags) > 0.5) if baseline_flags else False

    diagnoses = _metadata_value(patient_metadata, "known_diagnoses", "diagnoses", default=[])
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]

    return AFWindowFeatures(
        window_id=f"afib_{datetime.now().isoformat()}",
        patient_id=_metadata_value(patient_metadata, "patient_id", default="unknown"),
        recorded_at=datetime.now(),
        beat_count=len(beats_history),
        valid_beat_count=len(source),
        rr_intervals_ms=rr_intervals_ms,
        p_wave_present_flags=p_flags,
        window_duration_sec=sum(rr_intervals_ms) / 1000.0 if rr_intervals_ms else 0.0,
        rr_mean_ms=rr_stats.get("rr_mean"),
        rr_std_ms=rr_stats.get("rr_std"),
        rr_cv=rr_stats.get("rr_cv"),
        rmssd_sec=rr_stats.get("rmssd"),
        rr_cv_filtered=rr_cv_filtered,                    # <---- الحقل الجديد
        rmssd_filtered_sec=rmssd_filtered_sec,            # <---- الحقل الجديد
        ectopic_fraction=ectopic_fraction,
        mean_signal_quality=float(np.mean([_as_float(b.get("signal_quality_score"), 1.0) for b in source])),
        fibrillatory_baseline_detected=fibrillatory_baseline_detected,
        rhythm_classifier_af_probability=None,
        prior_af_history=any("fibrillation" in str(d).lower() for d in diagnoses),
        sex=_metadata_value(patient_metadata, "sex", "gender"),
        age=_metadata_value(patient_metadata, "age"),
        known_diagnoses=list(diagnoses) if diagnoses else [],
    )


def _print_afib_diagnostic(
    beat_index: int,
    w: Optional["AFWindowFeatures"],
    result: "DetectionResult",
    decision_details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a formatted AFib diagnostic dashboard to the terminal.
    Only prints when AFWindowFeatures was provided (non-None).
    """
    if w is None:
        return

    cv = w.rr_cv_filtered if w.rr_cv_filtered is not None else (w.rr_cv if w.rr_cv is not None else 0.0)
    rmssd = w.rmssd_filtered_sec if w.rmssd_filtered_sec is not None else (w.rmssd_sec if w.rmssd_sec is not None else 0.0)
    p_absent_fraction = (
        1.0 - (sum(w.p_wave_present_flags) / len(w.p_wave_present_flags))
        if w.p_wave_present_flags else 0.0
    )

    score = decision_details.get("final_score", 0.0) if decision_details else 0.0
    base_score = decision_details.get("base_score", 0.0) if decision_details else 0.0
    ectopic_penalty = decision_details.get("ectopic_penalty", 0.0) if decision_details else 0.0
    p_wave_bonus = decision_details.get("p_wave_bonus", 0.0) if decision_details else 0.0
    fibrillatory_bonus = decision_details.get("fibrillatory_bonus", 0.0) if decision_details else 0.0
    classifier_bonus = decision_details.get("classifier_bonus", 0.0) if decision_details else 0.0
    prior_af_bonus = decision_details.get("prior_af_bonus", 0.0) if decision_details else 0.0
    gate_failed = decision_details.get("gate_failed", None) if decision_details else None

    sep = "=" * 72
    dash = "-" * 72

    print()
    print(sep)
    print(f"  [AFIB DIAGNOSTIC DASHBOARD]  |  beat_index={beat_index}")
    print(sep)

    # ── Section 1: Window & Gates ────────────────────────────────
    print(f"  [1] WINDOW & GATES")
    print(dash)
    valid_ok = w.valid_beat_count >= 30
    print(f"    valid_beat_count ..: {w.valid_beat_count}  {'>= 30' if valid_ok else '< 30'}  |  {'PASS' if valid_ok else 'FAIL'}")
    dur_ok = w.window_duration_sec is None or w.window_duration_sec >= 30.0
    if w.window_duration_sec is not None:
        print(f"    window_duration_sec: {w.window_duration_sec:.1f}s  {'>= 30s' if dur_ok else '< 30s'}  |  {'PASS' if dur_ok else 'FAIL'}")
    else:
        print(f"    window_duration_sec: None (not checked)")
    cv_gate = cv >= 0.15
    print(f"    cv_filtered ........: {cv:.4f}  {'>= 0.15' if cv_gate else '< 0.15'}  |  {'PASS' if cv_gate else 'FAIL'}")
    rmssd_gate = rmssd >= 0.10
    print(f"    rmssd_filtered_sec .: {rmssd:.4f}  {'>= 0.10' if rmssd_gate else '< 0.10'}  |  {'PASS' if rmssd_gate else 'FAIL'}")
    ectopic_penalty_active = w.ectopic_fraction > 0.20
    print(f"    ectopic_fraction ...: {w.ectopic_fraction:.3f}  {'> 0.20' if ectopic_penalty_active else '<= 0.20'}  |  {'PENALTY' if ectopic_penalty_active else 'OK'}")
    print(f"    mean_signal_quality : {w.mean_signal_quality:.3f}  {'>= 0.60' if w.mean_signal_quality >= 0.60 else '< 0.60'}")

    # ── Section 2: Score Components ───────────────────────────────
    print()
    print(f"  [2] SCORE COMPONENTS")
    print(dash)
    print(f"    Base Score ..........: {base_score:.4f}  (from CV={cv:.4f})")
    print(f"    Ectopic Penalty .....: {ectopic_penalty:.4f}")
    print(f"    P-wave Absence Bonus : {p_wave_bonus:.4f}  (P-absent fraction={p_absent_fraction:.3f})")
    print(f"    Fibrillatory Baseline: {fibrillatory_bonus:.4f}")
    print(f"    Rhythm Classifier ...: {classifier_bonus:.4f}")
    print(f"    Prior AF History ....: {prior_af_bonus:.4f}")
    print(f"    {'-' * 40}")
    print(f"    FINAL SCORE .........: {score:.4f}  {'>= 0.60' if score >= 0.60 else '< 0.60'}  |  {'FIRED' if score >= 0.60 else 'REJECTED'}")

    # ── Section 3: Decision ───────────────────────────────────────
    print()
    print(f"  [3] DECISION")
    print(dash)
    if gate_failed:
        print(f"    GATE FAILED: {gate_failed}")
    elif result.triggered:
        print(f"    FIRED - AF pattern detected (confidence={result.confidence:.2f})")
    else:
        if score < 0.60:
            print(f"    REJECTED - Final score {score:.4f} < 0.60")
        else:
            print(f"    REJECTED - Confidence {result.confidence:.4f} below threshold")
    print(f"    Reason: {result.reason}")
    print(sep)
    print()


def detect_diseases_with_detector(
    beats_history: List[Dict[str, Any]],
    patient_metadata: Optional[Dict[str, Any]] = None,
    existing_events: Optional[List[Dict[str, Any]]] = None,
    db_connection: Any = None,
    session_id: Optional[str] = None,
    debug_afib: bool = False,
    debug_afl: bool = False,
) -> List[Dict[str, Any]]:
    if not beats_history:
        return []
    features = _build_disease_detector_features(beats_history, patient_metadata, existing_events)
    window_features = _build_window_features(beats_history, patient_metadata, existing_events)
    arvc_window = _build_arvc_window_features(
        beats_history, patient_metadata, existing_events,
        db_connection=db_connection, session_id=session_id
    )
    vt_window = _build_vt_window_features(
        beats_history, patient_metadata, existing_events,
        db_connection=db_connection, session_id=session_id
    )
    vf_window = _build_vf_window_features(
        beats_history, patient_metadata, existing_events
    )
    afl_window = _build_afl_window_features(
        beats_history, patient_metadata
    )
    afib_window = _build_afib_window_features(
        beats_history, patient_metadata
    )
    results = _DISEASE_DETECTOR.evaluate(
        features,
        window_features=window_features,
        arvc_window=arvc_window,
        vt_window=vt_window,
        vf_window=vf_window,
        afl_window=afl_window,
        afib_window=afib_window,
    )
    events = []
    for result in results:
        if result.triggered and float(result.confidence) >= _DISEASE_DETECTOR.min_confidence:
            events.append(_disease_result_to_event(result))
    # ── AFib diagnostic dashboard ──────────────────────────────
    if debug_afib and afib_window is not None:
        # Find the AFib result from results list
        afib_result = None
        for r in results:
            if "Atrial Fibrillation" in r.disease:
                afib_result = r
                break
        if afib_result is not None:
            # Build decision details from the window features
            cv = afib_window.rr_cv_filtered if afib_window.rr_cv_filtered is not None else (afib_window.rr_cv if afib_window.rr_cv is not None else 0.0)
            p_absent_fraction = (
                1.0 - (sum(afib_window.p_wave_present_flags) / len(afib_window.p_wave_present_flags))
                if afib_window.p_wave_present_flags else 0.0
            )
            base_score = 0.40 + min(0.15, 0.15 * ((cv - 0.15) / 0.15))
            ectopic_penalty = 0.15 if afib_window.ectopic_fraction > 0.20 else 0.0
            p_wave_bonus = 0.40 if p_absent_fraction >= 0.85 else (0.20 if p_absent_fraction >= 0.60 else 0.0)
            fibrillatory_bonus = 0.10 if afib_window.fibrillatory_baseline_detected else 0.0
            classifier_bonus = 0.0  # always None in this pipeline
            prior_af_bonus = 0.05 if afib_window.prior_af_history else 0.0
            final_score = base_score - ectopic_penalty + p_wave_bonus + fibrillatory_bonus + classifier_bonus + prior_af_bonus
            final_score = min(final_score, 1.0)

            gate_failed = None
            if afib_window.valid_beat_count < 30:
                gate_failed = f"valid_beat_count={afib_window.valid_beat_count} < 30"
            elif cv < 0.15:
                gate_failed = f"cv_filtered={cv:.4f} < 0.15 (regular rhythm)"
            elif afib_window.rmssd_filtered_sec is not None and afib_window.rmssd_filtered_sec < 0.10:
                gate_failed = f"rmssd_filtered_sec={afib_window.rmssd_filtered_sec:.4f} < 0.10"

            decision_details = {
                "base_score": base_score,
                "ectopic_penalty": ectopic_penalty,
                "p_wave_bonus": p_wave_bonus,
                "fibrillatory_bonus": fibrillatory_bonus,
                "classifier_bonus": classifier_bonus,
                "prior_af_bonus": prior_af_bonus,
                "final_score": final_score,
                "gate_failed": gate_failed,
            }
            # Get the last beat index from history
            last_idx = beats_history[-1].get("beat_index", 0) if beats_history else 0
            _print_afib_diagnostic(last_idx, afib_window, afib_result, decision_details)
    # ── AFL diagnostic dashboard ──────────────────────────────
    if debug_afl and afl_window is not None:
        afl_result = None
        for r in results:
            if "Flutter" in r.disease:
                afl_result = r
                break
        if afl_result is not None:
            last_idx = beats_history[-1].get("beat_index", 0) if beats_history else 0
            _print_afl_diagnostic(last_idx, afl_window, afl_result, beats_history)
    return events

from episode_manager import EPISODE_CONFIGS, EpisodePrimitive

# All event types configured as STATE primitive -- these are the ones that
# need an explicit "not present this cycle" pass, since RECURRENCE and
# CLUSTER primitives only ever fire on a positive detection and have no
# equivalent "explicitly absent" signal to report.
_STATE_EVENT_TYPES: List[str] = [
    event_type
    for event_type, config in EPISODE_CONFIGS.items()
    if config.primitive == EpisodePrimitive.STATE
]

_SEVERITY_CONFIDENCE_FALLBACK: Dict[str, float] = {
    "critical": 0.90,
    "high": 0.80,
    "moderate": 0.60,
    "low": 0.40,
    "info": 0.20,
}

def _infer_event_confidence(event: Dict[str, Any]) -> float:
    """
    Best-effort confidence extraction. Uses an explicit "confidence" field if
    the event already has one (disease_detector.py's DetectionResult-shaped
    events do), then "pattern_confidence" from detect_rhythm_patterns' metadata,
    then falls back to a severity-based approximation. This is a stated
    approximation, not a precise score.
    """
    if "confidence" in event and isinstance(event["confidence"], (int, float)):
        return float(event["confidence"])

    meta = event.get("metadata_json") or {}
    if isinstance(meta, dict):
        if isinstance(meta.get("pattern_confidence"), (int, float)):
            return float(meta["pattern_confidence"])
        if isinstance(meta.get("confidence"), (int, float)):
            return float(meta["confidence"])

    severity = str(event.get("severity", "low")).lower()
    return _SEVERITY_CONFIDENCE_FALLBACK.get(severity, 0.40)


def analyze_temporal_window(
    session_id: str,
    db_connection: Any,
    patient_metadata=None,
    window_size: int = 50,
    debug_afib: bool = False,
    debug_afl: bool = False,
    kafka_producer=None,
) -> List[Dict[str, Any]]:
    """
    Orchestrates the temporal analysis by fetching recent history and
    applying all detection rules. 
    Returns a list of detected rhythm events.
    """
    # db_connection is expected to be an instance of ECGDatabase (PostgreSQL version)
    history = db_connection.get_recent_beats(session_id, limit=window_size)
    
    if len(history) < 3:
        return [] # Not enough history
        
    events_detected = []
    
    # 1. Burden
    burdens = calculate_abnormal_burden(history)
    if burdens["pvc_burden"] >= 15.0:
        events_detected.append({
            "event_type": "HIGH_PVC_BURDEN",
            "severity": "moderate",
            "metadata_json": {
                **burdens,
                "reason": f"PVC burden {burdens['pvc_burden']:.1f}% exceeds threshold 15.0%",
                "ecg_findings": [f"PVC burden {burdens['pvc_burden']:.1f}%", f"APC burden {burdens['apc_burden']:.1f}%"],
            }
        })
        
    # 2. VT Runs (disabled - handled by disease_detector)
        
    ## 3. Tachy / Brady
    rate_events = detect_rate_abnormalities(history)
    events_detected.extend(rate_events)

    escape_event = detect_escape_beats(history)
    if escape_event:
        events_detected.append(escape_event)
        
    pause_event = detect_pauses(history)
    if pause_event:
        events_detected.append(pause_event)
        
    # 8. AV Block detection (1st, 2nd Mobitz I/II, 3rd degree)
    av_block_event = detect_av_block(
        history,
        existing_events=events_detected
    )
    if av_block_event:
        events_detected.append(av_block_event)

    # 3. Bigeminy / Trigeminy
    pattern_result = detect_rhythm_patterns(history)

    if pattern_result["bigeminy"]:
        events_detected.append({
            "event_type": "BIGEMINY",
            "severity": "moderate",
            "metadata_json": pattern_result
        })

    if pattern_result["trigeminy"]:
        events_detected.append({
            "event_type": "TRIGEMINY",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result["quadrigeminy"]:
        events_detected.append({
            "event_type": "QUADRIGEMINY",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result["couplet"]:
        events_detected.append({
            "event_type": "COUPLET",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result.get("atrial_bigeminy"):
        events_detected.append({
            "event_type": "ATRIAL_BIGEMINY",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result.get("atrial_trigeminy"):
        events_detected.append({
            "event_type": "ATRIAL_TRIGEMINY",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result.get("atrial_quadrigeminy"):
        events_detected.append({
            "event_type": "ATRIAL_QUADRIGEMINY",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result.get("atrial_couplet"):
        events_detected.append({
            "event_type": "ATRIAL_COUPLET",
            "severity": "low",
            "metadata_json": pattern_result
        })

    if pattern_result.get("atrial_triplet"):
        events_detected.append({
            "event_type": "ATRIAL_TRIPLET",
            "severity": "low",
            "metadata_json": pattern_result
        })

    # 3b. RR irregularity suggestive of AF (lightweight, non-agentic)
    rr_pattern_event = detect_rr_irregularity_pattern(history)
    if rr_pattern_event:
        events_detected.append(rr_pattern_event)

    # 4b. Standalone signal quality event (backend/display only)
    quality_event = detect_signal_quality_event(history)
    if quality_event:
        events_detected.append(quality_event)

    # 4. Pauses
    if detect_pauses(history):
        events_detected.append({
            "event_type": "PAUSE_DETECTED",
            "severity": "high",
            "metadata_json": {"rr_interval": history[-1]['rr_interval']}
        })

    disease_events = detect_diseases_with_detector(
        history,
        patient_metadata=patient_metadata,
        existing_events=events_detected,
        db_connection=db_connection,
        session_id=session_id,
        debug_afib=debug_afib,
        debug_afl=debug_afl,
    )
    events_detected = [
        event for event in events_detected
        if event.get("event_type") not in _DISEASE_MANAGED_EVENT_TYPES
    ]
    if disease_events:
        events_detected = _merge_disease_events(
            events_detected,
            disease_events
        )

    for event in events_detected:
        manager_decision = evaluate_event(event["event_type"])
        event["event_manager"] = manager_decision
    events_detected = [
        event for event in events_detected
        if event.get("event_manager", {}).get("known_event", False)
    ]
    # ── Lazy-init EpisodeManager on db_connection ──────────────────
    if not hasattr(db_connection, "_episode_manager"):
        db_connection._episode_manager = EpisodeManager(db_connection)
    episode_manager: EpisodeManager = db_connection._episode_manager

    last_time = history[-1]['timestamp']
    current_beat_index = history[-1].get('beat_index', 0)

    stored_events = []
    detected_types_this_cycle: set = set()

    # ── Pass 1: route every positively-detected event through the episode
    #    manager (RECURRENCE / STATE-true / CLUSTER, per its configured
    #    primitive) ─────────────────────────────────────────────────────
    for event in events_detected:
        event_type = event["event_type"]
        detected_types_this_cycle.add(event_type)

        result = route_event_through_episode_manager(
            episode_manager=episode_manager,
            session_id=session_id,
            event_type=event_type,
            beat_index=current_beat_index,
            timestamp=last_time,
            confidence=_infer_event_confidence(event),
            severity=event["severity"],
            metadata_json=event.get("metadata_json", {}),
            condition_true=True,
        )

        event["storage_status"] = result["storage_status"]
        event["episode_action"] = result["episode_action"]
        event["episode_id"] = result["episode_id"]
        event["episode"] = result["episode"]

        if result["storage_status"] == "created":
            stored_events.append(event)

        # Capture RECURRENCE / CLUSTER closures (when cooldown lapses or
        # cluster gap exceeded, the old episode is closed and a new one opens)
        if result.get("closed_episode") is not None:
            ce = result["closed_episode"]
            stored_events.append({
                "event_type": event_type,
                "severity": ce.get("severity", event.get("severity", "info")),
                "storage_status": "closed",
                "episode_action": "closed",
                "episode_id": result["closed_episode_id"],
                "episode": ce,
                "metadata_json": ce,
                "timestamp": ce.get("last_timestamp", last_time),
                "beat_index": ce.get("last_beat_index", current_beat_index),
            })

    # ── Pass 2: for every STATE-category event type NOT detected this
    #    cycle, explicitly report condition_true=False so a currently
    #    open episode (if any) closes promptly instead of only via
    #    cooldown timeout. ─────────────────────────────────────────────
    for event_type in _STATE_EVENT_TYPES:
        if event_type in detected_types_this_cycle:
            continue
        close_result = episode_manager.process_event(
            session_id=session_id,
            event_type=event_type,
            beat_index=current_beat_index,
            timestamp=last_time,
            confidence=0.0,
            severity="info",
            metadata_json={},
            condition_true=False,
        )
        # Capture closures for Kafka publishing (full episode data available)
        if close_result.get("action") == "closed" and close_result.get("episode") is not None:
            ep = close_result["episode"]
            stored_events.append({
                "event_type": event_type,
                "severity": ep.get("severity", "info"),
                "storage_status": "closed",
                "episode_action": "closed",
                "episode_id": close_result["episode_id"],
                "episode": ep,
                "metadata_json": ep,
                "timestamp": ep.get("last_timestamp", last_time),
                "beat_index": ep.get("last_beat_index", current_beat_index),
            })

    return stored_events
