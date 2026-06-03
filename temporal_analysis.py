from Event_Manager import evaluate_event
from typing import List, Dict, Any, Optional
# from Graduation project.ecg-project1 import history
# Constants based on typical MIT-BIH AAMI mapping
LABEL_N = 0  # Normal
LABEL_S = 1  # Supraventricular ectopic (APC)
LABEL_V = 2  # Ventricular ectopic (PVC)
LABEL_F = 3  # Fusion
LABEL_Q = 4  # Unknown
EVENT_COOLDOWNS = {
    "VT_RUN": 15,
    "BIGEMINY": 30,
    "TRIGEMINY": 30,
    "AFIB_SUSPECTED": 60,
    "TACHYCARDIA": 20,
    "BRADYCARDIA": 20
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

def calculate_abnormal_burden(beats_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates burden (percentage) of different types of abnormal beats in the window.
    """
    if not beats_history:
        return {"abnormal_burden": 0.0, "pvc_burden": 0.0, "apc_burden": 0.0}
        
    total = len(beats_history)
    abnormal = sum(1 for b in beats_history if b['is_abnormal'])
    pvcs = sum(1 for b in beats_history if b['predicted_label'] == LABEL_V)
    apcs = sum(1 for b in beats_history if b['predicted_label'] == LABEL_S)
    
    return {
        "abnormal_burden": abnormal / total,
        "pvc_burden": pvcs / total,
        "apc_burden": apcs / total
    }

def _get_labels_str(beats_history: List[Dict[str, Any]]) -> str:
    # Convert labels to string for easy regex/pattern matching. e.g. "020202"
    return "".join(str(b['predicted_label']) for b in beats_history)

def detect_rhythm_patterns(
    beats_history,
    min_bigeminy_cycles=3,
    min_trigeminy_cycles=2,
    min_confidence=0.70,
    min_quality=0.50
):

    labels = []
    
    for beat in beats_history:

        if beat.get('prediction_confidence', 0) < min_confidence:
            continue

        if beat.get('signal_quality_score', 0) < min_quality:
            continue

        labels.append(str(beat['predicted_label']))

    labels = "".join(labels)

    result = {
        "bigeminy": False,
        "trigeminy": False,
        "pattern_confidence": 0.0
    }

    # -----------------------------
    # BIGEMINY
    # -----------------------------
    bigeminy_pattern = "02" * min_bigeminy_cycles

    if bigeminy_pattern in labels:
        result["bigeminy"] = True
        result["pattern_confidence"] = 0.85

    # -----------------------------
    # TRIGEMINY
    # -----------------------------
    trigeminy_pattern = "002" * min_trigeminy_cycles

    if trigeminy_pattern in labels:
        result["trigeminy"] = True
        result["pattern_confidence"] = max(
            result["pattern_confidence"],
            0.80
        )

    return result
def detect_vt_runs(
    beats_history,
    min_consecutive_beats: int = 3,
    min_hr: float = 100,
    min_qrs_ms: float = 120,
    min_confidence: float = 0.70,
    min_quality: float = 0.50
):
    """
    Improved VT detection using:
    - consecutive ventricular beats
    - ventricular rate
    - wide QRS support
    - confidence gating
    - signal quality gating
    """

    if len(beats_history) < min_consecutive_beats:
        return None

    run = []

    for beat in reversed(beats_history):

        if beat['predicted_label'] != LABEL_V:
            break

        if beat.get('prediction_confidence', 0) < min_confidence:
            break

        if beat.get('signal_quality_score', 0) < min_quality:
            break

        run.append(beat)

    run = list(reversed(run))

    run_length = len(run)

    if run_length < min_consecutive_beats:
        return None

    # Average HR during run
    avg_hr = sum(
        b.get('heart_rate', 0) for b in run
    ) / run_length

    if avg_hr < min_hr:
        return None

    # Wide QRS support
    wide_qrs_count = sum(
        1 for b in run
        if b.get('qrs_width', 0) >= min_qrs_ms
    )

    wide_qrs_ratio = wide_qrs_count / run_length

    # Estimate duration
    start_time = run[0]['timestamp']
    end_time = run[-1]['timestamp']

    duration_sec = max(0, end_time - start_time)

    sustained = duration_sec >= 30

    # Severity logic
    if sustained:
        severity = "critical"
    elif run_length >= 5:
        severity = "high"
    else:
        severity = "moderate"

    return {
        "pattern": "VT_RUN",
        "run_length": run_length,
        "average_hr": avg_hr,
        "duration_sec": duration_sec,
        "wide_qrs_ratio": wide_qrs_ratio,
        "sustained": sustained,
        "severity": severity,
        "confidence": min(
            1.0,
            (
                (run_length / 10)
                + (avg_hr / 200)
                + wide_qrs_ratio
            ) / 3
        )
    }
def detect_pauses(beats_history: List[Dict[str, Any]], threshold_ms: float = 2000.0) -> bool:
    """
    Detects if the most recent beat had an abnormally long RR interval.
    """
    if not beats_history:
        return False
        
    last_beat = beats_history[-1]
    # Assuming rr_interval is in ms if we use threshold_ms
    # But if our pipeline uses seconds, adjust threshold
    rr = last_beat['rr_interval']
    if rr is None:
        return False
        
    if rr > 10: 
        # rr is probably in ms
        return rr > threshold_ms
    else:
        # rr is probably in seconds
        return rr > (threshold_ms / 1000.0)

def detect_rate_abnormalities(
    beats_history,
    tachy_threshold=100,
    brady_threshold=50,
    min_duration_beats=5,
    min_confidence=0.70,
    min_quality=0.50
):

    if len(beats_history) < min_duration_beats:
        return []

    events = []

    recent = beats_history[-min_duration_beats:]

    valid_beats = []

    for beat in recent:

        if beat.get('prediction_confidence', 0) < min_confidence:
            continue

        if beat.get('signal_quality_score', 0) < min_quality:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_duration_beats:
        return []

    hrs = [b['heart_rate'] for b in valid_beats]

    avg_hr = sum(hrs) / len(hrs)

    start_time = valid_beats[0]['timestamp']
    end_time = valid_beats[-1]['timestamp']

    duration_sec = end_time - start_time

    # -----------------------
    # TACHYCARDIA
    # -----------------------
    if all(hr >= tachy_threshold for hr in hrs):

        severity = (
            "high"
            if avg_hr >= 140
            else "moderate"
        )

        events.append({
            "event_type": "TACHYCARDIA",
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "duration_sec": duration_sec
            }
        })

    # -----------------------
    # BRADYCARDIA
    # -----------------------
    if all(hr <= brady_threshold for hr in hrs):

        severity = (
            "high"
            if avg_hr <= 40
            else "moderate"
        )

        events.append({
            "event_type": "BRADYCARDIA",
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "duration_sec": duration_sec
            }
        })

    return events


def process_event_with_cooldown(
    db_connection,
    session_id,
    event
):

    now_time = event["event_end_time"]

    active = db_connection.get_active_event(
        session_id,
        event["event_type"]
    )

    cooldown = EVENT_COOLDOWNS.get(
        event["event_type"],
        30
    )

    # ------------------------------------------------
    # ACTIVE EVENT EXISTS
    # ------------------------------------------------
    if active:

        last_time = active["event_end_time"]

        if (now_time - last_time) <= cooldown:

            db_connection.update_active_event(
                active["id"],
                new_end_time=now_time,
                metadata_json=event.get("metadata_json"),
                severity=event.get("severity")
            )

            return "updated"

        else:

            db_connection.close_event(
                active["id"],
                close_time=last_time
            )

    # ------------------------------------------------
    # CREATE NEW EVENT
    # ------------------------------------------------
    db_connection.insert_rhythm_event(event)

    return "created"
import math

def calculate_rr_irregularity(beats_history):

    rr_intervals = [
        b['rr_interval']
        for b in beats_history
        if b.get('rr_interval') is not None
        and b.get('rr_interval') > 0
    ]

    if len(rr_intervals) < 5:
        return None

    rr_mean = sum(rr_intervals) / len(rr_intervals)

    if rr_mean == 0:
        return None

    # Standard deviation
    rr_std = math.sqrt(
        sum((x - rr_mean) ** 2 for x in rr_intervals)
        / len(rr_intervals)
    )

    # Coefficient of variation
    rr_cv = rr_std / rr_mean

    # Successive RR differences
    diffs = [
        abs(rr_intervals[i] - rr_intervals[i - 1])
        for i in range(1, len(rr_intervals))
    ]

    rmssd = math.sqrt(
        sum(d ** 2 for d in diffs) / len(diffs)
    )

    irregularity_score = (
        (rr_cv * 0.5)
        + (rmssd / rr_mean * 0.5)
    )

    return {
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_cv": rr_cv,
        "rmssd": rmssd,
        "irregularity_score": irregularity_score
    }

def detect_afib(
    beats_history,
    min_window=15,
    irregularity_threshold=0.20,
    min_quality=0.50,
    min_confidence=0.50
):

    if len(beats_history) < min_window:
        return None

    valid_beats = []

    for beat in beats_history:

        if beat.get('signal_quality_score', 0) < min_quality:
            continue

        if beat.get('prediction_confidence', 0) < min_confidence:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    irregularity = calculate_rr_irregularity(valid_beats)

    if irregularity is None:
        return None

    score = irregularity["irregularity_score"]

    # ---------------------------------------------------
    # Rhythm pattern suppression
    # ---------------------------------------------------
    pattern_result = detect_rhythm_patterns(valid_beats)

    if pattern_result["bigeminy"]:
        return None

    if pattern_result["trigeminy"]:
        return None

    # ---------------------------------------------------
    # APC burden support
    # ---------------------------------------------------
    burdens = calculate_abnormal_burden(valid_beats)

    apc_burden = burdens["apc_burden"]

    # ---------------------------------------------------
    # Heart rate support
    # ---------------------------------------------------
    avg_hr = sum(
        b.get("heart_rate", 0)
        for b in valid_beats
    ) / len(valid_beats)

    # ---------------------------------------------------
    # Confidence fusion
    # ---------------------------------------------------
    afib_confidence = (
        (score * 0.5)
        + (min(apc_burden, 0.3) * 0.3)
        + (min(avg_hr / 150, 1.0) * 0.2)
    )

    if afib_confidence < irregularity_threshold:
        return None

    severity = (
        "high"
        if afib_confidence >= 0.35
        else "moderate"
    )

    return {
        "event_type": "AFIB_SUSPECTED",
        "severity": severity,
        "metadata_json": {
            "irregularity_score": score,
            "rr_cv": irregularity["rr_cv"],
            "rmssd": irregularity["rmssd"],
            "rr_std": irregularity["rr_std"],
            "apc_burden": apc_burden,
            "average_hr": avg_hr,
            "afib_confidence": afib_confidence
        }
    }

def augment_afib_stroke_risk(
    afib_event,
    patient_metadata=None
):

    if afib_event is None:
        return None

    risk_score = 0

    if patient_metadata:

        age = patient_metadata.get("age", 0)

        if age >= 75:
            risk_score += 2
        elif age >= 65:
            risk_score += 1

        if patient_metadata.get("hypertension"):
            risk_score += 1

        if patient_metadata.get("diabetes"):
            risk_score += 1

        if patient_metadata.get("heart_failure"):
            risk_score += 1

        if patient_metadata.get("prior_stroke"):
            risk_score += 2

        if patient_metadata.get("vascular_disease"):
            risk_score += 1

    risk_level = (
        "high"
        if risk_score >= 4
        else "moderate"
        if risk_score >= 2
        else "low"
    )

    afib_event["metadata_json"]["stroke_risk"] = risk_level
    afib_event["metadata_json"]["stroke_risk_score"] = risk_score
    if risk_score == 0:
     return None

    return afib_event

def augment_cardiomyopathy_risk(
    detected_events,
    patient_metadata=None
):

    risk_score = 0
    evidence = []

    event_types = {
        e["event_type"]
        for e in detected_events
    }

    # -----------------------------------------
    # VT
    # -----------------------------------------
    if "VT_RUN" in event_types:
        risk_score += 3
        evidence.append("ventricular tachycardia")

    # -----------------------------------------
    # PVC burden
    # -----------------------------------------
    if "HIGH_PVC_BURDEN" in event_types:
        risk_score += 2
        evidence.append("high PVC burden")

    # -----------------------------------------
    # BBB / wide conduction
    # -----------------------------------------
    if "POSSIBLE_BBB" in event_types:
        risk_score += 2
        evidence.append("wide QRS conduction abnormality")

    # -----------------------------------------
    # AFib
    # -----------------------------------------
    if "AFIB_SUSPECTED" in event_types:
        risk_score += 1
        evidence.append("AFib pattern")

    # -----------------------------------------
    # Long QT
    # -----------------------------------------
    if "POSSIBLE_LONG_QT" in event_types:
        risk_score += 1
        evidence.append("QT prolongation")

    # -----------------------------------------
    # Metadata augmentation
    # -----------------------------------------
    if patient_metadata:

        if patient_metadata.get("family_history_cardiomyopathy"):
            risk_score += 3
            evidence.append("family history")

        if patient_metadata.get("prior_cardiomyopathy"):
            risk_score += 4
            evidence.append("known cardiomyopathy")

        if patient_metadata.get("heart_failure"):
            risk_score += 2
            evidence.append("heart failure")

    risk_level = (
        "high"
        if risk_score >= 8
        else "moderate"
        if risk_score >= 4
        else "low"
    )
    if risk_score == 0:
     return None
    return {
        "event_type": "CARDIOMYOPATHY_RISK_AUGMENTED",
        "severity": risk_level,
        "metadata_json": {
            "risk_score": risk_score,
            "evidence": evidence
        }
    }

def augment_valvular_disease_risk(
    detected_events,
    patient_metadata=None
):

    risk_score = 0
    evidence = []

    event_types = {
        e["event_type"]
        for e in detected_events
    }

    # AFib strongly associated
    if "AFIB_SUSPECTED" in event_types:
        risk_score += 2
        evidence.append("AFib")

    # BBB / conduction changes
    if "POSSIBLE_BBB" in event_types:
        risk_score += 1
        evidence.append("conduction abnormality")

    # Tachy instability
    if "TACHYCARDIA" in event_types:
        risk_score += 1
        evidence.append("tachycardia")

    # Metadata
    if patient_metadata:

        if patient_metadata.get("known_valvular_disease"):
            risk_score += 4
            evidence.append("known valvular disease")

        if patient_metadata.get("rheumatic_heart_disease"):
            risk_score += 3
            evidence.append("rheumatic disease")

    risk_level = (
        "high"
        if risk_score >= 5
        else "moderate"
        if risk_score >= 3
        else "low"
    )
    if risk_score == 0:
     return None
    return {
        "event_type": "VALVULAR_DISEASE_RISK_AUGMENTED",
        "severity": risk_level,
        "metadata_json": {
            "risk_score": risk_score,
            "evidence": evidence
        }
    }
def augment_myocarditis_risk(
    detected_events,
    patient_metadata=None
):

    risk_score = 0
    evidence = []

    event_types = {
        e["event_type"]
        for e in detected_events
    }

    # VT
    if "VT_RUN" in event_types:
        risk_score += 2
        evidence.append("ventricular arrhythmia")

    # AV block
    if "POSSIBLE_AV_BLOCK" in event_types:
        risk_score += 2
        evidence.append("conduction abnormality")

    # Ischemic-like changes
    if "POSSIBLE_ISCHEMIC_PATTERN" in event_types:
        risk_score += 1
        evidence.append("ST/T abnormality")

    # AFib
    if "AFIB_SUSPECTED" in event_types:
        risk_score += 1
        evidence.append("atrial arrhythmia")

    # Metadata
    if patient_metadata:

        if patient_metadata.get("recent_viral_illness"):
            risk_score += 2
            evidence.append("recent viral illness")

        if patient_metadata.get("known_myocarditis"):
            risk_score += 4
            evidence.append("known myocarditis")

        if patient_metadata.get("fever"):
            risk_score += 1
            evidence.append("fever")

    risk_level = (
        "high"
        if risk_score >= 6
        else "moderate"
        if risk_score >= 3
        else "low"
    )

    if risk_score == 0:
        return None

    return {
        "event_type": "MYOCARDITIS_RISK_AUGMENTED",
        "severity": risk_level,
        "metadata_json": {
            "risk_score": risk_score,
            "evidence": evidence
        }
    }
def augment_heart_failure_risk(
    detected_events,
    patient_metadata=None
):

    risk_score = 0
    evidence = []

    event_types = {
        e["event_type"]
        for e in detected_events
    }

    # -------------------------------------------------
    # AFib
    # -------------------------------------------------
    if "AFIB_SUSPECTED" in event_types:
        risk_score += 2
        evidence.append("AFib pattern")

    # -------------------------------------------------
    # BBB / wide QRS
    # -------------------------------------------------
    if "POSSIBLE_BBB" in event_types:
        risk_score += 2
        evidence.append("wide QRS conduction abnormality")

    # -------------------------------------------------
    # VT
    # -------------------------------------------------
    if "VT_RUN" in event_types:
        risk_score += 2
        evidence.append("ventricular tachycardia")

    # -------------------------------------------------
    # Frequent PVC burden
    # -------------------------------------------------
    if "HIGH_PVC_BURDEN" in event_types:
        risk_score += 1
        evidence.append("high PVC burden")

    # -------------------------------------------------
    # Long QT
    # -------------------------------------------------
    if "POSSIBLE_LONG_QT" in event_types:
        risk_score += 1
        evidence.append("QT prolongation")

    # -------------------------------------------------
    # Brady/Tachy instability
    # -------------------------------------------------
    if (
        "TACHYCARDIA" in event_types
        or "BRADYCARDIA" in event_types
    ):
        risk_score += 1
        evidence.append("rate instability")

    # -------------------------------------------------
    # Patient metadata augmentation
    # -------------------------------------------------
    if patient_metadata:

        if patient_metadata.get("hypertension"):
            risk_score += 1
            evidence.append("hypertension")

        if patient_metadata.get("diabetes"):
            risk_score += 1
            evidence.append("diabetes")

        if patient_metadata.get("known_cad"):
            risk_score += 2
            evidence.append("known CAD")

        if patient_metadata.get("prior_heart_failure"):
            risk_score += 3
            evidence.append("prior heart failure")

    # -------------------------------------------------
    # Final risk level
    # -------------------------------------------------
    risk_level = (
        "high"
        if risk_score >= 7
        else "moderate"
        if risk_score >= 4
        else "low"
    )
    if risk_score == 0:
       return None


    return {
        "event_type": "POSSIBLE_HEART_FAILURE_PATTERN",
        "severity": risk_level,
        "metadata_json": {
            "hf_risk_score": risk_score,
            "evidence": evidence
        }
    }

def detect_possible_bbb(
    beats_history,
    min_wide_ratio=0.70,
    qrs_threshold=120,
    min_window=10,
    min_quality=0.50,
    min_confidence=0.70
):

    if len(beats_history) < min_window:
        return None

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    # Exclude ventricular runs
    ventricular_ratio = sum(
        1 for b in valid_beats
        if b["predicted_label"] == LABEL_V
    ) / len(valid_beats)

    if ventricular_ratio > 0.4:
        return None

    wide_beats = [
        b for b in valid_beats
        if b.get("qrs_width", 0) >= qrs_threshold
    ]

    wide_ratio = len(wide_beats) / len(valid_beats)

    if wide_ratio < min_wide_ratio:
        return None

    avg_qrs = sum(
        b.get("qrs_width", 0)
        for b in wide_beats
    ) / len(wide_beats)

    # Morphology consistency
    qrs_std = np.std([
        b.get("qrs_width", 0)
        for b in wide_beats
    ])

    morphology_consistency = max(
        0,
        1 - (qrs_std / avg_qrs)
    )

    severity = (
        "high"
        if avg_qrs >= 160
        else "moderate"
    )

    return {
        "event_type": "POSSIBLE_BBB",
        "severity": severity,
        "metadata_json": {
            "wide_qrs_ratio": wide_ratio,
            "average_qrs_width": avg_qrs,
            "morphology_consistency": morphology_consistency,
            "ventricular_ratio": ventricular_ratio
        }
    }
def detect_possible_flutter(
    beats_history,
    min_window=12,
    hr_threshold=110,
    rr_cv_min=0.05,
    rr_cv_max=0.18,
    min_quality=0.50,
    min_confidence=0.70
):

    if len(beats_history) < min_window:
        return None

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    irregularity = calculate_rr_irregularity(valid_beats)

    if irregularity is None:
        return None

    rr_cv = irregularity["rr_cv"]

    avg_hr = sum(
        b.get("heart_rate", 0)
        for b in valid_beats
    ) / len(valid_beats)

    # Flutter tends to be:
    # - tachycardic
    # - somewhat regular
    # - not chaotic like AFib

    if (
        avg_hr >= hr_threshold
        and rr_cv >= rr_cv_min
        and rr_cv <= rr_cv_max
    ):

        severity = (
            "high"
            if avg_hr >= 140
            else "moderate"
        )

        return {
            "event_type": "POSSIBLE_FLUTTER",
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "rr_cv": rr_cv
            }
        }

    return None
def detect_svt(
    beats_history,
    min_window=10,
    hr_threshold=140,
    qrs_threshold=120,
    rr_cv_max=0.10,
    min_quality=0.50,
    min_confidence=0.70
):

    if len(beats_history) < min_window:
        return None

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    avg_hr = sum(
        b.get("heart_rate", 0)
        for b in valid_beats
    ) / len(valid_beats)

    avg_qrs = sum(
        b.get("qrs_width", 0)
        for b in valid_beats
    ) / len(valid_beats)

    irregularity = calculate_rr_irregularity(valid_beats)

    if irregularity is None:
        return None

    rr_cv = irregularity["rr_cv"]

    # SVT:
    # - fast
    # - regular
    # - narrow QRS

    if (
        avg_hr >= hr_threshold
        and avg_qrs < qrs_threshold
        and rr_cv <= rr_cv_max
    ):

        severity = (
            "high"
            if avg_hr >= 180
            else "moderate"
        )

        return {
            "event_type": "SVT_SUSPECTED",
            "severity": severity,
            "metadata_json": {
                "average_hr": avg_hr,
                "average_qrs": avg_qrs,
                "rr_cv": rr_cv
            }
        }

    return None
def detect_possible_av_block(
    beats_history,
    pause_threshold_ms=2200,
    brady_threshold=50,
    min_window=8,
    min_quality=0.50,
    min_confidence=0.70
):

    if len(beats_history) < min_window:
        return None

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    pauses = []

    for beat in valid_beats:

        rr = beat.get("rr_interval")

        if rr is None:
            continue

        rr_ms = rr if rr > 10 else rr * 1000

        if rr_ms >= pause_threshold_ms:
            pauses.append(rr_ms)

    avg_hr = sum(
        b.get("heart_rate", 0)
        for b in valid_beats
    ) / len(valid_beats)

    if pauses and avg_hr <= brady_threshold:

        severity = (
            "high"
            if avg_hr <= 40
            else "moderate"
        )

        return {
            "event_type": "POSSIBLE_AV_BLOCK",
            "severity": severity,
            "metadata_json": {
                "pause_count": len(pauses),
                "average_hr": avg_hr,
                "longest_pause_ms": max(pauses)
            }
        }

    return None
import numpy as np

def estimate_qt_interval(
    samples,
    sampling_rate=125
):
    """
    Heuristic QT interval estimator.

    NOT clinical-grade.
    Used only for possible long-QT screening.
    """

    signal = np.array(samples)

    if len(signal) < 50:
        return None

    r_peak = np.argmax(signal)

    peak_value = signal[r_peak]

    # Approximate T-wave end:
    # find return toward baseline after R peak

    baseline = np.mean(signal[:20])

    threshold = baseline + (peak_value - baseline) * 0.15

    t_end = None

    for i in range(r_peak + 10, len(signal)):

        if signal[i] <= threshold:
            t_end = i
            break

    if t_end is None:
        return None

    qt_samples = t_end

    qt_ms = (qt_samples / sampling_rate) * 1000

    return qt_ms
import math

def calculate_qtc(
    qt_ms,
    rr_interval
):

    if qt_ms is None:
        return None

    if rr_interval is None or rr_interval <= 0:
        return None

    # convert RR to seconds if needed
    rr_sec = rr_interval / 1000 if rr_interval > 10 else rr_interval

    try:
        qtc = qt_ms / math.sqrt(rr_sec)
        return qtc

    except:
        return None
def detect_possible_long_qt(
    beats_history,
    qtc_threshold=470,
    severe_threshold=500,
    min_window=5,
    min_quality=0.50,
    min_confidence=0.70
):

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        qtc = beat.get("qtc")

        if qtc is None:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    avg_qtc = sum(
        b["qtc"]
        for b in valid_beats
    ) / len(valid_beats)

    prolonged_ratio = sum(
        1
        for b in valid_beats
        if b["qtc"] >= qtc_threshold
    ) / len(valid_beats)

    if prolonged_ratio < 0.7:
        return None

    severity = (
        "high"
        if avg_qtc >= severe_threshold
        else "moderate"
    )

    return {
        "event_type": "POSSIBLE_LONG_QT",
        "severity": severity,
        "metadata_json": {
            "average_qtc": avg_qtc,
            "prolonged_ratio": prolonged_ratio
        }
    }
def detect_possible_ischemia(
    beats_history,
    st_threshold=0.15,
    min_window=8,
    min_quality=0.50,
    min_confidence=0.70
):

    valid_beats = []

    for beat in beats_history:

        if beat.get("prediction_confidence", 0) < min_confidence:
            continue

        if beat.get("signal_quality_score", 0) < min_quality:
            continue

        if beat.get("st_deviation") is None:
            continue

        valid_beats.append(beat)

    if len(valid_beats) < min_window:
        return None

    abnormal_st = [
        b for b in valid_beats
        if abs(b["st_deviation"]) >= st_threshold
    ]

    abnormal_ratio = len(abnormal_st) / len(valid_beats)

    t_inversions = sum(
        1 for b in valid_beats
        if b.get("t_wave_inverted")
    )

    inversion_ratio = t_inversions / len(valid_beats)

    if (
        abnormal_ratio >= 0.6
        or inversion_ratio >= 0.6
    ):

        severity = (
            "high"
            if abnormal_ratio >= 0.8
            else "moderate"
        )

        return {
            "event_type": "POSSIBLE_ISCHEMIC_PATTERN",
            "severity": severity,
            "metadata_json": {
                "abnormal_st_ratio": abnormal_ratio,
                "t_inversion_ratio": inversion_ratio
            }
        }

    return None
def augment_cad_risk(
    detected_events,
    patient_metadata=None
):

    risk_score = 0
    evidence = []

    event_types = {
        e["event_type"]
        for e in detected_events
    }

    if "POSSIBLE_ISCHEMIC_PATTERN" in event_types:
        risk_score += 3
        evidence.append("ischemic ECG pattern")

    if "AFIB_SUSPECTED" in event_types:
        risk_score += 1
        evidence.append("AFib")

    if "VT_RUN" in event_types:
        risk_score += 2
        evidence.append("ventricular arrhythmia")

    if patient_metadata:

        if patient_metadata.get("smoker"):
            risk_score += 1
            evidence.append("smoking")

        if patient_metadata.get("hypertension"):
            risk_score += 1
            evidence.append("hypertension")

        if patient_metadata.get("diabetes"):
            risk_score += 1
            evidence.append("diabetes")

        if patient_metadata.get("hyperlipidemia"):
            risk_score += 1
            evidence.append("hyperlipidemia")

    risk_level = (
        "high"
        if risk_score >= 6
        else "moderate"
        if risk_score >= 3
        else "low"
    )
    if risk_score == 0:
        return None
    return {
        "event_type": "CAD_RISK_AUGMENTED",
        "severity": risk_level,
        "metadata_json": {
            "cad_risk_score": risk_score,
            "evidence": evidence
        }
    }
def analyze_temporal_window(session_id: str, db_connection: Any,patient_metadata=None, window_size: int = 50) -> List[Dict[str, Any]]:
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
    if burdens["pvc_burden"] > 0.15:
        events_detected.append({
            "event_type": "HIGH_PVC_BURDEN",
            "severity": "moderate",
            "metadata_json": burdens
        })
        
    # 2. VT Runs
    vt_run = detect_vt_runs(history)
    if vt_run:
        events_detected.append({
            "event_type": "VT_RUN",
            "severity": vt_run["severity"],
            "metadata_json": vt_run
        })
        
    ## 3. Tachy / Brady
    rate_events = detect_rate_abnormalities(history)
    events_detected.extend(rate_events)

    ## 4. AFib detection
    afib_event = detect_afib(history)

    if afib_event:
        events_detected.append(afib_event)
        
    # 5. BBB suspicion
    bbb_event = detect_possible_bbb(history)

    if bbb_event:
        events_detected.append(bbb_event)
    # 6. Possible Flutter
    flutter_event = detect_possible_flutter(history)

    if flutter_event:
        events_detected.append(flutter_event)

    # 7. SVT
    svt_event = detect_svt(history)

    if svt_event:
        events_detected.append(svt_event)

    # 8. Possible AV block
    av_block_event = detect_possible_av_block(history)

    if av_block_event:
        events_detected.append(av_block_event)

    # 9. Possible Long-QT
    long_qt_event = detect_possible_long_qt(history)

    if long_qt_event:
        events_detected.append(long_qt_event)
    # 10. Heart failure risk augmentation
    hf_event = augment_heart_failure_risk(
        events_detected,
        patient_metadata
    )

    if hf_event:
        events_detected.append(hf_event)
    # 11. Possible ischemia
    ischemia_event = detect_possible_ischemia(history)

    if ischemia_event:
        events_detected.append(ischemia_event)

    # 12. CAD risk augmentation
    cad_event = augment_cad_risk(
        events_detected,
        patient_metadata
    )

    if cad_event:
        events_detected.append(cad_event)
    # 13. Cardiomyopathy augmentation
    cardio_event = augment_cardiomyopathy_risk(
        events_detected,
        patient_metadata
    )
    
    if cardio_event:
        events_detected.append(cardio_event)
    
    # 14. Valvular disease augmentation
    valvular_event = augment_valvular_disease_risk(
        events_detected,
        patient_metadata
    )
    
    if valvular_event:
        events_detected.append(valvular_event)
    
    # 15. Myocarditis augmentation
    myocarditis_event = augment_myocarditis_risk(
        events_detected,
        patient_metadata
    )
    
    if myocarditis_event:
        events_detected.append(myocarditis_event)
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
        
    # 4. Pauses
    if detect_pauses(history):
        events_detected.append({
            "event_type": "PAUSE_DETECTED",
            "severity": "high",
            "metadata_json": {"rr_interval": history[-1]['rr_interval']}
        })
    for event in events_detected:
        manager_decision = evaluate_event(event["event_type"])
        event["event_manager"] = manager_decision
    # Store events in database
    last_time = history[-1]['timestamp']
    start_time = history[0]['timestamp']
    
    stored_events = []
    for event in events_detected:

        event_record = {
            "session_id": session_id,
            "event_type": event["event_type"],
            "event_start_time": start_time,
            "event_end_time": last_time,
            "severity": event["severity"],
            "metadata_json": event.get("metadata_json", {})
        }

        process_event_with_cooldown(
            db_connection,
            session_id,
            event_record
        )
    return events_detected