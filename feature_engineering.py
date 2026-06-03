import numpy as np
from typing import Dict, Any, List
from temporal_analysis import estimate_qt_interval, calculate_qtc

def calculate_heart_rate(rr_interval: float) -> float:
    """
    Calculates heart rate in BPM.
    Assuming rr_interval is provided in seconds. 
    If values are large, it assumes milliseconds.
    """
    if rr_interval <= 0:
        return 0.0
    if rr_interval > 10: # likely milliseconds
        return 60000.0 / rr_interval
    return 60.0 / rr_interval

def extract_amplitude_features(ecg_samples: np.ndarray) -> Dict[str, float]:
    """
    Extracts basic amplitude-based statistical features from the 187 sample window.
    """
    if len(ecg_samples) == 0:
        return {
            "amplitude_mean": 0.0,
            "amplitude_std": 0.0,
            "amplitude_min": 0.0,
            "amplitude_max": 0.0,
            "peak_to_peak": 0.0
        }
    
    return {
        "amplitude_mean": float(np.mean(ecg_samples)),
        "amplitude_std": float(np.std(ecg_samples)),
        "amplitude_min": float(np.min(ecg_samples)),
        "amplitude_max": float(np.max(ecg_samples)),
        "peak_to_peak": float(np.max(ecg_samples) - np.min(ecg_samples))
    }

def estimate_qrs_width(ecg_samples: np.ndarray, sampling_rate: int = 125) -> float:
    """
    Approximate QRS width estimator.
    The input array is typically 187 samples with the R-peak roughly near the center (index ~90 in many alignments).
    We find the maximum absolute peak, then trace outwards until amplitude drops below a threshold.
    Returns estimated width in milliseconds.
    """
    if len(ecg_samples) < 10:
        return 0.0
    
    # Detrend/center around mean locally for baseline
    centered = ecg_samples - np.mean(ecg_samples)
    abs_samples = np.abs(centered)
    
    # Assume the R peak is the max absolute value in the middle third
    start_idx = len(ecg_samples) // 3
    end_idx = 2 * len(ecg_samples) // 3
    if start_idx >= end_idx:
        peak_idx = np.argmax(abs_samples)
    else:
        peak_idx = start_idx + np.argmax(abs_samples[start_idx:end_idx])
    
    peak_val = abs_samples[peak_idx]
    
    # If it's a flatline essentially, width is 0
    if peak_val < 1e-4:
        return 0.0

    # Define threshold for QRS onset/offset (e.g., 20% of max peak)
    threshold = 0.20 * peak_val
    
    left_idx = peak_idx
    while left_idx > 0 and abs_samples[left_idx] > threshold:
        left_idx -= 1
        
    right_idx = peak_idx
    while right_idx < len(ecg_samples) - 1 and abs_samples[right_idx] > threshold:
        right_idx += 1
        
    width_samples = right_idx - left_idx
    # Convert samples to milliseconds
    width_ms = (width_samples / sampling_rate) * 1000.0
    return float(width_ms)
def estimate_st_deviation(
    ecg_samples,
    sampling_rate=125
):
    """
    Heuristic ST-segment deviation estimator.

    NOT clinical-grade.
    Used only for ischemia-pattern augmentation.
    """

    signal = np.array(ecg_samples)

    if len(signal) < 80:
        return None

    # R peak
    r_peak = np.argmax(np.abs(signal))

    # Baseline estimate
    baseline = np.mean(signal[:20])

    # Approximate J-point
    j_point = r_peak + int(0.04 * sampling_rate)

    # Approximate ST measurement point
    st_point = j_point + int(0.08 * sampling_rate)

    if st_point >= len(signal):
        return None

    st_value = signal[st_point]

    st_deviation = st_value - baseline

    return float(st_deviation)
def estimate_t_wave_abnormality(
    ecg_samples,
    sampling_rate=125
):

    signal = np.array(ecg_samples)

    if len(signal) < 100:
        return None

    r_peak = np.argmax(np.abs(signal))

    # T-wave region approximation
    t_start = r_peak + int(0.12 * sampling_rate)
    t_end = r_peak + int(0.32 * sampling_rate)

    if t_end >= len(signal):
        return None

    t_region = signal[t_start:t_end]

    if len(t_region) == 0:
        return None

    t_peak = np.max(t_region)
    t_min = np.min(t_region)

    # crude inversion heuristic
    inverted = abs(t_min) > abs(t_peak)

    return {
        "t_peak": float(t_peak),
        "t_min": float(t_min),
        "t_inverted": inverted
    }

def calculate_signal_quality(ecg_samples: np.ndarray) -> float:
    """
    Heuristic to estimate signal quality (0.0 to 1.0).
    Penalizes flatlines, clipping, and extreme jumps.
    """
    if len(ecg_samples) < 2:
        return 0.0
        
    score = 1.0
    
    std_val = np.std(ecg_samples)
    p2p = np.max(ecg_samples) - np.min(ecg_samples)
    diffs = np.abs(np.diff(ecg_samples))
    max_jump = np.max(diffs) if len(diffs) > 0 else 0
    
    # Penalty for flatline (variance too low)
    if std_val < 0.001:
        return 0.0
        
    # Penalty for massive amplitude jumps (likely artifact/motion)
    if max_jump > 2.0 * std_val:
        # reduce score based on how crazy the jump is
        score -= min(0.5, (max_jump / (2.0 * std_val)) * 0.1)
        
    # Penalty if p2p is absurdly large relative to normal normalized signals
    if p2p > 10.0:
        score -= 0.3
        
    return float(max(0.0, min(1.0, score)))

def process_beat(
    ecg_samples: List[float],
    rr_interval: float,
    predicted_label: int,
    confidence: float,
    session_id: str,
    timestamp: float,
    beat_index: int
) -> Dict[str, Any]:
    """
    Orchestrates feature extraction for a single beat.
    Returns a dictionary suitable for insertion into the database.
    """
    samples_arr = np.array(ecg_samples)
    
    amp_features = extract_amplitude_features(samples_arr)
    hr = calculate_heart_rate(rr_interval)
    qrs_width = estimate_qrs_width(samples_arr)
    sig_quality = calculate_signal_quality(samples_arr)
    qt_interval = estimate_qt_interval(samples_arr)
    st_deviation = estimate_st_deviation(samples_arr)
    t_wave_features = estimate_t_wave_abnormality(samples_arr)

    qtc = calculate_qtc(
        qt_interval,
        rr_interval
    )
    # We will assume labels > 0 are abnormal based on typical MIT-BIH (N=0, S=1, V=2, F=3, Q=4)
    is_abnormal = (predicted_label != 0)
    
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "beat_index": beat_index,
        "predicted_label": int(predicted_label),
        "prediction_confidence": float(confidence),
        "rr_interval": float(rr_interval),
        "heart_rate": hr,
        "qrs_width": qrs_width,
        "qt_interval": qt_interval,
        "qtc": qtc,
        "amplitude_mean": amp_features["amplitude_mean"],
        "amplitude_std": amp_features["amplitude_std"],
        "amplitude_min": amp_features["amplitude_min"],
        "amplitude_max": amp_features["amplitude_max"],
        "peak_to_peak": amp_features["peak_to_peak"],
        "signal_quality_score": sig_quality,
        "is_abnormal": bool(is_abnormal),
        "raw_feature_json": {
            # Storing extra agent-accessible features as needed
            "extracted_at": timestamp
        },
        "st_deviation": st_deviation,

        "t_wave_peak": (
            t_wave_features["t_peak"]
            if t_wave_features else None
        ),

        "t_wave_min": (
            t_wave_features["t_min"]
            if t_wave_features else None
        ),

        "t_wave_inverted": bool(
            t_wave_features["t_inverted"]
        )
        if t_wave_features else None
    }
