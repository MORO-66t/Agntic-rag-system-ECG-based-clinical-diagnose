"""
Feature packaging adapter.

Clinical ECG features are extracted in neurokit_feature_extractor.py. This file
only maps those NeuroKit-derived values into the database/pipeline schema and
adds runtime metadata such as model prediction, confidence, session, and beat id.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from neurokit_feature_extractor import extract_neurokit_morphology, extract_all_p_waves


CLINICAL_FS = 360
CNN_FS = 125
CNN_LEN = 187


def process_beat(
    cnn_samples: List[float],
    original_samples: List[float],
    rr_interval: float,
    predicted_label: int,
    confidence: float,
    session_id: str,
    timestamp: float,
    beat_index: int,
    t_peak_position: int = -1,
    p_peak_position: int = -1,
    rt_interval_ms: float = -1.0,
    pr_interval_ms_gt: float = -1.0,
    # ── NeuroKit2 context window (real delineation) ──
    context_samples: Optional[np.ndarray] = None,
    context_rpeaks: Optional[np.ndarray] = None,
    context_target_r: Optional[int] = None,
    beat_start: int = 0,
    context_info: Optional[Dict] = None,
    # ── Morphology debug mode ──
    debug_morphology: bool = False,
) -> Dict[str, Any]:
    """
    Return a beat feature row using neurokit_feature_extractor.py as the sole
    source for clinical morphology, interval, amplitude, and quality features.

    Parameters
    ----------
    debug_morphology : bool
        If True, the extractor returns detailed per-feature diagnostics
        explaining how each value was computed (which method succeeded/failed
        and why). Zero overhead when False (default).
    """
    _ = cnn_samples  # CNN samples are model input only; not a clinical feature source here.
    _ = pr_interval_ms_gt

    orig_arr = np.asarray(original_samples, dtype=np.float64)
    nk_result = extract_neurokit_morphology(
        orig_arr,
        sampling_rate=CLINICAL_FS,
        rr_interval=rr_interval,
        context_samples=context_samples,
        context_rpeaks=context_rpeaks,
        context_target_r=context_target_r,
        beat_start=beat_start,
        context_info=context_info,
        debug=debug_morphology,
    )
    nk_success = bool(nk_result.get("success", False))
    feature_source = nk_result.get("source", "neurokit2_failed")

    # ── Extract all P-waves for AV block detection ────────────
    all_p_waves = extract_all_p_waves(
        orig_arr,
        sampling_rate=CLINICAL_FS,
        beat_timestamp=timestamp,
    )

    result = {
        "session_id": session_id,
        "timestamp": timestamp,
        "beat_index": beat_index,
        "predicted_label": int(predicted_label),
        "prediction_confidence": float(confidence),
        "rr_interval": float(rr_interval),
        "heart_rate": nk_result.get("heart_rate"),
        "qrs_width": nk_result.get("qrs_width_ms"),
        "qrs_voltage": nk_result.get("qrs_voltage"),
        "q_amplitude": nk_result.get("q_amplitude"),
        "r_amplitude": nk_result.get("r_amplitude"),
        "s_amplitude": nk_result.get("s_amplitude"),
        "r_peak_idx": nk_result.get("r_peak_idx"),
        "qrs_onset": nk_result.get("qrs_onset"),
        "qrs_offset": nk_result.get("qrs_offset"),
        "q_peak": nk_result.get("q_peak"),
        "s_peak": nk_result.get("s_peak"),
        "qt_interval": nk_result.get("qt_interval_ms"),
        "qtc": nk_result.get("qtc_fridericia") or nk_result.get("qtc_bazett"),
        "qtc_bazett": nk_result.get("qtc_bazett"),
        "qtc_fridericia": nk_result.get("qtc_fridericia"),
        "amplitude_mean": nk_result.get("amplitude_mean"),
        "amplitude_std": nk_result.get("amplitude_std"),
        "amplitude_min": nk_result.get("amplitude_min"),
        "amplitude_max": nk_result.get("amplitude_max"),
        "peak_to_peak": nk_result.get("peak_to_peak"),
        "signal_quality_score": nk_result.get("signal_quality_score", 0.0),
        "is_abnormal": bool(predicted_label != 0),
        "st_deviation": nk_result.get("st_deviation"),
        "st_segment_ms": nk_result.get("st_segment_ms"),
        "t_wave_min": nk_result.get("t_wave_min"),
        "t_wave_inverted": nk_result.get("t_wave_inverted"),
        "t_wave_amplitude": nk_result.get("t_wave_amplitude"),
        "t_wave_polarity": nk_result.get("t_wave_polarity"),
        "tpeak_tend_interval_ms": nk_result.get("tpeak_tend_interval_ms"),
        "t_wave_width_ms": nk_result.get("t_wave_width_ms"),
        "p_wave_detected": nk_result.get("p_wave_detected", False),
        "p_wave_inverted": nk_result.get("p_wave_inverted"),
        "p_wave_width_ms": nk_result.get("p_wave_width_ms"),
        "p_wave_prominence": nk_result.get("p_wave_prominence"),
        "p_wave_amplitude": nk_result.get("p_wave_amplitude"),
        "p_wave_polarity": nk_result.get("p_wave_polarity"),
        "pr_interval_ms": nk_result.get("pr_interval_ms"),
        "pr_segment_ms": nk_result.get("pr_segment_ms"),
        "r_wave_inverted": None,
        "p_onset": nk_result.get("p_onset"),
        "p_peak": nk_result.get("p_peak"),
        "p_offset": nk_result.get("p_offset"),
        "t_onset": nk_result.get("t_onset"),
        "t_peak": nk_result.get("t_peak"),
        "t_offset": nk_result.get("t_offset"),
        "feature_source": feature_source,
        "u_wave_detected": nk_result.get("u_wave_detected", False),
        "u_wave_peak": nk_result.get("u_wave_peak"),
        "u_wave_amplitude": nk_result.get("u_wave_amplitude"),
        "delta_wave_detected": nk_result.get("delta_wave_detected", False),
        "delta_wave_slope": nk_result.get("delta_wave_slope"),
        "flutter_baseline_power": nk_result.get("flutter_baseline_power"),
        "flutter_baseline_dominant_hz": nk_result.get("flutter_baseline_dominant_hz"),
        "flutter_baseline_detected": nk_result.get("flutter_baseline_detected", False),
        "flutter_organization_index": nk_result.get("flutter_organization_index"),
        "qrs_axis_deg": nk_result.get("qrs_axis_deg"),
        "qtc_dispersion_ms": nk_result.get("qtc_dispersion_ms"),
        "electrical_alternans_detected": nk_result.get("electrical_alternans_detected"),
        "epsilon_wave_detected": nk_result.get("epsilon_wave_detected"),
        "spodick_sign_detected": nk_result.get("spodick_sign_detected"),
        "rhythm_classification": nk_result.get("rhythm_classification"),
        "hrv_mean_nn": nk_result.get("hrv_mean_nn"),
        "hrv_sdnn": nk_result.get("hrv_sdnn"),
        "hrv_rmssd": nk_result.get("hrv_rmssd"),
        "t_peak_position": int(t_peak_position) if t_peak_position != -1 else None,
        "p_peak_position": int(p_peak_position) if p_peak_position != -1 else None,
        "rt_interval_ms": float(rt_interval_ms) if rt_interval_ms != -1.0 else None,
        "context_samples": context_samples.tolist() if isinstance(context_samples, np.ndarray) else context_samples,
        "raw_feature_json": {
            "extracted_at": timestamp,
            "clinical_fs": CLINICAL_FS,
            "feature_contract": "all_clinical_features_from_neurokit_feature_extractor",
            "neurokit2_source": feature_source,
            "neurokit2_success": nk_success,
            "neurokit2_error": nk_result.get("error"),
            "advanced_feature_availability": nk_result.get("advanced_feature_availability", {}),
            "all_p_waves": all_p_waves,
            "flutter_baseline_periodogram": nk_result.get("flutter_baseline_periodogram"),
        },
    }

    # Morphology diagnostics (only present when debug_morphology=True)
    morph_diag = nk_result.get("_diagnostics")
    if morph_diag is not None:
        result["morphology_diagnostics"] = morph_diag

    return result
