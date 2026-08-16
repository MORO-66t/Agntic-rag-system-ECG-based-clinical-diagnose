"""
NeuroKit2-first ECG morphology extraction for the 360 Hz clinical branch.

This module is the primary source for morphology, interval, and amplitude
features. Some advanced features from the project roadmap need multi-lead ECG
or a longer rhythm strip; those are returned with explicit availability flags
instead of being silently guessed from a single Lead-II beat.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

try:
    import neurokit2 as nk
except Exception:  # pragma: no cover - handled at runtime in environments without NeuroKit2
    nk = None

# Diagnostic system (optional — zero overhead when debug=False)
try:
    from morphology_diagnostics import MorphologyDiagnosticCollector, NK_KEY_TO_FEATURE
    _HAS_DIAGNOSTICS = True
except ImportError:
    MorphologyDiagnosticCollector = None  # type: ignore
    NK_KEY_TO_FEATURE = {}  # type: ignore
    _HAS_DIAGNOSTICS = False


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _detect_r_peak(cleaned: np.ndarray, fs: int) -> int:
    heart_rate = 70.0
    min_distance = int((60.0 / heart_rate) * 0.5 * fs)
    peaks, _ = find_peaks(cleaned, distance=min_distance)
    if len(peaks) == 0:
        return int(len(cleaned) // 2)
    return int(peaks[-1] if len(peaks) > 1 else peaks[0])


def _peak_window(cleaned: np.ndarray, start: int, end: int, prefer_abs: bool = True) -> Optional[int]:
    start = max(0, start)
    end = min(len(cleaned), end)
    if end <= start:
        return None
    region = cleaned[start:end]
    if len(region) == 0:
        return None
    if prefer_abs:
        return int(start + np.argmax(np.abs(region)))
    return int(start + np.argmax(region))


def _threshold_bounds(
    cleaned: np.ndarray,
    peak: Optional[int],
    fs: int,
    max_ms: float,
    rel_height: float = 0.2,
) -> tuple[Optional[int], Optional[int]]:
    if peak is None:
        return None, None
    peak_value = cleaned[peak]
    baseline = float(np.median(cleaned[max(0, peak - int(max_ms / 1000 * fs)) : peak + 1]))
    threshold = baseline + (peak_value - baseline) * rel_height
    is_positive = peak_value >= baseline
    max_walk = int(max_ms / 1000 * fs)

    left = peak
    for idx in range(peak - 1, max(0, peak - max_walk), -1):
        if (is_positive and cleaned[idx] <= threshold) or ((not is_positive) and cleaned[idx] >= threshold):
            left = idx
            break
        left = idx

    right = peak
    for idx in range(peak + 1, min(len(cleaned) - 1, peak + max_walk)):
        if (is_positive and cleaned[idx] <= threshold) or ((not is_positive) and cleaned[idx] >= threshold):
            right = idx
            break
        right = idx

    return left, right


def _fallback_landmarks(cleaned: np.ndarray, r_peak: int, fs: int) -> Dict[str, Optional[int]]:
    """Beat-window landmark estimates used when NeuroKit delineation needs more context."""
    q_peak = _peak_window(cleaned, r_peak - int(0.08 * fs), r_peak, prefer_abs=False)
    s_peak = _peak_window(-cleaned, r_peak, r_peak + int(0.10 * fs), prefer_abs=False)

    qrs_onset = q_peak if q_peak is not None else max(0, r_peak - int(0.04 * fs))
    qrs_offset = s_peak if s_peak is not None else min(len(cleaned) - 1, r_peak + int(0.06 * fs))
    if qrs_onset is not None:
        qrs_onset = max(0, qrs_onset - int(0.02 * fs))
    if qrs_offset is not None:
        qrs_offset = min(len(cleaned) - 1, qrs_offset + int(0.02 * fs))

    p_peak = _peak_window(cleaned, r_peak - int(0.32 * fs), r_peak - int(0.08 * fs))
    p_onset, p_offset = _threshold_bounds(cleaned, p_peak, fs, max_ms=140)
    # Ensure p_offset does not cross into the QRS complex.
    # The threshold-crossing walk can overshoot into the QRS when the
    # P-wave and QRS are close together (common in 720-sample centered
    # windows). Clamp p_offset to at most qrs_onset - 1 sample.
    if p_offset is not None and qrs_onset is not None:
        p_offset = min(p_offset, max(0, qrs_onset - 1))
    
    p_peak_candidate = _peak_window(cleaned, r_peak - int(0.32 * fs), r_peak - int(0.08 * fs))
    # Reject candidates that are baseline noise or slope artifacts rather than a true P-wave.
    # Requires:
    # 1. Prominence >= 2.5 * local noise standard deviation
    # 2. Minimum peak amplitude prominence >= 0.03 mV relative to local baseline median
    # 3. Peak candidate is a true local extrema relative to immediate neighbors
    # 4. Distance to QRS onset is at least 40 ms
    p_peak = None
    if p_peak_candidate is not None:
        win_start = max(0, r_peak - int(0.32 * fs))
        win_end = max(win_start + 1, r_peak - int(0.08 * fs))
        window = cleaned[win_start:win_end]
        local_noise = float(np.std(window)) if len(window) else 0.0
        local_median = float(np.median(window)) if len(window) else 0.0
        candidate_prominence = abs(cleaned[p_peak_candidate] - local_median)
        
        # Local extrema check
        is_local_extrema = False
        if 0 < p_peak_candidate < len(cleaned) - 1:
            val = cleaned[p_peak_candidate]
            is_local_extrema = (val > cleaned[p_peak_candidate - 1] and val > cleaned[p_peak_candidate + 1]) or \
                               (val < cleaned[p_peak_candidate - 1] and val < cleaned[p_peak_candidate + 1])

        # Distance to QRS onset check
        dist_to_qrs = (qrs_onset - p_peak_candidate) if qrs_onset is not None else int(0.05 * fs)

        if (local_noise > 0 and 
            candidate_prominence >= 2.5 * local_noise and 
            candidate_prominence >= 0.03 and 
            is_local_extrema and 
            dist_to_qrs >= int(0.04 * fs)):
            p_peak = p_peak_candidate
    t_peak = _peak_window(cleaned, r_peak + int(0.12 * fs), r_peak + int(0.45 * fs))
    t_onset, t_offset = _threshold_bounds(cleaned, t_peak, fs, max_ms=220)

    return {
        "p_onset": p_onset,
        "p_peak": p_peak,
        "p_offset": p_offset,
        "q_peak": q_peak,
        "qrs_onset": qrs_onset,
        "qrs_offset": qrs_offset,
        "s_peak": s_peak,
        "t_onset": t_onset,
        "t_peak": t_peak,
        "t_offset": t_offset,
    }


def _nearest(positions, target, direction, max_d):
    """Nearest delineated landmark to ``target`` in the given direction, within max_d samples.

    ``positions`` is an array of sample indices for one fiducial (e.g. all
    T-onsets in the context window). Returns the closest one on the correct
    side of ``target``, or None if none is within ``max_d``.
    """
    positions = np.asarray(positions, dtype=float)
    positions = positions[~np.isnan(positions)]
    if len(positions) == 0:
        return None
    if direction == "before":
        candidates = positions[positions <= target]
    else:
        candidates = positions[positions >= target]
    if len(candidates) == 0:
        return None
    best = candidates[int(np.argmin(np.abs(candidates - target)))]
    if abs(best - target) > max_d:
        return None
    return float(best)


def _extract_target_landmarks(ctx_info, ctx_rpeaks, target_r, beat_start, fs, signal_len):
    """Pick the target beat's landmarks from a context-window delineation.

    ``ctx_info`` is the dict returned by ``nk.ecg_delineate`` over the whole
    context window (one entry per R-peak). ``ctx_rpeaks`` are the R-peak
    indices in context coordinates; ``target_r`` is the context-coordinate
    R-peak of the beat being stored. Landmarks are remapped into the single-
    beat (``original_samples``) frame via ``beat_start`` (the start index of
    ``original_samples`` within ``context_samples``) and returned as single-
    element arrays, matching the shape the rest of the function consumes.
    """
    ctx_rpeaks = np.asarray(ctx_rpeaks, dtype=int)
    k = int(np.argmin(np.abs(ctx_rpeaks - target_r)))   # index of stored beat in window
    max_pr = int(0.35 * fs)
    max_qrs = int(0.20 * fs)
    max_t = int(0.70 * fs)

    def pick(key, near, max_d):
        arr = np.asarray(ctx_info.get(key, []), dtype=float)
        val = _nearest(arr, ctx_rpeaks[k], near, max_d)
        if val is None:
            return np.array([], dtype=float)
        remapped = int(np.clip(val - beat_start, 0, signal_len - 1))
        return np.array([remapped], dtype=float)

    return {
        "ECG_R_Onsets":   pick("ECG_R_Onsets",   "before", max_qrs),
        "ECG_R_Offsets":  pick("ECG_R_Offsets",  "after",  max_qrs),
        "ECG_P_Onsets":   pick("ECG_P_Onsets",   "before", max_pr),
        "ECG_P_Peaks":    pick("ECG_P_Peaks",    "before", max_pr),
        "ECG_P_Offsets":  pick("ECG_P_Offsets",  "before", max_pr),
        "ECG_Q_Peaks":    pick("ECG_Q_Peaks",    "before", max_qrs),
        "ECG_S_Peaks":    pick("ECG_S_Peaks",    "after",  max_qrs),
        "ECG_T_Onsets":   pick("ECG_T_Onsets",   "after",  max_t),
        "ECG_T_Peaks":    pick("ECG_T_Peaks",    "after",  max_t),
        "ECG_T_Offsets":  pick("ECG_T_Offsets",  "after",  max_t),
    }


def extract_neurokit_morphology(
    original_samples,
    sampling_rate: int = 360,
    clean_method: str = "neurokit",
    rr_interval: Optional[float] = None,
    context_samples: Optional[np.ndarray] = None,   # multi-beat window (raw mV)
    context_rpeaks: Optional[np.ndarray] = None,   # R-peaks within context (relative to ctx start)
    context_target_r: Optional[int] = None,         # the beat being STORED, in context coords
    beat_start: int = 0,                          # start index of original_samples within context_samples
    context_info: Optional[Dict] = None,          # pre-computed nk.ecg_delineate result for the whole context
    debug: bool = False,                          # enable morphology diagnostics
) -> Dict[str, Any]:
    """
    Extract beat-level ECG features from a single clinical beat window.

    The returned keys are designed to map directly into feature_engineering.py
    and the beat_features database table.

    Context-window delineation (new way)
    ------------------------------------
    If ``context_samples`` (a multi-beat raw window) and ``context_rpeaks``
    (>=2 R-peaks inside it) are supplied, NeuroKit2 delineates the WHOLE window
    once and only the target beat's landmarks are kept, remapped into the
    ``original_samples`` frame. This lets NeuroKit2 succeed where a lone padded
    beat fails. Contract: ``original_samples == context_samples[beat_start :
    beat_start + len(original_samples)]`` and ``context_target_r`` is the
    context-coordinate R-peak of the beat being stored. When no context is
    given, the original padded single-beat path (and its heuristic fallback)
    is used unchanged. If ``context_info`` (a pre-computed
    ``nk.ecg_delineate`` result for the whole context window) is supplied, it
    is reused directly instead of re-delineating ``context_samples`` — this is
    the efficient whole-record variant used by the production pipeline.

    Parameters
    ----------
    debug : bool
        If True, return a ``_diagnostics`` key with a
        ``MorphologyDiagnosticCollector.to_dict()`` explaining how every
        feature was computed. Zero overhead when False (default).
    """
    # ── Diagnostic collector (zero overhead when debug=False) ──────────
    _diag: Optional[MorphologyDiagnosticCollector] = None
    if debug and _HAS_DIAGNOSTICS and MorphologyDiagnosticCollector is not None:
        _diag = MorphologyDiagnosticCollector(beat_index=0)

    if nk is None:
        if _diag is not None:
            _diag.record_neurokit2_failure("import", "neurokit2 is not installed")
        result = {"source": "neurokit2_unavailable", "success": False, "error": "neurokit2 is not installed"}
        if _diag is not None:
            result["_diagnostics"] = _diag.to_dict()
        return result

    signal = np.asarray(original_samples, dtype=np.float64)
    if signal.size < int(0.35 * sampling_rate):
        if _diag is not None:
            _diag.record_neurokit2_failure(
                "window_too_short",
                f"Beat window is too short for ECG delineation "
                f"({signal.size} samples < {int(0.35 * sampling_rate)} required)"
            )
        result = {"source": "neurokit2", "success": False, "error": "beat window is too short for ECG delineation"}
        if _diag is not None:
            result["_diagnostics"] = _diag.to_dict()
        return result

    rr_ms = None
    rr_value = _finite(rr_interval)
    if rr_value is not None:
        rr_ms = rr_value if rr_value > 10 else rr_value * 1000.0

    try:
        cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=sampling_rate, method=clean_method), dtype=np.float64)
        r_peak = _detect_r_peak(cleaned, sampling_rate)

        info = None
        used_context = False
        delineation_method = "cleaned_window"

        # ── Context-window NeuroKit2 delineation (new way) ───────────────
        # Delineate the whole multi-beat window once (all R-peaks) and keep
        # only the target beat's landmarks, remapped into the single-beat
        # (original_samples) frame. This is what lets NeuroKit2 succeed where
        # a lone padded beat fails.
        context_available = (context_rpeaks is not None and len(context_rpeaks) >= 2
                             and context_target_r is not None)
        if _diag is not None:
            _diag.record_context_window_available(context_available)

        if context_available:
            ctx_rpeaks_arr = np.asarray(context_rpeaks, dtype=int)
            target_r_ctx = int(context_target_r)
            ctx_info = None
            if context_info is not None:
                ctx_info = context_info
            elif context_samples is not None:
                ctx_cleaned = np.asarray(
                    nk.ecg_clean(np.asarray(context_samples, dtype=np.float64),
                                 sampling_rate=sampling_rate, method=clean_method),
                    dtype=np.float64,
                )
                try:
                    _, ctx_info = nk.ecg_delineate(
                        ctx_cleaned, rpeaks=ctx_rpeaks_arr,
                        sampling_rate=sampling_rate, method="dwt")
                    if _diag is not None:
                        _diag.record_neurokit2_attempted()
                        # ── Debug: capture raw NK context output ─────────────
                        _diag._raw_context_info = {}
                        for k in ["ECG_R_Onsets", "ECG_R_Offsets", "ECG_P_Onsets",
                                   "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks",
                                   "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks",
                                   "ECG_T_Offsets"]:
                            arr = np.asarray(ctx_info.get(k, []), dtype=float)
                            _diag._raw_context_info[k] = {
                                "len": len(arr),
                                "n_nan": int(np.sum(np.isnan(arr))),
                                "n_finite": int(np.sum(np.isfinite(arr))),
                                "values": arr[~np.isnan(arr)].tolist()[:6],
                                "all_nan": bool(np.all(np.isnan(arr))),
                            }
                except Exception as exc:
                    if _diag is not None:
                        _diag.record_neurokit2_failure("dwt_context", f"DWT context delineation exception: {str(exc)}")
                    ctx_info = None
            if ctx_info is not None:
                try:
                    # ── Debug: log target beat info ──────────────────────
                    if _diag is not None:
                        _diag._ctx_target_info = {
                            "target_r_ctx": int(target_r_ctx),
                            "n_ctx_rpeaks": len(ctx_rpeaks_arr),
                            "ctx_rpeaks": ctx_rpeaks_arr.tolist(),
                            "beat_start": beat_start,
                            "signal_len": len(signal),
                        }
                    info = _extract_target_landmarks(
                        ctx_info, ctx_rpeaks_arr, target_r_ctx,
                        beat_start, sampling_rate, len(signal))
                    # If the target beat's R could not be located, fall back.
                    if len(np.asarray(info.get("ECG_R_Onsets", []), dtype=float)) == 0:
                        if _diag is not None:
                            _diag.record_neurokit2_failure(
                                "context_no_r_onsets",
                                "Context-window DWT returned no R_Onsets for the target beat"
                            )
                        info = None
                    else:
                        used_context = True
                        delineation_method = "dwt"
                        r_peak = int(np.clip(target_r_ctx - beat_start, 0, len(signal) - 1))
                except Exception as exc:
                    if _diag is not None:
                        _diag.record_neurokit2_failure(
                            "dwt_context",
                            f"Context-window landmark extraction exception: {str(exc)}"
                        )
                    info = None  # fall through to padded single-beat path

        # ── Fallback: padded single-beat delineation (original behaviour) ─
        if info is None:
            # DWT delineation needs context around a beat. Beat windows cropped
            # from prev-R to next-R can be too short, so we pad only for
            # delineation and map indices back into the original beat window.
            # NeuroKit2's ecg_segment() requires len(signal) >= sampling_rate * 4
            # (1440 at 360 Hz), so we must pad enough to meet that.
            min_pad = max(int(1.0 * sampling_rate), sampling_rate * 2 - len(signal))
            # Ensure padded signal is at least 4*fs samples for NeuroKit2
            needed_total = sampling_rate * 4
            if len(signal) + 2 * min_pad < needed_total:
                min_pad = max(min_pad, (needed_total - len(signal) + 1) // 2)
            pad = min_pad
            padded_signal = np.pad(signal, (pad, pad), mode="edge")
            padded_cleaned = np.asarray(
                nk.ecg_clean(padded_signal, sampling_rate=sampling_rate, method=clean_method),
                dtype=np.float64,
            )
            padded_r_peak = int(r_peak + pad)
            delineation_method = "dwt"
            try:
                _, info = nk.ecg_delineate(
                    padded_cleaned,
                    rpeaks=np.asarray([padded_r_peak], dtype=int),
                    sampling_rate=sampling_rate,
                    method=delineation_method,
                )
                if _diag is not None:
                    _diag.record_neurokit2_attempted()
            except Exception as exc:
                if _diag is not None:
                    _diag.record_neurokit2_failure("dwt_padded", f"DWT padded delineation exception: {str(exc)}")
                delineation_method = "peak"
                try:
                    _, info = nk.ecg_delineate(
                        padded_cleaned,
                        rpeaks=np.asarray([padded_r_peak], dtype=int),
                        sampling_rate=sampling_rate,
                        method=delineation_method,
                    )
                    if _diag is not None:
                        _diag.record_neurokit2_attempted()
                except Exception as exc2:
                    if _diag is not None:
                        _diag.record_neurokit2_failure("peak_padded", f"Peak padded delineation exception: {str(exc2)}")
                    delineation_method = "cleaned_window"
                    info = None

        if info is None:
            # If we reach here with no recorded failure, it means NeuroKit2
            # ran but returned no landmarks (empty arrays). Record this.
            if _diag is not None and _diag._neurokit2_delineation_never_attempted:
                # Check if we actually attempted delineation (context or padded)
                if context_available or delineation_method in ("dwt", "peak"):
                    _diag.record_neurokit2_failure(
                        "dwt_padded",
                        "NeuroKit2 delineation returned no landmarks (empty arrays)"
                    )
                else:
                    _diag.record_neurokit2_failure(
                        "dwt_padded",
                        "NeuroKit2 delineation not available for this beat"
                    )

            landmarks = _fallback_landmarks(cleaned, r_peak, sampling_rate)
            info = {}

            for key in ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks",
                         "ECG_T_Peaks", "ECG_P_Onsets", "ECG_P_Offsets",
                         "ECG_T_Onsets", "ECG_T_Offsets", "ECG_R_Onsets",
                         "ECG_R_Offsets"]:
                info[key] = []

            def _set_if(key: str, fn_key: str) -> None:
                val = landmarks.get(fn_key)
                if val is not None:
                    info["ECG_" + key.replace("ECG_", "")] = np.array([val], dtype=float)

            _set_if("ECG_P_Peaks", "p_peak")
            _set_if("ECG_P_Onsets", "p_onset")
            _set_if("ECG_P_Offsets", "p_offset")
            _set_if("ECG_T_Peaks", "t_peak")
            _set_if("ECG_T_Onsets", "t_onset")
            _set_if("ECG_T_Offsets", "t_offset")
            _set_if("ECG_R_Onsets", "qrs_onset")
            _set_if("ECG_R_Offsets", "qrs_offset")
        else:
            if not used_context:
                pad_remap = {
                    "ECG_P_Peaks": -pad,
                    "ECG_P_Onsets": -pad,
                    "ECG_P_Offsets": -pad,
                    "ECG_Q_Peaks": -pad,
                    "ECG_S_Peaks": -pad,
                    "ECG_T_Peaks": -pad,
                    "ECG_T_Onsets": -pad,
                    "ECG_T_Offsets": -pad,
                    "ECG_R_Onsets": -pad,
                    "ECG_R_Offsets": -pad,
                }
                for nk_key, offset in pad_remap.items():
                    raw = np.asarray(info.get(nk_key, []), dtype=float)
                    raw = raw[~np.isnan(raw)]
                    if len(raw) == 0:
                        info[nk_key] = []
                    else:
                        info[nk_key] = np.clip(raw + offset, 0, len(signal) - 1)

        peak_map = {
            "ECG_P_Peaks": "p_peak",
            "ECG_Q_Peaks": "q_peak",
            "ECG_S_Peaks": "s_peak",
            "ECG_T_Peaks": "t_peak",
            "ECG_P_Onsets": "p_onset",
            "ECG_P_Offsets": "p_offset",
            "ECG_T_Onsets": "t_onset",
            "ECG_T_Offsets": "t_offset",
            "ECG_R_Onsets": "qrs_onset",
            "ECG_R_Offsets": "qrs_offset",
        }

        extract = {}
        extract["r_peak_idx"] = r_peak

        # Determine which NeuroKit2 method was ultimately used
        _nk2_dwt_succeeded = used_context or (delineation_method == "dwt")
        _peak_method_succeeded = delineation_method == "peak"

        for nk_key, store_key in peak_map.items():
            arr = np.asarray(info.get(nk_key, []), dtype=float)
            arr = arr[~np.isnan(arr)]
            value = int(arr[0]) if len(arr) > 0 else None
            extract[store_key] = value

            if _diag is not None:
                feature_name = store_key
                diag_feature = NK_KEY_TO_FEATURE.get(nk_key)
                if diag_feature is None:
                    diag_feature = store_key

                if _nk2_dwt_succeeded and value is not None:
                    _diag.set_feature_source(diag_feature, "neurokit2_dwt", float(value))
                elif _peak_method_succeeded and value is not None:
                    _diag.set_feature_source(diag_feature, "peak_based", float(value))
                elif delineation_method == "cleaned_window" and value is not None:
                    _diag.set_feature_source(diag_feature, "clean_window", float(value))
                elif value is not None:
                    _diag.set_feature_source(diag_feature, "neurokit2_dwt", float(value))
                else:
                    # Feature is missing
                    _diag.set_feature_missing(
                        diag_feature,
                        "No landmark returned by any extraction method"
                    )

        # ── PR interval ───────────────────────────────────────
        if extract.get("p_peak") is not None and r_peak is not None:
            pr_samples = r_peak - int(extract["p_peak"])
            pr_ms = float(pr_samples / sampling_rate * 1000.0)
            if pr_ms < 80.0:
                p_peak_r = _peak_window(cleaned, r_peak - int(0.32 * sampling_rate), r_peak - int(0.08 * sampling_rate))
                if p_peak_r is not None:
                    pr_samples = r_peak - p_peak_r
                    pr_ms = float(pr_samples / sampling_rate * 1000.0)
                    extract["p_peak"] = p_peak_r
            extract["pr_interval_ms"] = pr_ms if extract.get("p_peak") is not None else None
        else:
            extract["pr_interval_ms"] = None

        # ── PR segment ────────────────────────────────────────
        if extract.get("p_offset") is not None and extract.get("qrs_onset") is not None:
            pr_seg_samples = int(extract["qrs_onset"]) - int(extract["p_offset"])
            extract["pr_segment_ms"] = float(pr_seg_samples / sampling_rate * 1000.0)
        else:
            extract["pr_segment_ms"] = None

        # ── QRS duration ──────────────────────────────────────
        if extract.get("qrs_onset") is not None and extract.get("qrs_offset") is not None:
            qrs_samples = int(extract["qrs_offset"]) - int(extract["qrs_onset"])
            extract["qrs_width_ms"] = float(qrs_samples / sampling_rate * 1000.0)
        else:
            extract["qrs_width_ms"] = None

        # ── QT interval ───────────────────────────────────────
        if extract.get("qrs_onset") is not None and extract.get("t_offset") is not None:
            qt_samples = int(extract["t_offset"]) - int(extract["qrs_onset"])
            extract["qt_interval_ms"] = float(qt_samples / sampling_rate * 1000.0)
        else:
            extract["qt_interval_ms"] = None

        # ── QRS voltage (peak-to-peak) ────────────────────────
        qrs_region = cleaned[max(0, r_peak - int(0.04 * sampling_rate)): min(len(cleaned), r_peak + int(0.04 * sampling_rate))]
        if len(qrs_region) > 0:
            extract["qrs_voltage"] = float(np.max(qrs_region) - np.min(qrs_region))
            r_idx_rel = np.argmax(qrs_region)
            s_idx_rel = np.argmin(qrs_region)
            extract["r_amplitude"] = float(qrs_region[r_idx_rel])
            extract["s_amplitude"] = float(qrs_region[s_idx_rel])
        else:
            extract["qrs_voltage"] = extract["r_amplitude"] = extract["s_amplitude"] = None

        # ── Q wave ────────────────────────────────────────────
        if extract.get("qrs_onset") is not None and extract.get("q_peak") is not None:
            q_val = float(cleaned[int(extract["q_peak"])])
            extract["q_amplitude"] = q_val
            extract["q_wave_deep"] = q_val <= -0.30
        else:
            extract["q_amplitude"] = extract["q_wave_deep"] = None

        # ── ST deviation ──────────────────────────────────────
        st_offset = int(r_peak + 0.08 * sampling_rate)
        st_end = min(len(cleaned), int(r_peak + 0.20 * sampling_rate))
        if st_offset < st_end:
            st_segment = cleaned[st_offset:st_end]
            if len(st_segment) > 0:
                extract["st_deviation"] = float(np.mean(st_segment))
                t_peak_val = extract.get("t_peak")
                st_j = int(r_peak + 0.04 * sampling_rate)
                j80 = int(r_peak + 0.08 * sampling_rate)
                if j80 < len(cleaned) and t_peak_val is not None and j80 < int(t_peak_val) and st_j < j80:
                    slope_left = float(cleaned[st_j])
                    slope_right = float(cleaned[j80])
                    if abs(slope_right - slope_left) > 0.01:
                        extract["st_slope"] = "upsloping" if slope_right > slope_left else "downsloping"
                    else:
                        extract["st_slope"] = "flat"
        else:
            extract["st_deviation"] = None

        # ── T wave ────────────────────────────────────────────
        if extract.get("t_peak") is not None:
            t_idx = int(extract["t_peak"])
            t_val = float(cleaned[t_idx])
            extract["t_wave_amplitude"] = t_val
            extract["t_wave_inverted"] = t_val < 0.0
            baseline = float(np.median(cleaned[max(0, r_peak - int(0.15 * sampling_rate)): r_peak]))
            if t_val > baseline:
                extract["t_wave_polarity"] = "positive"
            elif t_val < baseline:
                extract["t_wave_polarity"] = "negative"
            else:
                extract["t_wave_polarity"] = "flat"

            if extract.get("t_onset") is not None and extract.get("t_offset") is not None:
                tw_samples = int(extract["t_offset"]) - int(extract["t_onset"])
                extract["t_wave_width_ms"] = float(tw_samples / sampling_rate * 1000.0)

            # Tpeak-Tend interval
            t_peak_idx = int(extract["t_peak"])
            if extract.get("t_offset") is not None:
                tend = int(extract["t_offset"])
                if tend > t_peak_idx:
                    tpe_samples = tend - t_peak_idx
                    extract["tpeak_tend_interval_ms"] = float(tpe_samples / sampling_rate * 1000.0)
        else:
            extract["t_wave_inverted"] = None
            extract["t_wave_amplitude"] = None
            extract["t_wave_polarity"] = None
            extract["t_wave_width_ms"] = None
            extract["tpeak_tend_interval_ms"] = None

        # ── P wave ────────────────────────────────────────────
        if extract.get("p_peak") is not None:
            p_idx = int(extract["p_peak"])
            p_val = float(cleaned[p_idx])
            extract["p_wave_detected"] = True
            extract["p_wave_amplitude"] = p_val
            baseline = float(np.median(cleaned[max(0, r_peak - int(0.15 * sampling_rate)): r_peak]))
            extract["p_wave_polarity"] = "positive" if p_val > baseline else ("negative" if p_val < baseline else "flat")

            if extract.get("p_onset") is not None and extract.get("p_offset") is not None:
                pw_samples = int(extract["p_offset"]) - int(extract["p_onset"])
                extract["p_wave_width_ms"] = float(pw_samples / sampling_rate * 1000.0)
                extract["p_wave_prominence"] = float(cleaned[int(extract["p_peak"])] - np.median(cleaned[int(extract["p_onset"]): int(extract["p_offset"])]))
        else:
            extract["p_wave_detected"] = False
            extract["p_wave_amplitude"] = None
            extract["p_wave_width_ms"] = None
            extract["p_wave_polarity"] = None
            extract["p_wave_prominence"] = None

        # ── Heart rate from RR ────────────────────────────────
        extract["heart_rate"] = (60000.0 / rr_ms) if rr_ms and rr_ms > 0 else None

        # ── Signal quality ────────────────────────────────────
        peak_amp = float(np.max(cleaned) - np.min(cleaned))
        extract["peak_to_peak"] = peak_amp
        extract["amplitude_mean"] = float(np.mean(cleaned))
        extract["amplitude_std"] = float(np.std(cleaned))
        extract["amplitude_min"] = float(np.min(cleaned))
        extract["amplitude_max"] = float(np.max(cleaned))

        baseline_noise = float(np.std(cleaned[:int(0.1 * sampling_rate)])) if len(cleaned) > int(0.1 * sampling_rate) else 0.0
        if peak_amp > 0:
            snr = peak_amp / max(baseline_noise, 1e-6)
            extract["signal_quality_score"] = float(min(1.0, max(0.0, 1.0 - 1.0 / max(snr, 1.0))))
        else:
            extract["signal_quality_score"] = 0.0

        # ── T wave min (nadir of the inverted T or minimum within T window) ──
        if extract.get("t_onset") is not None and extract.get("t_offset") is not None:
            t_start = int(extract["t_onset"])
            t_stop = int(extract["t_offset"])
            if t_stop > t_start:
                extract["t_wave_min"] = float(np.min(cleaned[t_start:t_stop]))
        else:
            extract["t_wave_min"] = None

        # ── Heart rate variability (placeholder - single beat) ──
        extract_tail = {
            **_u_wave(signal, extract.get("t_offset"), sampling_rate, float(np.median(cleaned[:int(0.1 * sampling_rate)]))),
            **_delta_wave(extract.get("qrs_onset"), extract.get("q_peak"), signal, sampling_rate),
            **_flutter_baseline(signal, extract.get("p_offset"), extract.get("qrs_onset"), sampling_rate),
            "qrs_axis_deg": None,
            "qtc_dispersion_ms": None,
            "electrical_alternans_detected": None,
            "epsilon_wave_detected": None,
            "spodick_sign_detected": None,
            "rhythm_classification": None,
            "hrv_mean_nn": None,
            "hrv_sdnn": None,
            "hrv_rmssd": None,
        }

        qt = extract["qt_interval_ms"]
        rr_sec = rr_ms / 1000.0 if rr_ms else None
        if qt is not None and rr_sec and rr_sec > 0:
            qt_sec = qt / 1000.0
            extract["qtc_bazett"] = float((qt_sec / np.sqrt(rr_sec)) * 1000.0)
            extract["qtc_fridericia"] = float((qt_sec / np.cbrt(rr_sec)) * 1000.0)

        extract["advanced_feature_availability"] = {
            "hrv_full": _single_lead_unavailable("requires a longer R-R series, not one beat"),
            "rhythm_classification": _single_lead_unavailable("handled in temporal_analysis over a beat window"),
            "qrs_axis": _single_lead_unavailable("requires multi-lead frontal plane ECG"),
            "qtc_dispersion": _single_lead_unavailable("requires QT values across multiple leads"),
            "electrical_alternans": _single_lead_unavailable("requires beat-to-beat amplitude series"),
            "epsilon_wave": _single_lead_unavailable("requires high-resolution/right precordial lead context"),
            "spodick_sign": _single_lead_unavailable("requires multi-lead ST/TP-segment slope assessment"),
        }
        extract.update(extract_tail)
        # Honest source labeling: real NeuroKit2 delineation (dwt or peak
        # method) is reported as "neurokit2_dwt"; the heuristic fallback is
        # reported as "cleaned_window" (never mislabeled as neurokit2_*).
        extract["source"] = "neurokit2_dwt" if delineation_method in ("dwt", "peak") else "cleaned_window"
        extract["success"] = True

        # ── Record diagnostics for derived features ──────────────
        if _diag is not None:
            _source_label = "neurokit2_dwt" if delineation_method in ("dwt", "peak") else "clean_window"
            # PR interval
            pr_val = extract.get("pr_interval_ms")
            if pr_val is not None:
                _diag.set_feature_source("pr_interval_ms", _source_label, float(pr_val))
            else:
                _diag.set_feature_missing("pr_interval_ms", "P-peak or R-peak not available for PR calculation")

            # QRS width
            qrs_val = extract.get("qrs_width_ms")
            if qrs_val is not None:
                _diag.set_feature_source("qrs_width_ms", _source_label, float(qrs_val))
            else:
                _diag.set_feature_missing("qrs_width_ms", "QRS onset/offset not available")

            # QT interval
            qt_val = extract.get("qt_interval_ms")
            if qt_val is not None:
                _diag.set_feature_source("qt_interval_ms", _source_label, float(qt_val))
            else:
                _diag.set_feature_missing("qt_interval_ms", "QRS onset or T offset not available")

            # Heart rate
            hr_val = extract.get("heart_rate")
            if hr_val is not None:
                _diag.set_feature_source("heart_rate", _source_label, float(hr_val))
            else:
                _diag.set_feature_missing("heart_rate", "RR interval not available")

            # P wave amplitude
            p_amp = extract.get("p_wave_amplitude")
            if p_amp is not None:
                _diag.set_feature_source("p_wave_amplitude", _source_label, float(p_amp))
            else:
                _diag.set_feature_missing("p_wave_amplitude", "P-peak not detected")

            # T wave amplitude
            t_amp = extract.get("t_wave_amplitude")
            if t_amp is not None:
                _diag.set_feature_source("t_wave_amplitude", _source_label, float(t_amp))
            else:
                _diag.set_feature_missing("t_wave_amplitude", "T-peak not detected")

            # R amplitude
            r_amp = extract.get("r_amplitude")
            if r_amp is not None:
                _diag.set_feature_source("r_amplitude", _source_label, float(r_amp))
            else:
                _diag.set_feature_missing("r_amplitude", "QRS region empty")

            # S amplitude
            s_amp = extract.get("s_amplitude")
            if s_amp is not None:
                _diag.set_feature_source("s_amplitude", _source_label, float(s_amp))
            else:
                _diag.set_feature_missing("s_amplitude", "QRS region empty")

            # Q amplitude
            q_amp = extract.get("q_amplitude")
            if q_amp is not None:
                _diag.set_feature_source("q_amplitude", _source_label, float(q_amp))
            else:
                _diag.set_feature_missing("q_amplitude", "QRS onset or Q-peak not available")

            # ST deviation
            st_val = extract.get("st_deviation")
            if st_val is not None:
                _diag.set_feature_source("st_deviation", _source_label, float(st_val))
            else:
                _diag.set_feature_missing("st_deviation", "ST segment too short or R-peak not available")

            # Attach diagnostics to result
            extract["_diagnostics"] = _diag.to_dict()

        return extract
    except Exception as exc:
        result = {"source": "neurokit2_dwt", "success": False, "error": str(exc)}
        if _diag is not None:
            _diag.record_neurokit2_failure("delineation_exception", f"Top-level extraction exception: {str(exc)}")
            result["_diagnostics"] = _diag.to_dict()
        return result


def _single_lead_unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _u_wave(signal: np.ndarray, t_offset: Optional[int], fs: int, baseline: float) -> Dict[str, Any]:
    if t_offset is None:
        return {"u_wave_detected": False, "u_wave_peak": None, "u_wave_amplitude": None}
    start = int(t_offset) + int(0.02 * fs)
    end = min(len(signal), int(t_offset) + int(0.22 * fs))
    if end <= start:
        return {"u_wave_detected": False, "u_wave_peak": None, "u_wave_amplitude": None}
    segment = signal[start:end]
    if len(segment) == 0:
        return {"u_wave_detected": False, "u_wave_peak": None, "u_wave_amplitude": None}
    if baseline is None:
        baseline = 0.0
    peak_idx = int(start + np.argmax(segment))
    amp = float(signal[peak_idx] - baseline)
    if amp <= 0.0:
        return {"u_wave_detected": False, "u_wave_peak": None, "u_wave_amplitude": None}
    return {"u_wave_detected": True, "u_wave_peak": peak_idx, "u_wave_amplitude": amp}


def _delta_wave(qrs_onset: Optional[int], q_peak: Optional[int], signal: np.ndarray, fs: int) -> Dict[str, Any]:
    if qrs_onset is None or q_peak is None:
        return {"delta_wave_detected": False, "delta_wave_slope": None}
    onset = int(qrs_onset)
    q_pk = int(q_peak)
    if q_pk <= onset:
        return {"delta_wave_detected": False, "delta_wave_slope": None}
    dur_ms = (q_pk - onset) / fs * 1000.0
    if dur_ms >= 40:
        slope = (signal[q_pk] - signal[onset]) / max(dur_ms, 1.0)
        return {"delta_wave_detected": True, "delta_wave_slope": float(slope)}
    return {"delta_wave_detected": False, "delta_wave_slope": None}


def _flutter_baseline(signal: np.ndarray, p_offset: Optional[int], qrs_onset: Optional[int], fs: int) -> Dict[str, Any]:
    """
    Detect flutter baseline oscillation power in the TP/TQ segment.
    """
    import traceback
    _PERIODOGRAM_LEN = 256   # zero-pad target -> ~1.4 Hz resolution at 360 Hz
    if p_offset is None or qrs_onset is None:
        return {
            "flutter_baseline_power": None,
            "flutter_baseline_dominant_hz": None,
            "flutter_baseline_detected": False,
            "flutter_organization_index": None,
            "flutter_baseline_periodogram": None,
        }
    tp_start = int(p_offset)
    tp_end = min(len(signal), int(qrs_onset))
    tp_len = tp_end - tp_start
    threshold = int(0.08 * fs)
    if tp_len < threshold:
        return {
            "flutter_baseline_power": None,
            "flutter_baseline_dominant_hz": None,
            "flutter_baseline_detected": False,
            "flutter_organization_index": None,
            "flutter_baseline_periodogram": None,
        }
    tp_segment = signal[tp_start:tp_end]
    
    if len(tp_segment) < 10:
        return {
            "flutter_baseline_power": None,
            "flutter_baseline_dominant_hz": None,
            "flutter_baseline_detected": False,
            "flutter_organization_index": None,
            "flutter_baseline_periodogram": None,
        }

    # For short TP segments, zero-pad to at least 128 samples to get
    # fine enough frequency resolution (< 3 Hz/bin) to resolve the
    # 4-9 Hz flutter band. Without padding, a ~30-sample segment at
    # 360 Hz gives 12 Hz/bin — too coarse to see 4-9 Hz.
    tp_for_spectrum = tp_segment - np.mean(tp_segment)
    min_nperseg = max(len(tp_for_spectrum), 128)
    # Zero-pad if needed, then use nperseg = min(padded_len, 128) for welch
    if len(tp_for_spectrum) < 128:
        padded = np.zeros(128)
        padded[:len(tp_for_spectrum)] = tp_for_spectrum
        tp_for_spectrum = padded
        nperseg = 128
    else:
        nperseg = min(len(tp_for_spectrum), 128)
    freqs, power = welch(tp_for_spectrum, fs=fs, nperseg=nperseg)
    band = (freqs >= 4.0) & (freqs <= 9.0)
    if not np.any(band):
        return {
            "flutter_baseline_power": None,
            "flutter_baseline_dominant_hz": None,
            "flutter_baseline_detected": False,
            "flutter_organization_index": None,
            "flutter_baseline_periodogram": None,
        }

    band_power = float(np.trapz(power[band], freqs[band]))
    dom_idx = int(np.argmax(power[band]))
    dom = float(freqs[band][dom_idx])
    total = float(np.trapz(power, freqs)) if len(freqs) else 0.0
    ratio = band_power / total if total > 0 else 0.0

    # Peak concentration — how much of the in-band power sits at the
    # dominant frequency vs. spread across the band.
    band_powers = power[band]
    mean_band_power = float(np.mean(band_powers)) if len(band_powers) else 0.0
    peak_power = float(band_powers[dom_idx])
    organization_index = (
        min(1.0, peak_power / (mean_band_power * len(band_powers)))
        if mean_band_power > 0
        else 0.0
    )
    # Fixed-length windowed periodogram for WINDOW-level averaging in
    # temporal_analysis.py — a single beat's segment is too short to
    # resolve the 4-9Hz band; averaging many beats' periodograms fixes this.
    try:
        centered = tp_segment - np.mean(tp_segment)
        windowed = centered * np.hanning(len(centered))
        padded = np.zeros(_PERIODOGRAM_LEN)
        n = min(len(windowed), _PERIODOGRAM_LEN)
        padded[:n] = windowed[:n]
        periodogram = (np.abs(np.fft.rfft(padded)) ** 2).tolist()
    except Exception:
        periodogram = None

    return {
        "flutter_baseline_power": band_power,
        "flutter_baseline_dominant_hz": dom,
        "flutter_baseline_detected": bool(ratio > 0.35),
        "flutter_organization_index": organization_index,
        "flutter_baseline_periodogram": periodogram,
    }

"""
 Run NK delineation with ALL R-peaks in the inter-beat window to find
 every P-wave (including orphan/non-conducted ones), not just the one
 nearest the current beat's R-peak.
 The inter-beat window (prev_R to next_R) typically contains ~1 beat of
 signal. The existing code in extract_neurokit_morphology() only asks NK
 for the P-wave belonging to the ONE R-peak near the centre. This function
 re-asks NK for ALL P-waves in the same window by giving it every R-peak
 detected by ecg_peaks().
 Returns a list of dicts, each with:
   - timestamp: absolute wall-clock time of P-peak
   - relative_sample: sample index within original_samples
   - amplitude: signal amplitude at P-peak
 Stored in raw_feature_json['all_p_waves'] for downstream use in
 AV-block detection (detect_av_block in temporal_analysis.py).
 """
def extract_all_p_waves(
    original_samples,
    sampling_rate: int = 360,
    beat_timestamp: float = 0.0,
    clean_method: str = "neurokit",
) -> Optional[list]:

    if nk is None:
        return None

    signal = np.asarray(original_samples, dtype=np.float64)
    if signal.size < int(0.35 * sampling_rate):
        return None

    try:
        # ========== أضف الـ Padding هنا - نفس منطق extract_neurokit_morphology ==========
        min_pad = max(int(1.0 * sampling_rate), sampling_rate * 2 - len(signal))
        needed_total = sampling_rate * 4
        if len(signal) + 2 * min_pad < needed_total:
            min_pad = max(min_pad, (needed_total - len(signal) + 1) // 2)
        pad = min_pad
        padded_signal = np.pad(signal, (pad, pad), mode="edge")
        padded_cleaned = np.asarray(
            nk.ecg_clean(padded_signal, sampling_rate=sampling_rate, method=clean_method),
            dtype=np.float64,
        )

        # ── ابحث عن الـ Peaks في الإشارة الممددة ──
        _, r_info = nk.ecg_peaks(padded_cleaned, sampling_rate=sampling_rate, method="neurokit")
        local_r_peaks = np.asarray(r_info.get("ECG_R_Peaks", []), dtype=int)
        local_r_peaks = local_r_peaks[(local_r_peaks >= 0) & (local_r_peaks < len(padded_cleaned))]

        # ── ارجع الفهارس للإشارة الأصلية (اطرح الـ Pad) ──
        local_r_peaks = local_r_peaks - pad
        local_r_peaks = local_r_peaks[(local_r_peaks >= 0) & (local_r_peaks < len(signal))]

        # لو لسه مفيش Peaks كافية (أقل من 2 عشان الـ Rate)، ارجع None من غير تحذير
        if len(local_r_peaks) < 2:
            return None

        # ── دلوقتي اشتغل على الـ Delineation باستخدام الـ R-peaks الأصلية (المصححة) ──
        # (ملحوظة: الأفضل تستخدم `padded_cleaned` هنا عشان السياق، بس ترجع الإحداثيات بعد الطرح)
        _, p_info = nk.ecg_delineate(
            padded_cleaned,  # استخدم الممددة عشان الدقة
            rpeaks=local_r_peaks + pad,  # رجّع الـ Indices للممددة عشان NeuroKit يفهم
            sampling_rate=sampling_rate,
            method="dwt",
        )
        
        p_indices = np.asarray(p_info.get("ECG_P_Peaks", []), dtype=float)
        p_indices = p_indices[~np.isnan(p_indices)]
        
        # ارجع الفهارس للإشارة الأصلية (اطرح الـ Pad)
        p_indices = p_indices - pad
        p_indices = p_indices[(p_indices >= 0) & (p_indices < len(signal))]

        if len(p_indices) == 0:
            return None

        return [
            {
                "timestamp": beat_timestamp + float(idx) / sampling_rate,
                "relative_sample": int(round(idx)),
                "amplitude": float(signal[int(round(idx))])
                if 0 <= int(round(idx)) < len(signal)
                else None,
            }
            for idx in p_indices
        ]

    except Exception:
        return None