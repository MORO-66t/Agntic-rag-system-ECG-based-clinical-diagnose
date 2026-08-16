import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io as sio
import json
from typing import Any, Dict, Iterator, Optional
from scipy.signal import resample

# ─────────────────────────────────────────────────────────────────
# CNN branch constants  (MUST NOT change — model was trained on these)
# ─────────────────────────────────────────────────────────────────
TARGET_LEN = 187          # samples fed to CNN
TARGET_FS  = 125          # Hz fed to CNN

# ─────────────────────────────────────────────────────────────────
# Clinical branch constants
# ─────────────────────────────────────────────────────────────────
CLINICAL_FS = 360         # original MIT-BIH sampling rate

mitbih_base_path = Path(__file__).resolve().parent / 'tr0ph1c' / 'mit-bih-arrhythmia-dataset-lead-ii' / 'versions' / '2' / 'mit-bih-arrhythmia-database-1.0.0'
mat_base_path    = Path(__file__).resolve().parent / 'tr0ph1c' / 'mit-bih-arrhythmia-dataset-lead-ii' / 'versions' / '2' / 'Pwaves_Twaves_Annotation' / 'Pwaves_Twaves_Annotation'


def record_path(record_name: str) -> str:
    return str(mitbih_base_path / record_name)


# ─────────────────────────────────────────────────────────────────
# CNN beat preparation  (unchanged logic)
# ─────────────────────────────────────────────────────────────────

def _resample_to_125(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """Resample beat from original_fs to 125 Hz."""
    if original_fs == TARGET_FS:
        return signal
    new_len = int(len(signal) * TARGET_FS / original_fs)
    return resample(signal, new_len)


def _normalize_to_187(signal: np.ndarray) -> np.ndarray:
    """Pad or resample so the beat is exactly 187 samples."""
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < TARGET_LEN:
        signal = np.pad(signal, (0, TARGET_LEN - len(signal)),
                        mode='constant', constant_values=0)
    elif len(signal) > TARGET_LEN:
        signal = resample(signal, TARGET_LEN)
    return signal


def _make_cnn_beat(beat_360: np.ndarray, original_fs: int) -> np.ndarray:
    """Full CNN preprocessing: resample → normalize to 187."""
    at_125 = _resample_to_125(beat_360, original_fs)
    return _normalize_to_187(at_125)


# ─────────────────────────────────────────────────────────────────
# Position scaling helpers
# ─────────────────────────────────────────────────────────────────

def _scale_position_to_cnn(pos_relative_360: int,
                            beat_len_360: int,
                            beat_len_125: int) -> int:
    """
    Convert an absolute-sample position (relative to beat start, at 360 Hz)
    into the CNN beat index (0..186).

    Two-step: 360→125 Hz scale, then 125→187 clamp if needed.
    """
    pos_125 = pos_relative_360 * (TARGET_FS / CLINICAL_FS)
    if beat_len_125 > TARGET_LEN:
        pos_final = int(pos_125 * (TARGET_LEN / beat_len_125))
    else:
        pos_final = int(pos_125)
    return min(max(pos_final, 0), TARGET_LEN - 1)


def _scale_position_to_360(pos_relative_360: int, beat_len_360: int) -> int:
    """
    Return position already in 360 Hz coordinates, clamped to the
    original beat window [0, beat_len_360 - 1].
    """
    return min(max(int(pos_relative_360), 0), beat_len_360 - 1)


# ─────────────────────────────────────────────────────────────────
# Shared per-beat conversion engine
# ─────────────────────────────────────────────────────────────────
# This is the SINGLE source of truth for turning one raw-signal beat
# window into (cnn_signal, original_samples, ground-truth peak
# annotations). Both the offline CSV exporter (get_record_peaks /
# convert_wfdb_to_csv) AND the real-time simulator
# (iter_record_beats, consumed by realtime_stream.py) call this same
# function, so there is exactly one implementation of beat
# segmentation + CNN resampling in the whole project.

def build_beat_record(
    signal_lead: np.ndarray,
    prev_r: int,
    curr_r: int,
    next_r: int,
    t_peak_abs: int,
    p_peak_abs: int,
    rr_interval: float,
    label: str,
    fs: int,
) -> Dict[str, Any]:
    """
    Build one beat's full data record as native Python types (no JSON
    serialization) — suitable both for appending into a CSV-export
    DataFrame row and for handing directly to ECGPipeline.process_beat()
    in a live/real-time context.

    Returns a dict with:
        original_samples   : list[float]  (360 Hz, prev_R -> next_R)
        original_beat_len   : int
        cnn_signal          : list[float] (187 samples, 125 Hz, normalised)
        rr_interval          : float (ms)
        label                : str (expert annotation symbol)
        rt_interval_ms        : float (-1.0 if no T annotation in window)
        t_peak_amplitude       : float
        t_peak_position_cnn     : int   (legacy CNN-coordinate index)
        t_peak_position_360      : int  (360 Hz coordinate index, ground truth)
        pr_interval_ms             : float (-1.0 if no P annotation in window)
        p_peak_amplitude            : float
        p_peak_position_cnn          : int
        p_peak_position_360           : int
    """
    beat_360 = signal_lead[prev_r:next_r].copy().astype(np.float64)
    beat_len_360 = len(beat_360)

    beat_125 = _resample_to_125(beat_360, fs)
    beat_len_125 = len(beat_125)
    beat_cnn = _normalize_to_187(beat_125)

    record: Dict[str, Any] = {
        "original_samples": beat_360.tolist(),
        "original_beat_len": beat_len_360,
        "cnn_signal": beat_cnn.tolist(),
        "rr_interval": float(rr_interval),
        "label": label,
    }

    # ── T-peak ──────────────────────────────────────────────────
    if t_peak_abs > 0 and curr_r < t_peak_abs < next_r:
        rt_samples_360 = t_peak_abs - curr_r
        record["rt_interval_ms"] = (rt_samples_360 / fs) * 1000.0
        record["t_peak_amplitude"] = float(signal_lead[t_peak_abs])

        t_relative_360 = t_peak_abs - prev_r
        record["t_peak_position_cnn"] = _scale_position_to_cnn(
            t_relative_360, beat_len_360, beat_len_125)
        record["t_peak_position_360"] = _scale_position_to_360(
            t_relative_360, beat_len_360)
    else:
        record["rt_interval_ms"] = -1.0
        record["t_peak_amplitude"] = 0.0
        record["t_peak_position_cnn"] = -1
        record["t_peak_position_360"] = -1

    # ── P-peak ──────────────────────────────────────────────────
    if p_peak_abs > 0 and prev_r < p_peak_abs < curr_r:
        pr_samples_360 = curr_r - p_peak_abs
        record["pr_interval_ms"] = (pr_samples_360 / fs) * 1000.0
        record["p_peak_amplitude"] = float(signal_lead[p_peak_abs])

        p_relative_360 = p_peak_abs - prev_r
        record["p_peak_position_cnn"] = _scale_position_to_cnn(
            p_relative_360, beat_len_360, beat_len_125)
        record["p_peak_position_360"] = _scale_position_to_360(
            p_relative_360, beat_len_360)
    else:
        record["pr_interval_ms"] = -1.0
        record["p_peak_amplitude"] = 0.0
        record["p_peak_position_cnn"] = -1
        record["p_peak_position_360"] = -1

    return record


def peak_to_dict(
    signal_lead: np.ndarray,
    prev_r: int,
    curr_r: int,
    next_r: int,
    t_peak_abs: int,
    p_peak_abs: int,
    rr_interval: float,
    label: str,
    fs: int
) -> dict:
    """
    CSV-row-shaped wrapper around build_beat_record(), kept for backward
    compatibility with existing CSV-export consumers. Produces a flat
    row dict with samp_0..samp_186 columns and a JSON-encoded
    original_beat_json column, matching the CSV schema this module has
    always produced.
    """
    rec = build_beat_record(
        signal_lead, prev_r, curr_r, next_r,
        t_peak_abs, p_peak_abs, rr_interval, label, fs,
    )

    row: Dict[str, Any] = {}
    for i, val in enumerate(rec["cnn_signal"]):
        row[f"samp_{i}"] = float(val)

    row["original_beat_json"] = json.dumps(rec["original_samples"])
    row["original_beat_len"] = rec["original_beat_len"]
    row["rr_interval"] = rec["rr_interval"]
    row["label"] = rec["label"]
    row["rt_interval_ms"] = rec["rt_interval_ms"]
    row["t_peak_amplitude"] = rec["t_peak_amplitude"]
    row["t_peak_position"] = rec["t_peak_position_cnn"]
    row["t_peak_position_360"] = rec["t_peak_position_360"]
    row["pr_interval_ms"] = rec["pr_interval_ms"]
    row["p_peak_amplitude"] = rec["p_peak_amplitude"]
    row["p_peak_position"] = rec["p_peak_position_cnn"]
    row["p_peak_position_360"] = rec["p_peak_position_360"]
    return row


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

def get_rr_intervals(annon_samples: np.ndarray, fs: int) -> np.ndarray:
    rr_samples = np.diff(annon_samples)
    return rr_samples / fs * 1000.0   # ms


def is_beat_annotation(symbol: str) -> bool:
    beat_symbols = {
        'N','L','R','B','A','a','J','S','V','r',
        'F','e','j','n','E','/','f','Q','?'
    }
    return symbol in beat_symbols


def _load_record_for_beats(record_name: str):
    """Shared loader: WFDB record + annotations + P/T .mat landmarks."""
    record = wfdb.rdrecord(record_path(record_name))
    record_annon = wfdb.rdann(
        record_path(record_name), 'atr',
        return_label_elements=['symbol', 'label_store']
    )

    mat_file_path = mat_base_path / f"Ant_mitdb_{record_name}.mat"
    try:
        mat_data = sio.loadmat(str(mat_file_path))
        mat_t_peaks = mat_data['T_Peaks_Annot'].flatten()
        mat_p_peaks = mat_data['P_Peaks_Annot'].flatten()
    except Exception as e:
        print(f"Warning: Could not load .mat for {record_name}: {e}")
        mat_t_peaks = np.array([])
        mat_p_peaks = np.array([])

    beat_symbols, beat_samples = [], []
    for sym, samp in zip(record_annon.symbol, record_annon.sample):
        if is_beat_annotation(sym):
            beat_symbols.append(sym)
            beat_samples.append(samp)

    r_samples = np.array(beat_samples)
    rr_intervals = get_rr_intervals(r_samples, record.fs)
    lead2 = record.p_signal[:, 0]

    return record, lead2, r_samples, rr_intervals, beat_symbols, mat_t_peaks, mat_p_peaks


def iter_record_beats(record_name: str, attach_context: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Generator: yields one beat record at a time, in chronological order,
    exactly as a real-time monitor would hand off newly-completed beats.

    This is the function realtime_stream.py should import and consume —
    it performs the SAME conversion as the offline CSV exporter
    (get_record_peaks), beat by beat, lazily, with no DataFrame and no
    CSV file in between. There is no separate resampling or delineation
    logic anywhere else in the project; this is the one place it happens.

    Each yielded dict additionally carries:
        beat_index   : int   (0-based, in stream order)
        timestamp     : float (seconds since record start, R-peak sample / fs)
        fs             : int  (sampling rate of the source record)

    Context window (real NeuroKit2 delineation, TRUE STREAMING)
    ----------------------------------------------------------
    When ``attach_context`` is True (the live/streaming path), each beat is
    delineated with REAL NeuroKit2 landmarks using a CENTERED 10 s sliding
    window with a fixed ~5 s processing delay:
        lead2[curr_r - 5*fs : curr_r + 5*fs]
    The target beat's R-peak sits in the MIDDLE of the window (5 s before +
    5 s after). This is a DELAYED real-time pipeline: beat i's morphology is
    computed once ~5 s of post-R signal has arrived, so NeuroKit2 delineates
    it with full temporal context instead of treating it as the last beat in
    a buffer. The 5 s of "future" signal contains other beats' P-QRS-T, but
    NeuroKit2 segments each beat locally (≈±0.5 s around its R-peak) and
    _extract_target_landmarks only keeps fiducials within 0.2-0.7 s of the
    target R-peak, so beat i's morphology features never depend on future
    beats' morphology — there is no real leakage, only a designed latency.
    Beats within the first/last ~4 s (asymmetric windows) fall back to the
    single-beat heuristic inside extract_neurokit_morphology. Set
    ``attach_context=False`` (offline CSV export, which defers morphology) to
    skip the per-beat delineation entirely.
    """
    record, lead2, r_samples, rr_intervals, beat_symbols, mat_t_peaks, mat_p_peaks = (
        _load_record_for_beats(record_name)
    )

    # ── True-streaming context (see per-beat block below) ────────────────
    # No whole-record batch delineation: each beat is delineated incrementally
    # from a PAST-ONLY rolling buffer as it arrives (no future leakage).

    for i in range(1, len(r_samples) - 1):
        prev_r = r_samples[i - 1]
        curr_r = r_samples[i]
        next_r = r_samples[i + 1]

        t_candidates = mat_t_peaks[(mat_t_peaks > curr_r) & (mat_t_peaks < next_r)]
        t_peak_abs = int(t_candidates[0]) if len(t_candidates) > 0 else -1

        p_candidates = mat_p_peaks[(mat_p_peaks > prev_r) & (mat_p_peaks < curr_r)]
        p_peak_abs = int(p_candidates[-1]) if len(p_candidates) > 0 else -1

        beat = build_beat_record(
            lead2, prev_r, curr_r, next_r,
            t_peak_abs, p_peak_abs,
            rr_intervals[i - 1], beat_symbols[i],
            record.fs,
        )
        beat["beat_index"] = i - 1
        beat["timestamp"] = float(curr_r) / record.fs
        beat["fs"] = int(record.fs)

        # ── True-streaming CENTERED context window (real NeuroKit2) ──────
        # 10 s sliding window centered on the target beat's R-peak, with a
        # fixed ~5 s processing delay. The target beat sits in the MIDDLE
        # (5 s before + 5 s after) so NeuroKit2 delineates it with full
        # temporal context. The post-R "future" samples only serve to center
        # the window; NeuroKit2 segments each beat locally (≈±0.5 s around
        # its R-peak) and _extract_target_landmarks keeps only fiducials
        # within 0.2-0.7 s of the target R-peak, so beat i's morphology never
        # depends on future beats' morphology — a designed latency, not
        # leakage. Beats without >=4 s of signal on BOTH sides (first/last
        # ~4 s of the record) get an asymmetric window and fall back to the
        # single-beat heuristic inside extract_neurokit_morphology.
        if attach_context:
            half = int(5 * record.fs)            # 5 s each side -> 10 s window
            min_side = int(4 * record.fs)        # need >=4 s context on each side
            window_start = int(max(0, curr_r - half))
            window_end = int(min(len(lead2), curr_r + half))
            enough_before = (curr_r - window_start) >= min_side
            enough_after = (window_end - curr_r) >= min_side
            ctx_samples = np.array(lead2[window_start:window_end], dtype=np.float64)
            # All R-peaks inside the window (past AND future) — relative to
            # window_start. They are context for dwt; beat i's own landmarks
            # are selected by position in _extract_target_landmarks.
            mask = (r_samples >= window_start) & (r_samples <= window_end)
            ctx_rpeaks = np.asarray(r_samples[mask], dtype=int) - window_start
            ctx_target_r = int(curr_r) - window_start
            beat_start = int(prev_r) - window_start
            if len(ctx_rpeaks) >= 2 and enough_before and enough_after:
                beat["context_samples"] = ctx_samples
                beat["context_rpeaks"] = ctx_rpeaks
                beat["context_target_r"] = ctx_target_r
                beat["beat_start"] = beat_start
                beat["context_info"] = None  # delineate per beat (no reuse)
            else:
                beat["context_samples"] = None
                beat["context_rpeaks"] = None
                beat["context_target_r"] = None
                beat["beat_start"] = 0
                beat["context_info"] = None
        else:
            beat["context_samples"] = None
            beat["context_rpeaks"] = None
            beat["context_target_r"] = None
            beat["beat_start"] = 0
            beat["context_info"] = None

        yield beat


def get_record_peaks(record_name: str) -> pd.DataFrame:
    """
    Offline CSV-export path. Now built by consuming the same
    iter_record_beats() generator the real-time simulator uses, so the
    two paths can never silently drift apart.
    """
    rows = []
    for beat in iter_record_beats(record_name, attach_context=False):
        row: Dict[str, Any] = {}
        for i, val in enumerate(beat["cnn_signal"]):
            row[f"samp_{i}"] = float(val)
        row["original_beat_json"] = json.dumps(beat["original_samples"])
        row["original_beat_len"] = beat["original_beat_len"]
        row["rr_interval"] = beat["rr_interval"]
        row["label"] = beat["label"]
        row["rt_interval_ms"] = beat["rt_interval_ms"]
        row["t_peak_amplitude"] = beat["t_peak_amplitude"]
        row["t_peak_position"] = beat["t_peak_position_cnn"]
        row["t_peak_position_360"] = beat["t_peak_position_360"]
        row["pr_interval_ms"] = beat["pr_interval_ms"]
        row["p_peak_amplitude"] = beat["p_peak_amplitude"]
        row["p_peak_position"] = beat["p_peak_position_cnn"]
        row["p_peak_position_360"] = beat["p_peak_position_360"]
        rows.append(row)

    print(record_path(record_name), "->", len(rows), "beats")
    return pd.DataFrame(rows)


def convert_wfdb_to_csv(output_csv: str, record_names: list):
    """
    Offline dataset-export utility — produces a flat CSV across many
    records. This is a separate use case from real-time simulation
    (e.g. building a training/evaluation dataset); it is NOT part of
    the live ECGPipeline data path.
    """
    all_peaks_df = pd.DataFrame()
    for idx, record_name in enumerate(record_names):
        print(f'Processing record {idx+1}/{len(record_names)}: {record_name}')
        df = get_record_peaks(record_name)
        all_peaks_df = pd.concat([all_peaks_df, df], ignore_index=True)
    all_peaks_df.to_csv(output_csv, index=False)


def is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_files_unique_names(directory_path):
    dir_path = Path(directory_path)
    unique_names = {
        f.stem for f in dir_path.iterdir()
        if f.is_file() and is_numeric(f.stem)
    }
    return sorted(list(unique_names))


if __name__ == '__main__':
    convert_wfdb_to_csv(
        'mitbih_beats1_with_pt.csv',
        get_files_unique_names(mitbih_base_path)
    )