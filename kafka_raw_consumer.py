"""
kafka_raw_consumer.py
======================
Replaces iter_record_beats() for a Kafka streaming context.
Consumes raw ECG signal chunks from Kafka, buffers them,
detects R-peaks, segments beats, builds context windows,
and yields beat dicts compatible with ECGPipeline.process_beat().

This is the Kafka equivalent of convert_wfdb_to_csv_W.iter_record_beats().
The existing functions (build_beat_record, _make_cnn_beat, etc.) are
imported and reused directly — no duplicate logic.

Architecture
------------
    Kafka (ecg.raw.signal)
         │  raw signal chunks (360 Hz, 1s chunks)
         ▼
    RawECGStreamProcessor
         │  buffers signal, detects R-peaks
         │  segments beats, builds context windows
         │  waits 5s for centered context
         ▼
    beat dict → ECGPipeline.process_beat()
         (identical format to iter_record_beats output)
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from config import KAFKA_CONFIG
from convert_wfdb_to_csv_W import (
    _make_cnn_beat,
    _resample_to_125,
    _normalize_to_187,
    build_beat_record,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (matching convert_wfdb_to_csv_W)
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_FS = 360        # Hz — raw signal sample rate
CONTEXT_WINDOW_SEC = 10   # total context window length
CONTEXT_HALF_SEC = 5      # seconds on each side
MIN_CONTEXT_SEC = 4       # minimum viable context on each side
CNN_TARGET_LEN = 187      # CNN input length
CNN_TARGET_FS = 125       # CNN input sample rate

# R-peak detection constants (simple threshold-based)
RPEAK_MIN_DISTANCE_MS = 200  # minimum ms between R-peaks (300 bpm max)
RPEAK_MIN_HEIGHT = 0.3       # minimum voltage for R-peak


def _detect_r_peaks(
    signal: np.ndarray,
    fs: int = CLINICAL_FS,
    existing_r_peaks: Optional[List[int]] = None,
) -> List[int]:
    """
    Simple R-peak detection using threshold + prominence.
    Used when the Kafka message does NOT include R-peak positions.

    This is a basic detector for demonstration. For production, replace
    with a more robust algorithm (Pan-Tompkins, NK2, or a trained model).

    Args:
        signal: Raw ECG signal buffer
        fs: Sampling rate in Hz
        existing_r_peaks: R-peak positions already detected (to avoid duplicates)

    Returns:
        List of R-peak sample positions (global indices)
    """
    # Simple energy-based R-peak detection
    # Bandpass approximation: subtract local mean
    window = int(0.2 * fs)  # 200ms window
    if len(signal) < window * 2:
        return []

    # Compute local energy
    squared = signal ** 2
    energy = np.convolve(squared, np.ones(window) / window, mode="same")

    # Threshold
    threshold = np.mean(energy) + 2.0 * np.std(energy)

    # Find peaks above threshold
    above = energy > threshold
    if not np.any(above):
        return []

    # Group contiguous regions → pick max in each
    diffs = np.diff(above.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1

    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(above)]])

    # Minimum R-peak spacing in samples
    min_distance = int(RPEAK_MIN_DISTANCE_MS * fs / 1000)

    existing_set = set(existing_r_peaks or [])
    new_peaks: List[int] = []
    for s, e in zip(starts, ends):
        region = signal[s:e]
        if len(region) == 0:
            continue
        peak_pos = s + int(np.argmax(np.abs(region)))
        # Check amplitude
        if abs(signal[peak_pos]) < RPEAK_MIN_HEIGHT:
            continue
        # Check minimum distance from existing peaks
        too_close = False
        for ep in existing_set:
            if abs(peak_pos - ep) < min_distance:
                too_close = True
                break
        for np_pos in new_peaks:
            if abs(peak_pos - np_pos) < min_distance:
                too_close = True
                break
        if not too_close:
            new_peaks.append(peak_pos)

    return sorted(new_peaks)


class RawECGStreamProcessor:
    """
    Consumes raw ECG signal chunks from Kafka, buffers them,
    detects R-peaks, segments beats, and yields beat dicts
    matching ECGPipeline.process_beat() input format.

    Usage:
        processor = RawECGStreamProcessor()
        for beat in processor.iter_beats():
            pipeline.process_beat(**beat)

    This class is the Kafka equivalent of iter_record_beats().
    """

    def __init__(
        self,
        fs: int = CLINICAL_FS,
        max_buffer_seconds: int = 30,
    ):
        self.fs = fs
        self.max_buffer_size = max_buffer_seconds * fs  # max samples to keep

        # Raw signal buffer (continuous, oldest samples are pruned)
        self.buffer: np.ndarray = np.array([], dtype=np.float64)

        # Global sample offset (total samples received)
        self.sample_offset: int = 0

        # Detected R-peak global positions (sorted)
        self.r_peaks: List[int] = []

        # R-peaks that have been processed into beats (removed from tracking)
        self.last_processed_r_index: int = -1

        # Processed beats awaiting context-window completion
        self.pending_beats: deque = deque()

        # Buffer pruning offset: samples before this have been pruned
        self.pruned_samples: int = 0

    def _global_to_buffer(self, global_pos: int) -> int:
        """Convert a global sample position to current buffer index."""
        return global_pos - self.pruned_samples

    def _buffer_to_global(self, buffer_pos: int) -> int:
        """Convert a buffer index to global sample position."""
        return buffer_pos + self.pruned_samples

    def _prune_buffer(self) -> None:
        """Remove old samples from the buffer to prevent unbounded growth."""
        if len(self.buffer) <= self.max_buffer_size:
            return
        # Keep only the most recent max_buffer_size samples
        prune_count = len(self.buffer) - self.max_buffer_size
        self.buffer = self.buffer[prune_count:]
        self.pruned_samples += prune_count

    def ingest_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Ingest a single Kafka message (raw signal chunk).

        Expected message format:
        {
            "session_id": "...",
            "patient_id": "...",
            "timestamp": 1742593200.0,
            "sample_index": 0,          # global sample index of first sample
            "sample_rate": 360,
            "lead": "II",
            "samples": [0.12, 0.13, ...],   # raw signal values
            "r_peaks": [180, 540],          # R-peak positions RELATIVE to chunk (optional)
            "is_final": false
        }
        """
        samples = np.array(chunk.get("samples", []), dtype=np.float64)
        if len(samples) == 0:
            return

        chunk_sample_index = chunk.get("sample_index", self.sample_offset)
        chunk_fs = chunk.get("sample_rate", self.fs)

        # Handle sample rate mismatch (resample if needed)
        if chunk_fs != self.fs:
            new_len = int(len(samples) * self.fs / chunk_fs)
            from scipy.signal import resample
            samples = np.asarray(resample(samples, new_len), dtype=np.float64)

        # Append to buffer
        # If there's a gap between current offset and chunk start, pad with zeros
        if chunk_sample_index > self.sample_offset:
            gap = chunk_sample_index - self.sample_offset
            self.buffer = np.concatenate([self.buffer, np.zeros(gap)])
            self.sample_offset = chunk_sample_index

        self.buffer = np.concatenate([self.buffer, samples])
        self.sample_offset += len(samples)

        # Process R-peaks from message, or detect them
        if chunk.get("r_peaks"):
            for rp_rel in chunk["r_peaks"]:
                rp_global = chunk_sample_index + rp_rel
                # Avoid duplicates
                if not self.r_peaks or rp_global > self.r_peaks[-1]:
                    self.r_peaks.append(rp_global)
        else:
            # Run R-peak detection on the new samples
            new_peaks = _detect_r_peaks(
                self.buffer[self._global_to_buffer(max(0, self.sample_offset - len(samples))):],
                self.fs,
                existing_r_peaks=self.r_peaks,
            )
            for p in new_peaks:
                gp = p + max(0, self.sample_offset - len(samples))
                if not self.r_peaks or gp > self.r_peaks[-1]:
                    self.r_peaks.append(gp)

        # Prune old buffer
        self._prune_buffer()

    def _build_beat_dict(
        self,
        prev_r_global: int,
        curr_r_global: int,
        next_r_global: int,
        timestamp: float,
        beat_index: int,
    ) -> Dict[str, Any]:
        """
        Build a beat dict identical to what iter_record_beats() yields.

        Reuses build_beat_record() from convert_wfdb_to_csv_W.py for the
        core CNN resampling, then adds context window fields.
        """
        # Extract beat signal from buffer
        prev_buf = self._global_to_buffer(prev_r_global)
        curr_buf = self._global_to_buffer(curr_r_global)
        next_buf = self._global_to_buffer(next_r_global)

        if next_buf > len(self.buffer) or prev_buf < 0:
            return None  # not enough data

        beat_signal = self.buffer[prev_buf:next_buf]

        # Use build_beat_record for CNN preparation
        cnn_signal = _make_cnn_beat(beat_signal, self.fs)

        beat_dict = {
            "original_samples": beat_signal.tolist(),
            "original_beat_len": len(beat_signal),
            "cnn_signal": cnn_signal.tolist(),
            "rr_interval": float((curr_r_global - prev_r_global) / self.fs * 1000.0),
            "label": "N",  # No expert annotation in stream; default to N
            "rt_interval_ms": -1.0,
            "t_peak_position_360": -1,
            "p_peak_position_360": -1,
            "pr_interval_ms": -1.0,
            "beat_index": beat_index,
            "timestamp": timestamp,
            "fs": self.fs,
        }

        # ── Build centered 10-second context window ────────────────
        half = int(CONTEXT_HALF_SEC * self.fs)
        min_side = int(MIN_CONTEXT_SEC * self.fs)

        window_start_g = curr_r_global - half
        window_end_g = curr_r_global + half

        window_start_buf = self._global_to_buffer(window_start_g)
        window_end_buf = self._global_to_buffer(window_end_g)

        enough_before = window_start_buf >= 0
        enough_after = window_end_buf <= len(self.buffer)

        if enough_before and enough_after:
            ctx = self.buffer[window_start_buf:window_end_buf].copy()

            # Find all R-peaks inside this window
            ctx_rpeaks = [
                rp - window_start_g
                for rp in self.r_peaks
                if window_start_g <= rp <= window_end_g
            ]
            ctx_target_r = curr_r_global - window_start_g
            beat_start = prev_r_global - window_start_g

            beat_dict["context_samples"] = ctx
            beat_dict["context_rpeaks"] = np.array(ctx_rpeaks, dtype=int)
            beat_dict["context_target_r"] = ctx_target_r
            beat_dict["beat_start"] = beat_start
            beat_dict["context_info"] = None
        else:
            # Asymmetric window — fallback to heuristic morphology
            beat_dict["context_samples"] = None
            beat_dict["context_rpeaks"] = None
            beat_dict["context_target_r"] = None
            beat_dict["beat_start"] = 0
            beat_dict["context_info"] = None

        return beat_dict

    def iter_beats(
        self,
        poll_function,
        session_id: str,
        max_beats: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generator that yields beat dicts from the Kafka stream.

        Args:
            poll_function: A callable that returns the next chunk dict
                           (typically KafkaConsumer.poll() or similar).
            session_id: Session identifier for filtering.
            max_beats: Stop after this many beats.

        Yields:
            Beat dicts compatible with ECGPipeline.process_beat().
        """
        beat_count = 0
        chunks_processed = 0
        consecutive_empty_polls = 0
        max_empty_polls = 100  # stop after 100 empty polls

        while True:
            # Poll for next chunk
            chunk = poll_function()
            chunks_processed += 1

            if chunk is None:
                consecutive_empty_polls += 1
                if consecutive_empty_polls >= max_empty_polls:
                    # Process any remaining pending beats
                    yield from self._flush_pending_beats(session_id, beat_count)
                    break
                time.sleep(0.1)
                continue

            consecutive_empty_polls = 0
            self.ingest_chunk(chunk)

            # Check if we have enough R-peaks to form a beat
            # We need: prev_r, curr_r, next_r, and 5s of signal after curr_r
            while len(self.r_peaks) >= 3:
                prev_r = self.r_peaks[0]
                curr_r = self.r_peaks[1]
                next_r = self.r_peaks[2]

                # Need 5 seconds of signal AFTER curr_r for centered context window
                needed_samples = curr_r + CONTEXT_HALF_SEC * self.fs
                if (self.sample_offset - self.pruned_samples) < self._global_to_buffer(needed_samples):
                    break  # wait for more data

                # Build the beat dict
                timestamp = float(curr_r) / self.fs
                # Add real-time epoch offset (same as realtime_stream.py does)
                # The timestamp should already be epoch if the website sends it

                beat_dict = self._build_beat_dict(
                    prev_r_global=prev_r,
                    curr_r_global=curr_r,
                    next_r_global=next_r,
                    timestamp=timestamp,
                    beat_index=beat_count,
                )

                # Remove the processed R-peak (prev_r is no longer needed)
                self.r_peaks.pop(0)

                if beat_dict is None:
                    continue

                beat_count += 1
                yield beat_dict

                if max_beats and beat_count >= max_beats:
                    return

            # If this is the final chunk, flush remaining beats
            if chunk.get("is_final", False):
                yield from self._flush_pending_beats(session_id, beat_count)
                break

    def _flush_pending_beats(
        self,
        session_id: str,
        start_beat_index: int,
    ) -> Iterator[Dict[str, Any]]:
        """
        Process any remaining R-peaks that couldn't get a full context window.
        Uses whatever signal is available (asymmetric context).
        """
        beat_index = start_beat_index
        while len(self.r_peaks) >= 2:
            prev_r = self.r_peaks[0]
            curr_r = self.r_peaks[1]

            # Determine next_r (use curr_r + average RR if no next peak)
            if len(self.r_peaks) >= 3:
                next_r = self.r_peaks[2]
            else:
                # Estimate next_r from average RR
                avg_rr = int((curr_r - prev_r) * 1.0) if len(self.r_peaks) >= 2 else int(0.8 * self.fs)
                next_r = curr_r + avg_rr

            timestamp = float(curr_r) / self.fs

            beat_dict = self._build_beat_dict(
                prev_r_global=prev_r,
                curr_r_global=curr_r,
                next_r_global=next_r,
                timestamp=timestamp,
                beat_index=beat_index,
            )

            self.r_peaks.pop(0)

            if beat_dict is None:
                continue

            beat_index += 1
            yield beat_dict