"""
kafka_mitbih_producer.py
=========================
Reads MIT-BIH records and streams raw signal chunks to Kafka.
Simulates what the website backend will do in production.

This is a TEST HARNESS — it lives in the ECG processing project
for testing purposes. The real website backend will have its own
simple producer that just streams raw signal.

Architecture
------------
    MIT-BIH record
         │
         ▼
    _load_record_for_beats()  (from convert_wfdb_to_csv_W.py)
         │  gets raw signal + R-peak positions + RR intervals
         ▼
    Streams 1-second chunks to ecg.raw.signal
         │  each chunk: 360 samples @ 360 Hz
         │  includes R-peak positions from annotations
         │  paced in real-time per RR interval (like realtime_stream.py)
         ▼
    Kafka topic: ecg.raw.signal
         │
         ▼
    ecg_kafka_service.py (consumes and processes)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from convert_wfdb_to_csv_W import _load_record_for_beats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE_SAMPLES = 360  # 1 second @ 360 Hz
CLINICAL_FS = 360


def iter_raw_signal_chunks(
    record_name: str,
    chunk_size: int = CHUNK_SIZE_SAMPLES,
) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields raw signal chunks from a MIT-BIH record.

    Each chunk is ~1 second of raw Lead II signal with R-peak positions
    relative to the chunk. This is exactly what the website backend
    will produce in production.

    Also yields per-chunk timing info for real-time pacing:
    - rr_intervals: list of RR intervals for beats whose R-peaks fall in this chunk
      Used by the producer to sleep the correct amount between chunks.

    Args:
        record_name: MIT-BIH record name (e.g., "202")
        chunk_size: Samples per chunk (default 360 = 1 second)

    Yields:
        Dict with keys matching the Kafka message schema:
        - session_id: str
        - patient_id: str
        - timestamp: float (epoch seconds)
        - sample_index: int (global sample position)
        - sample_rate: int (360)
        - lead: str ("II")
        - samples: List[float] (raw signal values)
        - r_peaks: List[int] (R-peak positions RELATIVE to this chunk)
        - rr_intervals: List[float] (RR intervals for beats in this chunk)
        - is_final: bool
    """
    record, lead2, r_samples, rr_intervals, beat_symbols, mat_t_peaks, mat_p_peaks = (
        _load_record_for_beats(record_name)
    )

    total_samples = len(lead2)
    session_id = f"mitbih-{record_name}"
    patient_id = f"patient-{record_name}"

    # Build a lookup: for each R-peak sample, get its RR interval
    # rr_intervals[i] is the interval between r_samples[i] and r_samples[i+1]
    r_peak_to_rr = {}
    for i in range(len(r_samples) - 1):
        r_peak_to_rr[int(r_samples[i])] = float(rr_intervals[i])
    # Last R-peak uses the previous RR interval as estimate
    if len(r_samples) > 0:
        r_peak_to_rr[int(r_samples[-1])] = float(rr_intervals[-2]) if len(rr_intervals) >= 2 else 800.0

    # Stream the raw signal in fixed-size chunks
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_samples = lead2[chunk_start:chunk_end].tolist()

        # Find R-peaks that fall within this chunk (relative positions) and their RR intervals
        r_peaks_in_chunk = []
        rr_in_chunk = []
        for r in r_samples:
            r_int = int(r)
            if chunk_start <= r_int < chunk_end:
                r_peaks_in_chunk.append(r_int - chunk_start)
                rr_in_chunk.append(r_peak_to_rr.get(r_int, 800.0))

        # Timestamp: approximate from sample position
        timestamp = float(chunk_start) / CLINICAL_FS

        is_final = chunk_end >= total_samples

        yield {
            "session_id": session_id,
            "patient_id": patient_id,
            "timestamp": timestamp,
            "sample_index": chunk_start,
            "sample_rate": CLINICAL_FS,
            "lead": "II",
            "samples": chunk_samples,
            "r_peaks": r_peaks_in_chunk,
            "rr_intervals": rr_in_chunk,  # for real-time pacing
            "is_final": is_final,
        }


def stream_mitbih_to_kafka(
    record_name: str,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "ecg.raw.signal",
    max_chunks: Optional[int] = None,
    realtime: bool = True,
) -> int:
    """
    Stream a MIT-BIH record to Kafka as raw signal chunks.

    Real-time pacing matches realtime_stream.py behavior:
    - After sending each chunk, sleep by the RR interval of the last
      detected R-peak. This paces the stream at the actual heart rate.
    - If no R-peak in a chunk, sleeps 0 (sends next chunk immediately)
      since multiple chunks are needed before the next beat.

    Args:
        record_name: MIT-BIH record name
        bootstrap_servers: Kafka broker address
        topic: Kafka topic to publish to
        max_chunks: Maximum number of chunks to send (None = all)
        realtime: If True, pace by RR intervals (like realtime_stream.py)

    Returns:
        Number of chunks sent
    """
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        compression_type="gzip",
        acks="all",
    )

    chunk_count = 0
    last_rr = 800.0  # default RR interval (75 bpm)
    last_chunk_time = time.time()

    try:
        for chunk in iter_raw_signal_chunks(record_name):
            if max_chunks is not None and chunk_count >= max_chunks:
                chunk["is_final"] = True
                producer.send(topic, key=chunk["session_id"], value=chunk)
                producer.flush()
                chunk_count += 1
                logger.info(
                    "Sent final chunk %d for %s (sample %d, %d R-peaks)",
                    chunk_count, record_name, chunk["sample_index"], len(chunk["r_peaks"]),
                )
                break

            producer.send(topic, key=chunk["session_id"], value=chunk)
            chunk_count += 1

            # Log progress periodically
            if chunk_count % 10 == 0 or chunk_count == 1:
                logger.info(
                    "Sent chunk %d for %s (sample %d, %d R-peaks in chunk, t=%.2fs)",
                    chunk_count, record_name, chunk["sample_index"],
                    len(chunk["r_peaks"]), chunk["timestamp"],
                )

            # ── Real-time pacing ─────────────────────────────────────
            # Match realtime_stream.py behavior: sleep by the last RR interval
            # This paces the stream at the actual heart rate.
            # realtime_stream.py does:
            #   if realtime and beat["rr_interval"] > 0:
            #       time.sleep(beat["rr_interval"] / 1000.0)
            #
            # In Kafka mode, we sleep by the RR interval of the last
            # detected R-peak. This ensures chunks arrive at the same
            # rate as beats in the original recording.
            if realtime:
                # Use the last RR interval from this chunk, or fall back
                rr_list = chunk.get("rr_intervals", [])
                if rr_list:
                    last_rr = rr_list[-1]
                
                now = time.time()
                elapsed = now - last_chunk_time
                # Each chunk represents ~1 second. After sending, we need
                # to wait the remaining time until the next R-peak would occur.
                # The pacing should match the heart rate: sleep by RR interval.
                sleep_for = max(0.0, (last_rr / 1000.0) - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_chunk_time = time.time()

        producer.flush()
        total_time = time.time() - last_chunk_time
        logger.info(
            "Finished streaming %s: %d chunks sent to %s",
            record_name, chunk_count, topic,
        )
    finally:
        producer.close()

    return chunk_count


def main():
    """CLI entry point for testing."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stream MIT-BIH records to Kafka")
    parser.add_argument("--record", default="202", help="MIT-BIH record name")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to send")
    parser.add_argument("--fast", action="store_true", help="Disable real-time pacing (send as fast as possible)")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka broker")
    parser.add_argument("--topic", default="ecg.raw.signal", help="Kafka topic")
    args = parser.parse_args()

    count = stream_mitbih_to_kafka(
        record_name=args.record,
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        max_chunks=args.max_chunks,
        realtime=not args.fast,
    )
    print(f"Done: {count} chunks sent")


if __name__ == "__main__":
    main()