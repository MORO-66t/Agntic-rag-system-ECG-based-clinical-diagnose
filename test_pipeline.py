"""
ECG Pipeline — Integration Test

Generates a synthetic ECG beat, runs it through the full pipeline,
and prints every stage of the output.

Usage:
    python test_pipeline.py
"""

from ecg_pipeline import ECGPipeline
import sys
import time
import json
import logging
import numpy as np

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("test_pipeline")


# ─────────────────────────────────────────────
# Synthetic ECG generator (187 samples, ~1.5 s)
# ─────────────────────────────────────────────

def generate_synthetic_beat(
    label: str = "normal",
    sampling_rate: int = 125,
    n_samples: int = 187,
) -> np.ndarray:
    """
    Generate a crude 187‑sample ECG‑like waveform.

    Parameters
    ----------
    label : str
        "normal" — narrow QRS, regular morphology
        "pvc"    — wide complex, inverted T‑wave
    """
    t = np.linspace(0, n_samples / sampling_rate, n_samples)

    if label == "normal":
        # P‑wave + QRS + T‑wave
        p_wave   = 0.15 * np.exp(-((t - 0.20) ** 2) / (2 * 0.01 ** 2))
        qrs      = 1.00 * np.exp(-((t - 0.40) ** 2) / (2 * 0.005 ** 2))
        q_dip    = -0.10 * np.exp(-((t - 0.37) ** 2) / (2 * 0.004 ** 2))
        s_dip    = -0.15 * np.exp(-((t - 0.43) ** 2) / (2 * 0.004 ** 2))
        t_wave   = 0.30 * np.exp(-((t - 0.65) ** 2) / (2 * 0.025 ** 2))
        signal   = p_wave + q_dip + qrs + s_dip + t_wave

    elif label == "pvc":
        # Wide QRS, no P‑wave, inverted T
        qrs      = 1.20 * np.exp(-((t - 0.40) ** 2) / (2 * 0.012 ** 2))
        s_dip    = -0.40 * np.exp(-((t - 0.48) ** 2) / (2 * 0.008 ** 2))
        t_wave   = -0.35 * np.exp(-((t - 0.70) ** 2) / (2 * 0.030 ** 2))
        signal   = qrs + s_dip + t_wave

    else:
        signal = np.random.randn(n_samples) * 0.1

    # Add small noise
    signal += np.random.randn(n_samples) * 0.01

    return signal.astype(np.float32)


# ─────────────────────────────────────────────
# Pretty‑printers
# ─────────────────────────────────────────────

SEPARATOR = "=" * 60

def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_prediction(pred: dict):
    print_section("PREDICTION")
    if pred is None:
        print("  (no prediction)")
        return
    print(f"  Label        : {pred.get('predicted_class')} (index {pred.get('predicted_label')})")
    print(f"  Confidence   : {pred.get('prediction_confidence', 0):.4f}")
    probs = pred.get("probabilities", [])
    if probs:
        labels = ["N", "S", "V", "F", "Q"]
        for lbl, p in zip(labels, probs):
            bar = "█" * int(p * 40)
            print(f"    {lbl}: {p:.4f} {bar}")


def print_features(feat: dict):
    print_section("EXTRACTED FEATURES")
    if feat is None:
        print("  (no features)")
        return
    keys_to_show = [
        "heart_rate", "rr_interval", "qrs_width",
        "qt_interval", "qtc",
        "st_deviation", "t_wave_inverted",
        "amplitude_mean", "peak_to_peak",
        "signal_quality_score", "is_abnormal",
    ]
    for k in keys_to_show:
        v = feat.get(k)
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")


def print_events(events: list):
    print_section("DETECTED EVENTS")
    if not events:
        print("  (no events detected)")
        return
    for i, ev in enumerate(events, 1):
        mgr = ev.get("event_manager", {})
        print(f"\n  ── Event {i} ──")
        print(f"  Type         : {ev.get('event_type')}")
        print(f"  Severity     : {ev.get('severity')}")
        print(f"  Trigger Agent: {mgr.get('trigger_agent', '?')}")
        print(f"  Priority     : {mgr.get('priority', '?')}")
        meta = ev.get("metadata_json", {})
        if meta:
            for mk, mv in meta.items():
                if isinstance(mv, float):
                    print(f"    {mk}: {mv:.4f}")
                else:
                    print(f"    {mk}: {mv}")


def print_agent_responses(responses: list):
    print_section("AGENT RESPONSES")
    if not responses:
        print("  (no agent responses)")
        return
    for i, resp in enumerate(responses, 1):
        print(f"\n  ── Response {i}: {resp.get('event_type')} ──")
        print(f"  Severity     : {resp.get('severity')}")
        print(f"  Priority     : {resp.get('priority')}")
        print(f"  Escalation   : {resp.get('escalation_level')}")
        print(f"\n  LLM Output:")
        text = resp.get("response", "")
        for line in text.split("\n"):
            print(f"    {line}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    
    print(SEPARATOR)
    print("  ECG Pipeline Integration Test")
    print(SEPARATOR)

    # ── Import pipeline ──
    from ecg_pipeline import ECGPipeline

    # ── Initialise ──
    # Set enable_agent=False if you want a fast run without LLM calls
    enable_agent = "--no-agent" not in sys.argv

    pipeline = ECGPipeline(
        model_path="ecg_cnn_model.keras",
        enable_agent=enable_agent,
    )

    session_id = f"test_session_{int(time.time())}"

    # ── Run multiple beats to build temporal context ──
    import random

    SCENARIO = "AFIB"
    # AFIB
    # VT
    # TACHYCARDIA
    # BRADYCARDIA
    # PVC_BURDEN
    # NORMAL

    n_beats = 40
    print("\n")
    print("=" * 60)
    print("SCENARIO:", SCENARIO)
    print("=" * 60)
    print("\n")

    for i in range(n_beats):

        # ----------------------------------
        # NORMAL
        # ----------------------------------
        if SCENARIO == "NORMAL":

            beat_type = "normal"

            rr_interval = 0.80

        # ----------------------------------
        # AFIB
        # ----------------------------------
        elif SCENARIO == "AFIB":

            beat_type = "normal"

            rr_interval = random.choice([
                0.45,
                0.62,
                0.81,
                1.05,
                0.55,
                0.95,
                0.72,
                1.20
            ])

        # ----------------------------------
        # TACHYCARDIA
        # ----------------------------------
        elif SCENARIO == "TACHYCARDIA":

            beat_type = "normal"

            rr_interval = 0.45

        # ----------------------------------
        # BRADYCARDIA
        # ----------------------------------
        elif SCENARIO == "BRADYCARDIA":

            beat_type = "normal"

            rr_interval = 1.40

        # ----------------------------------
        # PVC BURDEN
        # ----------------------------------
        elif SCENARIO == "PVC_BURDEN":

            beat_type = (
                "pvc"
                if i % 3 == 0
                else "normal"
            )

            rr_interval = 0.80

        # ----------------------------------
        # VT RUN
        # ----------------------------------
        elif SCENARIO == "VT":

            if i < 30:

                beat_type = "normal"

            else:

                beat_type = "pvc"

            rr_interval = 0.40

        else:

            beat_type = "normal"

            rr_interval = 0.80

        signal = generate_synthetic_beat(
            label=beat_type
        )

        ts = time.time() + i

        result = pipeline.process_beat(
            signal=signal.tolist(),
            session_id=session_id,
            timestamp=ts,
            rr_interval=rr_interval
        )

        print_prediction(
            result["beat_prediction"]
        )

        print_features(
            result["features"]
        )

        print_events(
            result["events"]
        )
    # ── Cleanup ──
    pipeline.close()

    print(f"\n{SEPARATOR}")
    print("  Test Complete")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
