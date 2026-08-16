"""
Morphology Diagnostic System — Full explainability for the extraction pipeline.

Provides dataclasses, a collector, and a formatter that capture exactly why
each morphological feature was computed the way it was. When debug mode is
disabled, zero overhead is added to the production path.

Usage
-----
    collector = MorphologyDiagnosticCollector(beat_index=1254)
    # ... during extraction, call:
    collector.record_neurokit2_failure("dwt_context", "No R_Onsets found")
    collector.set_feature_source("p_peak", "peak_based", 102.0)
    # At the end:
    print(MorphologyDiagnosticFormatter.format_beat(collector))
    log_text = MorphologyDiagnosticFormatter.format_beat_log(collector)
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Feature definitions — all morphological features we track
# ────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "p_onset",
    "p_peak",
    "p_offset",
    "q_peak",
    "r_onset",     # qrs_onset
    "r_offset",    # qrs_offset
    "s_peak",
    "t_onset",
    "t_peak",
    "t_offset",
    "pr_interval_ms",
    "qrs_width_ms",
    "qt_interval_ms",
    "p_wave_amplitude",
    "t_wave_amplitude",
    "r_amplitude",
    "s_amplitude",
    "q_amplitude",
    "st_deviation",
    "heart_rate",
]

# Mapping from internal NK keys to diagnostic feature names
NK_KEY_TO_FEATURE: Dict[str, str] = {
    "ECG_P_Onsets": "p_onset",
    "ECG_P_Peaks": "p_peak",
    "ECG_P_Offsets": "p_offset",
    "ECG_Q_Peaks": "q_peak",
    "ECG_R_Onsets": "r_onset",
    "ECG_R_Offsets": "r_offset",
    "ECG_S_Peaks": "s_peak",
    "ECG_T_Onsets": "t_onset",
    "ECG_T_Peaks": "t_peak",
    "ECG_T_Offsets": "t_offset",
}


# ────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────


@dataclass
class SourceAttempt:
    """Records what happened with one extraction source for a feature."""
    attempted: bool = False
    succeeded: bool = False
    reason: Optional[str] = None  # The exact failure reason (real exception or descriptive)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "reason": self.reason,
        }


@dataclass
class FeatureDiagnostic:
    """Complete diagnostic for a single morphological feature."""
    feature_name: str
    final_value: Optional[float] = None
    selected_source: str = "none"  # "neurokit2_dwt" | "peak_based" | "clean_window" | "none"
    neurokit2: SourceAttempt = field(default_factory=SourceAttempt)
    peak_based: SourceAttempt = field(default_factory=SourceAttempt)
    clean_window: SourceAttempt = field(default_factory=SourceAttempt)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "final_value": self.final_value,
            "selected_source": self.selected_source,
            "neurokit2": self.neurokit2.to_dict(),
            "peak_based": self.peak_based.to_dict(),
            "clean_window": self.clean_window.to_dict(),
            "warnings": self.warnings,
        }


# ────────────────────────────────────────────────────────────────
# Collector — instruments the extraction pipeline
# ────────────────────────────────────────────────────────────────


class MorphologyDiagnosticCollector:
    """
    Collects real-time diagnostics during morphology extraction.

    Usage:
        collector = MorphologyDiagnosticCollector(beat_index=1254)
        # ... extractor calls collector.record_*() methods ...
        diagnostics_dict = collector.to_dict()
    """

    def __init__(self, beat_index: int = 0):
        self.beat_index = beat_index
        self.features: Dict[str, FeatureDiagnostic] = {}

        # Method-level error tracking
        # These capture the global reasons why NeuroKit2 or peak-based failed
        self._neurokit2_dwt_context_error: Optional[str] = None
        self._neurokit2_dwt_padded_error: Optional[str] = None
        self._neurokit2_peak_padded_error: Optional[str] = None
        self._neurokit2_delineation_never_attempted: bool = True

        # Track whether context-window path was available
        self._context_window_available: bool = False

        # Initialize diagnostics for all features
        for name in FEATURE_NAMES:
            self.features[name] = FeatureDiagnostic(feature_name=name)

    # ── Method-level recorders ───────────────────────────────

    def record_neurokit2_failure(self, context: str, reason: str) -> None:
        """
        Record a NeuroKit2 delineation failure with the REAL exception reason.

        Parameters
        ----------
        context : str
            Where the failure occurred: 'dwt_context', 'dwt_padded', 'peak_padded',
            'import', 'window_too_short', or 'context_no_r_onsets'.
        reason : str
            The actual exception message or descriptive reason.
        """
        self._neurokit2_delineation_never_attempted = False
        if context == "dwt_context":
            if self._neurokit2_dwt_context_error is None:
                self._neurokit2_dwt_context_error = reason
        elif context == "dwt_padded":
            if self._neurokit2_dwt_padded_error is None:
                self._neurokit2_dwt_padded_error = reason
        elif context == "peak_padded":
            if self._neurokit2_peak_padded_error is None:
                self._neurokit2_peak_padded_error = reason
        elif context == "import":
            self._neurokit2_dwt_context_error = reason
            self._neurokit2_dwt_padded_error = reason
        elif context == "window_too_short":
            self._neurokit2_dwt_context_error = reason
            self._neurokit2_dwt_padded_error = reason
        elif context == "context_no_r_onsets":
            if self._neurokit2_dwt_context_error is None:
                self._neurokit2_dwt_context_error = reason

    def record_context_window_available(self, available: bool) -> None:
        self._context_window_available = available

    def record_neurokit2_attempted(self) -> None:
        """Mark that NeuroKit2 delineation was attempted (even if it returned no landmarks)."""
        self._neurokit2_delineation_never_attempted = False

    # ── Per-feature source assignment ─────────────────────────

    def set_feature_source(
        self,
        feature: str,
        source: str,
        value: Optional[float] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """
        Record which source was selected for a feature and propagate
        the method-level errors into the feature-level diagnostics.

        Parameters
        ----------
        feature : str
            Feature name (must be in FEATURE_NAMES).
        source : str
            'neurokit2_dwt', 'peak_based', or 'clean_window'.
        value : float or None
            The final extracted value.
        warnings : list of str, optional
            Any warnings for this feature.
        """
        diag = self.features.get(feature)
        if diag is None:
            return  # Unknown feature, skip

        diag.final_value = value
        diag.selected_source = source
        if warnings:
            diag.warnings.extend(warnings)

        # ── NeuroKit2 DWT attempt ──
        nk2_err = (
            self._neurokit2_dwt_context_error
            or self._neurokit2_dwt_padded_error
        )
        if not self._neurokit2_delineation_never_attempted or nk2_err is not None:
            diag.neurokit2.attempted = True
            if source == "neurokit2_dwt":
                diag.neurokit2.succeeded = True
            else:
                diag.neurokit2.succeeded = False
                diag.neurokit2.reason = nk2_err or "NeuroKit2 delineation failed (unknown reason)"

        # ── Peak-based attempt ──
        if not self._neurokit2_delineation_never_attempted or self._neurokit2_peak_padded_error is not None:
            # Peak-based was tried if DWT failed and we attempted peak method
            if self._neurokit2_peak_padded_error is not None:
                diag.peak_based.attempted = True
                diag.peak_based.succeeded = False
                diag.peak_based.reason = self._neurokit2_peak_padded_error
            elif source == "peak_based":
                diag.peak_based.attempted = True
                diag.peak_based.succeeded = True

        # ── Clean window attempt ──
        if source == "clean_window":
            diag.clean_window.attempted = True
            diag.clean_window.succeeded = True
        elif source in ("neurokit2_dwt", "peak_based"):
            diag.clean_window.attempted = False
            diag.clean_window.succeeded = False
            diag.clean_window.reason = "Higher-priority method already succeeded"

    def set_feature_missing(self, feature: str, reason: str) -> None:
        """
        Mark a feature as having no value from any source.
        """
        diag = self.features.get(feature)
        if diag is None:
            return
        diag.final_value = None
        diag.selected_source = "none"

        # Record method-level attempts
        nk2_err = (
            self._neurokit2_dwt_context_error
            or self._neurokit2_dwt_padded_error
        )
        if not self._neurokit2_delineation_never_attempted or nk2_err is not None:
            diag.neurokit2.attempted = True
            diag.neurokit2.succeeded = False
            diag.neurokit2.reason = nk2_err or "No landmark returned"
        if self._neurokit2_peak_padded_error is not None:
            diag.peak_based.attempted = True
            diag.peak_based.succeeded = False
            diag.peak_based.reason = self._neurokit2_peak_padded_error
        diag.clean_window.attempted = True
        diag.clean_window.succeeded = False
        diag.clean_window.reason = reason

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert all diagnostics to a JSON-serializable dict."""
        d = {
            "beat_index": self.beat_index,
            "features": {
                name: diag.to_dict() for name, diag in self.features.items()
            },
            # Method-level state needed for faithful reconstruction
            "neurokit2_delineation_never_attempted": self._neurokit2_delineation_never_attempted,
            "neurokit2_dwt_context_error": self._neurokit2_dwt_context_error,
            "neurokit2_dwt_padded_error": self._neurokit2_dwt_padded_error,
            "neurokit2_peak_padded_error": self._neurokit2_peak_padded_error,
            "context_window_available": self._context_window_available,
        }
        # Debug: raw NK context info (set during extraction when debug=True)
        if hasattr(self, "_raw_context_info") and self._raw_context_info is not None:
            d["raw_context_info"] = self._raw_context_info
        if hasattr(self, "_ctx_target_info") and self._ctx_target_info is not None:
            d["ctx_target_info"] = self._ctx_target_info
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MorphologyDiagnosticCollector":
        """
        Reconstruct a collector from its ``to_dict()`` serialization.
        This faithfully restores method-level state so that
        ``MorphologyDiagnosticFormatter.format_beat()`` displays the
        same information as the original extraction.
        """
        collector = cls(beat_index=data.get("beat_index", 0))

        # Restore method-level state
        collector._neurokit2_delineation_never_attempted = data.get(
            "neurokit2_delineation_never_attempted", True)
        collector._neurokit2_dwt_context_error = data.get("neurokit2_dwt_context_error")
        collector._neurokit2_dwt_padded_error = data.get("neurokit2_dwt_padded_error")
        collector._neurokit2_peak_padded_error = data.get("neurokit2_peak_padded_error")
        collector._context_window_available = data.get("context_window_available", False)

        # Restore per-feature diagnostics directly (bypass set_feature_source
        # to avoid the ordering dependency on method-level state).
        for feat_name, feat_diag in data.get("features", {}).items():
            if feat_name not in collector.features:
                continue
            diag = collector.features[feat_name]
            diag.final_value = feat_diag.get("final_value")
            diag.selected_source = feat_diag.get("selected_source", "none")

            # Restore each source attempt
            for src_key in ("neurokit2", "peak_based", "clean_window"):
                src_data = feat_diag.get(src_key, {})
                src = getattr(diag, src_key)
                src.attempted = src_data.get("attempted", False)
                src.succeeded = src_data.get("succeeded", False)
                src.reason = src_data.get("reason")

            diag.warnings = feat_diag.get("warnings", [])

        # Restore debug context info if present
        if "raw_context_info" in data:
            collector._raw_context_info = data["raw_context_info"]
        if "ctx_target_info" in data:
            collector._ctx_target_info = data["ctx_target_info"]

        return collector


# ────────────────────────────────────────────────────────────────
# Formatter — pretty-prints diagnostics to console / log
# ────────────────────────────────────────────────────────────────


class MorphologyDiagnosticFormatter:
    """
    Formats MorphologyDiagnosticCollector output for human reading.

    Two output modes:
        format_beat     — colorful/structured console output
        format_beat_log — compact log-friendly output
    """

    @staticmethod
    def format_beat(collector: MorphologyDiagnosticCollector) -> str:
        """Return a detailed per-beat diagnostic string for console display."""
        lines: List[str] = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append(f"║  MORPHOLOGY DIAGNOSTIC — Beat #{collector.beat_index}                          ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        # Print method-level summary first
        lines.append("")
        lines.append("  Extraction path summary:")
        if collector._context_window_available:
            lines.append("    Context window:    AVAILABLE (≥2 R-peaks)")
        else:
            lines.append("    Context window:    NOT AVAILABLE (<2 R-peaks or not provided)")

        # Debug: print raw NK context window output when available
        raw_info = getattr(collector, "_raw_context_info", None)
        if raw_info:
            lines.append("")
            lines.append("  ╔══ NeuroKit2 Context Window DWT Output ═══════════════════")
            for k, v in raw_info.items():
                status = "ALL NAN" if v.get("all_nan") else f"{v.get('n_finite')} finite / {v.get('n_nan')} NaN"
                vals_str = str(v.get("values", []))[:60]
                lines.append(f"  ║ {k:<20} {status:<20} first_few={vals_str}")
            lines.append("  ╚═══════════════════════════════════════════════════════════")

        ctx_target = getattr(collector, "_ctx_target_info", None)
        if ctx_target:
            lines.append(f"  Target R context coord: {ctx_target.get('target_r_ctx')}")
            lines.append(f"  Context R-peaks ({ctx_target.get('n_ctx_rpeaks')}): {ctx_target.get('ctx_rpeaks')}")
            lines.append(f"  beat_start={ctx_target.get('beat_start')}  signal_len={ctx_target.get('signal_len')}")

        lines.append("")

        for feature_name in FEATURE_NAMES:
            diag = collector.features.get(feature_name)
            if diag is None:
                continue

            # Skip features that had no extraction at all (unless they have data)
            if diag.selected_source == "none" and diag.neurokit2.reason is None:
                continue

            value_str = f"{diag.final_value}" if diag.final_value is not None else "None"

            lines.append(f"  {feature_name} (value={value_str})")
            lines.append(f"    Selected Source: {diag.selected_source}")
            lines.append("    " + "─" * 35)

            # NeuroKit2 DWT
            nk2_icon = "✓" if diag.neurokit2.succeeded else "✗"
            nk2_status = "SUCCESS" if diag.neurokit2.succeeded else "FAILED"
            if diag.neurokit2.attempted or diag.neurokit2.reason is not None:
                lines.append(f"    NeuroKit2 DWT:     {nk2_icon} {nk2_status}")
                if diag.neurokit2.reason is not None:
                    lines.append(f"      → {diag.neurokit2.reason}")
            else:
                lines.append(f"    NeuroKit2 DWT:     · Not attempted")

            # Peak-based
            pb_icon = "✓" if diag.peak_based.succeeded else "✗"
            pb_status = "SUCCESS" if diag.peak_based.succeeded else "FAILED"
            if diag.peak_based.attempted or diag.peak_based.reason is not None:
                lines.append(f"    Peak-based:        {pb_icon} {pb_status}")
                if diag.peak_based.reason is not None:
                    lines.append(f"      → {diag.peak_based.reason}")
            else:
                lines.append(f"    Peak-based:        · Not used")

            # Clean Window
            cw_icon = "✓" if diag.clean_window.succeeded else "·"
            if diag.clean_window.attempted:
                cw_status = "SUCCESS" if diag.clean_window.succeeded else "FAILED"
                lines.append(f"    Clean Window:      {cw_icon} {cw_status}")
                if diag.clean_window.reason is not None:
                    lines.append(f"      → {diag.clean_window.reason}")
            else:
                cw_reason = diag.clean_window.reason or "Higher-priority method already succeeded"
                lines.append(f"    Clean Window:      · Not used ({cw_reason})")

            # Warnings
            if diag.warnings:
                for w in diag.warnings:
                    lines.append(f"    ⚠ {w}")

            lines.append("")

        lines.append("╚" + "═" * 70 + "╝")
        return "\n".join(lines)

    @staticmethod
    def format_beat_log(collector: MorphologyDiagnosticCollector) -> str:
        """Return a compact log-friendly per-beat diagnostic string."""
        parts: List[str] = []
        parts.append(f"Beat #{collector.beat_index}")
        failures: List[str] = []
        fallbacks: List[str] = []

        for feature_name in FEATURE_NAMES:
            diag = collector.features.get(feature_name)
            if diag is None:
                continue
            if diag.selected_source != "neurokit2_dwt":
                fallbacks.append(f"{feature_name}={diag.selected_source}")
            if diag.neurokit2.reason is not None:
                brief = diag.neurokit2.reason.split(".")[0].split("\n")[0][:80]
                failures.append(f"{feature_name}:{brief}")

        if fallbacks:
            parts.append("fallbacks=[" + ", ".join(fallbacks) + "]")
        if failures:
            parts.append("reasons=[" + " | ".join(failures) + "]")
        return " | ".join(parts)


# ────────────────────────────────────────────────────────────────
# Convenience function to print diagnostics
# ────────────────────────────────────────────────────────────────


def print_morphology_diagnostics(collector: MorphologyDiagnosticCollector) -> None:
    """Print diagnostics to stdout and also log them."""
    console_text = MorphologyDiagnosticFormatter.format_beat(collector)
    print(console_text)
    log_text = MorphologyDiagnosticFormatter.format_beat_log(collector)
    logger.info(log_text)
