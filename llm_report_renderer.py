# -*- coding: utf-8 -*-
"""
A second, narrow LLM call whose only job is presentation: take the already-
generated `assessment` JSON (produced by the reasoning call in
pdf_semantic_rag.py — SYSTEM_PROMPT there is the clinical-reasoning model)
and rewrite it as the readable English report format, using ONLY the facts
already in that JSON.

WHY A SEPARATE CALL INSTEAD OF ASKING THE REASONING MODEL TO ALSO DO THIS:
Splitting reasoning and presentation into two calls means each prompt has
one job. The reasoning call (pdf_semantic_rag.py) still only ever has to
produce the strict JSON schema — nothing about its contract changes here.
This renderer call's only job is reformatting fields it's already been
handed; it is not asked to weigh evidence, rank conditions, or produce any
clinical judgment, so the failure modes it can have are narrower (wrong
formatting, dropped section, added filler) rather than a wrong diagnosis.
That said, "narrower risk" is not "no risk" — a model can still ignore the
template, add a claim that isn't in the input, or time out. Guardrails here:

1. The system prompt below explicitly forbids adding any clinical fact not
   already present in the input JSON, and forbids changing any number.
2. render_via_llm() checks the output actually contains the required
   section headers before accepting it; if not, it retries once with a
   stricter reminder.
3. If it still doesn't come back well-formed, render_via_llm() returns
   (None, error) rather than a guessed/partial report — the caller
   (pdf_semantic_rag.py) falls back to the deterministic template in
   clinical_report_formatter.py, which is always correct even though it
   reads slightly more mechanically. See response["formatted_report_source"]
   on PDFSemanticECGAgent.analyze()'s return value to see which path a
   given report actually came from.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

REPORT_RENDER_SYSTEM_PROMPT = """
You are a report-formatting component, not a clinical reasoning component.

You will be given one JSON object: a FINAL, ALREADY-COMPLETE clinical
assessment produced by a separate reasoning system. Every fact, finding,
condition name, confidence value, recommendation, question, and reference
in that JSON has already been decided. Your only job is to present it as
readable English prose in a fixed report format.

STRICT RULES

- Do not add any clinical finding, condition, risk level, recommendation,
  or question that is not already present in the input JSON.
- Do not change any number (confidence values, percentages, recheck
  minutes) from the input.
- Do not drop the missing-evidence or warnings content — every item there
  must still appear in your output, in substance (you may rephrase the
  wording, but not omit or soften it).
- Do not invent a diagnosis, medication, dosage, or citation that is not in
  the input.
- If a list in the input is empty, say so plainly in that section (e.g.
  "No additional findings were reported.") rather than inventing content to
  fill the section.
- Only include the "Important Notes (Missing Data / Warnings)" section if
  the input actually has missing-evidence or warning items; omit the whole
  section otherwise.
- Output ONLY the report itself — no greeting, no explanation of what you
  did, no markdown code fence, nothing before the title or after the final
  line.

REQUIRED OUTPUT FORMAT — use exactly these headers, in exactly this order,
in English:

## Clinical Decision Support Report

**Event Type:** <from input>
**Timestamp:** <from input, or "Not available">
**Severity Level:** <derived from risk_assessment.overall_urgency>
**Confidence:** <derived from event_confidence>

---

### 📋 Quick Summary
<1-3 sentences from the input's summary>

### 🔍 Key Findings
<bulleted list from evidence.supporting and any non-"not_applicable" risk_assessment entries>

### 🩺 Differential Diagnosis
<numbered list from most_likely_conditions, then bulleted alternatives from differential_diagnosis>

### ⚠️ Immediate Recommended Actions
<bulleted list from immediate_recommendations>

### ❓ Follow-up Questions
<bulleted list from recommended_questions>

### 📝 Important Notes (Missing Data / Warnings)
<bulleted list from evidence.missing and warnings — omit this whole section if both are empty>

### 📚 References Used
<bulleted list from references>

---

**Decision Summary:**
<one or two sentences reflecting backend_flags.notify_clinician_now, risk_assessment.overall_urgency, and follow_up>
"""

_REQUIRED_MARKERS = (
    "clinical decision support report",
    "quick summary",
    "key findings",
    "differential diagnosis",
    "immediate recommended actions",
    "follow-up questions",
    "references used",
    "decision summary",
)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"```\s*$", "", stripped)
    return stripped.strip()


def _looks_like_valid_report(text: Optional[str]) -> bool:
    """
    Cheap structural check, not a content check: confirms the model actually
    followed the required template rather than e.g. echoing the input JSON
    back, refusing, or dropping a section. This is a floor, not a guarantee
    of quality — it cannot verify the prose is factually faithful to the
    input, only that it has the shape we asked for.
    """
    if not text:
        return False
    candidate = _strip_code_fence(text)
    if not candidate:
        return False
    # If it starts with '{' it's almost certainly the model echoing JSON
    # back instead of rendering prose — reject outright rather than trust it.
    if candidate.lstrip().startswith("{"):
        return False
    lowered = candidate.lower()
    return all(marker in lowered for marker in _REQUIRED_MARKERS)


def build_render_prompt(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
) -> Dict[str, str]:
    """Build the {system, user} prompt pair for the rendering LLM call."""
    event = event or {}
    document_name = document_name or "Cardiac_Diagnostics_Comprehensive_KB.pdf"

    context = {
        "event_type": event.get("event_type"),
        "timestamp": event.get("timestamp") or event.get("created_at"),
        "document_name": document_name,
        "assessment": assessment,
    }
    try:
        context_json = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except Exception:
        context_json = str(context)

    user_prompt = f"""
Render the following assessment into the required report format. Follow
the STRICT RULES and REQUIRED OUTPUT FORMAT from the system prompt exactly.

INPUT
{context_json}
"""
    return {"system": REPORT_RENDER_SYSTEM_PROMPT, "user": user_prompt}


def render_via_llm(
    llm_client: Any,
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
    max_attempts: int = 2,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the formatter LLM and return (report_text, None) on success, or
    (None, error_message) if it never comes back well-formed within
    max_attempts. Never raises on a malformed response — only propagates an
    exception if the LLM client itself raises (e.g. network/auth failure),
    since that is a real failure the caller should know about rather than
    silently swallow.

    llm_client : anything with a .generate(system_prompt, user_prompt) method
        — normally the same LLMClient used for reasoning, or a distinct one
        if you want to point formatting at a different/cheaper model. See
        PDFSemanticECGAgent's `formatter_llm` constructor parameter.
    """
    if not assessment:
        return None, "no assessment to render"

    prompt = build_render_prompt(assessment, event=event, document_name=document_name)
    last_error = "renderer LLM did not produce a validly structured report"

    for attempt in range(max_attempts):
        text = llm_client.generate(prompt["system"], prompt["user"])
        if _looks_like_valid_report(text):
            return _strip_code_fence(text), None

        last_error = (
            "renderer LLM response was missing required section headers "
            "or was not prose (e.g. echoed JSON back)"
        )
        if attempt < max_attempts - 1:
            prompt = dict(prompt)
            prompt["user"] = (
                prompt["user"]
                + "\n\nYour previous response did not use the exact required "
                "section headers, or was not plain English prose. Respond "
                "again using EXACTLY the headers from the system prompt, in "
                "English, with no other content before or after."
            )

    return None, last_error


# =============================================================================
# PATIENT-FACING RENDERER
#
# This is a second, separate prompt from REPORT_RENDER_SYSTEM_PROMPT above —
# not a formatting option on the same prompt — because the rules for what's
# safe to tell a patient directly are fundamentally different from what's
# appropriate for a clinician, not just a matter of tone. Mixing "full
# clinical detail" and "no diagnostic percentages, no lab orders" into one
# prompt risks the model bleeding doctor-only content into the patient
# version under the same set of instructions. Two clean prompts, one call
# each, is safer than one prompt trying to do both.
#
# The doctor-facing render_via_llm() above only checks STRUCTURE (are the
# required headers present). For the patient version, structure isn't
# enough — a structurally perfect report could still leak a confidence
# percentage, a lab order, or a differential-diagnosis list into patient-
# facing text. _patient_report_has_disallowed_content() below scans for
# exactly that, and failing that scan invalidates the report regardless of
# whether the headers were right — see the AUDIENCE NOTE in
# clinical_report_formatter.py for why this content shouldn't reach a
# patient un-mediated by clinical staff in the first place.
# =============================================================================

PATIENT_REPORT_SYSTEM_PROMPT = """
You are writing a short update for a PATIENT about their own heart rhythm
monitor, based on a completed clinical assessment. You are not the patient's
doctor and you are not making treatment decisions — you are translating an
already-completed clinical assessment into plain, calm, accurate language
for the person wearing the monitor.

You will be given the same JSON assessment a clinician's version of this
report is built from. You must leave almost all of that detail out. Your
job is not to inform the patient the way you'd inform a clinician — it is
to tell them clearly what to do next, without alarming or confusing them
with content meant for clinical staff.

STRICT RULES — a patient must never see any of the following, even
rephrased or summarized:

- No differential diagnosis list or alternative conditions "considered".
  Mention at most the ONE primary finding, in plain words, if any.
- No confidence percentages or numeric scores of any kind.
- No ICD codes.
- No specific lab tests, medications, drug names, or dosages, even if they
  appear in immediate_recommendations — instead say their care team will
  decide what tests or treatment, if any, are needed.
- No citations, references, or mentions of the knowledge source/document.
- No clinical jargon without plain-language explanation — if you must
  reference a term the patient would recognize (e.g. "irregular heartbeat"),
  keep it at that level; do not use the technical condition name or subtype
  from the assessment as the primary way you describe it to them.

WHAT TO INCLUDE, ALWAYS IN PLAIN LANGUAGE:

1. What the monitor noticed — one or two plain-language sentences, not a
   clinical finding list.
2. What this means for them right now — calm, accurate, proportionate to
   the actual urgency in the assessment. Do not minimize a genuinely urgent
   or emergency finding to sound more reassuring than it is, and do not
   dramatize a routine/monitor-level finding into sounding like an
   emergency.
3. What they should do next — always frame this as contacting their care
   team / doctor / emergency services, matching the urgency level in the
   assessment (routine follow-up, contact your doctor soon, or seek
   emergency care now). Never tell the patient to take, adjust, or stop any
   medication themselves, and never suggest a specific test or treatment.
4. Up to 2-3 simple questions their care team might ask them, rephrased in
   the patient's own words if the assessment includes any — omit this
   section entirely if there are none worth surfacing.

Output ONLY the report itself — no greeting beyond what's in the format
below, no explanation of what you did, no markdown code fence.

REQUIRED OUTPUT FORMAT — use exactly these headers, in exactly this order:

## Heart Monitoring Update

**What We Noticed:**
<1-2 plain-language sentences>

**What This Means:**
<short, calm, accurate paragraph>

**What You Should Do:**
<clear next step, framed around contacting their care team, matching the true urgency>

**Questions Your Care Team Might Ask:**
<up to 3 simple bullet points, or omit this whole section if there are none>
"""

_PATIENT_REQUIRED_MARKERS = (
    "heart monitoring update",
    "what we noticed",
    "what this means",
    "what you should do",
)

# Patterns that must never appear in patient-facing text, regardless of
# whether the model was told not to include them — a second, independent
# check rather than trusting the prompt alone. Deliberately conservative:
# a false-positive here just means an unnecessary fallback to the
# guaranteed-clean deterministic template, which is a low-cost outcome;
# a false-negative means clinical detail reaching a patient un-mediated,
# which is not.
_DISALLOWED_PATIENT_PATTERNS = (
    re.compile(r"\b\d{1,3}\s?%"),                       # confidence percentages
    re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|mL|units?)\b", re.IGNORECASE),  # dosages
    re.compile(r"\b[A-Z]\d{2}(\.\d+)?\b"),               # ICD-10-style codes
    re.compile(r"differential diagnos", re.IGNORECASE),
    re.compile(r"\bicd-?10\b", re.IGNORECASE),
    re.compile(r"chunk_id", re.IGNORECASE),
)


def _patient_report_has_disallowed_content(text: str) -> Optional[str]:
    """Returns a description of the first disallowed pattern found, or None
    if the text is clean. Checked independently of the required-headers
    check — a report can pass one and fail the other."""
    for pattern in _DISALLOWED_PATIENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"disallowed content matched pattern {pattern.pattern!r}: {match.group(0)!r}"
    return None


def _looks_like_valid_patient_report(text: Optional[str]) -> bool:
    if not text:
        return False
    candidate = _strip_code_fence(text)
    if not candidate:
        return False
    if candidate.lstrip().startswith("{"):
        return False
    lowered = candidate.lower()
    if not all(marker in lowered for marker in _PATIENT_REQUIRED_MARKERS):
        return False
    if _patient_report_has_disallowed_content(candidate) is not None:
        return False
    return True


def build_patient_render_prompt(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build the {system, user} prompt pair for the patient-facing rendering
    call. Deliberately does NOT include document_name or retrieved chunks —
    the patient version has nothing to cite, so there's no reason to give
    the model that material to (mis)use.
    """
    event = event or {}
    context = {
        "event_type": event.get("event_type"),
        "assessment": assessment,
    }
    try:
        context_json = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except Exception:
        context_json = str(context)

    user_prompt = f"""
Write the patient update for the following completed clinical assessment.
Follow the STRICT RULES and REQUIRED OUTPUT FORMAT from the system prompt
exactly. Remember: this is going directly to the patient, not a clinician.

INPUT
{context_json}
"""
    return {"system": PATIENT_REPORT_SYSTEM_PROMPT, "user": user_prompt}


def render_patient_report_via_llm(
    llm_client: Any,
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    max_attempts: int = 2,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Patient-facing counterpart to render_via_llm() above. Same retry
    pattern, but validity here requires BOTH the required headers AND a
    clean pass of _patient_report_has_disallowed_content() — see the module
    docstring section above for why the extra content-safety check exists
    specifically for this audience and not the doctor-facing one.

    Never raises on a malformed or unsafe response — degrades to
    (None, error) so the caller falls back to the deterministic
    render_patient_summary_en()/_ar() template in
    clinical_report_formatter.py, which is guaranteed free of this content
    by construction rather than by prompting.
    """
    if not assessment:
        return None, "no assessment to render"

    prompt = build_patient_render_prompt(assessment, event=event)
    last_error = "renderer LLM did not produce a valid patient-safe report"

    for attempt in range(max_attempts):
        text = llm_client.generate(prompt["system"], prompt["user"])
        candidate = _strip_code_fence(text) if text else ""

        if _looks_like_valid_patient_report(text):
            return candidate, None

        disallowed = _patient_report_has_disallowed_content(candidate) if candidate else None
        if disallowed:
            last_error = f"patient report failed content-safety check: {disallowed}"
        else:
            last_error = (
                "patient report response was missing required section "
                "headers or was not prose"
            )

        if attempt < max_attempts - 1:
            prompt = dict(prompt)
            prompt["user"] = (
                prompt["user"]
                + "\n\nYour previous response either didn't use the exact "
                "required headers, or included content that must never be "
                "shown to a patient (percentages, dosages, codes, "
                "differential diagnoses, citations). Respond again, "
                "following the rules exactly."
            )

    return None, last_error