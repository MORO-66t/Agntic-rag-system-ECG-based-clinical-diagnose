# -*- coding: utf-8 -*-
"""
Deterministic template renderers for the agent's structured JSON assessment
(the `assessment` dict from PDFSemanticECGAgent.analyze()'s response — see
pdf_semantic_rag.py) into a readable Markdown report: a header block, then
Quick Summary / Key Findings / Differential Diagnosis / Immediate
Recommended Actions / Follow-up Questions / References Used / Decision
Summary.

Both an English (`_en`) and the original Arabic (`_ar`) version are provided.
The pipeline's default is now the English, LLM-rendered report — see
llm_report_renderer.py — with the `_en` functions here used ONLY as its
deterministic fallback (see the AUDIENCE + LLM-RENDERING notes below).

WHY THESE TEMPLATE FUNCTIONS STILL EXIST NOW THAT AN LLM RENDERS THE REPORT:
llm_report_renderer.py asks a second, narrow LLM call to turn the same
`assessment` dict into prose in this exact structure — a much easier, lower-
risk task than the original single call that also had to do the clinical
reasoning (it's pure rewriting of already-extracted fields, not new
judgment). But "lower risk" isn't "zero risk": that call can still time out,
error, or come back without the required section headers. When that
happens, pdf_semantic_rag.py falls back to render_report_or_fallback_en()
below, so there is always something displayable, guaranteed free of
hallucination, even if the LLM call fails outright. response["formatted_report_source"]
on the returned dict tells you which path produced the report you're
looking at ("llm" or "template_fallback").

AUDIENCE NOTE — please read before wiring this to an end-user screen:
The example report this was modeled on talks in clinician terms:
differential diagnoses with confidence percentages, "draw blood for
potassium/magnesium," guideline citations, "present to a cardiology
consultant within an hour." That is appropriate content for a clinician, a
triage nurse, or a reviewing physician — it is not written the way you would
talk to a patient directly. If this is meant to go straight to the person
wearing the monitor rather than to clinical staff, the differential-
diagnosis list, lab-draw instructions, and guideline citations should
likely be dropped or replaced with something like "your care team will
review this." This module renders the clinician-facing version (matching
what you asked for); render_patient_summary_en() / render_patient_summary_ar()
are much shorter, plain-language variants for the second audience, included
so the choice is explicit rather than accidental.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Display-name lookup: event_type -> Arabic name.
# Extend this as you add event types. Falls back to the raw event_type (or,
# if present, the primary disease name from the event record) if not listed.
# ---------------------------------------------------------------------------
EVENT_TYPE_DISPLAY_AR: Dict[str, str] = {
    "VT_RUN": "تسارع بطيني غير مستمر (NSVT)",
    "AFIB_DETECTED": "رجفان أذيني (Atrial Fibrillation)",
    "AFLUTTER_SUSPECTED": "رفرفة أذينية مشتبه بها (Atrial Flutter)",
    "SVT_SUSPECTED": "تسارع فوق بطيني مشتبه به (SVT)",
    "BRADYCARDIA": "بطء ضربات القلب (Bradycardia)",
    "TACHYCARDIA": "تسارع ضربات القلب (Tachycardia)",
    "POSSIBLE_AV_BLOCK": "حصار أذيني بطيني محتمل (AV Block)",
    "POSSIBLE_LONG_QT": "إطالة محتملة في QT (Long QT)",
    "POSSIBLE_ISCHEMIC_PATTERN": "نمط نقص تروية محتمل",
    "POSSIBLE_HEART_FAILURE_PATTERN": "نمط قصور قلبي محتمل",
    "PAUSE_DETECTED": "توقف في النظم (Pause)",
    "DISEASE_WPW_PREEXCITATION": "متلازمة وولف-باركنسون-وايت (WPW)",
    "DISEASE_BRUGADA_SYNDROME": "متلازمة بروجادا (Brugada)",
    "DISEASE_VENTRICULAR_FIBRILLATION": "رجفان بطيني (VF)",
}

URGENCY_DISPLAY_AR: Dict[str, str] = {
    "routine": "🟢 روتينية",
    "monitor": "🟡 متوسطة (تحتاج مراجعة)",
    "urgent": "🟠 عاجلة",
    "emergency": "🔴 طارئة",
}

RISK_LEVEL_DISPLAY_AR: Dict[str, Optional[str]] = {
    "low": "منخفض",
    "moderate": "متوسط",
    "high": "مرتفع",
    "critical": "حرج",
    "not_applicable": None,  # omitted from the report entirely, not shown as "N/A"
}

RISK_LABEL_AR: Dict[str, str] = {
    "stroke": "خطر السكتة الدماغية",
    "sudden_cardiac_death": "خطر الموت القلبي المفاجئ",
    "heart_failure": "خطر قصور القلب",
    "cardiogenic_shock": "خطر الصدمة القلبية",
}


# ---------------------------------------------------------------------------
# English equivalents of the four lookup tables above. Kept as separate
# dicts (not a single bilingual structure) so each language's report can be
# edited/extended independently.
# ---------------------------------------------------------------------------
EVENT_TYPE_DISPLAY_EN: Dict[str, str] = {
    "VT_RUN": "Nonsustained Ventricular Tachycardia (NSVT)",
    "AFIB_DETECTED": "Atrial Fibrillation (AFib)",
    "AFLUTTER_SUSPECTED": "Suspected Atrial Flutter",
    "SVT_SUSPECTED": "Suspected Supraventricular Tachycardia (SVT)",
    "BRADYCARDIA": "Bradycardia",
    "TACHYCARDIA": "Tachycardia",
    "POSSIBLE_AV_BLOCK": "Possible AV Block",
    "POSSIBLE_LONG_QT": "Possible Long QT",
    "POSSIBLE_ISCHEMIC_PATTERN": "Possible Ischemic Pattern",
    "POSSIBLE_HEART_FAILURE_PATTERN": "Possible Heart Failure Pattern",
    "PAUSE_DETECTED": "Pause Detected",
    "DISEASE_WPW_PREEXCITATION": "Wolff-Parkinson-White (WPW) Syndrome",
    "DISEASE_BRUGADA_SYNDROME": "Brugada Syndrome",
    "DISEASE_VENTRICULAR_FIBRILLATION": "Ventricular Fibrillation (VF)",
}

URGENCY_DISPLAY_EN: Dict[str, str] = {
    "routine": "🟢 Routine",
    "monitor": "🟡 Moderate (Needs Review)",
    "urgent": "🟠 Urgent",
    "emergency": "🔴 Emergency",
}

RISK_LEVEL_DISPLAY_EN: Dict[str, Optional[str]] = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "critical": "Critical",
    "not_applicable": None,  # omitted from the report entirely, not shown as "N/A"
}

RISK_LABEL_EN: Dict[str, str] = {
    "stroke": "Stroke Risk",
    "sudden_cardiac_death": "Sudden Cardiac Death Risk",
    "heart_failure": "Heart Failure Risk",
    "cardiogenic_shock": "Cardiogenic Shock Risk",
}


def _confidence_label_ar(confidence: Optional[float]) -> str:
    if confidence is None:
        return "غير محددة"
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "غير محددة"
    if value >= 80:
        return "عالية"
    if value >= 50:
        return "متوسطة"
    return "منخفضة"


def _event_display_name_ar(event_type: str, event: Optional[Dict[str, Any]]) -> str:
    if event_type in EVENT_TYPE_DISPLAY_AR:
        return EVENT_TYPE_DISPLAY_AR[event_type]
    metadata = (event or {}).get("metadata_json") or {}
    if isinstance(metadata, dict):
        disease = metadata.get("disease") or metadata.get("condition")
        if disease:
            return str(disease)
    return event_type


def _format_timestamp_ar(event: Optional[Dict[str, Any]]) -> str:
    ts = (event or {}).get("timestamp") or (event or {}).get("created_at")
    return str(ts) if ts else "غير متاح"


def _confidence_label_en(confidence: Optional[float]) -> str:
    if confidence is None:
        return "Not determined"
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "Not determined"
    if value >= 80:
        return "High"
    if value >= 50:
        return "Moderate"
    return "Low"


def _event_display_name_en(event_type: str, event: Optional[Dict[str, Any]]) -> str:
    if event_type in EVENT_TYPE_DISPLAY_EN:
        return EVENT_TYPE_DISPLAY_EN[event_type]
    metadata = (event or {}).get("metadata_json") or {}
    if isinstance(metadata, dict):
        disease = metadata.get("disease") or metadata.get("condition")
        if disease:
            return str(disease)
    return event_type or "Unknown Event"


def _format_timestamp_en(event: Optional[Dict[str, Any]]) -> str:
    ts = (event or {}).get("timestamp") or (event or {}).get("created_at")
    return str(ts) if ts else "Not available"


def _bullet_list(items: List[str], empty_text: str = "No additional data available.") -> str:
    if not items:
        return f"- {empty_text}\n"
    return "\n".join(f"- {item}" for item in items) + "\n"


def render_clinical_report_ar(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
) -> str:
    """
    Render one PDFSemanticECGAgent assessment (the Part-5-schema JSON dict —
    NOT the outer response envelope, NOT the raw llm_response string) into
    the clinician-facing Arabic Markdown report format.
    """
    assessment = assessment or {}
    event = event or {}
    document_name = document_name or "Cardiac_Diagnostics_Comprehensive_KB.pdf"

    event_type = event.get("event_type", "")
    severity_label = URGENCY_DISPLAY_AR.get(
        str(assessment.get("risk_assessment", {}).get("overall_urgency", "")).lower(),
        "غير محددة",
    )
    confidence = assessment.get("event_confidence")
    confidence_label = _confidence_label_ar(confidence)
    confidence_pct = f"{int(confidence)}%" if isinstance(confidence, (int, float)) else "غير محددة"

    lines: List[str] = []
    lines.append("## تقرير مساعد القرار السريري (Clinical Decision Support)\n")
    lines.append(f"**نوع الحدث:** {_event_display_name_ar(event_type, event)}  ")
    lines.append(f"**التوقيت:** {_format_timestamp_ar(event)}  ")
    lines.append(f"**مستوى الخطورة:** {severity_label}  ")
    lines.append(f"**نسبة الثقة:** {confidence_label} ({confidence_pct})\n")
    lines.append("---\n")

    lines.append("### 📋 ملخص سريع\n")
    lines.append((assessment.get("summary") or "لا يوجد ملخص متاح.") + "\n")

    lines.append("### 🔍 أبرز النتائج\n")
    findings: List[str] = []
    evidence = assessment.get("evidence") or {}
    for item in evidence.get("supporting") or []:
        if isinstance(item, dict) and item.get("finding"):
            findings.append(str(item["finding"]))
    risk = assessment.get("risk_assessment") or {}
    for key, label in RISK_LABEL_AR.items():
        level = str(risk.get(key, "")).lower()
        display = RISK_LEVEL_DISPLAY_AR.get(level)
        if display:
            findings.append(f"{label}: **{display}**")
    lines.append(_bullet_list(findings, empty_text="لا توجد بيانات إضافية."))

    lines.append("### 🩺 التشخيصات المحتملة (Differential Diagnosis)\n")
    diagnosis_lines: List[str] = []
    for idx, cond in enumerate(assessment.get("most_likely_conditions") or [], start=1):
        if not isinstance(cond, dict):
            continue
        name = cond.get("condition", "غير معروف")
        subtype = cond.get("subtype")
        conf = cond.get("confidence")
        conf_txt = f"{int(conf)}%" if isinstance(conf, (int, float)) else "غير محددة"
        subtype_txt = f" ({subtype})" if subtype and "Cannot determine" not in str(subtype) else ""
        diagnosis_lines.append(f"{idx}. **{name}{subtype_txt}** — نسبة الاحتمال: {conf_txt}")
    for diff in assessment.get("differential_diagnosis") or []:
        if not isinstance(diff, dict):
            continue
        diagnosis_lines.append(
            f"- **{diff.get('condition', 'غير معروف')}**: يُؤخذ بالاعتبار لأن "
            f"{diff.get('why_considered', 'لا يوجد تفصيل')}، لكنه أقل ترجيحًا لأن "
            f"{diff.get('why_ranked_lower', 'لا يوجد تفصيل')}."
        )
    lines.append(("\n".join(diagnosis_lines) if diagnosis_lines else "- لا توجد تشخيصات مقترحة.") + "\n")

    lines.append("### ⚠️ الإجراءات الفورية المقترحة\n")
    lines.append(_bullet_list(list(assessment.get("immediate_recommendations") or []), empty_text="لا توجد بيانات إضافية."))

    lines.append("### ❓ أسئلة للمتابعة (يُفضل الإجابة من قبل الطاقم الطبي)\n")
    question_lines = []
    for q in assessment.get("recommended_questions") or []:
        if isinstance(q, dict) and q.get("question"):
            question_lines.append(str(q["question"]))
    lines.append(_bullet_list(question_lines, empty_text="لا توجد بيانات إضافية."))

    caveats = list(evidence.get("missing") or []) + list(assessment.get("warnings") or [])
    if caveats:
        lines.append("### 📝 ملاحظات هامة (بيانات ناقصة أو تحذيرات)\n")
        lines.append(_bullet_list(caveats))

    lines.append("### 📚 المراجع المستخدمة\n")
    refs = assessment.get("references") or []
    if refs:
        ref_lines = [
            f"[{ref.get('citation_index', '?')}] {document_name} — chunk_id: {ref.get('chunk_id', 'غير متاح')}"
            for ref in refs
            if isinstance(ref, dict)
        ]
        lines.append(_bullet_list(ref_lines, empty_text="لا توجد بيانات إضافية."))
    else:
        lines.append(f"- مصدر المعرفة الطبية: {document_name}\n")

    lines.append("---\n")
    lines.append("**خلاصة القرار:**  ")
    urgency = str(risk.get("overall_urgency", "")).lower()
    notify_now = bool((assessment.get("backend_flags") or {}).get("notify_clinician_now"))
    recheck = (assessment.get("follow_up") or {}).get("recheck_in_minutes")

    if notify_now or urgency == "emergency":
        lines.append("🚨 يتطلب إخطار الطبيب/الطاقم الطبي فورًا دون تأخير.")
    elif urgency == "urgent":
        lines.append("يُنصح بعرض الحالة على استشاري قلب في أقرب وقت ممكن (خلال ساعات قليلة).")
    elif urgency == "monitor":
        recheck_txt = f" مع إعادة التقييم خلال {int(recheck)} دقيقة" if isinstance(recheck, (int, float)) else ""
        lines.append(
            f"يُنصح بعرض الحالة على استشاري قلب للمتابعة الروتينية{recheck_txt}. "
            "لا يستدعي الأمر تدخلاً إسعافيًا فوريًا ما لم تظهر أعراض جديدة على المريض."
        )
    else:
        lines.append("لا يستدعي إجراءً طارئًا حاليًا؛ يمكن المتابعة الروتينية.")

    return "\n".join(lines)


def render_report_or_fallback_ar(
    response: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
) -> str:
    """
    Convenience wrapper around render_clinical_report_ar() that takes the
    FULL response dict returned by PDFSemanticECGAgent.analyze() (not just
    the assessment sub-dict) and handles the cases where there is no usable
    assessment to render.
    """
    assessment = response.get("assessment") if isinstance(response, dict) else None
    if assessment is None:
        reason = (
            response.get("reason")
            if isinstance(response, dict) and response.get("reason")
            else "تعذر الحصول على تقييم مفهوم من النظام لهذا الحدث."
        )
        return (
            "## تقرير مساعد القرار السريري (Clinical Decision Support)\n\n"
            "⚠️ **تعذّر إنشاء تقرير آلي لهذا الحدث.**\n\n"
            f"{reason}\n\n"
            "يُرجى مراجعة الحدث يدويًا من قبل الطاقم الطبي."
        )
    return render_clinical_report_ar(assessment, event=event, document_name=document_name)


def render_patient_summary_ar(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Short, plain-language variant meant to be shown to the patient
    themselves rather than to clinical staff.
    """
    assessment = assessment or {}
    urgency = str((assessment.get("risk_assessment") or {}).get("overall_urgency", "")).lower()
    notify_now = bool((assessment.get("backend_flags") or {}).get("notify_clinician_now"))

    lines = ["## نتيجة مراقبة القلب\n"]
    lines.append((assessment.get("summary") or "تم رصد نمط في تخطيط القلب يستحق المراجعة.") + "\n")

    if notify_now or urgency == "emergency":
        lines.append("🚨 **يرجى التواصل مع طبيبك أو التوجه لأقرب طوارئ الآن.**")
    elif urgency == "urgent":
        lines.append("📞 يرجى التواصل مع طبيبك في أقرب وقت ممكن اليوم.")
    elif urgency == "monitor":
        lines.append("👍 لا داعي للقلق الفوري. سيقوم فريقك الطبي بمراجعة هذه النتيجة والتواصل معك إذا لزم الأمر.")
    else:
        lines.append("✅ هذه نتيجة روتينية ولا تستدعي أي إجراء من جانبك الآن.")

    questions = assessment.get("recommended_questions") or []
    if questions:
        lines.append("\nقد يسأل فريقك الطبي عن:")
        for q in questions[:3]:
            if isinstance(q, dict) and q.get("question"):
                lines.append(f"- {q['question']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# English renderers. Same section structure/order/emoji as the Arabic
# versions above, and the same structure the LLM in llm_report_renderer.py
# is instructed to produce.
# ---------------------------------------------------------------------------

REPORT_TITLE_EN = "## Clinical Decision Support Report"


def render_clinical_report_en(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
) -> str:
    """
    Render one PDFSemanticECGAgent assessment into the clinician-facing
    English Markdown report format.
    """
    assessment = assessment or {}
    event = event or {}
    document_name = document_name or "Cardiac_Diagnostics_Comprehensive_KB.pdf"

    event_type = event.get("event_type", "")
    severity_label = URGENCY_DISPLAY_EN.get(
        str(assessment.get("risk_assessment", {}).get("overall_urgency", "")).lower(),
        "Not determined",
    )
    confidence = assessment.get("event_confidence")
    confidence_label = _confidence_label_en(confidence)
    confidence_pct = f"{int(confidence)}%" if isinstance(confidence, (int, float)) else "Not determined"

    lines: List[str] = []
    lines.append(f"{REPORT_TITLE_EN}\n")
    lines.append(f"**Event Type:** {_event_display_name_en(event_type, event)}  ")
    lines.append(f"**Timestamp:** {_format_timestamp_en(event)}  ")
    lines.append(f"**Severity Level:** {severity_label}  ")
    lines.append(f"**Confidence:** {confidence_label} ({confidence_pct})\n")
    lines.append("---\n")

    lines.append("### 📋 Quick Summary\n")
    lines.append((assessment.get("summary") or "No summary available.") + "\n")

    lines.append("### 🔍 Key Findings\n")
    findings: List[str] = []
    evidence = assessment.get("evidence") or {}
    for item in evidence.get("supporting") or []:
        if isinstance(item, dict) and item.get("finding"):
            findings.append(str(item["finding"]))
    risk = assessment.get("risk_assessment") or {}
    for key, label in RISK_LABEL_EN.items():
        level = str(risk.get(key, "")).lower()
        display = RISK_LEVEL_DISPLAY_EN.get(level)
        if display:
            findings.append(f"{label}: **{display}**")
    lines.append(_bullet_list(findings))

    lines.append("### 🩺 Differential Diagnosis\n")
    diagnosis_lines: List[str] = []
    for idx, cond in enumerate(assessment.get("most_likely_conditions") or [], start=1):
        if not isinstance(cond, dict):
            continue
        name = cond.get("condition", "Unknown")
        subtype = cond.get("subtype")
        conf = cond.get("confidence")
        conf_txt = f"{int(conf)}%" if isinstance(conf, (int, float)) else "Not determined"
        subtype_txt = f" ({subtype})" if subtype and "Cannot determine" not in str(subtype) else ""
        diagnosis_lines.append(f"{idx}. **{name}{subtype_txt}** — Likelihood: {conf_txt}")
    for diff in assessment.get("differential_diagnosis") or []:
        if not isinstance(diff, dict):
            continue
        diagnosis_lines.append(
            f"- **{diff.get('condition', 'Unknown')}**: considered because "
            f"{diff.get('why_considered', 'no detail provided')}, but ranked lower because "
            f"{diff.get('why_ranked_lower', 'no detail provided')}."
        )
    lines.append(("\n".join(diagnosis_lines) if diagnosis_lines else "- No candidate diagnoses were proposed.") + "\n")

    lines.append("### ⚠️ Immediate Recommended Actions\n")
    lines.append(_bullet_list(list(assessment.get("immediate_recommendations") or [])))

    lines.append("### ❓ Follow-up Questions (for clinical staff to ask)\n")
    question_lines = []
    for q in assessment.get("recommended_questions") or []:
        if isinstance(q, dict) and q.get("question"):
            question_lines.append(str(q["question"]))
    lines.append(_bullet_list(question_lines))

    caveats = list(evidence.get("missing") or []) + list(assessment.get("warnings") or [])
    if caveats:
        lines.append("### 📝 Important Notes (Missing Data / Warnings)\n")
        lines.append(_bullet_list(caveats))

    lines.append("### 📚 References Used\n")
    refs = assessment.get("references") or []
    if refs:
        ref_lines = [
            f"[{ref.get('citation_index', '?')}] {document_name} — chunk_id: {ref.get('chunk_id', 'not available')}"
            for ref in refs
            if isinstance(ref, dict)
        ]
        lines.append(_bullet_list(ref_lines))
    else:
        lines.append(f"- Medical knowledge source: {document_name}\n")

    lines.append("---\n")
    lines.append("**Decision Summary:**  ")
    urgency = str(risk.get("overall_urgency", "")).lower()
    notify_now = bool((assessment.get("backend_flags") or {}).get("notify_clinician_now"))
    recheck = (assessment.get("follow_up") or {}).get("recheck_in_minutes")

    if notify_now or urgency == "emergency":
        lines.append("🚨 Requires immediate notification of the physician/clinical staff without delay.")
    elif urgency == "urgent":
        lines.append("Recommend presenting this case to a cardiology consultant as soon as possible (within a few hours).")
    elif urgency == "monitor":
        recheck_txt = f", with reassessment in {int(recheck)} minutes" if isinstance(recheck, (int, float)) else ""
        lines.append(
            f"Recommend presenting this case to a cardiology consultant for routine follow-up{recheck_txt}. "
            "This does not require immediate emergency intervention unless the patient develops new symptoms."
        )
    else:
        lines.append("No emergency action is required at this time; routine follow-up is sufficient.")

    return "\n".join(lines)


def render_report_or_fallback_en(
    response: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
) -> str:
    """
    English counterpart to render_report_or_fallback_ar(). This is the
    function pdf_semantic_rag.py falls back to when the LLM-based renderer
    in llm_report_renderer.py fails or times out.
    """
    assessment = response.get("assessment") if isinstance(response, dict) else None
    if assessment is None:
        reason = (
            response.get("reason")
            if isinstance(response, dict) and response.get("reason")
            else "The system could not produce a usable assessment for this event."
        )
        return (
            f"{REPORT_TITLE_EN}\n\n"
            "⚠️ **Unable to generate an automated report for this event.**\n\n"
            f"{reason}\n\n"
            "Please review this event manually with clinical staff."
        )
    return render_clinical_report_en(assessment, event=event, document_name=document_name)


def render_patient_summary_en(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
) -> str:
    """
    English counterpart to render_patient_summary_ar() — short,
    plain-language variant meant for the patient themselves.
    """
    assessment = assessment or {}
    urgency = str((assessment.get("risk_assessment") or {}).get("overall_urgency", "")).lower()
    notify_now = bool((assessment.get("backend_flags") or {}).get("notify_clinician_now"))

    lines = ["## Heart Monitoring Result\n"]
    lines.append((assessment.get("summary") or "A pattern in your heart tracing was detected that needs review.") + "\n")

    if notify_now or urgency == "emergency":
        lines.append("🚨 **Please contact your doctor or go to the nearest emergency room now.**")
    elif urgency == "urgent":
        lines.append("📞 Please contact your doctor as soon as possible today.")
    elif urgency == "monitor":
        lines.append("👍 No immediate concern. Your care team will review this result and reach out if needed.")
    else:
        lines.append("✅ This is a routine result and does not require any action from you right now.")

    questions = assessment.get("recommended_questions") or []
    if questions:
        lines.append("\nYour care team may ask you about:")
        for q in questions[:3]:
            if isinstance(q, dict) and q.get("question"):
                lines.append(f"- {q['question']}")

    return "\n".join(lines)
