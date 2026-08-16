from dataclasses import dataclass
from typing import List


@dataclass
class EventRule:
    trigger_agent: bool
    needs_rag: bool
    needs_questions: bool
    needs_patient_memory: bool
    priority: str
    cooldown_minutes: int
    escalation_level: str
    category: str
    requires_reviewer: bool
    possible_conditions: List[str]


EVENT_RULES = {
    "VT_RUN": EventRule(
        True, True, True, True,
        "critical", 1, "emergency",
        "arrhythmia", False,
        ["ventricular_tachycardia"]
    ),

    "AFIB_DETECTED": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "arrhythmia", True,
        ["atrial_fibrillation", "stroke_risk"]
    ),

    "AFLUTTER_SUSPECTED": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "arrhythmia", True,
        ["atrial_flutter"]
    ),

    "SVT_SUSPECTED": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "arrhythmia", True,
        ["supraventricular_tachycardia"]
    ),

    # NOTE (architecture review, Part 7 of the agent redesign): PVC burden is a
    # backend rate/percentage finding, not a diagnosis — there is nothing for
    # the LLM to differentially reason about at the routine 15% threshold
    # defined in temporal_analysis.calculate_abnormal_burden(). It stays a
    # deterministic, backend-only finding. If a future higher/clinically
    # significant burden threshold (or a burden+confirmed-symptom
    # combination) is added, give it its own event type with
    # trigger_agent=True rather than flipping this one back on.
    "HIGH_PVC_BURDEN": EventRule(
        False, False, False, False,
        "moderate", 30, "monitor",
        "ventricular", False,
        ["frequent_pvc", "ventricular_ectopy"]
    ),

    "BRADYCARDIA": EventRule(
        False, False, False, False,
        "moderate", 30, "monitor",
        "rate", False,
        ["sinus_bradycardia", "heart_block"]
    ),

    "TACHYCARDIA": EventRule(
        False, False, False, False,
        "moderate", 30, "monitor",
        "rate", False,
        ["sinus_tachycardia", "svt"]
    ),

    "POSSIBLE_BBB": EventRule(
        True, True, False, True,
        "low", 60, "monitor",
        "conduction", False,
        ["bundle_branch_block"]
    ),

    "POSSIBLE_AV_BLOCK": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "conduction", True,
        ["atrioventricular_block"]
    ),

    "POSSIBLE_LONG_QT": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "electrical", True,
        ["long_qt_syndrome"]
    ),

    "POSSIBLE_ISCHEMIC_PATTERN": EventRule(
        True, True, True, True,
        "high", 15, "urgent",
        "ischemia", True,
        ["ischemia", "coronary_artery_disease"]
    ),

    "CAD_RISK_AUGMENTED": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "ischemia", True,
        ["coronary_artery_disease"]
    ),

    "POSSIBLE_HEART_FAILURE_PATTERN": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "heart_failure", True,
        ["heart_failure"]
    ),

    "CARDIOMYOPATHY_RISK_AUGMENTED": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "cardiomyopathy", True,
        ["cardiomyopathy"]
    ),

    "VALVULAR_DISEASE_RISK_AUGMENTED": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "valvular", True,
        ["valvular_disease"]
    ),

    "MYOCARDITIS_CLINICAL_CORRELATION": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "inflammatory", True,
        ["myocarditis"]
    ),

    "DISEASE_WPW_PREEXCITATION": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "conduction", True,
        ["wpw_syndrome", "pre_excitation"]
    ),

    "DISEASE_BRUGADA_SYNDROME": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "channelopathy", True,
        ["brugada_syndrome"]
    ),

    "DISEASE_HCM": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "cardiomyopathy", True,
        ["hypertrophic_cardiomyopathy"]
    ),

    "DISEASE_ARVC": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "cardiomyopathy", True,
        ["arrhythmogenic_right_ventricular_cardiomyopathy"]
    ),

    "DISEASE_PULMONARY_EMBOLISM": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "pulmonary_vascular", True,
        ["pulmonary_embolism"]
    ),

    "DISEASE_PERICARDITIS": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "inflammatory", True,
        ["pericarditis"]
    ),

    "DISEASE_CARDIAC_TAMPONADE": EventRule(
        True, True, True, True,
        "critical", 10, "emergency",
        "inflammatory", True,
        ["cardiac_tamponade"]
    ),

    "DISEASE_LVH_HYPERTENSION": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "structural", True,
        ["left_ventricular_hypertrophy", "hypertension"]
    ),

    "DISEASE_PULMONARY_HYPERTENSION": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "pulmonary_vascular", True,
        ["pulmonary_hypertension"]
    ),

    "DISEASE_CARDIAC_AMYLOIDOSIS": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "cardiomyopathy", True,
        ["cardiac_amyloidosis"]
    ),

    "DISEASE_VENTRICULAR_FIBRILLATION": EventRule(
        True, True, True, True,
        "critical", 1, "emergency",
        "arrhythmia", False,
        ["ventricular_fibrillation"]
    ),

    "DISEASE_DETECTED": EventRule(
        True, True, True, True,
        "moderate", 60, "monitor",
        "disease_detector", True,
        ["ecg_detected_disease"]
    ),

    "PAUSE_DETECTED": EventRule(
        True, True, True, True,
        "high", 30, "urgent",
        "conduction", True,
        ["atrioventricular_block", "sinus_node_dysfunction"]
    ),

    "PROLONGED_ASYSTOLE": EventRule(
        True, True, True, True,
        "critical", 1, "emergency",
        "conduction", True,
        ["asystole", "severe_sinus_arrest"]
    ),

    "EXTREME_TACHYCARDIA": EventRule(
        True, True, True, True,
        "critical", 1, "emergency",
        "rate", True,
        ["extreme_tachycardia"]
    ),

    "EXTREME_BRADYCARDIA": EventRule(
        True, True, True, True,
        "critical", 1, "emergency",
        "rate", True,
        ["extreme_bradycardia"]
    ),

    "BIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "TRIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "QUADRIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "COUPLET": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "ATRIAL_BIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "ATRIAL_TRIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "ATRIAL_QUADRIGEMINY": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "ATRIAL_COUPLET": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),

    "ATRIAL_TRIPLET": EventRule(
        False, False, False, False,
        "low", 60, "none",
        "pattern", False,
        []
    ),


    # ── AV Block subtypes ─────────────────────────────────────
    "FIRST_DEGREE_AV_BLOCK": EventRule(
        False, False, False, False,
        "low", 60, "monitor",
        "conduction", False,
        ["first_degree_av_block"]
    ),

    "MOBITZ_I_AV_BLOCK": EventRule(
        True, True, True, True,
        "moderate", 30, "monitor",
        "conduction", False,
        ["mobitz_i_wenckebach"]
    ),

    "MOBITZ_II_AV_BLOCK": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "conduction", True,
        ["mobitz_ii_av_block"]
    ),

    "SECOND_DEGREE_AV_BLOCK_2TO1": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "conduction", True,
        ["second_degree_av_block_2to1"]
    ),

    "HIGH_GRADE_AV_BLOCK": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "conduction", True,
        ["high_grade_av_block"]
    ),

    "THIRD_DEGREE_AV_BLOCK": EventRule(
        True, True, True, True,
        "critical", 5, "emergency",
        "conduction", True,
        ["third_degree_av_block", "complete_heart_block"]
    ),
}


def evaluate_event(event_type: str) -> dict:

    rule = EVENT_RULES.get(event_type)

    if not rule:
        return {
            "known_event": False
        }

    return {
        "known_event": True,
        "trigger_agent": rule.trigger_agent,
        "needs_rag": rule.needs_rag,
        "needs_questions": rule.needs_questions,
        "needs_patient_memory": rule.needs_patient_memory,
        "priority": rule.priority,
        "cooldown_minutes": rule.cooldown_minutes,
        "cooldown_seconds": rule.cooldown_minutes * 60,
        "escalation_level": rule.escalation_level,
        "category": rule.category,
        "requires_reviewer": rule.requires_reviewer,
        "possible_conditions": rule.possible_conditions
    }
