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
        True, True, False, True,
        "critical", 1, "emergency",
        "arrhythmia", False,
        ["ventricular_tachycardia"]
    ),

    "AFIB_SUSPECTED": EventRule(
        True, True, True, True,
        "high", 10, "urgent",
        "arrhythmia", True,
        ["atrial_fibrillation", "stroke_risk"]
    ),

    "POSSIBLE_FLUTTER": EventRule(
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

    "HIGH_PVC_BURDEN": EventRule(
        True, True, False, True,
        "moderate", 30, "monitor",
        "ventricular", False,
        ["frequent_pvc", "ventricular_ectopy"]
    ),

    "BRADYCARDIA": EventRule(
        True, True, True, True,
        "moderate", 30, "monitor",
        "rate", False,
        ["sinus_bradycardia", "heart_block"]
    ),

    "TACHYCARDIA": EventRule(
        True, True, True, True,
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

    "MYOCARDITIS_RISK_AUGMENTED": EventRule(
        True, True, True, True,
        "high", 60, "urgent",
        "inflammatory", True,
        ["myocarditis"]
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
    )
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
        "escalation_level": rule.escalation_level,
        "category": rule.category,
        "requires_reviewer": rule.requires_reviewer,
        "possible_conditions": rule.possible_conditions
    }