EVENT_ALIASES = {
"AFIB_SUSPECTED": "AFIB",
"POSSIBLE_FLUTTER": "AFLUTTER",
"HIGH_PVC_BURDEN": "PVC",
"VT_RUN": "VT",
"BRADYCARDIA": "BRADYCARDIA",
"TACHYCARDIA": "TACHYCARDIA",
"POSSIBLE_AV_BLOCK": "AV_BLOCK",
"POSSIBLE_BBB": "BBB",
"POSSIBLE_LONG_QT": "LONG_QT",
"POSSIBLE_ISCHEMIC_PATTERN": "ISCHEMIA",
"CAD_RISK_AUGMENTED": "CAD",
"POSSIBLE_HEART_FAILURE_PATTERN": "HEART_FAILURE",
"CARDIOMYOPATHY_RISK_AUGMENTED": "CARDIOMYOPATHY",
"VALVULAR_DISEASE_RISK_AUGMENTED": "VALVULAR_DISEASE",
"MYOCARDITIS_RISK_AUGMENTED": "MYOCARDITIS"
}

class EventQueryBuilder:

    EVENT_MAP = {

        "AFIB": {

            "condition_id": "atrial_fibrillation",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "AFLUTTER": {

            "condition_id": "atrial_flutter",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "PVC": {

            "condition_id": "premature_ventricular_contractions",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "VT": {

            "condition_id": "ventricular_tachycardia",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "BRADYCARDIA": {

            "condition_id": "bradycardia",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "TACHYCARDIA": {

            "condition_id": "tachycardia",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "HEART_FAILURE": {

            "condition_id": "heart_failure",

            "core_sections": [

                "overview",

                "symptoms",

                "causes",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "HEART_ATTACK": {

            "condition_id": "heart_attack",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        },

        "LONG_QT": {

            "condition_id": "long_qt_syndrome",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "AV_BLOCK": {

            "condition_id": "atrioventricular_block",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "BUNDLE_BRANCH_BLOCK": {

            "condition_id": "left_bundle_branch_block",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "followup_questions"
            ]
        },

        "CARDIOMYOPATHY": {

            "condition_id": "cardiomyopathy",

            "core_sections": [

                "overview",

                "symptoms",

                "causes",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "MYOCARDITIS": {

            "condition_id": "myocarditis",

            "core_sections": [

                "overview",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "CAD": {

            "condition_id": "coronary_artery_disease",

            "core_sections": [

                "overview",

                "symptoms",

                "risk_factors",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "VALVULAR_DISEASE": {

            "condition_id": "valvular_disease",

            "core_sections": [

                "overview",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions"
            ]
        },

        "PE_RHS": {

            "condition_id": "pulmonary_embolism_right_heart_strain",

            "core_sections": [

                "overview",

                "ecg_clues",

                "symptoms",

                "causes",

                "complications",

                "diagnosis",

                "treatment",

                "followup_questions",

                "when_to_seek_help"
            ]
        }
    }

    @classmethod
    def build(cls, event_type):

        event_type = EVENT_ALIASES.get(
            event_type,
            event_type
        )

        return cls.EVENT_MAP.get(
            event_type.upper()
        )