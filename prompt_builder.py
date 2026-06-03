from system_prompt import SYSTEM_PROMPT

def format_beats(beats):

    lines = []

    for b in beats:

        lines.append(
            f"""
Beat {b['beat_index']}

HR={b.get('heart_rate')}
RR={b.get('rr_interval')}

QRS={b.get('qrs_width')}
QT={b.get('qt_interval')}
QTc={b.get('qtc')}

ST_Deviation={b.get('st_deviation')}

T_Wave_Peak={b.get('t_wave_peak')}
T_Wave_Min={b.get('t_wave_min')}
T_Wave_Inverted={b.get('t_wave_inverted')}

Amplitude_Mean={b.get('amplitude_mean')}
Amplitude_Std={b.get('amplitude_std')}
Amplitude_Min={b.get('amplitude_min')}
Amplitude_Max={b.get('amplitude_max')}

Peak_to_Peak={b.get('peak_to_peak')}

Signal_Quality={b.get('signal_quality_score')}

Prediction_Label={b.get('predicted_label')}
Prediction_Confidence={b.get('prediction_confidence')}

Abnormal={b.get('is_abnormal')}
"""
        )

    return "\n".join(lines)

class PromptBuilder:

    @staticmethod
    def build(context):

        event = context["event"]

        beats = format_beats(
            context["beats"]
        )
        recent_events = context["recent_events"]
        knowledge = context["knowledge"]

        knowledge_text = ""

        for chunk in knowledge["knowledge"]:

            knowledge_text += f"""

SECTION:
{chunk["section"]}

CONTENT:
{chunk["content"]}

"""

        user_prompt = f"""

EVENT INFORMATION

{event}

==================================================

 RECENT EVENT
 
 {recent_events}
 
 ==================================================

RECENT ECG BEATS

{beats}

==================================================
STATISTICS

{context["statistics"]}

===================================================
MEDICAL KNOWLEDGE

{knowledge_text}



==================================================

Perform full reasoning.



"""

        return {
            "system": SYSTEM_PROMPT,
            "user": user_prompt
        }
    
