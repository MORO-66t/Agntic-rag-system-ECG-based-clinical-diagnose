from database import ECGDatabase
from context_builder import ContextBuilder
from Event_Manager import evaluate_event
from beat_statistics import BeatStatistics

class AgentContextBuilder:

    def __init__(self):

        self.db = ECGDatabase()
        self.context_builder = ContextBuilder()

    def build(
        self,
        session_id,
        event_type
    ):

        event_info = evaluate_event(event_type)

        beats = self.db.get_recent_beats(
            session_id,
            limit=10
        )
        recent_events = (
        self.db.get_recent_events(
            session_id,
            limit=10
        )
        )
        recent_events = (
            self.db.get_recent_events(
                session_id
            )
        )
        
        stats = BeatStatistics.build(
            beats
        )

        knowledge = self.context_builder.build(
            event_type
        )
        if knowledge is None:

            knowledge = {
                "knowledge": [],
                "knowledge_sections": {},
                "condition_id": None
            }

        return {
        "event": event_info,
        "beats": beats,
        "knowledge": knowledge,
        "statistics": stats,
        "recent_events": recent_events
    }