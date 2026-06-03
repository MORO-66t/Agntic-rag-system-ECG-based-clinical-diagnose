from database import ECGDatabase
from event_query_builder import EventQueryBuilder
import database
class ContextBuilder:

    def __init__(self):

        self.db = ECGDatabase()

    def build(
        self,
        event_type
    ):

        event_data = EventQueryBuilder.build(
            event_type
        )
        # recent_events = self.db.get_session_events(
        #     session_id
        # )
    #     recent_events = (
    #     self.db.get_recent_events(
    #         session_id
    #     )
    # )

        if not event_data:

            return None

        chunks = self.db.get_condition_sections(
            condition_id=event_data["condition_id"],
            sections=event_data["core_sections"]
        )

        return {

            "event_type":
                event_type,

            "condition_id":
                event_data["condition_id"],
            # "recent_events": recent_events,

            "knowledge":
                [dict(x) for x in chunks],

            "knowledge_sections":
                {
                    x["section"]: x["content"]
                    for x in chunks
                }
        }