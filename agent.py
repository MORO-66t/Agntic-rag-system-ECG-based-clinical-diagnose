from agent_context_builder import (
    AgentContextBuilder
)
from database import ECGDatabase

from prompt_builder import (
    PromptBuilder
)

from llm_client import (
    LLMClient
)


class ECGAgent:

    def __init__(self):

        self.context_builder = (
            AgentContextBuilder()
        )

        self.llm = (
            LLMClient()
        )
        self.db = ECGDatabase()
    def analyze(

        self,

        session_id,

        event_type

    ):

        context = (
            self.context_builder.build(
                session_id,
                event_type
            )
        )

        prompt = (
            PromptBuilder.build(
                context
            )
        )
        print("BEFORE LLM")
        response = (
            self.llm.generate(
                prompt["system"],
                prompt["user"]
            )
        )
        print(prompt["user"][:4000])
        print("AFTER LLM")
        print("RESPONSE LENGTH:", len(str(response)))
        response = (
            self.llm.generate(
                prompt["system"],
                prompt["user"]
            )
        )

        self.db.insert_agent_interaction(

            patient_id="demo_patient",

            session_id=session_id,

            event_type=event_type,

            prompt_json={
                "system": prompt["system"],
                "user": prompt["user"]
            },

            retrieved_chunks=
                context["knowledge"],

            response_json={
                "response": response
            }
        )

        return response